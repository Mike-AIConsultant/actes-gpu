"""Get one model from the shelf onto local disk in a form faster-whisper can load.

Used twice: at image build time for the models that are baked in, and at run time the first
time somebody picks one that is not. Both paths go through `ensure()`, so a lazily fetched
model is byte-for-byte what the build would have produced.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from registry import CACHE_DIR, CONVERT_IGNORE, CT2_FILES, MODELS, canonical

_LOCK = threading.Lock()


def _log(msg):
    print(f"[models] {msg}", flush=True)


# Hugging Face rate-limits anonymous downloads per IP, and both a shared CI runner and a
# RunPod machine are shared IPs. A 429 is a "come back in a minute", not a real failure, so
# it is retried rather than allowed to fail a build or a meeting.
DOWNLOAD_ATTEMPTS = int(os.environ.get("SHELF_DOWNLOAD_ATTEMPTS", "5"))
DOWNLOAD_BACKOFF_S = float(os.environ.get("SHELF_DOWNLOAD_BACKOFF_S", "20"))


def _download(repo, dst, allow=None, ignore=None):
    """One download from the Hub, retried on the transient failures.

    The RunPod template sets HF_HUB_OFFLINE=1 on purpose: it guarantees that loading the
    baked models (whisper, sortformer, pyannote) can never quietly reach for the network.
    That flag would also block the shelf, so it is lifted for the length of this one call
    and put back afterwards. Nothing else in the process ever sees it unset.
    """
    from huggingface_hub import snapshot_download

    saved = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        last = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                snapshot_download(repo, local_dir=str(dst),
                                  allow_patterns=allow, ignore_patterns=ignore,
                                  # gentler than the default 8: fewer parallel requests is
                                  # the difference between being rate-limited and not
                                  max_workers=4)
                return
            except Exception as e:
                last = e
                transient = any(w in f"{e.__class__.__name__}: {e}"
                                for w in ("429", "Too Many Requests", "504", "503", "502",
                                          "Timeout", "timed out", "Connection",
                                          "IncompleteRead", "ChunkedEncoding"))
                if attempt == DOWNLOAD_ATTEMPTS or not transient:
                    raise
                wait = DOWNLOAD_BACKOFF_S * (2 ** (attempt - 1))
                _log(f"{repo}: {e.__class__.__name__} on attempt {attempt}, "
                     f"retrying in {wait:.0f}s")
                time.sleep(wait)
        raise last
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _convert(src, dst):
    """Plain Whisper checkpoint -> CTranslate2 float16, plus the tokenizer.json that
    faster-whisper looks for (several of these repos ship only the slow tokenizer files).

    --force because dst is created by the caller: without it the converter refuses to write
    into a directory that already exists.
    """
    subprocess.run(
        ["ct2-transformers-converter", "--model", str(src), "--output_dir", str(dst),
         "--quantization", "float16", "--force",
         "--copy_files", "preprocessor_config.json"],
        check=True,
    )
    if not (Path(dst) / "tokenizer.json").exists():
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(str(src)).save_pretrained(str(dst))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        s = Path(src) / name
        if s.exists() and not (Path(dst) / name).exists():
            shutil.copy2(s, Path(dst) / name)


def _build(key, spec, dst):
    """Produce a ready-to-load model directory at dst. Writes to a temp dir and renames,
    so a half-finished download can never look like a good model."""
    tmp = Path(f"{dst}.tmp-{os.getpid()}")
    raw = Path(f"{dst}.raw-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(raw, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        if spec["format"] == "ct2":
            _log(f"{key}: downloading {spec['repo']} (already CTranslate2)")
            _download(spec["repo"], tmp, allow=CT2_FILES)
        else:
            raw.mkdir(parents=True, exist_ok=True)
            try:
                _log(f"{key}: downloading {spec['repo']}")
                _download(spec["repo"], raw, ignore=CONVERT_IGNORE)
                _log(f"{key}: converting to CTranslate2 float16")
                _convert(raw, tmp)
            finally:
                shutil.rmtree(raw, ignore_errors=True)
        if not (tmp / "model.bin").exists():
            raise RuntimeError(f"{key}: no model.bin after build")
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(raw, ignore_errors=True)
    _log(f"{key}: ready at {dst} in {time.time() - t0:.0f}s")


def ensure(key):
    """Returns (path, seconds_spent_fetching). Zero seconds means it was already there."""
    key = canonical(key)
    if key is None:
        raise KeyError("unknown model key")
    spec = MODELS[key]

    # Several keys can share one set of weights (mixt and es are the same model with a
    # different language hint), so the path, not the key, is what is looked for on disk.
    baked = spec.get("baked")
    dst = Path(baked) if baked else Path(CACHE_DIR) / key
    if (dst / "model.bin").exists():
        return str(dst), 0.0

    with _LOCK:                       # a second job on the same worker waits, does not refetch
        if (dst / "model.bin").exists():
            return str(dst), 0.0
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _build(key, spec, dst)
        return str(dst), round(time.time() - t0, 1)


if __name__ == "__main__":
    # build time: python convert.py ca mixt gl
    for k in sys.argv[1:]:
        ensure(k)
