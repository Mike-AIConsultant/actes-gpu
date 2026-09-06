"""Get one model from the shelf onto local disk in a form faster-whisper can load.

Used twice: at image build time for the models that are baked in, and at run time the first
time somebody picks one that is not. Both paths go through `ensure()`, so a lazily fetched
model is byte-for-byte what the build would have produced.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from registry import CACHE_DIR, CT2_FILES, MODELS

_LOCK = threading.Lock()


def _log(msg):
    print(f"[models] {msg}", flush=True)


def _download(repo, dst, allow=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=str(dst), allow_patterns=allow)


def _convert(src, dst):
    """Plain Whisper checkpoint -> CTranslate2 float16, plus the tokenizer.json that
    faster-whisper looks for (several of these repos ship only the slow tokenizer files)."""
    subprocess.run(
        ["ct2-transformers-converter", "--model", str(src), "--output_dir", str(dst),
         "--quantization", "float16", "--copy_files", "preprocessor_config.json"],
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
    tmp = Path(str(dst) + f".tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        if spec["format"] == "ct2":
            _log(f"{key}: downloading {spec['repo']} (already CTranslate2)")
            _download(spec["repo"], tmp, allow=CT2_FILES)
        else:
            raw = Path(str(dst) + f".raw-{os.getpid()}")
            shutil.rmtree(raw, ignore_errors=True)
            raw.mkdir(parents=True, exist_ok=True)
            try:
                _log(f"{key}: downloading {spec['repo']}")
                _download(spec["repo"], raw, allow=spec.get("allow"))
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
    _log(f"{key}: ready at {dst} in {time.time() - t0:.0f}s")


def ensure(key):
    """Returns (path, seconds_spent_fetching). Zero seconds means it was already there."""
    spec = MODELS[key]
    baked = spec.get("baked")
    if baked and (Path(baked) / "model.bin").exists():
        return baked, 0.0

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
    # build time: python convert.py ca mixed-es gl
    for k in sys.argv[1:]:
        ensure(k)
