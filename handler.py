"""actes-cat GPU service: one RunPod serverless job = one meeting.

It does the three expensive steps and nothing else:
  1. fetch the audio (a signed, short lived URL served by the Arsys box)
  2. ffmpeg to 16 kHz mono
  3. transcribe with the BSC Catalan model on the card
  4. separate speakers, Sortformer by default, pyannote when Sortformer hits its ceiling

It stores nothing. Everything it downloads lives under /tmp and is deleted before the job
returns. All of the words, all of the timings and all of the speaker turns go back to Arsys,
which owns the database, the audio and the acta.
"""
import base64
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import runpod

SR = 16000
ASR_MODEL_DIR = os.environ.get("ASR_MODEL_DIR", "/models/bsc-punct")
SORTFORMER_PATH = os.environ.get(
    "SORTFORMER_PATH", "/models/sortformer/diar_streaming_sortformer_4spk-v2.nemo"
)
PYANNOTE_REPO = os.environ.get(
    "PYANNOTE_REPO", "pyannote-community/speaker-diarization-community-1"
)
# Sortformer v2 can only tell four people apart. If it reports exactly four, "four" may really
# mean "four or more", so we do not trust it and run pyannote instead, which has no ceiling.
SORTFORMER_CEILING = 4

# Same settings as the CPU worker on Arsys. Only device and compute_type differ.
TRANSCRIBE_OPTS = dict(
    beam_size=5,
    vad_filter=True,
    word_timestamps=True,
    condition_on_previous_text=False,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
)

_ASR = None
_SORTFORMER = None
_PYANNOTE = None


# ------------------------------------------------------------------ models (loaded once)
def get_asr():
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel

        _ASR = WhisperModel(ASR_MODEL_DIR, device="cuda", compute_type="float16")
    return _ASR


def get_sortformer():
    global _SORTFORMER
    if _SORTFORMER is None:
        from nemo.collections.asr.models import SortformerEncLabelModel

        m = SortformerEncLabelModel.restore_from(
            restore_path=SORTFORMER_PATH, map_location="cuda", strict=False
        )
        m.eval()
        # The streaming settings NVIDIA publishes as "high latency", the accurate end of the
        # range. This is an offline meeting, so there is no reason to trade accuracy for lag.
        try:
            mods = m.sortformer_modules
            mods.chunk_len = 124
            mods.chunk_right_context = 1
            mods.fifo_len = 124
            mods.spkcache_len = 188
            mods.spkcache_refresh_rate = 144
            mods.spkcache_update_period = 144
        except Exception as e:  # a newer NeMo may rename these; defaults still work
            print(f"sortformer: could not set streaming params ({e}), using defaults", flush=True)
        _SORTFORMER = m
    return _SORTFORMER


def get_pyannote():
    global _PYANNOTE
    if _PYANNOTE is None:
        import torch
        from pyannote.audio import Pipeline

        p = Pipeline.from_pretrained(PYANNOTE_REPO)
        p.to(torch.device("cuda"))
        _PYANNOTE = p
    return _PYANNOTE


# ------------------------------------------------------------------ audio in
def fetch_audio(inp, workdir: Path):
    """Returns (path, bytes, seconds_spent)."""
    t0 = time.time()
    dst = workdir / "input.bin"
    if inp.get("audio_b64"):
        dst.write_bytes(base64.b64decode(inp["audio_b64"]))
    elif inp.get("audio_url"):
        import requests

        with requests.get(inp["audio_url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    else:
        raise ValueError("no audio_url and no audio_b64 in the job input")
    return dst, dst.stat().st_size, round(time.time() - t0, 2)


def to_wav16k(src: Path, workdir: Path) -> Path:
    wav = workdir / "audio.16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(wav)],
        check=True,
        capture_output=True,
    )
    return wav


def audio_seconds(wav: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(wav)).duration)


# ------------------------------------------------------------------ transcription
def transcribe(wav: Path, language="ca"):
    model = get_asr()
    segments, _info = model.transcribe(str(wav), language=language, **TRANSCRIBE_OPTS)
    words = []
    for i, seg in enumerate(segments):
        if seg.words:
            for w in seg.words:
                words.append(
                    {"start": round(w.start, 2), "end": round(w.end, 2), "text": w.word,
                     "segment_id": i}
                )
        else:
            # alignment produced nothing for this segment: keep the text as one long "word",
            # exactly as the CPU worker does, so downstream code sees the same shape
            words.append(
                {"start": round(float(seg.start), 2), "end": round(float(seg.end), 2),
                 "text": " " + seg.text.strip(), "segment_id": i}
            )
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


# ------------------------------------------------------------------ speakers
def _parse_sortformer(preds):
    """NeMo has moved this return shape around between releases, so accept every form it has
    used: a list of "start end speaker" strings, a list of (start, end, label) tuples, or a
    frame level tensor of per speaker probabilities. Returns [(start, end, label), ...]."""
    if preds is None:
        return []
    # diarize() returns one entry per input file
    if isinstance(preds, (list, tuple)) and len(preds) == 1 and isinstance(preds[0], (list, tuple)):
        preds = preds[0]

    turns = []
    for item in preds:
        if isinstance(item, str):
            parts = item.replace(",", " ").split()
            if len(parts) >= 3:
                turns.append((float(parts[0]), float(parts[1]), str(parts[2])))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            turns.append((float(item[0]), float(item[1]), str(item[2])))
            continue
    if turns:
        return turns

    # last resort: a tensor shaped (frames, speakers) of probabilities
    try:
        import numpy as np

        arr = preds
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return []
        frame_s = 0.08  # Sortformer emits one frame every 80 ms
        active = arr > 0.5
        for spk in range(arr.shape[1]):
            on = None
            for f in range(arr.shape[0]):
                if active[f, spk] and on is None:
                    on = f
                elif not active[f, spk] and on is not None:
                    turns.append((on * frame_s, f * frame_s, f"speaker_{spk}"))
                    on = None
            if on is not None:
                turns.append((on * frame_s, arr.shape[0] * frame_s, f"speaker_{spk}"))
        turns.sort(key=lambda t: t[0])
    except Exception as e:
        print(f"sortformer: could not parse output: {e}", flush=True)
    return turns


def diarize_sortformer(wav: Path):
    model = get_sortformer()
    import torch

    with torch.inference_mode():
        try:
            preds = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=False)
        except TypeError:
            # older / newer NeMo without that keyword
            preds = model.diarize(audio=[str(wav)], batch_size=1)
    return _parse_sortformer(preds)


def diarize_pyannote(wav: Path):
    import numpy as np
    import soundfile as sf
    import torch

    pipe = get_pyannote()
    samples, sr = sf.read(str(wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    out = pipe({"waveform": waveform, "sample_rate": sr})
    # pyannote 4.x hands back a DiarizeOutput, 3.x an Annotation
    ann = getattr(out, "speaker_diarization", out)
    return [(float(seg.start), float(seg.end), str(label))
            for seg, _track, label in ann.itertracks(yield_label=True)]


def count_speakers(turns):
    return len({spk for _s, _e, spk in turns})


def diarize(wav: Path, engine="auto"):
    """Returns (turns, engine_used, per_engine_seconds)."""
    seconds = {}
    if engine == "none":
        return [], "none", seconds

    if engine in ("auto", "sortformer"):
        t0 = time.time()
        turns = diarize_sortformer(wav)
        seconds["sortformer_s"] = round(time.time() - t0, 2)
        n = count_speakers(turns)
        if engine == "sortformer" or (turns and n < SORTFORMER_CEILING):
            return turns, "sortformer", seconds
        # n == 4 means "four, or more than four": Sortformer cannot say. n == 0 means it
        # found nobody. Either way, ask pyannote, which has no ceiling.
        print(f"sortformer reported {n} speakers, falling back to pyannote", flush=True)

    t0 = time.time()
    turns = diarize_pyannote(wav)
    seconds["pyannote_s"] = round(time.time() - t0, 2)
    return turns, "pyannote", seconds


# ------------------------------------------------------------------ self test
def selftest():
    """Runs a short synthetic file through everything so one cheap call proves the image is
    sound, without shipping real audio anywhere."""
    out = {"ok": True, "steps": {}}
    work = Path(tempfile.mkdtemp(prefix="selftest-"))
    try:
        wav = work / "tone.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "sine=frequency=200:duration=12", "-ac", "1", "-ar", str(SR), str(wav)],
            check=True, capture_output=True,
        )
        for name, fn in (
            ("asr", lambda: len(transcribe(wav))),
            ("sortformer", lambda: len(diarize_sortformer(wav))),
            ("pyannote", lambda: len(diarize_pyannote(wav))),
        ):
            t0 = time.time()
            try:
                out["steps"][name] = {"ok": True, "n": fn(), "s": round(time.time() - t0, 2)}
            except Exception as e:
                out["ok"] = False
                out["steps"][name] = {"ok": False, "error": f"{e.__class__.__name__}: {e}",
                                      "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out["gpu"] = gpu_name()
    out["where"] = where_am_i()
    return out


def where_am_i():
    """Which RunPod datacentre this worker is in. Written into every job's timings so the
    claim that the audio was processed inside the European Union is auditable per meeting."""
    return {
        "datacenter": os.environ.get("RUNPOD_DC_ID") or os.environ.get("RUNPOD_DATACENTER_ID") or "unknown",
        "region": os.environ.get("RUNPOD_REGION") or "",
        "worker_id": os.environ.get("RUNPOD_POD_ID") or "",
    }


def gpu_name():
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ the job
def handler(job):
    inp = job.get("input") or {}
    if inp.get("selftest"):
        return selftest()

    t_all = time.time()
    work = Path(tempfile.mkdtemp(prefix="job-"))
    timings = {}
    try:
        src, size_bytes, timings["download_s"] = fetch_audio(inp, work)

        t0 = time.time()
        wav = to_wav16k(src, work)
        timings["convert_s"] = round(time.time() - t0, 2)
        try:
            src.unlink()
        except OSError:
            pass
        audio_s = audio_seconds(wav)

        t0 = time.time()
        get_asr()
        timings["asr_load_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        words = transcribe(wav, language=inp.get("language") or "ca")
        timings["transcribe_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        turns, engine, engine_s = diarize(wav, inp.get("diar_engine") or "auto")
        timings["diarize_s"] = round(time.time() - t0, 2)
        timings.update(engine_s)

        timings["total_s"] = round(time.time() - t_all, 2)
        return {
            "ok": True,
            "words": words,
            "turns": [[round(a, 2), round(b, 2), spk] for a, b, spk in turns],
            "diar_engine": engine,
            "speakers_raw": count_speakers(turns),
            "audio_s": round(audio_s, 2),
            "audio_bytes": size_bytes,
            "gpu": gpu_name(),
            "where": where_am_i(),
            "timings": timings,
        }
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": f"{e.__class__.__name__}: {e}",
                "trace": traceback.format_exc()[-2000:], "timings": timings}
    finally:
        # nothing this job touched survives it
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST"):
        print(json.dumps(handler({"input": json.loads(os.environ["LOCAL_TEST"])}), indent=1)[:4000])
    else:
        runpod.serverless.start({"handler": handler})
