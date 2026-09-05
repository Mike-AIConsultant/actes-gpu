# actes-cat GPU service: BSC Catalan faster-whisper + NVIDIA streaming Sortformer + pyannote.
# Everything is baked in so a cold start never downloads a model.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/hf \
    MODELS_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- python deps -------------------------------------------------------------
# torch first (biggest, most stable layer) so later edits do not rebuild it
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir \
        "faster-whisper==1.2.1" \
        soundfile numpy runpod requests huggingface_hub
# NeMo ASR brings Sortformer. cuda-python is pinned: newer builds break nemo's import.
RUN pip3 install --no-cache-dir "nemo_toolkit[asr]==2.4.0" || \
    pip3 install --no-cache-dir "nemo_toolkit[asr]"
RUN pip3 install --no-cache-dir "pyannote.audio>=4.0"

# ---- models ------------------------------------------------------------------
# BSC-LT Catalan faster-whisper (CTranslate2), ~3 GB. Same model as the CPU worker.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h', \
  local_dir='/models/bsc-punct', \
  allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.json','preprocessor_config.json'])"

# NVIDIA streaming Sortformer v2 (4 speaker ceiling, 2.8 GB of card, 8.6 s on 47 min audio).
# Only the .nemo checkpoint: the gguf and the figures are dead weight in the image.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('nvidia/diar_streaming_sortformer_4spk-v2', local_dir='/models/sortformer', \
  allow_patterns=['*.nemo'])"

# pyannote Community-1, open mirror, no token. Used when Sortformer hits its 4 speaker ceiling.
# Downloaded into the HF cache (not a local dir) so Pipeline.from_pretrained works offline.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('pyannote-community/speaker-diarization-community-1', \
  allow_patterns=['config.yaml','*/pytorch_model.bin','*/*.npz'])"

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
