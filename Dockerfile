# actes-cat GPU service: the Iberian speech-model shelf + NVIDIA streaming Sortformer + pyannote.
#
# What is baked in and what is not (the rule, and the reason):
#   baked      a model that would otherwise have to be CONVERTED at run time, which costs
#              minutes on every fresh worker. Converting once here is the only sane place.
#   on demand  a model the publisher already ships in CTranslate2 form. Fetching one of those
#              is a plain 3.1 GB file copy, about a minute on a RunPod machine, once per
#              worker, and keeping them out holds the image (and so the cold start) down.
# The one exception is Basque, which needs converting AND is rare on a Catalan meeting site,
# so it is left on demand and pays about six minutes the first time somebody picks it.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/models/hf \
    MODELS_DIR=/models \
    PYTHONPATH=/app

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
# transformers is what ct2-transformers-converter reads a plain Whisper checkpoint with
RUN pip3 install --no-cache-dir "transformers>=4.44"

# ---- speaker models ----------------------------------------------------------
# NVIDIA streaming Sortformer v2 (4 speaker ceiling, 2.8 GB of card, 8.6 s on 47 min audio).
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('nvidia/diar_streaming_sortformer_4spk-v2', local_dir='/models/sortformer', \
  allow_patterns=['*.nemo'])"

# pyannote Community-1, open mirror, no token. Used when Sortformer hits its 4 speaker ceiling.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('pyannote-community/speaker-diarization-community-1', \
  allow_patterns=['config.yaml','*/pytorch_model.bin','*/*.npz'])"

# ---- the speech-model shelf --------------------------------------------------
WORKDIR /app
COPY registry.py convert.py /app/

# ca: the Catalan default, already CTranslate2, baked because every job that does not choose
#     anything uses it. Skipping it would mean a download on the commonest path.
# mixed-es / es: BSC "Languages of Spain", punctuated. Needs converting, and Catalan mixed
#     with Spanish is the second commonest meeting here.
# gl: Galician turbo. Needs converting, but turbo is small (1.6 GB converted), so baking it
#     is nearly free and removes a run-time conversion.
RUN python /app/convert.py ca mixed-es gl && du -sh /models/*

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
