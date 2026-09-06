# actes-cat Qwen acta writer: Qwen3.8-27B baked into the stock RunPod vLLM worker image.
#
# Separate from ../Dockerfile (the speech image) on purpose. Nothing here touches that image,
# so transcription cold starts are unaffected. This image is only the LLM that writes the acta
# (meeting minutes) text once a transcript exists.
#
# Baking, not a network volume: on a fresh RunPod machine the model would otherwise download
# from Hugging Face before it could answer anything, and that download is unpredictable
# (30 seconds on some machines, 25+ minutes on others, unusable). A network volume fixes the
# download but locks every worker to one datacentre, shrinking the EU pool from five locations
# to one. Baking is free and keeps all five.
#
# Quantisation: RedHatAI/Qwen3.8-27B-INT4 (compressed-tensors W4A16), 19.45 GB. This is the
# build that actually fits a 24 GB card: vLLM's own startup log measured weights + non-torch at
# 17.0 GiB, peak activation 0.39 GiB, KV cache 5.19 GiB (73,728+ tokens). Do NOT swap in
# barrydeen/Qwen3.8-27B-AWQ-4bit (27.74 GB) -- it leaves 48 DeltaNet layers in BF16 and does not
# fit a 24 GB card. See the "qwen round 2" worklog entry for the full measurement.
FROM runpod/worker-v1-vllm:v2.26.0

# Where the model bakes to. Deliberately NOT /runpod-volume (the base image's default): that
# path only exists when a network volume is mounted at runtime, which this image does not use.
# /models lives inside the image layer itself, so the weights are there with no volume at all.
ARG MODEL_NAME=RedHatAI/Qwen3.8-27B-INT4
ARG BASE_PATH=/models

ENV MODEL_NAME=$MODEL_NAME \
    BASE_PATH=$BASE_PATH \
    HF_HOME=${BASE_PATH}/huggingface-cache/hub \
    HUGGINGFACE_HUB_CACHE=${BASE_PATH}/huggingface-cache/hub \
    HF_DATASETS_CACHE=${BASE_PATH}/huggingface-cache/datasets \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    # Runtime defaults. The endpoint template can still override any of these; baking them
    # here just means a plain `docker run` of this image also comes up sane.
    MAX_MODEL_LEN=65536 \
    GPU_MEMORY_UTILIZATION=0.96 \
    MAX_NUM_SEQS=2 \
    VLLM_STARTUP_TIMEOUT=3000
    # ENFORCE_EAGER is deliberately left UNSET. With it on, generation measured 12.6 tok/s;
    # with CUDA graphs (the default when unset), 49-50 tok/s. Do not set this env var.

# Same snapshot_download pattern the speech image uses for the speaker models
# (../Dockerfile:42-51), except here it is the base image's own download_model.py, which also
# writes /local_model_args.json so main.py points `vllm serve` at the local snapshot instead of
# a Hugging Face repo id. No HF_TOKEN needed: this model is public.
RUN python3 /src/download_model.py

# Add the RunPod datacentre to every response, mirroring where_am_i() in ../handler.py, so a
# European-mode acta has proof it was actually written inside the EU.
COPY handler.py /src/handler.py
