#!/usr/bin/env bash
# Creates (or recreates) the RunPod serverless endpoint that writes the acta (meeting minutes)
# with Qwen3.8-27B. Separate endpoint from the speech one (create-endpoint.sh): different image,
# different card budget, different idle timing. Modelled on create-endpoint.sh, same EU-member-
# state-only pinning and the same GraphQL trap.
#
#   RUNPOD_API_KEY=... ./create-qwen-endpoint.sh
#
# Prints the endpoint id. Put it in /root/actes-cat/.env as QWEN_ENDPOINT_ID.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
IMAGE="${IMAGE:-ghcr.io/mike-aiconsultant/actes-gpu:qwen-latest}"
NAME="${NAME:-actes-cat-qwen}"

api() { curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }

echo "creating template for $IMAGE"
TEMPLATE=$(api -X POST https://rest.runpod.io/v1/templates -d "$(cat <<JSON
{
  "name": "$NAME-$(date +%s)",
  "imageName": "$IMAGE",
  "isServerless": true,
  "containerDiskInGb": 60,
  "env": {
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "MAX_MODEL_LEN": "65536",
    "GPU_MEMORY_UTILIZATION": "0.96",
    "MAX_NUM_SEQS": "2",
    "VLLM_STARTUP_TIMEOUT": "3000"
  }
}
JSON
)")
TEMPLATE_ID=$(echo "$TEMPLATE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "template: $TEMPLATE_ID"

echo "creating endpoint (EU member states only, 24 GB cards only)"
# 24 GB cards ONLY, unlike the speech endpoint. The model is baked to fit a 24 GB card with
# room to spare (17.0 GiB weights + non-torch, 5.19 GiB KV cache); a 48 GB card would just cost
# more for the same job. NVIDIA L40S and RTX A6000 (both 48 GB) are deliberately left out.
ENDPOINT=$(api -X POST https://rest.runpod.io/v1/endpoints -d "$(cat <<JSON
{
  "name": "$NAME",
  "templateId": "$TEMPLATE_ID",
  "computeType": "GPU",
  "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA L4"],
  "dataCenterIds": ["EU-FR-1", "EU-NL-1", "EU-SE-1", "EU-RO-1", "EU-CZ-1"],
  "minCudaVersion": "12.8",
  "workersMin": 0,
  "workersMax": 1,
  "idleTimeout": 90,
  "flashboot": true,
  "executionTimeoutMs": 1800000,
  "scalerType": "QUEUE_DELAY",
  "scalerValue": 4
}
JSON
)")
ENDPOINT_ID=$(echo "$ENDPOINT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# The REST API accepts dataCenterIds, returns 200, and does nothing with it (proved twice now,
# once on the speech endpoint and once in round 2 of this build). Only the GraphQL mutation
# actually pins the locations. AMPERE_24 + ADA_24 is the GraphQL name for exactly the same
# 24 GB tier as the gpuTypeIds list above -- do not widen this to AMPERE_48/ADA_48.
echo "pinning locations and GPU tier through GraphQL (REST ignores dataCenterIds)"
Q="mutation { saveEndpoint(input: { id: \"$ENDPOINT_ID\", name: \"$NAME\", templateId: \"$TEMPLATE_ID\", gpuIds: \"AMPERE_24,ADA_24\", locations: \"EU-FR-1,EU-NL-1,EU-SE-1,EU-RO-1,EU-CZ-1\", idleTimeout: 90, scalerType: \"QUEUE_DELAY\", scalerValue: 4, workersMin: 0, workersMax: 1 }) { id locations gpuIds } }"
echo "GraphQL pinning response:"
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"query":sys.argv[1]}))' "$Q")" \
  https://api.runpod.io/graphql

echo
echo "reading it back through REST too (both must agree, see the qwen round 2 worklog entry)"
api "https://rest.runpod.io/v1/endpoints/$ENDPOINT_ID"

echo
echo "ENDPOINT_ID=$ENDPOINT_ID"
echo "Now check a real job's response.where.datacenter: an EU-* value is the only proof pinning worked."
