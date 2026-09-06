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
    "MAX_MODEL_LEN": "40960",
    "GPU_MEMORY_UTILIZATION": "0.96",
    "MAX_NUM_SEQS": "2",
    "VLLM_STARTUP_TIMEOUT": "3000"
  }
}
JSON
)")
TEMPLATE_ID=$(echo "$TEMPLATE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "template: $TEMPLATE_ID"

echo "creating endpoint (EU member states only, 48 GB cards)"
# 48 GB cards (RTX A6000, A40), NOT 24 GB. Proved the hard way on 2026-09-06 with a diagnostic
# pod (gpu/qwen -- see the "qwen image" worklog entry): a 24 GB card only has room for the
# weights (17.9 GiB) plus activation plus CUDA graphs with almost nothing left for a 40,960
# token KV cache, so vLLM refuses to start and the serverless worker restarts forever with NO
# visible error through any API (ready -> running -> ready, queue never drains). RunPod's own
# 24 GB "4090" and "L4" tiers on serverless do not have the headroom a bare-metal 4090 pod does.
# A 48 GB card gives 25.6 GiB of KV cache, nine times what one request needs, for 12 cents/hr
# more ($1.22 vs $1.10). Do not shrink this back to 24 GB without re-measuring on THIS specific
# serverless tier (not a diagnostic pod) first.
ENDPOINT=$(api -X POST https://rest.runpod.io/v1/endpoints -d "$(cat <<JSON
{
  "name": "$NAME",
  "templateId": "$TEMPLATE_ID",
  "computeType": "GPU",
  "gpuTypeIds": ["NVIDIA RTX A6000", "NVIDIA A40"],
  "dataCenterIds": ["EU-FR-1", "EU-NL-1", "EU-SE-1", "EU-RO-1", "EU-CZ-1"],
  "minCudaVersion": "13.0",
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
# actually pins the locations. AMPERE_48 is the GraphQL name matching RTX A6000 / A40 -- do not
# narrow this back to a 24 GB tier without re-measuring MAX_MODEL_LEN on serverless (not a pod).
echo "pinning locations and GPU tier through GraphQL (REST ignores dataCenterIds)"
Q="mutation { saveEndpoint(input: { id: \"$ENDPOINT_ID\", name: \"$NAME\", templateId: \"$TEMPLATE_ID\", gpuIds: \"AMPERE_48\", locations: \"EU-FR-1,EU-NL-1,EU-SE-1,EU-RO-1,EU-CZ-1\", idleTimeout: 90, scalerType: \"QUEUE_DELAY\", scalerValue: 4, workersMin: 0, workersMax: 1 }) { id locations gpuIds } }"
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
