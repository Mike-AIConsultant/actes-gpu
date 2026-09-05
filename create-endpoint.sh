#!/usr/bin/env bash
# Creates (or recreates) the RunPod serverless endpoint that does the transcription.
#
# EU ONLY, and deliberately EU MEMBER STATES only: Iceland and Norway are in the European
# Economic Area but not in the European Union, and the privacy page says European Union.
#
#   RUNPOD_API_KEY=... ./create-endpoint.sh
#
# Prints the endpoint id. Put it in /root/actes-cat/.env as GPU_ENDPOINT_ID.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
IMAGE="${IMAGE:-ghcr.io/mike-aiconsultant/actes-gpu:latest}"
NAME="${NAME:-actes-cat-gpu}"

api() { curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@"; }

echo "creating template for $IMAGE"
TEMPLATE=$(api -X POST https://rest.runpod.io/v1/templates -d "$(cat <<JSON
{
  "name": "$NAME-$(date +%s)",
  "imageName": "$IMAGE",
  "isServerless": true,
  "containerDiskInGb": 60,
  "env": {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
}
JSON
)")
TEMPLATE_ID=$(echo "$TEMPLATE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
echo "template: $TEMPLATE_ID"

echo "creating endpoint (EU member states only)"
ENDPOINT=$(api -X POST https://rest.runpod.io/v1/endpoints -d "$(cat <<JSON
{
  "name": "$NAME",
  "templateId": "$TEMPLATE_ID",
  "computeType": "GPU",
  "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                 "NVIDIA L4", "NVIDIA L40S", "NVIDIA RTX A6000"],
  "dataCenterIds": ["EU-FR-1", "EU-NL-1", "EU-SE-1", "EU-RO-1", "EU-CZ-1"],
  "workersMin": 0,
  "workersMax": 2,
  "idleTimeout": 30,
  "flashboot": true,
  "executionTimeoutMs": 1800000,
  "scalerType": "QUEUE_DELAY",
  "scalerValue": 4
}
JSON
)")
ENDPOINT_ID=$(echo "$ENDPOINT" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# The REST API accepts dataCenterIds, returns 200, and does nothing with it. Verified 2026-09-06:
# the endpoint came back with no restriction at all. Only the GraphQL mutation actually pins the
# locations, so it is applied here as well and must not be removed.
echo "pinning locations through GraphQL (REST ignores dataCenterIds)"
Q="mutation { saveEndpoint(input: { id: \"$ENDPOINT_ID\", name: \"$NAME\", templateId: \"$TEMPLATE_ID\", gpuIds: \"AMPERE_24,ADA_24\", locations: \"EU-FR-1,EU-NL-1,EU-SE-1,EU-RO-1,EU-CZ-1\", idleTimeout: 30, scalerType: \"QUEUE_DELAY\", scalerValue: 4, workersMin: 0, workersMax: 2 }) { id locations gpuIds } }"
curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys;print(json.dumps({"query":sys.argv[1]}))' "$Q")" \
  https://api.runpod.io/graphql

echo
echo "ENDPOINT_ID=$ENDPOINT_ID"
echo "Now check a real job's gpu_datacenter: an EU-* value is the only proof the pinning worked."
