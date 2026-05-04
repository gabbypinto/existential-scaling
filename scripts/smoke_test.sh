#!/usr/bin/env bash
# Pull the llama.cpp image, start a model slot, wait for it to load,
# then run a quick smoke test to confirm the model is responding correctly.
#
# Usage:
#   bash scripts/smoke_test.sh --slot 1
#   bash scripts/smoke_test.sh --slot 1 --stop-after
#
# --slot        which slot to test: 1,2,3,4  (default: 1)
# --stop-after  stop the container after the test passes
# --timeout     seconds to wait for model to load  (default: 900)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

SLOT=1
STOP_AFTER=false
TIMEOUT=900

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)       SLOT="$2";    shift 2 ;;
    --stop-after) STOP_AFTER=true; shift ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found"; exit 1; }
_env_val() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '[:space:]'; }

MODEL=$(_env_val "MODEL_${SLOT}")
PORT=$(_env_val "PORT_${SLOT}")

[[ -n "$MODEL" ]] || { echo "ERROR: MODEL_${SLOT} not set in .env"; exit 1; }
[[ -n "$PORT"  ]] || { echo "ERROR: PORT_${SLOT} not set in .env";  exit 1; }

MODEL_YAML="$PROJECT_ROOT/src/configs/model.yaml"
[[ -f "$MODEL_YAML" ]] || { echo "ERROR: src/configs/model.yaml not found"; exit 1; }

LLM_SERVICE="llm_${SLOT}"
HEALTH_URL="http://localhost:${PORT}/v1/models"

echo ""
echo "============================================"
echo "  Slot    : $SLOT  ($LLM_SERVICE, port $PORT)"
echo "  Model   : $MODEL"
echo "============================================"
echo ""

cd "$PROJECT_ROOT"

# 1. Pull latest llama.cpp image
echo "[1/3] Pulling llama.cpp image..."
docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
echo ""

# 2. Start container — downloads model from HF and converts to GGUF on first run
echo "[2/3] Starting $LLM_SERVICE (this may take several minutes on first run)..."
docker compose up -d "$LLM_SERVICE"
echo ""

ELAPSED=0
INTERVAL=10
while true; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "      -> Ready! (${ELAPSED}s elapsed)"
    break
  fi
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo ""
    echo "ERROR: Model did not load within ${TIMEOUT}s."
    echo "       Check logs: docker compose logs $LLM_SERVICE"
    exit 1
  fi
  printf "      -> Not ready yet (HTTP %s), retrying in %ds... [%ds/%ds]\r" \
    "$HTTP_CODE" "$INTERVAL" "$ELAPSED" "$TIMEOUT"
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""

# 3. Smoke test — single completion using model.yaml parameters
echo "[3/3] Running smoke test (using src/configs/model.yaml params)..."
RESPONSE=$(python3 - <<EOF
import json, yaml, urllib.request

with open("$MODEL_YAML") as f:
    cfg = yaml.safe_load(f)

enable_thinking = cfg.get("enable_thinking", False)
temperature     = cfg.get("thinking_temp", 0.6) if enable_thinking else cfg.get("nonthinking_temp", 0.1)
# cap max_tokens for smoke test — enough for a think + short answer
max_tokens      = min(cfg.get("thinking_budget", 0) + cfg.get("max_output_tokens", 512), 2048)

payload = {
    "model":              "$MODEL",
    "messages":           [{"role": "user", "content": "Reply with one word: hello"}],
    "max_tokens":         max_tokens,
    "temperature":        temperature,
    "top_p":              cfg.get("top_p", 1.0),
    "repetition_penalty": cfg.get("repetition_penalty", 1.0),
}

req = urllib.request.Request(
    "http://localhost:$PORT/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as r:
    print(r.read().decode())
EOF
)

CONTENT=$(echo "$RESPONSE" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
print(msg.get('content') or msg.get('reasoning_content') or '')
" 2>/dev/null || echo "")

echo ""
if [[ -n "$CONTENT" ]]; then
  echo "============================================"
  echo "  SMOKE TEST PASSED"
  echo "  Model response: $CONTENT"
  echo "============================================"
else
  echo "============================================"
  echo "  SMOKE TEST FAILED — no valid response"
  echo "  Raw response:"
  echo "  $RESPONSE"
  echo "============================================"
  exit 1
fi

if $STOP_AFTER; then
  echo ""
  echo "Stopping $LLM_SERVICE..."
  docker compose stop "$LLM_SERVICE"
fi
