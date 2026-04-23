#!/usr/bin/env bash
# Start a vLLM slot, wait for it to load, then run an eval.
#
# Usage:
#   bash scripts/start_eval.sh --slot 1 --benchmark gpqa [--limit 2] [--follow]
#
# --slot       which vLLM slot to use: 1,2,3,4  (default: 1)
#              reads MODEL_N / PORT_N / GPU_IDS_N from .env
# --benchmark  yaml name under src/configs/benchmarks/  (required)
# --model      model yaml under src/configs/            (default: model)
# --limit      only run first N problems — smoke test
# --follow     tail eval logs after launching
# --timeout    seconds to wait for vLLM to be ready     (default: 600)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# ---------- defaults ----------
SLOT=1
MODEL_CFG="model"
BENCH_CFG=""
LIMIT=""
FOLLOW=false
TIMEOUT=600

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)       SLOT="$2";       shift 2 ;;
    --benchmark)  BENCH_CFG="$2";  shift 2 ;;
    --model)      MODEL_CFG="$2";  shift 2 ;;
    --limit)      LIMIT="$2";      shift 2 ;;
    --follow)     FOLLOW=true;     shift   ;;
    --timeout)    TIMEOUT="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$BENCH_CFG" ]]; then
  echo "Usage: $0 --slot <1-4> --benchmark <name> [--model model] [--limit N] [--follow]"
  echo ""
  echo "Available benchmarks:"
  ls "$PROJECT_ROOT/src/configs/benchmarks/"*.yaml 2>/dev/null \
    | xargs -n1 basename | sed 's/\.yaml$//' | sed 's/^/  /'
  exit 1
fi

# ---------- validate configs ----------
MODEL_YAML="$PROJECT_ROOT/src/configs/${MODEL_CFG}.yaml"
BENCH_YAML="$PROJECT_ROOT/src/configs/benchmarks/${BENCH_CFG}.yaml"
[[ -f "$MODEL_YAML" ]] || { echo "ERROR: model config not found: $MODEL_YAML"; exit 1; }
[[ -f "$BENCH_YAML" ]] || { echo "ERROR: benchmark config not found: $BENCH_YAML"; exit 1; }

# ---------- read slot vars from .env ----------
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found"; exit 1; }

_env_val() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '[:space:]'; }

MODEL=$(_env_val "MODEL_${SLOT}")
PORT=$(_env_val "PORT_${SLOT}")

[[ -n "$MODEL" ]] || { echo "ERROR: MODEL_${SLOT} not set in .env"; exit 1; }
[[ -n "$PORT"  ]] || { echo "ERROR: PORT_${SLOT} not set in .env"; exit 1; }

# slot N → vllm service name (slot 1 = vllm, slot 2 = vllm_2, ...)
VLLM_SERVICE=$([ "$SLOT" -eq 1 ] && echo "vllm" || echo "vllm_${SLOT}")
HEALTH_URL="http://localhost:${PORT}/v1/models"

RUN_CMD="python run_eval.py --model configs/${MODEL_CFG}.yaml --benchmark configs/benchmarks/${BENCH_CFG}.yaml"
[[ -n "$LIMIT" ]] && RUN_CMD="$RUN_CMD --limit $LIMIT"

CONTAINER_NAME="${USER:-eval}_eval_slot${SLOT}_${BENCH_CFG}"

echo ""
echo "============================================"
echo "  Slot         : $SLOT  ($VLLM_SERVICE, port $PORT)"
echo "  Model        : $MODEL"
echo "  Benchmark    : $BENCH_CFG"
[[ -n "$LIMIT" ]] && echo "  Limit        : $LIMIT problems"
echo "============================================"
echo ""

# ---------- 1. start vLLM ----------
cd "$PROJECT_ROOT"
echo "[1/3] Starting $VLLM_SERVICE..."
docker compose up -d "$VLLM_SERVICE"

# ---------- 2. wait for vLLM to be healthy ----------
echo "[2/3] Waiting for vLLM to be ready at $HEALTH_URL ..."
ELAPSED=0
INTERVAL=10
while true; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "      -> Ready! (${ELAPSED}s elapsed)"
    break
  fi
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "ERROR: vLLM did not become ready within ${TIMEOUT}s."
    echo "       Check logs: docker compose logs $VLLM_SERVICE"
    exit 1
  fi
  printf "      -> Not ready yet (HTTP %s), retrying in %ds... [%ds/%ds]\r" \
    "$HTTP_CODE" "$INTERVAL" "$ELAPSED" "$TIMEOUT"
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

# ---------- 3. launch eval ----------
echo "[3/3] Launching eval container: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker compose run \
  --detach \
  --name "$CONTAINER_NAME" \
  --no-deps \
  -e MODEL="$MODEL" \
  -e PORT="$PORT" \
  eval \
  bash -c "pip install -q -r /app/requirements.txt && $RUN_CMD"

echo ""
echo "============================================"
echo "  Eval running in: $CONTAINER_NAME"
echo ""
echo "  Watch logs:"
echo "    docker logs -f $CONTAINER_NAME"
echo ""
echo "  Results will appear in:"
echo "    logs/${BENCH_CFG}/$(echo "$MODEL" | cut -d/ -f2 | tr '[:upper:]' '[:lower:]')/"
echo "============================================"
echo ""

if $FOLLOW; then
  docker logs -f "$CONTAINER_NAME"
fi
