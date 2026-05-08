#!/usr/bin/env bash
# Run all benchmarks sequentially on a given slot.
# Each benchmark waits for the previous to finish before starting.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh --slot 3
#   bash scripts/run_all_benchmarks.sh --slot 4 --skip aime24,mmlu
#   bash scripts/run_all_benchmarks.sh --slot 4 --model model --limit 2
#
# --slot      which LLM service slot: 1,2,3,4  (default: 1)
# --model     model yaml under src/configs/   (default: model)
# --skip      comma-separated list of benchmarks to skip
# --limit     only run first N problems per benchmark (for testing)
# --timeout   seconds to wait for LLM service to be ready (default: 900)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

SLOT=1
MODEL_CFG="model"
LIMIT=""
TIMEOUT=900
SKIP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)    SLOT="$2";      shift 2 ;;
    --model)   MODEL_CFG="$2"; shift 2 ;;
    --limit)   LIMIT="$2";     shift 2 ;;
    --timeout) TIMEOUT="$2";   shift 2 ;;
    --skip)    SKIP="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

_should_skip() {
  local bench="$1"
  [[ -z "$SKIP" ]] && return 1
  IFS=',' read -ra skip_list <<< "$SKIP"
  for s in "${skip_list[@]}"; do
    [[ "$s" == "$bench" ]] && return 0
  done
  return 1
}

[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found"; exit 1; }
_env_val() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true; }

MODEL=$(_env_val "MODEL_${SLOT}")
PORT=$(_env_val "PORT_${SLOT}")

[[ -n "$MODEL" ]] || { echo "ERROR: MODEL_${SLOT} not set in .env"; exit 1; }
[[ -n "$PORT"  ]] || { echo "ERROR: PORT_${SLOT} not set in .env";  exit 1; }

MODEL_YAML="$PROJECT_ROOT/src/configs/${MODEL_CFG}.yaml"
[[ -f "$MODEL_YAML" ]] || { echo "ERROR: model config not found: $MODEL_YAML"; exit 1; }

LLM_SERVICE="llm_${SLOT}"
HEALTH_URL="http://localhost:${PORT}/v1/models"
MODEL_SHORT=$(echo "$MODEL" | cut -d/ -f2 | tr '[:upper:]' '[:lower:]')
EVAL_CONTAINER="${USER:-eval}_eval_${MODEL_SHORT}"

BENCHMARKS=(aime24 aime25 gpqa lcb lcb_pro piqa_global scicode aa_omniscience matharena_apex global_mmlu_lite mmlu)

ACTIVE=()
for b in "${BENCHMARKS[@]}"; do
  _should_skip "$b" || ACTIVE+=("$b")
done

echo ""
echo "============================================"
echo "  Slot      : $SLOT  ($LLM_SERVICE, port $PORT)"
echo "  Model     : $MODEL"
echo "  Benchmarks: ${ACTIVE[*]}"
[[ -n "$LIMIT" ]] && echo "  Limit     : $LIMIT problems each"
echo "============================================"
echo ""

cd "$PROJECT_ROOT"

# 1. Start LLM service
echo "[1/2] Starting $LLM_SERVICE..."
docker compose up -d "$LLM_SERVICE"
echo ""

# 2. Wait for LLM service to be healthy
echo "[2/2] Waiting for LLM service at $HEALTH_URL ..."
ELAPSED=0
INTERVAL=10
while true; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "      -> Ready! (${ELAPSED}s elapsed)"
    break
  fi
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "ERROR: LLM service did not become ready within ${TIMEOUT}s."
    echo "       Check logs: docker compose logs $LLM_SERVICE"
    exit 1
  fi
  printf "      -> Not ready yet (HTTP %s), retrying in %ds... [%ds/%ds]\r" \
    "$HTTP_CODE" "$INTERVAL" "$ELAPSED" "$TIMEOUT"
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""

# 3. Run each benchmark sequentially (synchronous — waits for completion)
TOTAL=${#BENCHMARKS[@]}
PASSED=()
FAILED=()

for i in "${!BENCHMARKS[@]}"; do
  BENCH="${BENCHMARKS[$i]}"
  NUM=$((i + 1))
  BENCH_YAML="$PROJECT_ROOT/src/configs/benchmarks/${BENCH}.yaml"

  if _should_skip "$BENCH"; then
    echo "  -> Skipping $BENCH"
    continue
  fi

  [[ -f "$BENCH_YAML" ]] || { echo "WARNING: $BENCH_YAML not found, skipping"; FAILED+=("$BENCH"); continue; }

  RUN_CMD="python run_eval.py --model configs/${MODEL_CFG}.yaml --benchmark configs/benchmarks/${BENCH}.yaml"
  [[ -n "$LIMIT" ]] && RUN_CMD="$RUN_CMD --limit $LIMIT"

  echo ""
  echo "============================================"
  echo "  Benchmark $NUM/$TOTAL: $BENCH"
  echo "============================================"
  echo ""

  # Run synchronously (no --detach) so script waits for completion
  docker rm -f "$EVAL_CONTAINER" 2>/dev/null || true
  if docker compose run \
      --rm \
      --name "$EVAL_CONTAINER" \
      --no-deps \
      -e MODEL="$MODEL" \
      -e PORT="$PORT" \
      eval \
      bash -c "pip install -q -r /app/requirements.txt && $RUN_CMD"; then
    PASSED+=("$BENCH")
    echo ""
    echo "  -> $BENCH complete"
  else
    FAILED+=("$BENCH")
    echo ""
    echo "  -> $BENCH FAILED (exit code $?)"
  fi
done

echo ""
echo "============================================"
echo "  All benchmarks done"
echo "  Passed : ${PASSED[*]:-none}"
echo "  Failed : ${FAILED[*]:-none}"
echo "============================================"
echo ""
