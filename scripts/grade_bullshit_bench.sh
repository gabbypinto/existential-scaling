#!/usr/bin/env bash
# Grade BullshitBench responses with an LLM judge running in one of the .env slots.
# Starts the judge slot (if not already up), waits for it, then runs
# src/grade_bullshit_bench.py inside the eval container.
#
# Usage:
#   bash scripts/grade_bullshit_bench.sh --slot 4                  # judge = MODEL_4 / PORT_4
#   bash scripts/grade_bullshit_bench.sh --slot 4 --limit 3        # smoke test: grade 3 responses
#   bash scripts/grade_bullshit_bench.sh --slot 4 --dry-run        # list what would be graded
#   bash scripts/grade_bullshit_bench.sh --slot 4 --overwrite      # re-grade everything
#   bash scripts/grade_bullshit_bench.sh --slot 4 --no-think       # judge is a thinking model; turn it off
#
# --slot       slot whose MODEL_N/PORT_N is the JUDGE (default: 4)
# --logs-dir   directory to grade, relative to repo root (default: logs/bullshit_bench)
# --limit N    grade at most N responses
# --overwrite  re-grade responses that already have a score
# --no-think   pass enable_thinking=false to the judge chat template
# --dry-run    don't call the judge, just list ungraded responses
# --timeout    seconds to wait for the judge to load (default: 900)
# --env-file   path to env file (default: .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

SLOT=4
LOGS_DIR="logs/bullshit_bench"
TIMEOUT=900
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)      SLOT="$2";      shift 2 ;;
    --logs-dir)  LOGS_DIR="$2";  shift 2 ;;
    --limit)     EXTRA+=(--limit "$2"); shift 2 ;;
    --overwrite) EXTRA+=(--overwrite);  shift ;;
    --no-think)  EXTRA+=(--judge-no-think); shift ;;
    --dry-run)   EXTRA+=(--dry-run);    shift ;;
    --timeout)   TIMEOUT="$2";   shift 2 ;;
    --env-file)  ENV_FILE="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "ERROR: env file not found: $ENV_FILE"; exit 1; }

# read MODEL_N / PORT_N for the judge slot straight from the env file
MODEL=$(grep -E "^MODEL_${SLOT}=" "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '"' || true)
PORT=$(grep  -E "^PORT_${SLOT}="  "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '"' || true)
[[ -n "$MODEL" && -n "$PORT" ]] || { echo "ERROR: MODEL_${SLOT}/PORT_${SLOT} not set in $ENV_FILE"; exit 1; }

LLM_SERVICE="llm_${SLOT}"
HEALTH_URL="http://localhost:${PORT}/v1/models"
CONTAINER_NAME="${USER:-eval}_grade_bullshit_slot${SLOT}"

echo ""
echo "============================================"
echo "  Judge slot  : $SLOT  ($LLM_SERVICE, port $PORT)"
echo "  Judge model : $MODEL"
echo "  Logs dir    : $LOGS_DIR"
echo "============================================"
echo ""

cd "$PROJECT_ROOT"
echo "[1/3] Starting $LLM_SERVICE..."
docker compose --env-file "$ENV_FILE" up -d "$LLM_SERVICE"

echo "[2/3] Waiting for judge at $HEALTH_URL ..."
ELAPSED=0; INTERVAL=10
while true; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  [[ "$HTTP_CODE" == "200" ]] && { echo "      -> Ready! (${ELAPSED}s)"; break; }
  if [[ $ELAPSED -ge $TIMEOUT ]]; then
    echo "ERROR: judge did not become ready within ${TIMEOUT}s. Check: docker compose logs $LLM_SERVICE"; exit 1
  fi
  printf "      -> Not ready (HTTP %s), retrying in %ds... [%ds/%ds]\r" "$HTTP_CODE" "$INTERVAL" "$ELAPSED" "$TIMEOUT"
  sleep $INTERVAL; ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[3/3] Grading in eval container: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
# logs/ is mounted at /app/src/logs and the working dir is /app/src, so LOGS_DIR resolves as-is.
docker compose --env-file "$ENV_FILE" run --rm -T --no-deps \
  --name "$CONTAINER_NAME" \
  -e PYTHONUNBUFFERED=1 \
  eval \
  bash -c "pip install -q -r /app/requirements.txt && python grade_bullshit_bench.py --logs-dir '$LOGS_DIR' --judge-port $PORT --judge-model '$MODEL' ${EXTRA[*]:-}"

echo ""
echo "Done. Per-run summaries updated under $LOGS_DIR/<model>/<prompt>/summary.json"
echo "Roll up with: python scripts/extract_metrics.py --logs-dir logs --benchmarks bullshit_bench"
