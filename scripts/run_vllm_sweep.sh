#!/usr/bin/env bash
# Run benchmarks x system prompts for one model on vLLM, many requests in flight.
#
# The model and server settings come from the env file (VLLM_* block), like the
# llama.cpp slots; flags choose what to run. Starts the `vllm` compose service,
# waits for it, runs src/run_sweep.py in the eval container (one shared request
# pool across every benchmark/prompt/round/question), then stops the server.
# Runs in a detached screen session; output is also saved under logs/vllm_sweep/.
#
# Usage:
#   bash scripts/run_vllm_sweep.sh --benchmarks aime24,aime25,gpqa
#   bash scripts/run_vllm_sweep.sh --benchmarks aime25 --prompts Baseline --limit 2 --num-rounds 1
#
# --benchmarks   comma-separated benchmarks under src/configs/benchmarks/   (required)
# --prompts      comma-separated prompt keys       (default: every key in src/configs/prompts.yaml)
# --model-cfg    model yaml stem under src/configs/ (default: model)
# --concurrency  requests in flight                (default: VLLM_CONCURRENCY, else 32)
# --limit        only the first N problems per benchmark
# --num-rounds   override each benchmark's num_rounds
# --overwrite    re-run benchmark/prompt pairs that already have a summary.json
# --timeout      seconds to wait for vLLM to become healthy (default: 1800)
# --env-file     env file with the VLLM_* settings  (default: .env)
# --force        don't ask when llama.cpp containers are running
# --foreground   run in this terminal instead of a screen session
#
# Env file (VLLM_* block):
#   VLLM_MODEL             HF model id, safetensors (required)
#   VLLM_PORT              host port               (default 20100)
#   VLLM_GPU_IDS           CUDA_VISIBLE_DEVICES    (default 0)
#   VLLM_GPU_LABEL         container name suffix   (default 0)
#   VLLM_GPU_MEM_UTIL      share of GPU memory     (default 0.8)
#   VLLM_REASONING_PARSER  auto | none | <parser>  (default auto, picked from the model name)
#   VLLM_CONCURRENCY       requests in flight      (default 32)
#   VLLM_IMAGE             server image            (default vllm/vllm-openai:v0.31.0)
#   VLLM_EXTRA_ARGS        extra `vllm serve` flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORIG_ARGS=("$@")

BENCHMARKS=""
PROMPTS=""
MODEL_CFG="model"
CONCURRENCY=""
LIMIT=""
NUM_ROUNDS=""
OVERWRITE=false
TIMEOUT=1800
ENV_FILE="$PROJECT_ROOT/.env"
FORCE=false
FOREGROUND=false
INNER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks)  BENCHMARKS="$2";  shift 2 ;;
    --prompts)     PROMPTS="$2";     shift 2 ;;
    --model-cfg)   MODEL_CFG="$2";   shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --limit)       LIMIT="$2";       shift 2 ;;
    --num-rounds)  NUM_ROUNDS="$2";  shift 2 ;;
    --overwrite)   OVERWRITE=true;   shift   ;;
    --timeout)     TIMEOUT="$2";     shift 2 ;;
    --env-file)    ENV_FILE="$2";    shift 2 ;;
    --force)       FORCE=true;       shift   ;;
    --foreground)  FOREGROUND=true;  shift   ;;
    --_inner)      INNER=true;       shift   ;;   # internal: running inside screen
    -h|--help)     sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$BENCHMARKS" ]] || { echo "ERROR: --benchmarks is required (e.g. --benchmarks aime24,gpqa)"; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "ERROR: env file not found: $ENV_FILE"; exit 1; }
[[ -f "$PROJECT_ROOT/src/configs/${MODEL_CFG}.yaml" ]] || { echo "ERROR: model config not found: src/configs/${MODEL_CFG}.yaml"; exit 1; }

# value of KEY in the env file, without inline comments or whitespace
_env_val() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/[[:space:]]#.*$//' | tr -d '[:space:]' || true; }

VLLM_MODEL="$(_env_val VLLM_MODEL)"
VLLM_PORT="$(_env_val VLLM_PORT)";                   VLLM_PORT="${VLLM_PORT:-20100}"
VLLM_GPU_LABEL="$(_env_val VLLM_GPU_LABEL)";         VLLM_GPU_LABEL="${VLLM_GPU_LABEL:-0}"
VLLM_REASONING_PARSER="$(_env_val VLLM_REASONING_PARSER)"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-auto}"
# VLLM_EXTRA_ARGS may contain spaces, so read it without stripping them
ENV_EXTRA_ARGS="$(grep -E '^VLLM_EXTRA_ARGS=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/^"//; s/"$//' || true)"
CONCURRENCY="${CONCURRENCY:-$(_env_val VLLM_CONCURRENCY)}"
CONCURRENCY="${CONCURRENCY:-32}"

[[ -n "$VLLM_MODEL" ]] || { echo "ERROR: VLLM_MODEL is not set in $ENV_FILE"; exit 1; }

# Reasoning parser: lets vLLM separate thinking from the answer and enforce thinking_token_budget
if [[ "$VLLM_REASONING_PARSER" == "auto" ]]; then
  lower="$(echo "$VLLM_MODEL" | tr '[:upper:]' '[:lower:]')"
  case "$lower" in
    *deepseek-r1*|*r1-distill*|*qwq*) PARSER="deepseek_r1" ;;
    *qwen3*)                          PARSER="qwen3" ;;
    *gemma-4*|*gemma4*)               PARSER="gemma4" ;;
    *)                                PARSER="" ;;
  esac
elif [[ "$VLLM_REASONING_PARSER" == "none" ]]; then
  PARSER=""
else
  PARSER="$VLLM_REASONING_PARSER"
fi

MAX_NUM_SEQS=$(( CONCURRENCY > 32 ? CONCURRENCY : 32 ))
EXTRA_ARGS="$ENV_EXTRA_ARGS${PARSER:+ --reasoning-parser $PARSER}"
CONTAINER="${USER}_vllm_GPU_${VLLM_GPU_LABEL}"
MODEL_SHORT="$(basename "$VLLM_MODEL" | tr '[:upper:]' '[:lower:]')"
SCREEN_NAME="$(echo "$MODEL_SHORT" | tr '.-' '__')_vllm"

# ---------- outer: pre-flight, then hand off to a screen session ----------
if ! $INNER; then
  echo ""
  echo "============================================"
  echo "  vLLM sweep — $(date)"
  echo "  Model       : $VLLM_MODEL"
  echo "  Parser      : ${PARSER:-none}"
  echo "  Benchmarks  : $BENCHMARKS"
  echo "  Prompts     : ${PROMPTS:-all}"
  echo "  Concurrency : $CONCURRENCY   (port $VLLM_PORT)"
  [[ -n "$LIMIT" ]] && echo "  Limit       : $LIMIT"
  echo "============================================"
  echo ""

  OTHERS=$(docker ps --format '{{.Names}}' | grep -E "^${USER}_(llm|bench)_" || true)
  if [[ -n "$OTHERS" ]] && ! $FORCE; then
    echo "WARNING: other model containers are running and share GPU memory/bandwidth:"
    echo "$OTHERS" | sed 's/^/    /'
    echo "  vLLM may fail to start with --gpu-memory-utilization set. Stop them first, e.g.:"
    echo "    docker compose stop llm_1 llm_2 llm_3"
    read -rp "  Continue anyway? [y/N] " ans
    [[ "$ans" == [yY]* ]] || exit 1
  fi

  if ! $FOREGROUND; then
    command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not found — install it or use --foreground"; exit 1; }
    mkdir -p "$PROJECT_ROOT/logs/vllm_sweep"
    LOG_FILE="$PROJECT_ROOT/logs/vllm_sweep/${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S).log"
    if screen -ls 2>/dev/null | grep -qF ".$SCREEN_NAME"; then
      echo "ERROR: screen '$SCREEN_NAME' already exists (a sweep for this model is running?)."
      echo "       Attach: screen -r $SCREEN_NAME    or kill it: screen -S $SCREEN_NAME -X quit"
      exit 1
    fi
    ARGS_Q="$(printf '%q ' "${ORIG_ARGS[@]}")"
    screen -dmS "$SCREEN_NAME" bash -c \
      "bash '$SCRIPT_DIR/$(basename "$0")' $ARGS_Q --force --_inner 2>&1 | tee -a '$LOG_FILE'; echo ''; echo '=== Done: $SCREEN_NAME — press Enter to close ==='; read"
    echo "  Launched in screen '$SCREEN_NAME'"
    echo "    Attach:  screen -r $SCREEN_NAME     (detach: Ctrl-A D)"
    echo "    Log:     ${LOG_FILE#$PROJECT_ROOT/}"
    echo "    Results: logs/<benchmark>/$MODEL_SHORT/<prompt>/"
    exit 0
  fi
fi

# ---------- inner (or --foreground): start vLLM, run the sweep, stop vLLM ----------
cd "$PROJECT_ROOT"

stop_server() {
  echo ""
  echo "Stopping vLLM ($CONTAINER)..."
  docker compose --env-file "$ENV_FILE" --profile vllm stop vllm >/dev/null 2>&1 || true
}
trap stop_server EXIT

# Fix HF cache dirs that Docker may have created as root
if [[ -d "$HOME/.cache/huggingface" ]]; then
  sudo -n chown -R "$USER:$USER" "$HOME/.cache/huggingface" 2>/dev/null || true
fi

echo "[1/3] Starting vLLM: $VLLM_MODEL"
export VLLM_MAX_NUM_SEQS="$MAX_NUM_SEQS"
export VLLM_EXTRA_ARGS="$EXTRA_ARGS"
docker compose --env-file "$ENV_FILE" --profile vllm up -d --force-recreate vllm

echo "[2/3] Waiting for http://localhost:$VLLM_PORT/health (first load can take 10-15 min)..."
ELAPSED=0
while true; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" || true)
  [[ "$CODE" == "200" ]] && { echo "      -> Ready (${ELAPSED}s)"; break; }
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: $CONTAINER exited during startup. Last log lines:"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: vLLM not healthy after ${TIMEOUT}s. Last log lines:"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
  fi
  sleep 10
  ELAPSED=$((ELAPSED + 10))
done

SWEEP_CMD="python run_sweep.py --model configs/${MODEL_CFG}.yaml --benchmarks $BENCHMARKS --concurrency $CONCURRENCY"
[[ -n "$PROMPTS" ]]    && SWEEP_CMD+=" --prompts $PROMPTS"
[[ -n "$LIMIT" ]]      && SWEEP_CMD+=" --limit $LIMIT"
[[ -n "$NUM_ROUNDS" ]] && SWEEP_CMD+=" --num-rounds $NUM_ROUNDS"
$OVERWRITE             && SWEEP_CMD+=" --overwrite"

echo "[3/3] Running sweep: $SWEEP_CMD"
docker compose --env-file "$ENV_FILE" run --rm -T --no-deps \
  -e MODEL="$VLLM_MODEL" \
  -e PORT="$VLLM_PORT" \
  -e ENGINE=vllm \
  -e PYTHONUNBUFFERED=1 \
  eval \
  bash -c "pip install -q -r /app/requirements.txt && $SWEEP_CMD"

echo ""
echo "Sweep finished. Results: logs/<benchmark>/$MODEL_SHORT/<prompt>/"
