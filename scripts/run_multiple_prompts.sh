#!/usr/bin/env bash
# Run a full system-prompt sweep across all configured model slots.
# Launches one detached screen session per slot — all models run in parallel.
#
# Usage:
#   bash scripts/run_multiple_prompts.sh
#   bash scripts/run_multiple_prompts.sh --limit 2           # smoke test
#   bash scripts/run_multiple_prompts.sh --slots 1,3         # specific slots only
#
# --slots       comma-separated slots to run (default: auto-detect from .env)
# --benchmarks  comma-separated benchmarks  (default: aime24,aime25,gpqa,global_mmlu_lite)
# --prompts     path to prompt variants YAML (default: src/configs/prompts.yaml)
# --limit       only run first N problems per benchmark
# --timeout     seconds to wait for LLM service ready (default: 900)
# --env-file    path to env file (default: .env)
# --model       model config yaml stem under src/configs/ (default: model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BENCHMARKS_RAW="aime24,aime25,gpqa,global_mmlu_lite"
PROMPTS_FILE="$PROJECT_ROOT/src/configs/prompts.yaml"
LIMIT=""
TIMEOUT=900
ENV_FILE="$PROJECT_ROOT/.env"
MODEL_CFG="model"
SLOTS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slots)      SLOTS_OVERRIDE="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS_RAW="$2"; shift 2 ;;
    --prompts)    PROMPTS_FILE="$2";   shift 2 ;;
    --limit)      LIMIT="$2";          shift 2 ;;
    --timeout)    TIMEOUT="$2";        shift 2 ;;
    --env-file)   ENV_FILE="$2";       shift 2 ;;
    --model)      MODEL_CFG="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$ENV_FILE" ]]     || { echo "ERROR: .env not found: $ENV_FILE"; exit 1; }
[[ -f "$PROMPTS_FILE" ]] || { echo "ERROR: prompts file not found: $PROMPTS_FILE"; exit 1; }
command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not found — install it first"; exit 1; }


_env_val() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true; }

# Derive a clean screen session name from a model identifier (same as run_batch_job.sh):
#   unsloth/Qwen3.5-9B-GGUF → qwen3_5_9b
_session_name() {
  echo "$1" \
    | cut -d/ -f2 \
    | sed -E 's/-GGUF$//; s/-Q[0-9].*//' \
    | tr '[:upper:]' '[:lower:]' \
    | tr '.-' '__'
}

# Fix HF cache dirs that Docker may have created as root
if [[ -d "$HOME/.cache/huggingface" ]]; then
  echo "Fixing HF cache permissions..."
  sudo chown -R "$USER:$USER" "$HOME/.cache/huggingface" 2>/dev/null \
    || echo "WARNING: could not fix HF cache permissions — if you see PermissionError, run: sudo chown -R \$USER:\$USER ~/.cache/huggingface"
fi

# Determine which slots to run
if [[ -n "$SLOTS_OVERRIDE" ]]; then
  IFS=',' read -ra SLOTS <<< "$SLOTS_OVERRIDE"
else
  SLOTS=()
  for i in $(seq 1 8); do
    [[ -n "$(_env_val "MODEL_${i}")" ]] && SLOTS+=("$i")
  done
fi

[[ ${#SLOTS[@]} -gt 0 ]] || { echo "ERROR: no model slots found in $ENV_FILE"; exit 1; }

echo ""
echo "============================================"
echo "  Benchmark sweep — $(date)"
echo "  Benchmarks : $BENCHMARKS_RAW"
echo "  Prompts    : $PROMPTS_FILE"
echo "  Slots      : ${SLOTS[*]}"
[[ -n "$LIMIT" ]] && echo "  Limit      : $LIMIT"
echo "============================================"
echo ""

PARENT_PID=$$
LAUNCHED=()

for SLOT in "${SLOTS[@]}"; do
  MODEL=$(_env_val "MODEL_${SLOT}")
  PORT=$(_env_val "PORT_${SLOT}")

  if [[ -z "$MODEL" ]]; then echo "WARNING: MODEL_${SLOT} not set, skipping"; continue; fi
  if [[ -z "$PORT"  ]]; then echo "WARNING: PORT_${SLOT} not set, skipping";  continue; fi

  MODEL_SHORT=$(echo "$MODEL" | cut -d/ -f2 | tr '[:upper:]' '[:lower:]')
  SCREEN_NAME=$(_session_name "$MODEL")
  SLOT_SCRIPT="/tmp/sweep_slot_${SLOT}_${PARENT_PID}.sh"
  DONE_FILE="/tmp/sweep_slot_${SLOT}_${PARENT_PID}.done"

  # Compute container-internal path to the prompts file (./src mounts to /app/src)
  PROMPTS_CONTAINER=$(python3 -c "import os; print(os.path.relpath('$PROMPTS_FILE', '$PROJECT_ROOT/src'))" 2>/dev/null || echo "configs/prompts.yaml")

  # Write the per-slot script (outer vars expand now; inner vars are escaped)
  cat > "$SLOT_SCRIPT" << SLOT_SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"

PASS=()
FAIL=()

echo "[slot $SLOT] Starting llm_$SLOT ($MODEL)..."
docker compose --env-file "$ENV_FILE" up -d "llm_$SLOT"

HEALTH_URL="http://localhost:$PORT/v1/models"
ELAPSED=0
while true; do
  HTTP_CODE=\$(curl -s -o /dev/null -w "%{http_code}" "\$HEALTH_URL" || true)
  if [[ "\$HTTP_CODE" == "200" ]]; then
    echo "[slot $SLOT] Ready! (\${ELAPSED}s)"
    break
  fi
  if [[ \$ELAPSED -ge $TIMEOUT ]]; then
    echo "[slot $SLOT] ERROR: LLM not ready after ${TIMEOUT}s"
    docker compose --env-file "$ENV_FILE" logs "llm_$SLOT" | tail -20
    exit 1
  fi
  sleep 10
  ELAPSED=\$((ELAPSED + 10))
done

# Discover prompt keys from top-level YAML keys (no host python/venv needed)
mapfile -t PROMPT_KEYS <<< "\$(grep -E '^[A-Za-z_][A-Za-z0-9_]*:' "$PROMPTS_FILE" | cut -d: -f1)"

IFS=',' read -ra BENCHMARKS <<< "$BENCHMARKS_RAW"

for BENCH in "\${BENCHMARKS[@]}"; do
  echo "[slot $SLOT] ---- Benchmark: \$BENCH ----"
  for PROMPT_KEY in "\${PROMPT_KEYS[@]}"; do
    LOG_DIR="logs/\${BENCH}/$MODEL_SHORT/\$PROMPT_KEY"
    echo "[slot $SLOT]   \$PROMPT_KEY -> \$LOG_DIR"

    PROMPT_KEY_SLUG=\$(echo "\$PROMPT_KEY" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/_\$//')
    CONTAINER_NAME="eval_${MODEL_SHORT}_\${BENCH}_\${PROMPT_KEY_SLUG}"
    docker rm -f "\$CONTAINER_NAME" 2>/dev/null || true

    _EVAL_CMD="python run_eval.py --model configs/${MODEL_CFG}.yaml --benchmark configs/benchmarks/\${BENCH}.yaml --prompts-file $PROMPTS_CONTAINER --prompt-key \${PROMPT_KEY} --log-dir \${LOG_DIR}$([ -n "$LIMIT" ] && echo " --limit $LIMIT")"

    if docker compose --env-file "$ENV_FILE" run \\
        --rm \\
        -T \\
        --name "\$CONTAINER_NAME" \\
        --no-deps \\
        -e MODEL="$MODEL" \\
        -e PORT="$PORT" \\
        -e PYTHONUNBUFFERED=1 \\
        eval \\
        bash -c "pip install -q -r /app/requirements.txt && \$_EVAL_CMD"; then
      PASS+=("\$BENCH/\$PROMPT_KEY")
    else
      FAIL+=("\$BENCH/\$PROMPT_KEY")
      echo "[slot $SLOT] FAILED: \$BENCH/\$PROMPT_KEY"
    fi
  done
done

echo ""
echo "[slot $SLOT] Stopping llm_$SLOT..."
docker compose --env-file "$ENV_FILE" stop "llm_$SLOT"

echo "[slot $SLOT] DONE. Passed: \${#PASS[@]}, Failed: \${#FAIL[@]}"
printf "passed=%s failed=%s\n" "\${#PASS[@]}" "\${#FAIL[@]}" > "$DONE_FILE"
SLOT_SCRIPT_EOF

  chmod +x "$SLOT_SCRIPT"

  # Kill any existing screen with this name before relaunching
  if screen -ls 2>/dev/null | grep -qF ".$SCREEN_NAME"; then
    echo "  WARNING: screen '$SCREEN_NAME' already exists — killing it"
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
    sleep 0.3
  fi

  screen -dmS "$SCREEN_NAME" bash -c "bash '$SLOT_SCRIPT'; echo ''; echo '=== Done: $SCREEN_NAME — press Enter to close ==='; read"
  echo "  Slot $SLOT → screen '$SCREEN_NAME'  ($MODEL)"
  LAUNCHED+=("$SCREEN_NAME")
done

echo ""
echo "============================================"
echo "  ${#LAUNCHED[@]} screen session(s) launched"
echo ""
echo "  Per-slot screens:"
echo "    List:     screen -ls"
for S in "${LAUNCHED[@]}"; do
  echo "    Attach:   screen -r $S"
done
echo "    Detach:   Ctrl-A D"
echo ""
echo "  Per-combo containers (one active at a time per slot):"
echo "    List:     docker ps --filter name=eval_"
echo "    Follow:   docker logs -f eval_<model>_<bench>_<prompt>"
echo ""
echo "  Logs: logs/{benchmark}/{model}/{prompt_key}/summary.json"
echo "============================================"
