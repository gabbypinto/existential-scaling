#!/usr/bin/env bash
# Launch benchmark jobs for all configured .env slots, each in its own named screen session.
# Session names are derived from the model (e.g. qwen2.5-coder-7b, qwen2.5-coder-14b-instruct).
#
# Usage:
#   bash scripts/run_batch_job.sh
#   bash scripts/run_batch_job.sh --skip lcb_pro,scicode
#   bash scripts/run_batch_job.sh --slots 1,3,5
#
# --skip     benchmarks to skip (default: lcb_pro)
# --slots    comma-separated slot numbers to run (default: all slots with MODEL_N set)
# --env-file path to .env file (default: project root .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP=""
ENV_FILE="$PROJECT_ROOT/.env"
SLOTS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)     SKIP="$2";           shift 2 ;;
    --slots)    SLOTS_OVERRIDE="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found: $ENV_FILE"; exit 1; }
command -v screen >/dev/null 2>&1 || { echo "ERROR: screen not found — install it first"; exit 1; }

_env_val() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true; }

# Derive a clean screen session name from a model identifier:
#   bartowski/Qwen2.5-Coder-14B-Instruct-GGUF → qwen2_5_coder_14b_instruct
#   ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF      → qwen2_5_coder_3b
#   Qwen/Qwen2.5-Coder-7B                     → qwen2_5_coder_7b
_session_name() {
  echo "$1" \
    | cut -d/ -f2 \
    | sed -E 's/-GGUF$//; s/-Q[0-9].*//' \
    | tr '[:upper:]' '[:lower:]' \
    | tr '.-' '__'
}

# Determine which slots to run
if [[ -n "$SLOTS_OVERRIDE" ]]; then
  IFS=',' read -ra SLOTS <<< "$SLOTS_OVERRIDE"
else
  SLOTS=()
  for N in 1 2 3 4 5 6; do
    MODEL=$(_env_val "MODEL_${N}")
    [[ -n "$MODEL" ]] && SLOTS+=("$N")
  done
fi

[[ ${#SLOTS[@]} -gt 0 ]] || { echo "ERROR: No configured slots found in $ENV_FILE"; exit 1; }

echo ""
echo "============================================"
echo "  Batch benchmark launcher"
echo "  Slots  : ${SLOTS[*]}"
echo "  Skip   : ${SKIP:-none}"
echo "============================================"
echo ""

LAUNCHED=()
for SLOT in "${SLOTS[@]}"; do
  MODEL=$(_env_val "MODEL_${SLOT}")
  if [[ -z "$MODEL" ]]; then
    echo "  Slot $SLOT: MODEL_${SLOT} not set — skipping"
    continue
  fi

  SESSION=$(_session_name "$MODEL")

  # Warn and kill any existing session with this name
  if screen -ls 2>/dev/null | grep -qF ".$SESSION"; then
    echo "  WARNING: screen '$SESSION' already exists — killing it"
    screen -S "$SESSION" -X quit 2>/dev/null || true
    sleep 0.3
  fi

  RUN_CMD="bash '$SCRIPT_DIR/run_all_benchmarks.sh' --slot $SLOT"
  [[ -n "$SKIP" ]] && RUN_CMD+=" --skip '$SKIP'"

  # Keep session open after job finishes so output remains visible
  screen -dmS "$SESSION" bash -c "$RUN_CMD; echo ''; echo '=== Done: $SESSION — press Enter to close ==='; read"

  echo "  Slot $SLOT → screen '$SESSION'  ($MODEL)"
  LAUNCHED+=("$SESSION")
done

echo ""
echo "============================================"
echo "  ${#LAUNCHED[@]} session(s) started"
echo ""
echo "  Attach to a session:"
for S in "${LAUNCHED[@]}"; do
  echo "    screen -r $S"
done
echo ""
echo "  Detach from a session: Ctrl-A D"
echo "  List all sessions:     screen -ls"
echo "============================================"
echo ""
