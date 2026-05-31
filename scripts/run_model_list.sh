#!/usr/bin/env bash
# Run all benchmarks for a list of models, sequentially or in parallel.
# Each model gets its own temp env + model config, then run_all_benchmarks.sh is called.
# If a model entry has no "weights" field, the model is auto-converted from safetensors to GGUF
# before the LLM service starts.
#
# Usage:
#   bash scripts/run_model_list.sh [--list model_list.json] [--limit N] [--dry-run] [--parallel]
#
# --list      path to model list JSON  (default: model_list.json in project root)
# --limit     only run first N problems per benchmark (smoke test)
# --dry-run   print what would run without executing
# --parallel  run all models simultaneously, one per slot (slot = position in JSON, 1-indexed)
#             logs go to logs/parallel_<model>.log; requires enough GPU slots in docker-compose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIST_FILE="$PROJECT_ROOT/model_list.json"
LIMIT=""
DRY_RUN=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)     LIST_FILE="$2"; shift 2 ;;
    --limit)    LIMIT="$2";    shift 2 ;;
    --dry-run)  DRY_RUN=true;  shift   ;;
    --parallel) PARALLEL=true; shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$LIST_FILE" ]] || { echo "ERROR: model list not found: $LIST_FILE"; exit 1; }

# Parse top-level config from the JSON using Python (no jq dependency)
_jq() { python3 -c "import json; d=json.load(open('$LIST_FILE')); print($1)"; }
SLOT=$(_jq "d.get('slot',1)")
SKIP=$(_jq "d.get('skip','')")
MODEL_COUNT=$(_jq "len(d['models'])")

TMP_MODEL_YAML="$PROJECT_ROOT/src/configs/_model_run_tmp.yaml"
TMP_ENV="$PROJECT_ROOT/.env._run_tmp"

cleanup() { rm -f "$TMP_MODEL_YAML" "$TMP_ENV"; }
trap cleanup EXIT

echo ""
echo "============================================"
echo "  Model list : $(basename "$LIST_FILE")"
echo "  Models     : $MODEL_COUNT"
echo "  Slot       : $SLOT"
[[ -n "$SKIP"  ]] && echo "  Skip       : $SKIP"
[[ -n "$LIMIT" ]] && echo "  Limit      : $LIMIT (smoke test)"
echo "============================================"
echo ""

LOG_FILE="$PROJECT_ROOT/logs/model_run_log.txt"
mkdir -p "$(dirname "$LOG_FILE")"

# ===========================================================================
# PARALLEL MODE — one slot per model (slot = idx+1), all run simultaneously
# ===========================================================================
if $PARALLEL; then
  declare -a PAR_LOCAL_GGUFS
  declare -a PAR_SHORT_NAMES

  # Initialize model names; GGUFs must already exist (convert locally with convert_to_gguf.sh)
  for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
    HF_REPO=$(python3 -c "
import json
with open('$LIST_FILE') as f:
    d = json.load(f)
print(d['models'][$IDX].get('hf_repo',''))
")
    PAR_SHORT_NAMES[$IDX]=$(echo "$HF_REPO" | cut -d/ -f2)
    PAR_LOCAL_GGUFS[$IDX]=""
  done

  # Phase 2: generate per-model yamls + accumulate shared env
  SHARED_ENV="$PROJECT_ROOT/.env._parallel_tmp"
  BASE_ENV_FILE="$PROJECT_ROOT/$(_jq "d.get('env_file','.env')")"
  cp "$BASE_ENV_FILE" "$SHARED_ENV"

  declare -a PAR_TMP_YAMLS
  for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
    SLOT=$((IDX + 1))
    TMP_YAML_NAME="_model_tmp_par_${SLOT}"
    TMP_YAML_PATH="$PROJECT_ROOT/src/configs/${TMP_YAML_NAME}.yaml"
    PAR_TMP_YAMLS[$IDX]="$TMP_YAML_NAME"

    TMP_VARS=$(mktemp)
    python3 "$SCRIPT_DIR/_generate_model_config.py" \
      "$LIST_FILE" "$IDX" "$PROJECT_ROOT" "$TMP_YAML_PATH" "$SHARED_ENV" \
      "${PAR_LOCAL_GGUFS[$IDX]:-}" "$SLOT" > "$TMP_VARS"
    rm -f "$TMP_VARS"
  done

  par_cleanup() {
    rm -f "$SHARED_ENV"
    for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
      SLOT=$((IDX + 1))
      rm -f "$PROJECT_ROOT/src/configs/_model_tmp_par_${SLOT}.yaml"
    done
  }
  trap par_cleanup EXIT

  if $DRY_RUN; then
    echo ""
    for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
      SLOT=$((IDX + 1))
      echo "  [dry-run] slot $SLOT → ${PAR_SHORT_NAMES[$IDX]}"
      echo "           cmd: run_all_benchmarks.sh --slot $SLOT --model ${PAR_TMP_YAMLS[$IDX]}"
    done
    echo ""
    exit 0
  fi

  # Phase 3: fork all benchmark runs
  echo ""
  echo "  Starting $MODEL_COUNT parallel eval jobs..."
  echo ""
  declare -a PAR_PIDS
  for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
    SLOT=$((IDX + 1))
    LOG="$PROJECT_ROOT/logs/parallel_${PAR_SHORT_NAMES[$IDX]}.log"
    echo "  Slot $SLOT: ${PAR_SHORT_NAMES[$IDX]} → $LOG"

    RUN_CMD=(
      bash "$SCRIPT_DIR/run_all_benchmarks.sh"
      --slot    "$SLOT"
      --env-file "$SHARED_ENV"
      --model   "${PAR_TMP_YAMLS[$IDX]}"
    )
    [[ -n "$SKIP"  ]] && RUN_CMD+=(--skip  "$SKIP")
    [[ -n "$LIMIT" ]] && RUN_CMD+=(--limit "$LIMIT")

    "${RUN_CMD[@]}" > "$LOG" 2>&1 &
    PAR_PIDS[$IDX]=$!
  done

  echo ""
  echo "  All jobs running. Follow with:"
  echo "    tail -f logs/parallel_*.log"
  echo ""

  # Phase 4: wait and collect results
  declare -a PAR_RESULTS
  for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
    if wait "${PAR_PIDS[$IDX]}"; then
      PAR_RESULTS[$IDX]="OK"
    else
      PAR_RESULTS[$IDX]="FAILED"
    fi
    FINISH_TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  [$FINISH_TS] ${PAR_SHORT_NAMES[$IDX]} → ${PAR_RESULTS[$IDX]}"
    echo "[$FINISH_TS] DONE(parallel): ${PAR_SHORT_NAMES[$IDX]} → ${PAR_RESULTS[$IDX]}" >> "$LOG_FILE"
  done

  echo ""
  printf '╔══════════════════════════════════════════════════╗\n'
  printf '║  %-48s║\n' "ALL PARALLEL JOBS COMPLETE"
  for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
    printf '║  %-48s║\n' "  Slot $((IDX+1)): ${PAR_SHORT_NAMES[$IDX]} — ${PAR_RESULTS[$IDX]}"
  done
  printf '╚══════════════════════════════════════════════════╝\n'
  echo ""
  exit 0
fi

# ===========================================================================
# SEQUENTIAL MODE (default)
# ===========================================================================
for IDX in $(seq 0 $((MODEL_COUNT - 1))); do
  # First pass: get model identity without generating temp files yet
  MODEL_INFO=$(python3 -c "
import json
with open('$LIST_FILE') as f:
    d = json.load(f)
m = d['models'][$IDX]
print(m.get('hf_repo',''))
print(m.get('weights',''))
print(m.get('quant','q8_0'))
")
  HF_REPO=$(echo "$MODEL_INFO" | sed -n '1p')
  WEIGHTS=$(echo "$MODEL_INFO" | sed -n '2p')
  QUANT=$(echo "$MODEL_INFO" | sed -n '3p')
  MODEL_SHORT_NAME=$(echo "$HF_REPO" | cut -d/ -f2)

  echo ""
  echo "============================================"
  echo "  Model $((IDX + 1))/$MODEL_COUNT: $MODEL_SHORT_NAME"
  echo "============================================"
  echo ""

  LOCAL_GGUF=""

  # --- Generate temp config files ---
  TMP_VARS=$(mktemp)
  python3 "$SCRIPT_DIR/_generate_model_config.py" \
    "$LIST_FILE" "$IDX" "$PROJECT_ROOT" "$TMP_MODEL_YAML" "$TMP_ENV" "$LOCAL_GGUF" > "$TMP_VARS"
  # shellcheck disable=SC1090
  source "$TMP_VARS"
  rm -f "$TMP_VARS"

  if $DRY_RUN; then
    echo "  [dry-run] hf_repo  : $HF_REPO"
    if [[ -n "$LOCAL_GGUF" ]]; then
      echo "  [dry-run] local    : $(basename "$LOCAL_GGUF") (converted)"
    else
      echo "  [dry-run] weights  : $WEIGHTS"
    fi
    echo "  [dry-run] slot     : $SLOT  skip: ${SKIP:-none}"
    echo "  [dry-run] cmd      : run_all_benchmarks.sh --slot $SLOT --env-file .env._run_tmp --model _model_run_tmp"
    continue
  fi

  RUN_CMD=(
    bash "$SCRIPT_DIR/run_all_benchmarks.sh"
    --slot    "$SLOT"
    --env-file "$TMP_ENV"
    --model   "_model_run_tmp"
  )
  [[ -n "$SKIP"  ]] && RUN_CMD+=(--skip  "$SKIP")
  [[ -n "$LIMIT" ]] && RUN_CMD+=(--limit "$LIMIT")

  START_TIME=$(date +%s)
  "${RUN_CMD[@]}" || true
  END_TIME=$(date +%s)
  ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))
  FINISH_TS=$(date '+%Y-%m-%d %H:%M:%S')

  NEXT_MSG="All models complete!"
  [[ -n "$NEXT_MODEL_NAME" ]] && NEXT_MSG="Next: $NEXT_MODEL_NAME"

  printf '\n'
  printf '╔══════════════════════════════════════════════════╗\n'
  printf '║  %-48s║\n' "DONE: $MODEL_SHORT_NAME"
  printf '║  %-48s║\n' "Finished: $FINISH_TS  (${ELAPSED_MIN}m)"
  printf '║  %-48s║\n' "$NEXT_MSG"
  printf '╚══════════════════════════════════════════════════╝\n'
  printf '\n'

  echo "[$FINISH_TS] DONE: $MODEL_SHORT_NAME  (${ELAPSED_MIN}m)  skip=$SKIP" >> "$LOG_FILE"
done

echo "All $MODEL_COUNT model(s) complete. Log: $LOG_FILE"
echo ""
