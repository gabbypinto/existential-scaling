#!/usr/bin/env bash
# Measure decode speed under concurrent requests, vLLM vs llama.cpp.
#
# Starts a dedicated benchmark server container, sweeps concurrency levels with
# `vllm bench serve` (same client for both engines), saves one JSON per level,
# then removes the server. Does not touch the docker-compose llm_N slots.
#
# Usage:
#   bash scripts/bench_concurrency.sh --engine vllm --model google/gemma-3-12b-it
#   bash scripts/bench_concurrency.sh --engine llamacpp --gguf models/gemma-3-12b-it-Q8_0.gguf \
#        --tokenizer google/gemma-3-12b-it
#   python3 scripts/summarize_bench.py          # side-by-side table of all runs
#
# --engine        vllm | llamacpp                                    (required)
# --model         HF model id to serve (vllm)                        (required for vllm)
# --gguf          path to a .gguf file (llamacpp)                    (required for llamacpp)
# --tokenizer     HF model id for the random prompts / token counts  (default: --model; required for llamacpp)
# --name          results label                                      (default: derived from model / gguf)
# --conc          comma-separated concurrency levels                 (default: 1,2,4,8,16,32)
# --in / --out    input / output tokens per request                  (default: 512 / 256)
# --image         server image          (default: vllm -> $VLLM_IMAGE_DEFAULT, llamacpp -> $LLAMA_IMAGE_DEFAULT)
# --client-image  image that runs `vllm bench serve`                 (default: $VLLM_IMAGE_DEFAULT)
# --port          host port for the benchmark server                 (default: 20010)
# --gpu-mem       vLLM --gpu-memory-utilization (share of unified memory) (default: 0.8)
# --server-args   extra server flags, e.g. "--language-model-only"   (default: none)
# --timeout       seconds to wait for the server to become healthy   (default: 1800)
# --env-file      env file to read HF_TOKEN from                     (default: .env)
# --force         don't ask before running alongside other llm containers
# --dry-run       print the docker commands instead of running them
#
# Results: logs/bench_concurrency/<engine>/<name>/conc_<N>.json + meta.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pinned NGC image from NVIDIA's DGX Spark playbook. Gemma 4 needs vllm/vllm-openai:gemma4-cu130.
VLLM_IMAGE_DEFAULT="nvcr.io/nvidia/vllm:26.05.post1-py3"
LLAMA_IMAGE_DEFAULT="ghcr.io/ggml-org/llama.cpp:server-cuda"

ENGINE=""
MODEL=""
GGUF=""
TOKENIZER=""
NAME=""
CONC="1,2,4,8,16,32"
IN_LEN=512
OUT_LEN=256
IMAGE=""
CLIENT_IMAGE="$VLLM_IMAGE_DEFAULT"
PORT=20010
GPU_MEM=0.8
SERVER_ARGS=""
TIMEOUT=1800
ENV_FILE="$PROJECT_ROOT/.env"
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)       ENGINE="$2";       shift 2 ;;
    --model)        MODEL="$2";        shift 2 ;;
    --gguf)         GGUF="$2";         shift 2 ;;
    --tokenizer)    TOKENIZER="$2";    shift 2 ;;
    --name)         NAME="$2";         shift 2 ;;
    --conc)         CONC="$2";         shift 2 ;;
    --in)           IN_LEN="$2";       shift 2 ;;
    --out)          OUT_LEN="$2";      shift 2 ;;
    --image)        IMAGE="$2";        shift 2 ;;
    --client-image) CLIENT_IMAGE="$2"; shift 2 ;;
    --port)         PORT="$2";         shift 2 ;;
    --gpu-mem)      GPU_MEM="$2";      shift 2 ;;
    --server-args)  SERVER_ARGS="$2";  shift 2 ;;
    --timeout)      TIMEOUT="$2";      shift 2 ;;
    --env-file)     ENV_FILE="$2";     shift 2 ;;
    --force)        FORCE=true;        shift   ;;
    --dry-run)      DRY_RUN=true;      shift   ;;
    -h|--help)      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------- validate ----------
case "$ENGINE" in
  vllm)
    [[ -n "$MODEL" ]] || { echo "ERROR: --engine vllm needs --model <hf-id>"; exit 1; }
    TOKENIZER="${TOKENIZER:-$MODEL}"
    IMAGE="${IMAGE:-$VLLM_IMAGE_DEFAULT}"
    NAME="${NAME:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"
    ;;
  llamacpp)
    [[ -n "$GGUF" ]]      || { echo "ERROR: --engine llamacpp needs --gguf <path>"; exit 1; }
    [[ -n "$TOKENIZER" ]] || { echo "ERROR: --engine llamacpp needs --tokenizer <hf-id> (the model's HF repo)"; exit 1; }
    $DRY_RUN || [[ -f "$GGUF" ]] || { echo "ERROR: gguf not found: $GGUF"; exit 1; }
    GGUF_DIR="$(cd "$(dirname "$GGUF")" 2>/dev/null && pwd || dirname "$GGUF")"
    GGUF_BASE="$(basename "$GGUF")"
    IMAGE="${IMAGE:-$LLAMA_IMAGE_DEFAULT}"
    NAME="${NAME:-$(basename "$GGUF_BASE" .gguf | tr '[:upper:]' '[:lower:]')}"
    ;;
  *) echo "ERROR: --engine must be vllm or llamacpp"; exit 1 ;;
esac

IFS=',' read -ra LEVELS <<< "$CONC"
MAXCONC=0
for c in "${LEVELS[@]}"; do (( c > MAXCONC )) && MAXCONC=$c; done
SEQ_LEN=$((IN_LEN + OUT_LEN + 256))   # per-request context, with headroom
read -ra EXTRA_SERVER_ARGS <<< "$SERVER_ARGS"

_env_val() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true; }
HF_TOKEN="${HF_TOKEN:-$(_env_val HF_TOKEN)}"
# exported so `docker run -e HF_TOKEN` passes it through without putting it on the command line
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

RESULT_DIR="$PROJECT_ROOT/logs/bench_concurrency/$ENGINE/$NAME"
CONTAINER="${USER}_bench_${ENGINE}"
BASE_URL="http://127.0.0.1:$PORT"

# Run or (with --dry-run) print a command given as an array
run() {
  if $DRY_RUN; then printf '  '; printf '%q ' "$@"; echo; else "$@"; fi
}

echo ""
echo "============================================"
echo "  Concurrency benchmark — $(date)"
echo "  Engine     : $ENGINE ($IMAGE)"
echo "  Model      : ${MODEL:-$GGUF}"
echo "  Tokenizer  : $TOKENIZER"
echo "  Levels     : ${LEVELS[*]}  (in=$IN_LEN, out=$OUT_LEN)"
[[ -n "$SERVER_ARGS" ]] && echo "  Server args: $SERVER_ARGS"
echo "  Results    : ${RESULT_DIR#$PROJECT_ROOT/}"
$DRY_RUN && echo "  DRY RUN — commands are printed, not run"
echo "============================================"
echo ""

# ---------- pre-flight: other servers share the unified memory and bandwidth ----------
if ! $DRY_RUN; then
  OTHERS=$(docker ps --format '{{.Names}}' | grep -E "^${USER}_(llm|bench|vllm)_" | grep -vx "$CONTAINER" || true)
  if [[ -n "$OTHERS" ]] && ! $FORCE; then
    echo "WARNING: other model containers are running:"
    echo "$OTHERS" | sed 's/^/    /'
    echo "  They share memory and bandwidth with the benchmark, which distorts the"
    echo "  numbers and can stop vLLM from starting. Stop them first, e.g.:"
    echo "    docker compose stop llm_1 llm_2 llm_3"
    if [[ -t 0 ]]; then
      read -rp "  Continue anyway? [y/N] " ans
      [[ "$ans" == [yY]* ]] || exit 1
    else
      echo "  Not interactive — re-run with --force to continue anyway."; exit 1
    fi
  fi
  mkdir -p "$RESULT_DIR"
fi

# ---------- 1. start the server ----------
cleanup() {
  $DRY_RUN && return
  echo ""
  echo "Removing $CONTAINER..."
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[1/3] Starting $ENGINE server: $CONTAINER"
$DRY_RUN || docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [[ "$ENGINE" == "vllm" ]]; then
  # --entrypoint vllm + "serve" works for both NGC images (no server entrypoint)
  # and vllm/vllm-openai images. No --enforce-eager: it disables CUDA graphs.
  SERVER_CMD=(docker run -d --name "$CONTAINER" --gpus all --ipc=host
    -p "127.0.0.1:$PORT:8000"
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN
    --entrypoint vllm "$IMAGE"
    serve "$MODEL" --served-model-name "$NAME" --port 8000
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$SEQ_LEN"
    --max-num-seqs "$MAXCONC"
    ${EXTRA_SERVER_ARGS[@]+"${EXTRA_SERVER_ARGS[@]}"})
else
  # Same flags as the docker-compose llm_N slots, plus one slot per concurrent request
  SERVER_CMD=(docker run -d --name "$CONTAINER" --gpus all --ipc=host
    -p "127.0.0.1:$PORT:8080"
    -v "$GGUF_DIR:/bench_models:ro"
    "$IMAGE"
    --model "/bench_models/$GGUF_BASE" -ngl 999 --flash-attn auto --no-mmproj
    -np "$MAXCONC" --ctx-size "$((MAXCONC * SEQ_LEN))"
    --host 0.0.0.0 --port 8080 --alias "$NAME"
    ${EXTRA_SERVER_ARGS[@]+"${EXTRA_SERVER_ARGS[@]}"})
fi
run "${SERVER_CMD[@]}"

# ---------- 2. wait for health, then warm up ----------
echo "[2/3] Waiting for $BASE_URL/health$([[ $ENGINE == vllm ]] && echo " (first vLLM load can take 10-15 min)")..."
if ! $DRY_RUN; then
  ELAPSED=0
  while true; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" || true)
    [[ "$CODE" == "200" ]] && { echo "      -> Ready (${ELAPSED}s)"; break; }
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "ERROR: $CONTAINER exited during startup. Last log lines:"
      docker logs --tail 30 "$CONTAINER" 2>&1 || true
      exit 1
    fi
    if (( ELAPSED >= TIMEOUT )); then
      echo "ERROR: not healthy after ${TIMEOUT}s. Last log lines:"
      docker logs --tail 30 "$CONTAINER" 2>&1 || true
      exit 1
    fi
    printf "      -> HTTP %s, waiting... [%ds/%ds]\r" "$CODE" "$ELAPSED" "$TIMEOUT"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
  done
  # vLLM JIT-compiles on the first request; keep that out of the measurements
  curl -s -o /dev/null "$BASE_URL/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\": \"$NAME\", \"prompt\": \"Hello\", \"max_tokens\": 16}" || true
fi

if ! $DRY_RUN; then
  cat > "$RESULT_DIR/meta.json" <<EOF
{
  "engine": "$ENGINE",
  "name": "$NAME",
  "model": "${MODEL:-$GGUF}",
  "tokenizer": "$TOKENIZER",
  "image": "$IMAGE",
  "input_len": $IN_LEN,
  "output_len": $OUT_LEN,
  "server_args": "$SERVER_ARGS",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
fi

# ---------- 3. concurrency sweep ----------
echo "[3/3] Sweeping concurrency: ${LEVELS[*]}"
for C in "${LEVELS[@]}"; do
  N=$(( 4 * C > 16 ? 4 * C : 16 ))
  echo ""
  echo "  ---- concurrency $C ($N requests) ----"
  CLIENT_CMD=(docker run --rm --network host
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -v "$RESULT_DIR:/results"
    -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN
    --entrypoint vllm "$CLIENT_IMAGE"
    bench serve --backend openai --base-url "$BASE_URL" --endpoint /v1/completions
    --model "$NAME" --tokenizer "$TOKENIZER"
    --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" --ignore-eos
    --max-concurrency "$C" --num-prompts "$N"
    --save-result --result-dir /results --result-filename "conc_$C.json")
  run "${CLIENT_CMD[@]}"
done

# the client container writes as root; hand the results back to the user
if ! $DRY_RUN; then
  docker run --rm -v "$RESULT_DIR:/results" --entrypoint chown "$CLIENT_IMAGE" \
    -R "$(id -u):$(id -g)" /results >/dev/null 2>&1 || true
fi

echo ""
echo "============================================"
echo "  Done -> ${RESULT_DIR#$PROJECT_ROOT/}"
echo "  Summary: python3 scripts/summarize_bench.py"
echo "============================================"
