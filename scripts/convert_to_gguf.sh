#!/usr/bin/env bash
# Convert a HuggingFace model to GGUF using llama.cpp, then delete the HF cache.
#
# Usage:
#   bash scripts/convert_to_gguf.sh --model Qwen/Qwen2.5-Coder-7B --family qwen2.5_coder
#   bash scripts/convert_to_gguf.sh --model google/gemma-4-E4B-it --family gemma4 --quant f16
#   bash scripts/convert_to_gguf.sh --model Qwen/Qwen2.5-Coder-7B   # flat models/ output
#
# --model    HuggingFace repo ID or full URL  (required)
# --family   subdir under models/ to store the GGUF  (optional; default: flat models/)
# --quant    q8_0 or f16                      (default: q8_0)
# --llama    path to llama.cpp repo           (default: ~/llama.cpp)
#
# Output: models/<family>/<MODEL_SHORT>-<quant>.gguf
# After conversion the safetensors HF cache for this model is deleted automatically.
#
# To use the result, add to .env:
#   LOCAL_MODEL_N=/app/models/<family>/<MODEL_SHORT>-<quant>.gguf

set -euo pipefail

MODEL=""
QUANT="q8_0"
FAMILY=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2";     shift 2 ;;
    --quant)   QUANT="$2";     shift 2 ;;
    --family)  FAMILY="$2";    shift 2 ;;
    --llama)   LLAMA_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$MODEL" ]] || {
  echo "Usage: $0 --model <hf-repo-or-url> [--family <subdir>] [--quant q8_0|f16] [--llama ~/llama.cpp]"
  exit 1
}

# Strip full URL to repo ID (e.g. https://huggingface.co/Qwen/Qwen3.5-4B → Qwen/Qwen3.5-4B)
MODEL="${MODEL#https://huggingface.co/}"
MODEL="${MODEL%/}"

# Validate quant
case "$QUANT" in
  f16|q8_0) ;;
  *) echo "ERROR: --quant must be f16 or q8_0 (got: $QUANT)"; exit 1 ;;
esac

MODEL_SHORT=$(echo "$MODEL" | cut -d/ -f2)

if [[ -n "$FAMILY" ]]; then
  OUTDIR="$PROJECT_ROOT/models/$FAMILY"
else
  OUTDIR="$PROJECT_ROOT/models"
fi
OUTFILE="${OUTDIR}/${MODEL_SHORT}-${QUANT}.gguf"

echo ""
echo "============================================"
echo "  Model  : $MODEL"
echo "  Quant  : $QUANT"
echo "  Output : $OUTFILE"
echo "  llama  : $LLAMA_DIR"
echo "============================================"
echo ""

# Clone llama.cpp if not present
if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "Cloning llama.cpp into $LLAMA_DIR ..."
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

CONVERT="$LLAMA_DIR/convert_hf_to_gguf.py"
[[ -f "$CONVERT" ]] || { echo "ERROR: convert_hf_to_gguf.py not found in $LLAMA_DIR"; exit 1; }

# Install conversion dependencies (quiet, skip if already installed)
echo "Installing conversion deps..."
python3 -m pip install -q -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"

mkdir -p "$OUTDIR"

# Download model to local HF cache (instant if already cached)
echo "Downloading $MODEL to local cache..."
MODEL_DIR=$(python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
print(snapshot_download('$MODEL', token=token))
")
echo "Model at: $MODEL_DIR"

echo ""
echo "Converting → $OUTFILE ..."
echo ""
python3 "$CONVERT" "$MODEL_DIR" --outfile "$OUTFILE" --outtype "$QUANT"

echo ""
echo "Done: $OUTFILE"
echo "Size: $(du -sh "$OUTFILE" | cut -f1)"

# Delete the HF cache for this model (safetensors no longer needed after conversion)
CACHE_REPO_DIR=$(dirname "$(dirname "$MODEL_DIR")")
if [[ "$CACHE_REPO_DIR" == *"huggingface/hub/models--"* ]]; then
  echo ""
  echo "Cleaning HF cache: $CACHE_REPO_DIR"
  rm -rf "$CACHE_REPO_DIR"
  echo "Cache deleted."
else
  echo ""
  echo "WARNING: Unexpected cache path '$CACHE_REPO_DIR' — skipping cleanup."
fi

echo ""
echo "Add to .env to use this model:"
if [[ -n "$FAMILY" ]]; then
  echo "  LOCAL_MODEL_N=/app/models/${FAMILY}/${MODEL_SHORT}-${QUANT}.gguf"
else
  echo "  LOCAL_MODEL_N=/app/models/${MODEL_SHORT}-${QUANT}.gguf"
fi
