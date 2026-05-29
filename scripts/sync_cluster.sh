#!/bin/bash

# Usage: bash scripts/sync_cluster.sh [--host <ssh-host>] [--env-file <path>]
# Default host: mlat_cluster_07
# Default env:  .env (synced as .env on remote)
# Example (Spark): bash scripts/sync_cluster.sh --host mlat_spark_02 --env-file .env.dgx_spark

CLUSTER="mlat_cluster_07"
ENV_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)     CLUSTER="$2";  shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

REMOTE_DIR="~/existential-scaling"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve env file — default to .env, otherwise use the specified file
if [[ -z "$ENV_FILE" ]]; then
  ENV_SRC="$PROJECT_ROOT/.env"
else
  ENV_SRC="$PROJECT_ROOT/$ENV_FILE"
fi

[[ -f "$ENV_SRC" ]] || { echo "ERROR: env file not found: $ENV_SRC"; exit 1; }

rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
  "$PROJECT_ROOT/src" \
  "$PROJECT_ROOT/scripts" \
  "$PROJECT_ROOT/docker-compose.yml" \
  "$PROJECT_ROOT/Dockerfile" \
  "$PROJECT_ROOT/requirements.txt" \
  "$CLUSTER:$REMOTE_DIR/"

# Sync env file as .env on the remote
rsync -avz "$ENV_SRC" "$CLUSTER:$REMOTE_DIR/.env"
