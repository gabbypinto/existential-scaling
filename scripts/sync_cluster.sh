#!/bin/bash

# this is just whatever you set hostname to on the ssh config
# prob should change to some env value or something
CLUSTER="mlat_cluster_07"
REMOTE_DIR="~/existential-scaling"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz \
  "$PROJECT_ROOT/src" \
  "$PROJECT_ROOT/scripts" \
  "$PROJECT_ROOT/.env" \
  "$PROJECT_ROOT/docker-compose.yml" \
  "$PROJECT_ROOT/requirements.txt" \
  "$CLUSTER:$REMOTE_DIR/"
