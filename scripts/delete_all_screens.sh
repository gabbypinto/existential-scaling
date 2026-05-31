#!/usr/bin/env bash
# Kill all screen sessions and clean up dead ones.
# Usage: bash scripts/delete_all_screens.sh

set -euo pipefail

SESSIONS=$(screen -ls 2>/dev/null | grep -E '^\s+[0-9]+\.' | awk '{print $1}' || true)

if [[ -z "$SESSIONS" ]]; then
  echo "No screen sessions found."
  exit 0
fi

echo "Killing screen sessions:"
while IFS= read -r session; do
  echo "  $session"
  screen -S "$session" -X quit 2>/dev/null || true
done <<< "$SESSIONS"

# Clean up any dead sessions left behind
screen -wipe 2>/dev/null || true

echo "Done."
