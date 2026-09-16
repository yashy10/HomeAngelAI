#!/usr/bin/env bash
set -euo pipefail

cd /workspace/labs/homeangel-ai

env_file="${HOMEANGEL_ENV_FILE:-/workspace/labs/homeangel-ai/.env.local}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

bind_addr="${HOMEANGEL_CONSOLE_BIND:-127.0.0.1}"
port="${HOMEANGEL_CONSOLE_PORT:-8765}"

echo "Serving HomeAngel demo console at http://${bind_addr}:${port}/console/"
echo "Serving local telemetry from /workspace/labs/homeangel-ai/telemetry.json"
echo "Serving local events from /workspace/labs/homeangel-ai/events.log"

exec python3 /workspace/labs/homeangel-ai/console/server.py \
  --bind "${bind_addr}" \
  --port "${port}"
