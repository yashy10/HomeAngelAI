#!/usr/bin/env bash
set -euo pipefail

cd /workspace/labs/homeangel-ai

bind_addr="${HOMEANGEL_CONSOLE_BIND:-127.0.0.1}"
port="${HOMEANGEL_CONSOLE_PORT:-8765}"

echo "Serving HomeAngel demo console at http://${bind_addr}:${port}/console/"
echo "Serving local telemetry from /workspace/labs/homeangel-ai/telemetry.json"
echo "Serving local events from /workspace/labs/homeangel-ai/events.log"

exec python3 /workspace/labs/homeangel-ai/console/server.py \
  --bind "${bind_addr}" \
  --port "${port}"
