#!/usr/bin/env bash
set -euo pipefail

if [[ -f "$HOME/pyneat/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/pyneat/bin/activate"
fi

exec python3 /workspace/labs/homeangel-ai/vlm_server.py "$@"
