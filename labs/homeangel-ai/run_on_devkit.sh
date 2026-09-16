#!/usr/bin/env bash
set -euo pipefail

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  args=(--config /workspace/labs/homeangel-ai/config.devkit.yaml)
fi

if command -v devkit-run >/dev/null 2>&1; then
  exec devkit-run /workspace/labs/homeangel-ai/build/homeangel-ai "${args[@]}"
fi

if declare -F dk >/dev/null 2>&1; then
  exec dk /workspace/labs/homeangel-ai/build/homeangel-ai "${args[@]}"
fi

if [[ -n "${DEVKIT_SYNC_DEVKIT_IP:-}" ]]; then
  exec ssh -T \
    -p "${DEVKIT_SYNC_DEVKIT_PORT:-22}" \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    "${DEVKIT_SYNC_DEVKIT_USER:-sima}@${DEVKIT_SYNC_DEVKIT_IP}" \
    bash --noprofile --norc -s -- \
    /workspace/labs/homeangel-ai/build/homeangel-ai "${args[@]}" <<'REMOTE_RUN'
set -euo pipefail
target="$1"
shift
cd /workspace/labs/homeangel-ai
export LD_LIBRARY_PATH="/workspace/labs/homeangel-ai/libcompat:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/neat/runtime:/usr/lib/aarch64-linux-gnu/neat/gst-plugins:${LD_LIBRARY_PATH:-}"
exec "${target}" "$@"
REMOTE_RUN
fi

echo "DevKit runner not found. Run from the Neat SDK container shell and use:" >&2
echo "  dk /workspace/labs/homeangel-ai/build/homeangel-ai --config /workspace/labs/homeangel-ai/config.devkit.yaml" >&2
exit 1
