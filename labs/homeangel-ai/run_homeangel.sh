#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${HOMEANGEL_APP_ROOT:-/workspace/labs/homeangel-ai}"
ENV_FILE="${HOMEANGEL_ENV_FILE:-${APP_ROOT}/.env.local}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

library_paths=(
  /workspace/labs/homeangel-ai/libcompat
  /lib/aarch64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  /usr/lib/aarch64-linux-gnu/neat/runtime
  /usr/lib/aarch64-linux-gnu/neat/gst-plugins
)

if [[ -d /opt/toolchain/aarch64/modalix/usr/lib ]]; then
  library_paths+=(
    /opt/toolchain/aarch64/modalix/usr/lib
    /opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu
    /opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/runtime
    /opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/neat/gst-plugins
  )
fi

joined_paths="$(IFS=:; echo "${library_paths[*]}")"
export LD_LIBRARY_PATH="${joined_paths}:${LD_LIBRARY_PATH:-}"

exec "${APP_ROOT}/build/homeangel-ai" "$@"
