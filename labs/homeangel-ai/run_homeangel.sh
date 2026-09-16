#!/usr/bin/env bash
set -euo pipefail

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

exec /workspace/labs/homeangel-ai/build/homeangel-ai "$@"
