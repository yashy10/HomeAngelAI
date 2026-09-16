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

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  args=(--config "${APP_ROOT}/config.devkit.yaml")
fi

if command -v devkit-run >/dev/null 2>&1; then
  exec devkit-run "${APP_ROOT}/run_homeangel.sh" "${args[@]}"
fi

if declare -F dk >/dev/null 2>&1; then
  exec dk "${APP_ROOT}/run_homeangel.sh" "${args[@]}"
fi

if [[ -n "${DEVKIT_SYNC_DEVKIT_IP:-}" ]]; then
  exec ssh -T \
    -p "${DEVKIT_SYNC_DEVKIT_PORT:-22}" \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    "${DEVKIT_SYNC_DEVKIT_USER:-sima}@${DEVKIT_SYNC_DEVKIT_IP}" \
    bash --noprofile --norc -s -- \
    "${APP_ROOT}/run_homeangel.sh" "${args[@]}" <<'REMOTE_RUN'
set -euo pipefail
target="$1"
shift
exec "${target}" "$@"
REMOTE_RUN
fi

echo "DevKit runner not found. Run from the Neat SDK container shell and use:" >&2
echo "  dk /workspace/labs/homeangel-ai/run_homeangel.sh --config /workspace/labs/homeangel-ai/config.devkit.yaml" >&2
exit 1
