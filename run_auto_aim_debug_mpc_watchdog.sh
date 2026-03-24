#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
BIN_PATH="${ROOT_DIR}/build/auto_aim_debug_mpc"
DEFAULT_CONFIG="${ROOT_DIR}/configs/standard3.yaml"
RESTART_DELAY="${RESTART_DELAY:-1}"
STARTUP_DELAY="${STARTUP_DELAY:-3}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_PATH="${LOG_DIR}/auto_aim_debug_mpc_watchdog.log"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [config_path] [-- extra_args...]

Examples:
  $(basename "$0")
  $(basename "$0") configs/standard3.yaml

Environment:
  RESTART_DELAY   Seconds to wait before restart (default: 1)
  STARTUP_DELAY   Seconds to wait before first launch (default: 8)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Error: executable not found: ${BIN_PATH}"
  echo "Build first: make -C build -j\$(nproc)"
  exit 1
fi

mkdir -p "${LOG_DIR}"
touch "${LOG_PATH}"
if [[ -t 1 ]]; then
  exec > >(tee -a "${LOG_PATH}") 2>&1
else
  exec >>"${LOG_PATH}" 2>&1
fi

cd "${ROOT_DIR}"

CONFIG_INPUT="${1:-${DEFAULT_CONFIG}}"
if [[ "${CONFIG_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
elif [[ -f "${CONFIG_INPUT}" ]]; then
  CONFIG_PATH="${CONFIG_INPUT}"
elif [[ -f "${ROOT_DIR}/${CONFIG_INPUT}" ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_INPUT}"
else
  echo "Error: config file not found: ${CONFIG_INPUT}"
  exit 1
fi

EXTRA_ARGS=()
if [[ $# -gt 1 ]]; then
  if [[ "${2}" == "--" ]]; then
    EXTRA_ARGS=("${@:3}")
  else
    EXTRA_ARGS=("${@:2}")
  fi
fi

STOP_REQUESTED=0
on_stop() {
  STOP_REQUESTED=1
}
trap on_stop INT TERM

echo "[watchdog] Command: ${BIN_PATH} ${CONFIG_PATH} ${EXTRA_ARGS[*]-}"
echo "[watchdog] RESTART_DELAY=${RESTART_DELAY}s"
echo "[watchdog] STARTUP_DELAY=${STARTUP_DELAY}s"
echo "[watchdog] ROOT_DIR=${ROOT_DIR}"
echo "[watchdog] LOG_PATH=${LOG_PATH}"
echo "[watchdog] DISPLAY=${DISPLAY:-<empty>}"
echo "[watchdog] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<empty>}"
echo "[watchdog] Press Ctrl+C to stop."

if [[ "${STARTUP_DELAY}" != "0" ]]; then
  echo "[watchdog] initial startup delay ${STARTUP_DELAY}s..."
  sleep "${STARTUP_DELAY}"
fi

while true; do
  if [[ "${STOP_REQUESTED}" -eq 1 ]]; then
    echo "[watchdog] stop requested, exiting."
    break
  fi

  START_TS="$(date '+%F %T')"
  echo "[${START_TS}] start auto_aim_debug_mpc"

  set +e
  "${BIN_PATH}" "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
  EXIT_CODE=$?
  set -e

  END_TS="$(date '+%F %T')"
  if [[ "${STOP_REQUESTED}" -eq 1 ]]; then
    echo "[${END_TS}] stopped."
    break
  fi

  echo "[${END_TS}] exited with code ${EXIT_CODE}, restart in ${RESTART_DELAY}s..."
  sleep "${RESTART_DELAY}"
done
