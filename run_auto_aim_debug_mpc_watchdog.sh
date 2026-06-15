#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
BIN_PATH="${ROOT_DIR}/build/auto_aim_debug_mpc"
DEFAULT_CONFIG="${ROOT_DIR}/configs/standard3.yaml"
RESTART_DELAY="${RESTART_DELAY:-1}"
STARTUP_DELAY="${STARTUP_DELAY:-3}"
AUTO_OPEN_WEB="${AUTO_OPEN_WEB:-1}"
WEB_OPEN_TIMEOUT="${WEB_OPEN_TIMEOUT:-20}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_PATH="${LOG_DIR}/auto_aim_debug_mpc_watchdog.log"
WEB_HELPER_PID=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [config_path] [-- extra_args...]

Examples:
  $(basename "$0")
  $(basename "$0") configs/standard3.yaml

Environment:
  RESTART_DELAY   Seconds to wait before restart (default: 1)
  STARTUP_DELAY   Seconds to wait before first launch (default: 3)
  AUTO_OPEN_WEB   Auto open local web debugger in browser (default: 1)
  WEB_OPEN_TIMEOUT Seconds to wait for web debugger readiness (default: 20)
EOF
}

yaml_value() {
  local key="$1"
  local default_value="$2"
  local value=""

  if [[ -f "${CONFIG_PATH}" ]]; then
    value="$(
      awk -F ':' -v key="${key}" '
        $0 !~ /^[[:space:]]*#/ && $1 ~ ("^[[:space:]]*" key "[[:space:]]*$") {
          sub(/^[^:]*:[[:space:]]*/, "", $0)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
          gsub(/^"/, "", $0)
          gsub(/"$/, "", $0)
          print
          exit
        }
      ' "${CONFIG_PATH}" 2>/dev/null
    )"
  fi

  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${default_value}"
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_web_settings() {
  EFFECTIVE_DISABLE_WEB="$(yaml_value disable_web false)"
  EFFECTIVE_WEB_HOST="$(yaml_value web_host 127.0.0.1)"
  EFFECTIVE_WEB_PORT="$(yaml_value web_port 8090)"

  local i=0
  while [[ ${i} -lt ${#EXTRA_ARGS[@]} ]]; do
    local arg="${EXTRA_ARGS[i]}"
    case "${arg}" in
      --disable-web)
        EFFECTIVE_DISABLE_WEB=true
        ;;
      --disable-web=*)
        EFFECTIVE_DISABLE_WEB="${arg#*=}"
        ;;
      --web-host=*)
        EFFECTIVE_WEB_HOST="${arg#*=}"
        ;;
      --web-host)
        if [[ $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
          EFFECTIVE_WEB_HOST="${EXTRA_ARGS[i + 1]}"
          i=$((i + 1))
        fi
        ;;
      --web-port=*)
        EFFECTIVE_WEB_PORT="${arg#*=}"
        ;;
      --web-port)
        if [[ $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
          EFFECTIVE_WEB_PORT="${EXTRA_ARGS[i + 1]}"
          i=$((i + 1))
        fi
        ;;
    esac
    i=$((i + 1))
  done

  case "${EFFECTIVE_WEB_HOST}" in
    ""|0.0.0.0|localhost)
      WEB_OPEN_HOST="127.0.0.1"
      ;;
    *)
      WEB_OPEN_HOST="${EFFECTIVE_WEB_HOST}"
      ;;
  esac

  WEB_URL="http://${WEB_OPEN_HOST}:${EFFECTIVE_WEB_PORT}/"
  WEB_HEALTH_URL="http://${WEB_OPEN_HOST}:${EFFECTIVE_WEB_PORT}/healthz"
}

browser_open_available() {
  command -v xdg-open >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1
}

open_browser_url() {
  local url="$1"

  if command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "${url}" >/dev/null 2>&1 &
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    nohup python3 -m webbrowser "${url}" >/dev/null 2>&1 &
    return 0
  fi

  return 1
}

http_ready() {
  local url="$1"

  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 1 "${url}" >/dev/null 2>&1
    return $?
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -q -T 1 -O - "${url}" >/dev/null 2>&1
    return $?
  fi

  return 1
}

wait_and_open_web() {
  local url="$1"
  local health_url="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))

  echo "[watchdog] waiting for web debugger at ${url}"
  while (( SECONDS < deadline )); do
    if [[ "${STOP_REQUESTED}" -eq 1 ]]; then
      return 0
    fi

    if http_ready "${health_url}"; then
      if open_browser_url "${url}"; then
        echo "[watchdog] opened browser: ${url}"
      else
        echo "[watchdog] browser opener not available, open manually: ${url}"
      fi
      return 0
    fi

    sleep 0.5
  done

  echo "[watchdog] web debugger not ready within ${timeout_s}s, open manually if needed: ${url}"
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
  if [[ -n "${WEB_HELPER_PID}" ]]; then
    kill "${WEB_HELPER_PID}" >/dev/null 2>&1 || true
  fi
}
trap on_stop INT TERM

resolve_web_settings

echo "[watchdog] Command: ${BIN_PATH} ${CONFIG_PATH} ${EXTRA_ARGS[*]-}"
echo "[watchdog] RESTART_DELAY=${RESTART_DELAY}s"
echo "[watchdog] STARTUP_DELAY=${STARTUP_DELAY}s"
echo "[watchdog] AUTO_OPEN_WEB=${AUTO_OPEN_WEB}"
echo "[watchdog] WEB_OPEN_TIMEOUT=${WEB_OPEN_TIMEOUT}s"
echo "[watchdog] ROOT_DIR=${ROOT_DIR}"
echo "[watchdog] LOG_PATH=${LOG_PATH}"
echo "[watchdog] DISPLAY=${DISPLAY:-<empty>}"
echo "[watchdog] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<empty>}"
echo "[watchdog] WEB_URL=${WEB_URL}"
echo "[watchdog] Press Ctrl+C to stop."

if [[ "${STARTUP_DELAY}" != "0" ]]; then
  echo "[watchdog] initial startup delay ${STARTUP_DELAY}s..."
  sleep "${STARTUP_DELAY}"
fi

if is_truthy "${AUTO_OPEN_WEB}" && ! is_truthy "${EFFECTIVE_DISABLE_WEB}"; then
  if browser_open_available; then
    wait_and_open_web "${WEB_URL}" "${WEB_HEALTH_URL}" "${WEB_OPEN_TIMEOUT}" &
    WEB_HELPER_PID=$!
  else
    echo "[watchdog] browser opener not found, open manually: ${WEB_URL}"
  fi
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
