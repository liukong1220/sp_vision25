#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

BIN_NAME="${1:-auto_debug}"
DEFAULT_CONFIG="${ROOT_DIR}/configs/sentry.yaml"
CONFIG_PATH="${2:-${DEFAULT_CONFIG}}"
RESTART_DELAY="${RESTART_DELAY:-2}"
WEB_READY_TIMEOUT="${WEB_READY_TIMEOUT:-10}"
AUTO_OPEN_BROWSER="${AUTO_OPEN_BROWSER:-1}"

ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
if [[ -z "${AMENT_PREFIX_PATH:-}" && -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
  set -u
fi
if [[ -z "${COLCON_PREFIX_PATH:-}" && -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${WORKSPACE_ROOT}/install/setup.bash"
  set -u
fi

resolve_binary() {
  if [[ -x "${ROOT_DIR}/build/${BIN_NAME}" ]]; then
    echo "${ROOT_DIR}/build/${BIN_NAME}"
  elif [[ -x "${WORKSPACE_ROOT}/install/sp_vision25/lib/sp_vision25/${BIN_NAME}" ]]; then
    echo "${WORKSPACE_ROOT}/install/sp_vision25/lib/sp_vision25/${BIN_NAME}"
  else
    return 1
  fi
}

BIN_PATH="$(resolve_binary || true)"
if [[ -z "${BIN_PATH}" ]]; then
  echo "[debug_watchdog] ERROR: executable not found for '${BIN_NAME}' (checked build/ and install/)" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[debug_watchdog] ERROR: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

yaml_value() {
  local key="$1" default="$2" value=""
  value="$(awk -F ':' -v key="$key" '
    $0 !~ /^[[:space:]]*#/ && $1 ~ ("^[[:space:]]*" key "[[:space:]]*$") {
      sub(/^[^:]*:[[:space:]]*/, "");
      gsub(/^[[:space:]]+|[[:space:]]+$/, "");
      gsub(/^"/, ""); gsub(/"$/, "");
      print; exit
    }' "${CONFIG_PATH}" 2>/dev/null || true)"
  echo "${value:-${default}}"
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

WEB_HOST="$(yaml_value web_host "127.0.0.1")"
WEB_PORT="$(yaml_value web_port "8090")"
DISABLE_WEB="$(yaml_value disable_web "false")"

wait_for_web() {
  local url="http://127.0.0.1:${WEB_PORT}/healthz"
  local deadline=$((SECONDS + WEB_READY_TIMEOUT))
  echo "[debug_watchdog] 等待 Web 服务就绪 (${url}) ..."
  while [[ ${SECONDS} -lt ${deadline} ]]; do
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "${url}" >/dev/null 2>&1; then
      echo "[debug_watchdog] Web 服务已就绪"
      return 0
    fi
    if command -v wget >/dev/null 2>&1 && wget -q -T 1 -O - "${url}" >/dev/null 2>&1; then
      echo "[debug_watchdog] Web 服务已就绪"
      return 0
    fi
    sleep 1
  done
  echo "[debug_watchdog] WARN: Web 服务等待超时"
  return 1
}

open_browser() {
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    return 0
  fi
  local url="http://127.0.0.1:${WEB_PORT}/"
  echo "[debug_watchdog] 打开浏览器: ${url}"
  if command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "${url}" >/dev/null 2>&1 &
  elif command -v firefox >/dev/null 2>&1; then
    nohup firefox "${url}" >/dev/null 2>&1 &
  fi
}

CHILD_PID=""
STOP_REQUESTED=0

cleanup() {
  STOP_REQUESTED=1
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] 停止调试程序 PID=${CHILD_PID}"
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${CHILD_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "${CHILD_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "=========================================="
echo "调试看门狗已启动 (按 Ctrl+C 退出)"
echo "程序: ${BIN_PATH}"
echo "配置: ${CONFIG_PATH}"
echo "Web: http://${WEB_HOST}:${WEB_PORT}/"
echo "重启延迟: ${RESTART_DELAY} 秒"
echo "工作目录: ${ROOT_DIR}"
echo "=========================================="

cd "${ROOT_DIR}"
while (( ! STOP_REQUESTED )); do
  echo "[$(date '+%H:%M:%S')] 启动调试程序..."
  "${BIN_PATH}" "${CONFIG_PATH}" &
  CHILD_PID="$!"

  if is_truthy "${AUTO_OPEN_BROWSER}" && ! is_truthy "${DISABLE_WEB}"; then
    wait_for_web && open_browser || true
  fi

  set +e
  wait "${CHILD_PID}"
  EXIT_STATUS=$?
  set -e
  CHILD_PID=""

  if (( STOP_REQUESTED )); then
    break
  fi
  echo "[$(date '+%H:%M:%S')] 程序已退出 (status=${EXIT_STATUS})，${RESTART_DELAY} 秒后重启..."
  sleep "${RESTART_DELAY}"
done
