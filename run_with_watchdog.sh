#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

BIN_NAME="${1:-sentry}"
DEFAULT_CONFIG="${ROOT_DIR}/configs/standard3.yaml"
if [[ "${BIN_NAME}" == "sentry" || "${BIN_NAME}" == sentry_* ]]; then
  DEFAULT_CONFIG="${ROOT_DIR}/configs/sentry.yaml"
fi
CONFIG_PATH="${2:-${DEFAULT_CONFIG}}"
RESTART_DELAY="${RESTART_DELAY:-2}"

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
  echo "[vision_watchdog] ERROR: executable not found for '${BIN_NAME}' (checked build/ and install/)" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[vision_watchdog] ERROR: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

CHILD_PID=""
STOP_REQUESTED=0

cleanup() {
  STOP_REQUESTED=1
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] 停止视觉程序 PID=${CHILD_PID}"
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
echo "视觉看门狗已启动 (按 Ctrl+C 退出)"
echo "程序: ${BIN_PATH}"
echo "配置: ${CONFIG_PATH}"
echo "重启延迟: ${RESTART_DELAY} 秒"
echo "工作目录: ${ROOT_DIR}"
echo "=========================================="

cd "${ROOT_DIR}"
while (( ! STOP_REQUESTED )); do
  echo "[$(date '+%H:%M:%S')] 启动视觉程序..."
  "${BIN_PATH}" "${CONFIG_PATH}" &
  CHILD_PID="$!"

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
