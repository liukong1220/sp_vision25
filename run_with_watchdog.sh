#!/usr/bin/env bash
# 文件名: run_with_watchdog.sh
# 功能:
# 1. 在新终端中启动一个“比赛运行型”程序（如 standard_mpc / sentry）
# 2. 程序退出后自动重启
# 3. 不附带浏览器 / Web 调试等待逻辑，尽量减少无关开销
#
# 适用场景:
# - standard_mpc
# - sentry
# - standard
# - mt_standard
# - 其他比赛主入口程序

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_NAME="${1:-standard_mpc}"
BIN_PATH="${ROOT_DIR}/build/${BIN_NAME}"

# 第二个参数允许显式传配置文件；未提供时按主程序给常用默认值
DEFAULT_CONFIG="${ROOT_DIR}/configs/standard3.yaml"
if [[ "${BIN_NAME}" == "sentry" ]]; then
  DEFAULT_CONFIG="${ROOT_DIR}/configs/sentry.yaml"
fi
CONFIG_PATH="${2:-${DEFAULT_CONFIG}}"

RESTART_DELAY="${RESTART_DELAY:-2}"

start_program_in_terminal() {
    local title="${BIN_NAME}"
    local inner_cmd="cd \"${ROOT_DIR}\" && \"${BIN_PATH}\" \"${CONFIG_PATH}\"; echo '程序已退出，窗口将关闭'; sleep 2"

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="${title}" -- bash -c "${inner_cmd}" &
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --title="${title}" --command="bash -c \"${inner_cmd}\"" &
    elif command -v konsole >/dev/null 2>&1; then
        konsole --hold -p tabtitle="${title}" -e bash -c "${inner_cmd}" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -T "${title}" -e bash -c "${inner_cmd}" &
    else
        echo "[比赛看门狗] 错误: 找不到可用的终端模拟器"
        exit 1
    fi

    sleep 2
    local pid
    pid=$(pgrep -f "${BIN_PATH}" | head -1 || true)
    echo "${pid}"
}

main() {
    [[ ! -x "${BIN_PATH}" ]] && { echo "错误: ${BIN_PATH} 不可执行"; exit 1; }

    echo "=========================================="
    echo "比赛看门狗已启动 (按 Ctrl+C 彻底退出)"
    echo "程序: ${BIN_PATH}"
    echo "配置: ${CONFIG_PATH}"
    echo "重启延迟: ${RESTART_DELAY} 秒"
    echo "工作目录: ${ROOT_DIR}"
    echo "=========================================="

    while true; do
        echo "[$(date '+%H:%M:%S')] 启动比赛程序..."

        local prog_pid
        prog_pid=$(start_program_in_terminal)
        if [[ -z "${prog_pid}" ]]; then
            echo "[比赛看门狗] 警告: 无法获取程序 PID，可能启动失败，等待后重试"
            sleep "${RESTART_DELAY}"
            continue
        fi

        while kill -0 "${prog_pid}" 2>/dev/null; do
            sleep 1
        done

        echo "[$(date '+%H:%M:%S')] 程序已退出，${RESTART_DELAY} 秒后重启..."
        sleep "${RESTART_DELAY}"
    done
}

main
