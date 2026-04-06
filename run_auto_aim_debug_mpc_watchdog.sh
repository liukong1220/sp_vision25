#!/usr/bin/env bash
# 文件名: run_auto_aim_with_restart.sh
# 功能: 每次在新终端中运行 auto_aim_debug_mpc，退出后自动重启，并自动打开浏览器
# 修正: 确保工作目录为项目根目录，避免相对路径错误

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${ROOT_DIR}/build/auto_aim_debug_mpc"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/standard3.yaml}"

RESTART_DELAY="${RESTART_DELAY:-2}"
WEB_READY_TIMEOUT="${WEB_READY_TIMEOUT:-10}"

# 从配置文件读取 web 设置（简单解析）
WEB_HOST="0.0.0.0"
WEB_PORT="8090"
DISABLE_WEB="false"

yaml_value() {
    local key="$1" default="$2" value=""
    [[ -f "${CONFIG_PATH}" ]] && value=$(awk -F ':' -v key="$key" '$0 !~ /^[[:space:]]*#/ && $1 ~ ("^[[:space:]]*" key "[[:space:]]*$") {sub(/^[^:]*:[[:space:]]*/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); gsub(/^"/, ""); gsub(/"$/, ""); print; exit}' "${CONFIG_PATH}" 2>/dev/null)
    echo "${value:-${default}}"
}

resolve_web_settings() {
    WEB_HOST=$(yaml_value web_host "0.0.0.0")
    WEB_PORT=$(yaml_value web_port "8090")
    DISABLE_WEB=$(yaml_value disable_web "false")
}

wait_for_web() {
    local url="http://127.0.0.1:${WEB_PORT}/healthz"
    local deadline=$((SECONDS + WEB_READY_TIMEOUT))
    echo "[自启动] 等待 Web 服务就绪 (${url}) ..."
    while [[ ${SECONDS} -lt ${deadline} ]]; do
        if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "${url}" >/dev/null 2>&1; then
            echo "[自启动] Web 服务已就绪"
            return 0
        fi
        if command -v wget >/dev/null 2>&1 && wget -q -T 1 -O - "${url}" >/dev/null 2>&1; then
            echo "[自启动] Web 服务已就绪"
            return 0
        fi
        sleep 1
    done
    echo "[自启动] 警告: Web 服务未就绪，仍尝试打开浏览器"
    return 1
}

open_browser() {
    local url="http://127.0.0.1:${WEB_PORT}/"
    echo "[自启动] 打开浏览器: ${url}"
    if command -v firefox >/dev/null 2>&1; then
        nohup firefox "${url}" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then
        nohup xdg-open "${url}" >/dev/null 2>&1 &
    elif command -v python3 >/dev/null 2>&1; then
        nohup python3 -m webbrowser "${url}" >/dev/null 2>&1 &
    else
        echo "[自启动] 无法自动打开浏览器，请手动访问: ${url}"
    fi
}

# 在新终端中启动程序，并返回程序的 PID（通过 pgrep 获取）
start_program_in_terminal() {
    local title="auto_aim_debug_mpc"
    # 关键修正：先 cd 到项目根目录，再运行程序，确保相对路径 assets/ 能被正确找到
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
        echo "[自启动] 错误: 找不到可用的终端模拟器"
        exit 1
    fi

    # 等待程序启动，然后获取其 PID
    sleep 2
    local pid=$(pgrep -f "${BIN_PATH}" | head -1 || true)
    echo "${pid}"
}

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# 主循环
main() {
    [[ ! -x "${BIN_PATH}" ]] && { echo "错误: ${BIN_PATH} 不可执行"; exit 1; }
    resolve_web_settings

    echo "=========================================="
    echo "自启动脚本已启动 (按 Ctrl+C 彻底退出)"
    echo "程序: ${BIN_PATH}"
    echo "配置: ${CONFIG_PATH}"
    echo "Web: http://${WEB_HOST}:${WEB_PORT}/"
    echo "重启延迟: ${RESTART_DELAY} 秒"
    echo "工作目录: ${ROOT_DIR}"
    echo "=========================================="

    local browser_opened=0

    while true; do
        echo "[$(date '+%H:%M:%S')] 启动程序..."

        local prog_pid=$(start_program_in_terminal)
        if [[ -z "${prog_pid}" ]]; then
            echo "[自启动] 警告: 无法获取程序 PID，可能启动失败，等待后重试"
            sleep "${RESTART_DELAY}"
            continue
        fi

        if [[ ${browser_opened} -eq 0 ]] && ! is_truthy "${DISABLE_WEB}"; then
            wait_for_web
            open_browser
            browser_opened=1
        fi

        # 等待程序退出
        while kill -0 "${prog_pid}" 2>/dev/null; do
            sleep 1
        done

        echo "[$(date '+%H:%M:%S')] 程序已退出，${RESTART_DELAY} 秒后重启..."
        sleep "${RESTART_DELAY}"
        browser_opened=0
    done
}

main
