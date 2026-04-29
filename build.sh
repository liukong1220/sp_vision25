#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录，并切换到项目根目录，避免从别的路径执行时找不到 CMakeLists.txt。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 构建目录默认使用 build，也允许通过外部环境变量覆盖。
BUILD_DIR="${BUILD_DIR:-build}"
# 默认使用 Release 构建，也允许外部传入 BUILD_TYPE=Debug 进行覆盖。
BUILD_TYPE="${BUILD_TYPE:-Release}"

# 检查是否存在项目的 CMakeLists.txt，避免误在错误目录执行脚本。
if [[ ! -f "CMakeLists.txt" ]]; then
  echo "[build] 错误：当前目录未找到 CMakeLists.txt"
  exit 1
fi

# 检查处理器核心数，用于给 cmake --build 自动分配并行编译任务数。
CPU_COUNT="$(nproc)"

# 适当保留 1 个核心给系统，避免编译时整机完全卡死。
# 如果机器核心数较少，则至少保留 1 个并行任务。
if (( CPU_COUNT <= 2 )); then
  BUILD_JOBS=1
else
  BUILD_JOBS="$(( CPU_COUNT - 1 ))"
fi

# 同时读取总内存，防止核心很多但内存较小时并行数过高导致编译抖动或 OOM。
MEM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
MEM_TOTAL_GB="$(( MEM_TOTAL_KB / 1024 / 1024 ))"

# 内存较小时进一步限制并行度，让脚本更稳。
if (( MEM_TOTAL_KB < 8 * 1024 * 1024 )); then
  BUILD_JOBS=1
elif (( MEM_TOTAL_KB < 16 * 1024 * 1024 )) && (( BUILD_JOBS > 4 )); then
  BUILD_JOBS=4
fi

echo "[build] 项目目录: ${PROJECT_DIR}"
echo "[build] 构建目录: ${BUILD_DIR}"
echo "[build] 构建类型: ${BUILD_TYPE}"
echo "[build] CPU 核心数: ${CPU_COUNT}"
echo "[build] 内存大小: ${MEM_TOTAL_GB} GB"
echo "[build] 并行编译任务数: -j${BUILD_JOBS}"

# 第一步：生成构建系统文件。
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

# 第二步：执行并行编译。
cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"

echo "[build] 编译完成"
