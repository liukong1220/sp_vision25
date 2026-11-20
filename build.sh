#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ ! -f "CMakeLists.txt" ]]; then
  echo "[build] ERROR: CMakeLists.txt not found in ${PROJECT_DIR}" >&2
  exit 1
fi

source_compat() {
  local script_path="$1"
  set +u
  # shellcheck disable=SC1090
  source "${script_path}"
  set -u
}

ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
if [[ -z "${AMENT_PREFIX_PATH:-}" && -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  source_compat "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
  echo "[build] loaded ROS2 environment: /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
fi
if [[ -z "${COLCON_PREFIX_PATH:-}" && -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  source_compat "${WORKSPACE_ROOT}/install/setup.bash"
  echo "[build] loaded workspace overlay: ${WORKSPACE_ROOT}/install/setup.bash"
fi

CPU_COUNT="$(nproc)"
MEM_AVAILABLE_MB="$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
BUILD_MEM_PER_JOB_MB="${BUILD_MEM_PER_JOB_MB:-1024}"
AUTO_JOBS=$(( MEM_AVAILABLE_MB / BUILD_MEM_PER_JOB_MB ))
MAX_CPU_JOBS=$(( CPU_COUNT > 2 ? CPU_COUNT - 1 : 1 ))
(( AUTO_JOBS > MAX_CPU_JOBS )) && AUTO_JOBS="${MAX_CPU_JOBS}"
(( AUTO_JOBS < 1 )) && AUTO_JOBS=1
BUILD_JOBS="${BUILD_JOBS:-${AUTO_JOBS}}"

echo "[build] project: ${PROJECT_DIR}"
echo "[build] build dir: ${BUILD_DIR}"
echo "[build] build type: ${BUILD_TYPE}"
echo "[build] cpu: ${CPU_COUNT}, mem available: ${MEM_AVAILABLE_MB} MiB, jobs: -j${BUILD_JOBS}"

cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_TESTING=OFF "$@"
cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"
echo "[build] done"
