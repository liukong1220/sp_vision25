# 仿真共享内存适配层。
#
# 这一层只依赖 OpenCV / Eigen 和 sp_vision25 自己的 tools，
# 刻意不依赖 io 里的相机与串口实现：仿真入口不允许初始化物理相机和串口。
# 之所以还要 io/serial/include，是因为 io/gimbal/gimbal.hpp 里的
# io::GimbalState 定义要复用（只用类型，不打开串口），而那个头文件
# 顺带 include 了 serial/serial.h。

add_library(sim_io STATIC
  clock_bridge.cpp
  shared_memory_client.cpp
  sim_camera.cpp
  sim_gimbal.cpp
  sim_ground_truth.cpp
)

target_include_directories(sim_io PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/io/serial/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sim_io PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  tools
)

target_compile_options(sim_io PRIVATE -Wall -Wextra)
