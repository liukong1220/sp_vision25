cmake_minimum_required(VERSION 3.16.3)

project(sp_vision25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

# `sp_msgs` 的 rosidl typesupport 在链接接口里会直接引用 `fastcdr`、`rmw::rmw`
# 这类 imported target。由于下面的 `io / auto_aim / auto_buff` 是并列子目录，
# 这些 target 必须在顶层作用域先被导入，兄弟目录之间才能共同可见。
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(geometry_msgs QUIET)
find_package(tf2 QUIET)
find_package(tf2_ros QUIET)
find_package(tf2_geometry_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(fastrtps_cmake_module QUIET)
find_package(fastcdr QUIET CONFIG)
find_package(rmw QUIET)
find_package(sp_msgs QUIET)

set(SP_VISION_HAS_ROS2 OFF)
if(
  ament_cmake_FOUND AND
  rclcpp_FOUND AND
  std_msgs_FOUND AND
  geometry_msgs_FOUND AND
  tf2_FOUND AND
  tf2_ros_FOUND AND
  tf2_geometry_msgs_FOUND AND
  rosidl_typesupport_cpp_FOUND AND
  fastrtps_cmake_module_FOUND AND
  fastcdr_FOUND AND
  rmw_FOUND AND
  sp_msgs_FOUND)
  set(SP_VISION_HAS_ROS2 ON)
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(standard src/standard.cpp)
add_executable(mt_standard src/mt_standard.cpp)
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(auto_debug src/auto_debug.cpp)
add_executable(auto_buff_debug src/auto_buff_debug.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)

target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(mt_standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools debug_visualization io)
target_link_libraries(auto_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools debug_visualization io)
target_link_libraries(auto_buff_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

set(SP_VISION_RUNTIME_TARGETS
  standard
  mt_standard
  standard_mpc
  auto_aim_debug_mpc
  auto_debug
  auto_buff_debug
  auto_buff_debug_mpc
  uav
  uav_debug
)

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

list(APPEND SP_VISION_RUNTIME_TARGETS
  capture
  calibrate_camera
  calibrate_handeye
  calibrate_robotworld_handeye
  split_video
)

# 只有在找到 ROS 环境时才会添加 sentry 节点
if(SP_VISION_HAS_ROS2)
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools debug_visualization io)
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)

    list(APPEND SP_VISION_RUNTIME_TARGETS
      sentry
      sentry_bp
      sentry_debug
      sentry_multithread
    )
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()

if(ament_cmake_FOUND)
    # 安装运行时资源：
    # 1. 可执行文件统一进 lib/${PROJECT_NAME}
    # 2. configs/assets 安装到 share/${PROJECT_NAME}
    # 这样 ros2 run / colcon install 后仍能找到配置和模型文件。
    install(
      TARGETS ${SP_VISION_RUNTIME_TARGETS}
      DESTINATION lib/${PROJECT_NAME}
    )

    install(
      DIRECTORY configs assets docs
      DESTINATION share/${PROJECT_NAME}
    )

    install(
      FILES readme.md run_with_watchdog.sh debug_mpc_watchdog.sh
      DESTINATION share/${PROJECT_NAME}
    )

    ament_package()
endif()
