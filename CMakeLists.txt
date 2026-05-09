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

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_aim_test_web tests/auto_aim_test_web.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_aim_test_web ${OpenCV_LIBS} fmt::fmt yaml-cpp tools debug_visualization io auto_aim auto_buff)
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(minimum_vision_system ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

# 这些测试/调试程序本身就是现场定位问题常用的入口，
# 迁入工作区后也一并安装到 lib/${PROJECT_NAME}，方便直接 `ros2 run sp_vision25 ...`。
list(APPEND SP_VISION_RUNTIME_TARGETS
  auto_aim_test
  auto_aim_test_web
  auto_buff_test
  camera_detect_test
  camera_test
  camera_thread_test
  cboard_test
  fire_test
  detector_video_test
  gimbal_response_test
  multi_usbcamera_test
  usbcamera_detect_test
  usbcamera_test
  handeye_test
  dm_test
  minimum_vision_system
  gimbal_test
  planner_test
  planner_test_offline
)

# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(SP_VISION_HAS_ROS2)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools debug_visualization io)
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)

    list(APPEND SP_VISION_RUNTIME_TARGETS
      sentry
      sentry_bp
      sentry_debug
      sentry_multithread
      publish_test
      subscribe_test
      topic_loop_test
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
