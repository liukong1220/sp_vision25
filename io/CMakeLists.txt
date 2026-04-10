cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

set(HIKROBOT_SDK_HINT_PATHS
  $ENV{HIKROBOT_SDK_ROOT}
  $ENV{MVCAM_SDK_ROOT}
  $ENV{MVCAM_COMMON_RUNENV}
  /opt/MVS
  /usr/local/MVS
  /opt/hikrobot/MVS
  /opt/HikRobot/MVS
  ${CMAKE_SOURCE_DIR}/../src/dependencies/hik_camera_ros2_driver/hikSDK
  ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot
)

find_path(HIKROBOT_INCLUDE_DIR
  NAMES MvCameraControl.h
  PATHS ${HIKROBOT_SDK_HINT_PATHS}
  PATH_SUFFIXES include MvImport/include
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(HIKROBOT_LIB_SUFFIXES lib lib/64 lib/amd64 MvImport/lib/64 MvImport/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(HIKROBOT_LIB_SUFFIXES lib lib/arm64 MvImport/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

find_library(HIKROBOT_MVCAMERA_LIBRARY
  NAMES MvCameraControl
  PATHS ${HIKROBOT_SDK_HINT_PATHS}
  PATH_SUFFIXES ${HIKROBOT_LIB_SUFFIXES}
)

if(NOT HIKROBOT_INCLUDE_DIR OR NOT HIKROBOT_MVCAMERA_LIBRARY)
  message(FATAL_ERROR
    "HikRobot SDK not found. Set HIKROBOT_SDK_ROOT to the SDK root directory, "
    "or place the SDK under sp_vision25/io/hikrobot or src/dependencies/hik_camera_ros2_driver/hikSDK.")
endif()

get_filename_component(HIKROBOT_LIBRARY_DIR "${HIKROBOT_MVCAMERA_LIBRARY}" DIRECTORY)
message(STATUS "Using HikRobot SDK include dir: ${HIKROBOT_INCLUDE_DIR}")
message(STATUS "Using HikRobot SDK library dir: ${HIKROBOT_LIBRARY_DIR}")

set(MINDVISION_SDK_HINT_PATHS
  $ENV{MINDVISION_SDK_ROOT}
  $ENV{MVSDK_ROOT}
  /opt/MindVision
  /usr/local/MindVision
  ${CMAKE_CURRENT_SOURCE_DIR}/mindvision
)

find_path(MINDVISION_INCLUDE_DIR
  NAMES CameraApi.h
  PATHS ${MINDVISION_SDK_HINT_PATHS}
  PATH_SUFFIXES include
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(MINDVISION_LIB_SUFFIXES lib lib/64 lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(MINDVISION_LIB_SUFFIXES lib lib/arm64)
endif()

find_library(MINDVISION_SDK_LIBRARY
  NAMES MVSDK
  PATHS ${MINDVISION_SDK_HINT_PATHS}
  PATH_SUFFIXES ${MINDVISION_LIB_SUFFIXES}
)

if(MINDVISION_INCLUDE_DIR AND MINDVISION_SDK_LIBRARY)
  set(SP_VISION_HAS_MINDVISION_SDK ON)
  get_filename_component(MINDVISION_LIBRARY_DIR "${MINDVISION_SDK_LIBRARY}" DIRECTORY)
  message(STATUS "Using MindVision SDK include dir: ${MINDVISION_INCLUDE_DIR}")
  message(STATUS "Using MindVision SDK library dir: ${MINDVISION_LIBRARY_DIR}")
else()
  set(SP_VISION_HAS_MINDVISION_SDK OFF)
  message(WARNING
    "MindVision SDK not found. MindVision camera support will be disabled. "
    "Set MINDVISION_SDK_ROOT if you need it.")
endif()

set(IO_SOURCES
  hikrobot/hikrobot.cpp
  usbcamera/usbcamera.cpp
  camera.cpp
  cboard.cpp
  dm_imu/dm_imu.cpp
  gimbal/gimbal.cpp
)

if(SP_VISION_HAS_MINDVISION_SDK)
  list(APPEND IO_SOURCES mindvision/mindvision.cpp)
endif()

# 创建目标 io
add_library(io STATIC 
    ${IO_SOURCES}
)

# hikrobot
target_include_directories(io PUBLIC "${HIKROBOT_INCLUDE_DIR}")
target_link_directories(io PUBLIC "${HIKROBOT_LIBRARY_DIR}")

# mindvision
if(SP_VISION_HAS_MINDVISION_SDK)
  target_include_directories(io PUBLIC "${MINDVISION_INCLUDE_DIR}")
  target_link_directories(io PUBLIC "${MINDVISION_LIBRARY_DIR}")
  target_link_libraries(io "${MINDVISION_SDK_LIBRARY}")
  target_compile_definitions(io PUBLIC SP_VISION_HAS_MINDVISION_SDK=1)
endif()

target_link_libraries(io "${HIKROBOT_MVCAMERA_LIBRARY}" usb-1.0 yaml-cpp serial)

################## 检查 ROS 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(fastrtps_cmake_module QUIET)
find_package(fastcdr QUIET CONFIG)
find_package(rmw QUIET)
find_package(sp_msgs QUIET)

# 如果找不到 ROS 环境，则跳过 ROS 相关的部分
if(NOT ament_cmake_FOUND OR NOT rclcpp_FOUND OR NOT std_msgs_FOUND OR NOT rosidl_typesupport_cpp_FOUND OR NOT fastrtps_cmake_module_FOUND OR NOT fastcdr_FOUND OR NOT rmw_FOUND OR NOT sp_msgs_FOUND)
    message(WARNING "ROS2 not found, skipping ROS2 specific code.")
else()
    message(STATUS "ROS2 environment found, compiling ROS2-related code.")

    # `sp_msgs` 是标准 rosidl 接口包，它导出的 fastrtps typesupport
    # 在生成阶段会直接依赖 `fastcdr` 和 `rmw::rmw` 这些 imported target。
    # `sp_vision25` 目前还是普通 CMake 工程，不是完整 ament 包，
    # 所以这里显式导入底层依赖，避免出现 “Target fastcdr not found”。
    # 添加 ROS2 相关的文件
    target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)

    # 设置 ROS2 依赖
    include_directories(${std_msgs_INCLUDE_DIRS})
    ament_target_dependencies(io rclcpp std_msgs sp_msgs)
endif()
