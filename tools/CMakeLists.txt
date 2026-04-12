cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    debug.cpp
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    path.cpp
    plotter.cpp
    runtime_params.cpp
    web_debugger.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
    pid.cpp
    crc.cpp
)

add_library(debug_visualization OBJECT
    debug_visualization.cpp
)

# 记录源码根目录，供运行时资源路径解析使用。
target_compile_definitions(tools PRIVATE SP_VISION_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_include_directories(tools PRIVATE ${PROJECT_SOURCE_DIR}/io/serial/include)
target_link_libraries(tools PUBLIC yaml-cpp)
