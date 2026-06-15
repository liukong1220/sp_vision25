cmake_minimum_required(VERSION 3.16.3)

project(sp_vision_unit_tests)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
# Project root so headers like "tools/math_tools.hpp" resolve
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

enable_testing()

set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
set(AUTO_AIM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tasks/auto_aim)

# --- math_tools_test ---
add_executable(math_tools_test
    math_tools_test.cpp
    ${TOOLS_DIR}/math_tools.cpp
)
target_link_libraries(math_tools_test GTest::gtest_main ${OpenCV_LIBS} fmt::fmt)
add_test(NAME math_tools_test COMMAND math_tools_test)

# --- crc_test ---
add_executable(crc_test
    crc_test.cpp
    ${TOOLS_DIR}/crc.cpp
)
target_link_libraries(crc_test GTest::gtest_main)
add_test(NAME crc_test COMMAND crc_test)

# --- pid_test ---
add_executable(pid_test
    pid_test.cpp
    ${TOOLS_DIR}/pid.cpp
    ${TOOLS_DIR}/math_tools.cpp
)
target_link_libraries(pid_test GTest::gtest_main ${OpenCV_LIBS} fmt::fmt)
add_test(NAME pid_test COMMAND pid_test)

# --- trajectory_test ---
add_executable(trajectory_test
    trajectory_test.cpp
    ${TOOLS_DIR}/trajectory.cpp
)
target_link_libraries(trajectory_test GTest::gtest_main)
add_test(NAME trajectory_test COMMAND trajectory_test)

# --- ekf_test ---
add_executable(ekf_test
    ekf_test.cpp
    ${TOOLS_DIR}/extended_kalman_filter.cpp
)
target_link_libraries(ekf_test GTest::gtest_main)
add_test(NAME ekf_test COMMAND ekf_test)

# --- voter_test ---
add_executable(voter_test
    voter_test.cpp
    ${AUTO_AIM_DIR}/voter.cpp
    ${AUTO_AIM_DIR}/armor.cpp
)
target_link_libraries(voter_test GTest::gtest_main ${OpenCV_LIBS} fmt::fmt)
add_test(NAME voter_test COMMAND voter_test)
