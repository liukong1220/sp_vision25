#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "io/ros2/ros2.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

int main(int argc, char ** argv)
{
  tools::Exiter exiter;
  io::ROS2 ros2;

  double i = 0;
  while (!exiter.exit()) {
    io::VisionTargetState data;
    data.tracking = true;
    data.nav_hold = true;
    data.fire_permitted = false;
    data.target_id = 7;
    // 测试消息里显式带一个建议点编号，方便直接验证行为树回不回退到 fallback 点。
    data.suggested_goal_index = 2;
    data.confidence = 1.0;
    data.target_distance = i + 1.0;
    data.target_yaw = 0.1 * i;
    data.target_pitch = -0.01 * i;
    data.target_position_gimbal = Eigen::Vector3d{i, i + 1.0, 2.0};
    ros2.publish(data);
    i++;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (i > 1000) break;
  }
  return 0;
}
