#include "publish2nav.hpp"

#include <memory>

namespace io
{

Publish2Nav::Publish2Nav() : Node("vision_target_publisher")
{
  // 统一把视觉对外接口收敛到 `vision/target`，
  // 这样行为树、调试脚本和 rviz/ros2 topic 都只需要盯一个话题。
  publisher_ = this->create_publisher<sp_msgs::msg::VisionTargetMsg>("vision/target", 10);

  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node initialized.");
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node shutting down.");
}

void Publish2Nav::send_data(const VisionTargetState & data)
{
  // 这里把内部结构体显式展开成 ROS 消息，
  // 目的是让每个字段都能在 topic echo 里直观看见，方便比赛现场调试。
  sp_msgs::msg::VisionTargetMsg message;
  message.timestamp = this->now();
  message.tracking = data.tracking;
  message.nav_hold = data.nav_hold;
  message.fire_permitted = data.fire_permitted;
  message.target_id = data.target_id;
  message.suggested_goal_index = data.suggested_goal_index;
  message.confidence = static_cast<float>(data.confidence);
  message.target_distance = static_cast<float>(data.target_distance);
  message.target_yaw = static_cast<float>(data.target_yaw);
  message.target_pitch = static_cast<float>(data.target_pitch);

  message.target_position_gimbal.x = data.target_position_gimbal.x();
  message.target_position_gimbal.y = data.target_position_gimbal.y();
  message.target_position_gimbal.z = data.target_position_gimbal.z();

  message.target_position_map.x = data.target_position_map.x();
  message.target_position_map.y = data.target_position_map.y();
  message.target_position_map.z = data.target_position_map.z();

  publisher_->publish(message);
}

void Publish2Nav::start()
{
  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

}  // namespace io
