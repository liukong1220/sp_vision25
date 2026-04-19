#include "publish2nav.hpp"

#include <cmath>
#include <memory>

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tools/yaml.hpp"

namespace io
{

Publish2Nav::Publish2Nav(const std::string & config_path)
: Node("vision_target_publisher")
{
  loadVisionFusionConfig(config_path);

  // 统一把视觉对外接口收敛到 `vision/target`，
  // 这样行为树、调试脚本和 rviz/ros2 topic 都只需要盯一个话题。
  publisher_ = this->create_publisher<sp_msgs::msg::VisionTargetMsg>("vision/target", 10);
  target_point_map_publisher_ =
    this->create_publisher<geometry_msgs::msg::PointStamped>("vision/target_point_map", 10);
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());

  RCLCPP_INFO(
    this->get_logger(),
    "vision_target_publisher node initialized (map_frame=%s, gimbal_frame=%s, pose_timeout=%.3fs, nav_hold_min_confidence=%.2f, publish_target_point_map=%d).",
    map_frame_.c_str(), gimbal_frame_.c_str(), pose_timeout_s_, nav_hold_min_confidence_,
    publish_target_point_map_);
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node shutting down.");
}

void Publish2Nav::send_data(const VisionTargetState & data)
{
  ensureTfListener();

  // 这里把内部结构体显式展开成 ROS 消息，
  // 目的是让每个字段都能在 topic echo 里直观看见，方便比赛现场调试。
  sp_msgs::msg::VisionTargetMsg message;
  message.timestamp = this->now();
  message.tracking = data.tracking;
  message.nav_hold = data.nav_hold;
  message.fire_permitted = data.fire_permitted;
  message.target_id = data.target_id;
  message.confidence = static_cast<float>(data.confidence);
  message.target_distance = static_cast<float>(data.target_distance);
  message.target_yaw = static_cast<float>(data.target_yaw);
  message.target_pitch = static_cast<float>(data.target_pitch);

  message.target_position_gimbal.x = data.target_position_gimbal.x();
  message.target_position_gimbal.y = data.target_position_gimbal.y();
  message.target_position_gimbal.z = data.target_position_gimbal.z();

  geometry_msgs::msg::PointStamped target_point_map;
  bool has_target_point_map = false;
  if (data.has_target_position_map && isFiniteVector(data.target_position_map) &&
    !data.target_position_map_frame.empty())
  {
    target_point_map.header.stamp = message.timestamp;
    target_point_map.header.frame_id = data.target_position_map_frame;
    target_point_map.point.x = data.target_position_map.x();
    target_point_map.point.y = data.target_position_map.y();
    target_point_map.point.z = data.target_position_map.z();
    has_target_point_map = true;
  } else {
    has_target_point_map = tryProjectTargetPoint(data.target_position_gimbal, target_point_map);
  }

  if (has_target_point_map) {
    message.target_position_map = target_point_map.point;
    message.has_target_position_map = true;
    message.target_position_map_frame = target_point_map.header.frame_id;

    if (publish_target_point_map_) {
      target_point_map_publisher_->publish(target_point_map);
    }
  } else {
    message.has_target_position_map = false;
    message.target_position_map_frame.clear();
  }

  const bool allow_nav_hold =
    message.tracking && message.has_target_position_map &&
    data.confidence >= nav_hold_min_confidence_;
  message.nav_hold = data.nav_hold && allow_nav_hold;

  publisher_->publish(message);
}

void Publish2Nav::start()
{
  ensureTfListener();
  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

void Publish2Nav::loadVisionFusionConfig(const std::string & config_path)
{
  if (config_path.empty()) {
    return;
  }

  const auto yaml = tools::load(config_path);
  const auto vision_fusion = yaml["vision_fusion"];
  if (!vision_fusion) {
    return;
  }

  if (vision_fusion["map_frame"]) {
    map_frame_ = vision_fusion["map_frame"].as<std::string>();
  }
  if (vision_fusion["gimbal_frame"]) {
    gimbal_frame_ = vision_fusion["gimbal_frame"].as<std::string>();
  }
  if (vision_fusion["pose_timeout_s"]) {
    pose_timeout_s_ = vision_fusion["pose_timeout_s"].as<double>();
  }
  if (vision_fusion["nav_hold_min_confidence"]) {
    nav_hold_min_confidence_ = vision_fusion["nav_hold_min_confidence"].as<double>();
  }
  if (vision_fusion["publish_target_point_map"]) {
    publish_target_point_map_ = vision_fusion["publish_target_point_map"].as<bool>();
  }
}

void Publish2Nav::ensureTfListener()
{
  if (!tf_listener_) {
    tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this->shared_from_this(), false);
  }
}

bool Publish2Nav::tryProjectTargetPoint(
  const Eigen::Vector3d & target_position_gimbal, geometry_msgs::msg::PointStamped & point_map)
{
  if (!isFiniteVector(target_position_gimbal)) {
    return false;
  }

  geometry_msgs::msg::PointStamped point_gimbal;
  point_gimbal.header.stamp = this->now();
  point_gimbal.header.frame_id = gimbal_frame_;
  point_gimbal.point.x = target_position_gimbal.x();
  point_gimbal.point.y = target_position_gimbal.y();
  point_gimbal.point.z = target_position_gimbal.z();

  try {
    const auto transform = tf_buffer_->lookupTransform(map_frame_, gimbal_frame_, tf2::TimePointZero);
    if (pose_timeout_s_ > 0.0 && (transform.header.stamp.sec != 0U ||
      transform.header.stamp.nanosec != 0U))
    {
      const auto age_s = (this->now() - rclcpp::Time(transform.header.stamp)).seconds();
      if (age_s > pose_timeout_s_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Skip target_position_map projection because TF %s -> %s is stale: age=%.3fs timeout=%.3fs",
          map_frame_.c_str(), gimbal_frame_.c_str(), age_s, pose_timeout_s_);
        return false;
      }
    }

    tf2::doTransform(point_gimbal, point_map, transform);
    point_map.header.frame_id = map_frame_;
    point_map.header.stamp = this->now();
    return std::isfinite(point_map.point.x) && std::isfinite(point_map.point.y) &&
           std::isfinite(point_map.point.z);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Failed to project target_position_gimbal from %s to %s: %s",
      gimbal_frame_.c_str(), map_frame_.c_str(), ex.what());
    return false;
  }
}

bool Publish2Nav::isFiniteVector(const Eigen::Vector3d & vector) const
{
  return std::isfinite(vector.x()) && std::isfinite(vector.y()) && std::isfinite(vector.z());
}

}  // namespace io
