#include "publish2nav.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

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
  // `ROS2::start()` 会在独立线程 spin 这个节点，这里显式告知 tf2 Buffer，
  // 避免带 timeout 的 lookupTransform 被误判为“没有单独线程喂 TF”。
  tf_buffer_->setUsingDedicatedThread(true);
  static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

  RCLCPP_INFO(
    this->get_logger(),
    "vision_target_publisher node initialized (map_frame=%s, gimbal_frame=%s, mount_frame=%s, camera_frame=%s, pose_timeout=%.3fs, transform_timeout=%.3fs, nav_hold_min_confidence=%.2f, publish_target_point_map=%d).",
    map_frame_.c_str(), gimbal_frame_.c_str(), mount_frame_.c_str(), camera_frame_.c_str(), pose_timeout_s_,
    transform_timeout_s_, nav_hold_min_confidence_, publish_target_point_map_);
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "vision_target_publisher node shutting down.");
}

void Publish2Nav::send_data(const VisionTargetState & data)
{
  ensureTfListener();
  publishCalibrationTfIfNeeded();
  const auto observation_stamp = resolveObservationStamp(data);

  // 这里把内部结构体显式展开成 ROS 消息，
  // 目的是让每个字段都能在 topic echo 里直观看见，方便比赛现场调试。
  sp_msgs::msg::VisionTargetMsg message;
  message.timestamp = observation_stamp;
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
    target_point_map.header.stamp = observation_stamp;
    target_point_map.header.frame_id = data.target_position_map_frame;
    target_point_map.point.x = data.target_position_map.x();
    target_point_map.point.y = data.target_position_map.y();
    target_point_map.point.z = data.target_position_map.z();
    has_target_point_map = true;
  } else {
    has_target_point_map =
      tryProjectTargetPoint(data.target_position_gimbal, observation_stamp, target_point_map);
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
  if (vision_fusion["camera_frame"]) {
    camera_frame_ = vision_fusion["camera_frame"].as<std::string>();
  }
  if (vision_fusion["mount_frame"]) {
    mount_frame_ = vision_fusion["mount_frame"].as<std::string>();
  }
  if (vision_fusion["gimbal_frame"]) {
    gimbal_frame_ = vision_fusion["gimbal_frame"].as<std::string>();
  }
  if (vision_fusion["calib_gimbal_frame"]) {
    calib_gimbal_frame_ = vision_fusion["calib_gimbal_frame"].as<std::string>();
  } else {
    calib_gimbal_frame_ = gimbal_frame_;
  }
  if (vision_fusion["pose_timeout_s"]) {
    pose_timeout_s_ = vision_fusion["pose_timeout_s"].as<double>();
  }
  if (vision_fusion["transform_timeout_s"]) {
    transform_timeout_s_ = vision_fusion["transform_timeout_s"].as<double>();
  }
  if (vision_fusion["transform_time_backoff_s"]) {
    transform_time_backoff_s_ = vision_fusion["transform_time_backoff_s"].as<double>();
  }
  if (vision_fusion["nav_hold_min_confidence"]) {
    nav_hold_min_confidence_ = vision_fusion["nav_hold_min_confidence"].as<double>();
  }
  if (vision_fusion["publish_target_point_map"]) {
    publish_target_point_map_ = vision_fusion["publish_target_point_map"].as<bool>();
  }
  if (vision_fusion["publish_calib_tf"]) {
    publish_calib_tf_ = vision_fusion["publish_calib_tf"].as<bool>();
  }

  if (yaml["R_camera2gimbal"]) {
    const auto data = yaml["R_camera2gimbal"].as<std::vector<double>>();
    if (data.size() == 9) {
      R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
      has_camera_to_gimbal_calib_ = true;
    }
  }
  if (yaml["t_camera2gimbal"]) {
    const auto data = yaml["t_camera2gimbal"].as<std::vector<double>>();
    if (data.size() == 3) {
      t_camera2gimbal_ = Eigen::Map<const Eigen::Vector3d>(data.data());
      has_camera_to_gimbal_calib_ = has_camera_to_gimbal_calib_ && true;
    } else {
      has_camera_to_gimbal_calib_ = false;
    }
  } else {
    has_camera_to_gimbal_calib_ = false;
  }
}

builtin_interfaces::msg::Time Publish2Nav::resolveObservationStamp(const VisionTargetState & data) const
{
  rclcpp::Time ros_stamp = this->now();
  if (data.has_observation_time) {
    auto age = std::chrono::steady_clock::now() - data.observation_time;
    if (age < std::chrono::steady_clock::duration::zero()) {
      age = std::chrono::steady_clock::duration::zero();
    }
    ros_stamp = ros_stamp - rclcpp::Duration::from_nanoseconds(
      std::chrono::duration_cast<std::chrono::nanoseconds>(age).count());
  }

  if (transform_time_backoff_s_ > 0.0) {
    ros_stamp = ros_stamp - rclcpp::Duration::from_seconds(transform_time_backoff_s_);
  }
  builtin_interfaces::msg::Time stamp_msg;
  const auto stamp_ns = std::max<int64_t>(0, ros_stamp.nanoseconds());
  stamp_msg.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
  stamp_msg.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
  return stamp_msg;
}

void Publish2Nav::publishCalibrationTfIfNeeded()
{
  if (
    calib_tf_published_ || !publish_calib_tf_ || !static_tf_broadcaster_ ||
    !has_camera_to_gimbal_calib_ || camera_frame_.empty() || mount_frame_.empty() ||
    calib_gimbal_frame_.empty())
  {
    return;
  }

  try {
    const auto mount_to_camera_msg =
      tf_buffer_->lookupTransform(mount_frame_, camera_frame_, tf2::TimePointZero);

    tf2::Transform mount_to_camera_tf;
    tf2::fromMsg(mount_to_camera_msg.transform, mount_to_camera_tf);

    tf2::Matrix3x3 camera_to_gimbal_basis(
      R_camera2gimbal_(0, 0), R_camera2gimbal_(0, 1), R_camera2gimbal_(0, 2),
      R_camera2gimbal_(1, 0), R_camera2gimbal_(1, 1), R_camera2gimbal_(1, 2),
      R_camera2gimbal_(2, 0), R_camera2gimbal_(2, 1), R_camera2gimbal_(2, 2));
    tf2::Transform gimbal_to_camera_tf(
      camera_to_gimbal_basis.inverse(),
      -(camera_to_gimbal_basis.inverse() *
      tf2::Vector3(t_camera2gimbal_.x(), t_camera2gimbal_.y(), t_camera2gimbal_.z())));

    const tf2::Transform mount_to_calib_gimbal_tf = mount_to_camera_tf * gimbal_to_camera_tf;

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = now();
    transform.header.frame_id = mount_frame_;
    transform.child_frame_id = calib_gimbal_frame_;
    transform.transform = tf2::toMsg(mount_to_calib_gimbal_tf);

    static_tf_broadcaster_->sendTransform(transform);
    calib_tf_published_ = true;

    const auto & origin = mount_to_calib_gimbal_tf.getOrigin();
    RCLCPP_INFO(
      this->get_logger(),
      "Published calibration TF %s -> %s using camera frame %s, offset=[%.4f, %.4f, %.4f]",
      mount_frame_.c_str(), calib_gimbal_frame_.c_str(), camera_frame_.c_str(), origin.x(),
      origin.y(), origin.z());
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Waiting for TF %s -> %s before publishing calibration frame %s: %s",
      mount_frame_.c_str(), camera_frame_.c_str(), calib_gimbal_frame_.c_str(), ex.what());
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
  const Eigen::Vector3d & target_position_gimbal, const builtin_interfaces::msg::Time & stamp,
  geometry_msgs::msg::PointStamped & point_map)
{
  if (!isFiniteVector(target_position_gimbal)) {
    return false;
  }

  geometry_msgs::msg::PointStamped point_gimbal;
  point_gimbal.header.stamp = stamp;
  point_gimbal.header.frame_id = gimbal_frame_;
  point_gimbal.point.x = target_position_gimbal.x();
  point_gimbal.point.y = target_position_gimbal.y();
  point_gimbal.point.z = target_position_gimbal.z();

  try {
    geometry_msgs::msg::TransformStamped transform;
    const auto requested_time = rclcpp::Time(point_gimbal.header.stamp);
    try {
      transform = tf_buffer_->lookupTransform(
        map_frame_, point_gimbal.header.frame_id, requested_time,
        rclcpp::Duration::from_seconds(std::max(0.0, transform_timeout_s_)));
    } catch (const tf2::TransformException & exact_ex) {
      transform = tf_buffer_->lookupTransform(
        map_frame_, point_gimbal.header.frame_id, tf2::TimePointZero);
      const auto latest_time = rclcpp::Time(transform.header.stamp);
      const double fallback_gap_s = std::abs((requested_time - latest_time).seconds());
      if (fallback_gap_s > std::max(0.05, transform_timeout_s_)) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Use latest TF for target_position_map projection because timestamped TF %s -> %s is unavailable: %s (gap=%.3fs)",
          map_frame_.c_str(), point_gimbal.header.frame_id.c_str(), exact_ex.what(),
          fallback_gap_s);
      } else {
        RCLCPP_DEBUG_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Use nearby latest TF for target_position_map projection: requested %s -> %s gap=%.3fs",
          map_frame_.c_str(), point_gimbal.header.frame_id.c_str(), fallback_gap_s);
      }
    }

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
    point_map.header.stamp = point_gimbal.header.stamp;
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
