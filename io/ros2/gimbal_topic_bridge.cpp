#include "gimbal_topic_bridge.hpp"

#include <algorithm>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace io
{

GimbalTopicBridge::GimbalTopicBridge()
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
    owns_rclcpp_context_ = true;
  }

  node_ = std::make_shared<rclcpp::Node>("vision_gimbal_bridge");

  state_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "serial/gimbal_state", 10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      gimbalStateCallback(msg);
    });

  cmd_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>("serial/gimbal_cmd", 10);

  spin_thread_ = std::thread([this]() { rclcpp::spin(node_); });

  tools::logger()->info(
    "[GimbalTopicBridge] Ready: sub serial/gimbal_state, pub serial/gimbal_cmd");
}

GimbalTopicBridge::~GimbalTopicBridge()
{
  quit_ = true;
  if (owns_rclcpp_context_ && rclcpp::ok()) rclcpp::shutdown();
  if (spin_thread_.joinable()) spin_thread_.join();
}

GimbalMode GimbalTopicBridge::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState GimbalTopicBridge::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string GimbalTopicBridge::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

void GimbalTopicBridge::gimbalStateCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() != 11) {
    tools::logger()->warn(
      "[GimbalTopicBridge] Bad gimbal_state size: {} (expect 11)", msg->data.size());
    return;
  }

  const auto t = std::chrono::steady_clock::now();

  // 四元数样本入缓存（wxyz，与原 read_thread 行为一致）
  Eigen::Quaterniond q(msg->data[1], msg->data[2], msg->data[3], msg->data[4]);
  {
    std::lock_guard<std::mutex> lock(sample_mutex_);
    if (!latest_sample_.has_value()) {
      latest_sample_ = {q, t};
    } else {
      prev_sample_ = latest_sample_;
      latest_sample_ = {q, t};
    }
    sample_new_ = true;
  }
  sample_cv_.notify_all();

  // 状态缓存
  {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.yaw = static_cast<float>(msg->data[5]);
    state_.yaw_vel = static_cast<float>(msg->data[6]);
    state_.pitch = static_cast<float>(msg->data[7]);
    state_.pitch_vel = static_cast<float>(msg->data[8]);
    state_.bullet_speed = static_cast<float>(msg->data[9]);
    state_.bullet_count = static_cast<uint16_t>(msg->data[10]);
  }

  // mode 映射（含融合扩展 4/5/6，仲裁记录表见 Start/README.md）。
  // 4/5 映射为 IDLE 仅是内部状态记录；实际抑制发送在 send() 丢包实现
  // （基准主程序不读 mode，见文件头"仲裁落地点说明"）。
  const auto mode_value = static_cast<int>(msg->data[0]);
  GimbalMode new_mode;
  bool send_suppressed = false;
  bool warn = false;
  switch (mode_value) {
    case 0:
      new_mode = GimbalMode::IDLE;
      break;
    case 1:
      new_mode = GimbalMode::AUTO_AIM;
      break;
    case 2:
      new_mode = GimbalMode::SMALL_BUFF;
      break;
    case 3:
      new_mode = GimbalMode::BIG_BUFF;
      break;
    case 4:  // NAV_OVERRIDE：导航接管（仲裁：导航优先）
      new_mode = GimbalMode::IDLE;
      send_suppressed = true;
      break;
    case 5:  // CONFLICT：冲突，视觉降级停发
      new_mode = GimbalMode::IDLE;
      send_suppressed = true;
      warn = true;
      break;
    default:  // 6 及非法值：按自瞄回退 + 告警（沿用原协议处理）
      new_mode = GimbalMode::AUTO_AIM;
      warn = true;
      break;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = new_mode;
    send_suppressed_ = send_suppressed;
  }
  if (warn) {
    tools::logger()->warn("[GimbalTopicBridge] mode={} (nav_override/conflict/invalid)", mode_value);
  }
}

Eigen::Quaterniond GimbalTopicBridge::q(std::chrono::steady_clock::time_point t)
{
  // 与原 io::Gimbal::q 相同的插值与等待策略：
  // 区间内 slerp；过早取前样本；过晚则最多等待 100ms 新样本（原版经
  // queue_.pop_for(100ms) 实现，此处用条件变量等价实现），超时返回最近姿态。
  constexpr auto kWaitTimeout = std::chrono::milliseconds(100);
  std::unique_lock<std::mutex> lock(sample_mutex_);

  while (!quit_) {
    if (prev_sample_.has_value() && latest_sample_.has_value()) {
      const auto & [q_a, t_a] = *prev_sample_;
      const auto & [q_b, t_b] = *latest_sample_;
      if (t_a < t && t <= t_b) {
        const auto t_ab = tools::delta_time(t_a, t_b);
        if (std::abs(t_ab) > 1e-6) {
          const auto t_ac = tools::delta_time(t_a, t);
          const double k = std::clamp(t_ac / t_ab, 0.0, 1.0);
          last_returned_q_ = q_a.slerp(k, q_b).normalized();
          return last_returned_q_;
        }
        last_returned_q_ = q_b.normalized();
        return last_returned_q_;
      }
      if (t <= t_a) {
        last_returned_q_ = q_a.normalized();
        return last_returned_q_;
      }
    }

    if (latest_sample_.has_value()) {
      last_returned_q_ = std::get<0>(*latest_sample_).normalized();
    }

    // 未覆盖 t：等新样本（对齐原版 pop_for 的阻塞等待），超时则返回最近姿态。
    if (sample_cv_.wait_for(lock, kWaitTimeout, [this] { return sample_new_ || quit_.load(); })) {
      sample_new_ = false;
      continue;  // 新样本到达，回到循环头重新判断插值区间
    }
    return last_returned_q_;  // 超时
  }

  return last_returned_q_;
}

void GimbalTopicBridge::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  // 仲裁落地点：mode=4/5 时丢包（基准主程序不读 mode、无条件调用 send，
  // 必须在此处抑制，否则导航接管/冲突降级对视觉侧完全不生效）。
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (send_suppressed_) return;
  }

  const uint8_t mode = control ? (fire ? 2 : 1) : 0;
  std_msgs::msg::Float64MultiArray msg;
  msg.data.reserve(7);
  msg.data.push_back(static_cast<double>(mode));
  msg.data.push_back(static_cast<double>(yaw));
  msg.data.push_back(static_cast<double>(yaw_vel));
  msg.data.push_back(static_cast<double>(yaw_acc));
  msg.data.push_back(static_cast<double>(pitch));
  msg.data.push_back(static_cast<double>(pitch_vel));
  msg.data.push_back(static_cast<double>(pitch_acc));
  cmd_pub_->publish(msg);
}

void GimbalTopicBridge::send(io::VisionToGimbal cmd)
{
  send(
    cmd.mode != 0, cmd.mode == 2, cmd.yaw, cmd.yaw_vel, cmd.yaw_acc, cmd.pitch, cmd.pitch_vel,
    cmd.pitch_acc);
}

}  // namespace io
