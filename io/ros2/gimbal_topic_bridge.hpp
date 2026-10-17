#ifndef IO__GIMBAL_TOPIC_BRIDGE_HPP
#define IO__GIMBAL_TOPIC_BRIDGE_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "io/gimbal/gimbal.hpp"

namespace io
{

// ============================================================
// 串口融合改造（视觉侧）
// ------------------------------------------------------------
// 作用：替代 io::Gimbal 的串口读写层，接口与 io::Gimbal 完全兼容：
//   mode() / state() / str() / q(t) / send(...)
// 使 auto_aim_debug_mpc.cpp 只需把
//     io::Gimbal gimbal(config_path);
// 换成
//     io::GimbalTopicBridge gimbal;
// 其余代码（gimbal.q(t) / gimbal.state() / gimbal.send(...)）零改动。
//
// 数据流：
//   订阅 /serial/gimbal_state (导航发布) -> 姿态队列+状态缓存（复用原逻辑语义）
//   gimbal.send(...)                  -> 发布 /serial/gimbal_cmd（导航转发到串口）
//
// 话题格式（与 nav_bridge/README.md 一致）：
//   gimbal_state[11]: mode q_w q_x q_y q_z yaw yaw_vel pitch pitch_vel bullet_speed bullet_count
//   gimbal_cmd[7]:    mode yaw yaw_vel yaw_acc pitch pitch_vel pitch_acc
//   单位一律弧度制。
//
// mode 映射记录（仲裁：导航优先；详见 Start/README.md 决策表）：
//   0->IDLE 1->AUTO_AIM 2->SMALL_BUFF 3->BIG_BUFF
//   4->NAV_OVERRIDE(桥接层丢包，不发布 gimbal_cmd)
//   5->CONFLICT(桥接层丢包并告警)
//   6/其他->按 AUTO_AIM 回退并告警（沿用原协议非法值处理）
//
// 仲裁落地点说明（重要）：
//   基准程序 auto_aim_debug_mpc.cpp 不读取 gimbal.mode()，主循环无条件决策
//   并调用 send()（见该文件 L194/202/209/549，全文无 mode 分支）。因此
//   mode 4/5 的"视觉侧停止发送/降级"必须在桥接层 send() 内丢包实现，
//   不能依赖主程序根据 mode 分支——否则仲裁只对电控端单侧生效。
// ============================================================
class GimbalTopicBridge
{
public:
  GimbalTopicBridge();
  ~GimbalTopicBridge();

  // ---- 与 io::Gimbal 兼容的只读接口 ----
  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  // ---- 与 io::Gimbal::send 兼容的两种重载 ----
  // 注：mode=4/5 状态下 send() 直接丢包（见上"仲裁落地点说明"）
  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  void send(io::VisionToGimbal cmd);

private:
  void gimbalStateCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  mutable std::mutex mutex_;        // 保护 mode_/state_/send_suppressed_
  mutable std::mutex sample_mutex_; // 保护姿态样本缓存（沿用原双锁设计）
  GimbalMode mode_ = GimbalMode::AUTO_AIM;
  bool send_suppressed_ = false;    // mode=4/5 时为 true，send() 直接丢包
  GimbalState state_{};

  using TimedQuaternion = std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>;
  Eigen::Quaterniond last_returned_q_ = Eigen::Quaterniond::Identity();
  std::optional<TimedQuaternion> prev_sample_;
  std::optional<TimedQuaternion> latest_sample_;
  // 等待语义对齐原 io::Gimbal::q()（原版经 queue_.pop_for(100ms) 阻塞等新样本，
  // 桥接版无串口队列，改用条件变量实现同样的"最多等 100ms"行为）：
  std::condition_variable sample_cv_;
  bool sample_new_ = false;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr state_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
  std::thread spin_thread_;
  std::atomic<bool> quit_{false};
  bool owns_rclcpp_context_ = false;
};

}  // namespace io

#endif  // IO__GIMBAL_TOPIC_BRIDGE_HPP
