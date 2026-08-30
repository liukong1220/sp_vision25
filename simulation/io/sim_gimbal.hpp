#ifndef SIMULATION_IO__SIM_GIMBAL_HPP
#define SIMULATION_IO__SIM_GIMBAL_HPP

// 仿真云台：把共享内存的 pose 通道当作云台反馈，把控制量编码成 GimbalCmd 发回。
//
// 两个关键点：
//   1. 状态必须来自与图像严格同帧的 pose（update(bundle) 只接受 SimCamera 的
//      last_bundle()），不能用“处理完成时的最新姿态”，否则解算用的旋转和图像
//      对不上，误差会被当成目标运动吃进 EKF。
//   2. 命令编码里 yaw/pitch 的正负与零点做成可配置仿射映射（scale/offset），
//      默认值已由 probe 模式单轴实测确认（见下）。
//        simulator 解码: yaw_rad = radians(yaw_deg), pitch_rad = radians(-pitch_deg - 90)
//        这里编码:       yaw_deg = deg(yaw),         pitch_deg = deg(pitch) - 90
//
//      pitch 为什么不是解码式的直接逆变换：simulator 把解出来的 pitch 塞进
//      `Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0)`，那是绕 Bevy +X 轴转，
//      而 Bevy 前向是 -Z，绕 +X 正转是抬头；反馈这边按 ROS 的 ZYX 分解，绕 +Y
//      正转是低头。两个约定天生反向，所以要让"下发 pitch = 反馈 pitch"，
//      编码必须多取一次反：pitch_scale = +1 而不是 -1。
//      实测（probe --axis=pitch）：scale=-1 时 fb_pitch 恒等于 -cmd_pitch，
//      改成 +1 后 fb_pitch == cmd_pitch。yaw 两个约定同向，scale=+1 即可。

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <string>

#include "io/gimbal/gimbal.hpp"
#include "shared_memory_client.hpp"

namespace sim_io
{
// 禁止开火的原因，按位组合。任一位置起即锁定开火。
enum SafetyFault : std::uint32_t
{
  FAULT_NONE = 0,
  FAULT_STARTUP = 1u << 0,        // 还没拿到有效状态
  FAULT_HEARTBEAT_LOST = 1u << 1, // 仿真端心跳超时
  FAULT_NO_NEW_FRAME = 1u << 2,   // 长时间没有新帧
  FAULT_TARGET_LOST = 1u << 3,    // 目标丢失
  FAULT_CLOCK_JUMP = 1u << 4,     // 检测到 wall clock 跳变
  FAULT_FRAME_FAULT = 1u << 5,    // 帧号倒退 / pose 不同帧
  FAULT_FIRE_DISABLED = 1u << 6,  // 配置层面禁止开火（默认）
  FAULT_STATE_STALE = 1u << 7,    // 同帧姿态过期
};

std::string describe_faults(std::uint32_t faults);

struct SimGimbalConfig
{
  // 命令映射（默认为 simulator 解码式的逆变换）
  double yaw_scale = 1.0;
  double yaw_offset_deg = 0.0;
  double pitch_scale = 1.0;
  double pitch_offset_deg = -90.0;

  // simulator 的 [projectile] speed。共享内存没有这个字段，必须与仿真端配置一致。
  double bullet_speed = 25.0;

  bool allow_fire = false;           // 默认禁止开火
  double state_timeout_ms = 200.0;   // 同帧姿态超过此龄视为过期
  double safe_stop_period_ms = 20.0; // 空闲时重发安全停止命令的周期
  double vel_lpf_alpha = 0.35;       // 角速度低通系数

  // 反馈四元数的坐标系修正，绕 ROS Y 轴右乘（单位：度）。
  //
  // 仿真端发布反馈时多乘了一个绕自身 X 轴的 90° 滚转（capture.rs 里的
  // `Quat::from_euler(EulerRot::ZYX, 0, 0, PI/2)`），随后整体做 to_ros_quat：
  //     q_pub = A·(R_muzzle · Rx(90°))·A⁻¹ = (A·R_muzzle·A⁻¹) · (A·Rx(90°)·A⁻¹)
  // A 把 Bevy 的 X 轴映射到 ROS 的 -Y（A 的第一列是 (0,-1,0)），所以后一项就是
  // 绕 ROS -Y 轴转 90°，即 Ry(-90°)。于是真正的枪口姿态是
  //     R_muzzle_ros = q_pub · Ry(+90°)
  // 不做这个修正会有两个后果：
  //   1. 下发 pitch=0 时反馈 pitch 恰好是 -90°，正好落在 ZYX 欧拉角的万向锁
  //      奇点上，yaw 会解出完全无意义的值；
  //   2. 这个四元数还会交给 solver.set_R_gimbal2world()，PnP 的世界坐标会整体
  //      偏 90°。所以这是正确性问题，不只是日志好看不好看。
  double feedback_pitch_fix_deg = 90.0;
};

class SimGimbal
{
public:
  SimGimbal(SharedMemoryClient & client, SimGimbalConfig config = {});

  // 用与图像同帧的 pose 束更新状态。必须在 solver.set_R_gimbal2world() 之前调用。
  void update(const FrameBundle & bundle, std::chrono::steady_clock::time_point steady_ts);

  bool has_state() const { return has_state_; }
  std::uint64_t frame_seq() const { return frame_seq_; }

  // 同帧云台姿态，Rust 侧发布顺序是 [w,x,y,z]。
  // 已按 feedback_pitch_fix_deg 修正过坐标系，可直接喂 solver。
  const Eigen::Quaterniond & q() const { return q_; }

  // 共享内存里的原始四元数，未做任何修正。只用于排查坐标系问题。
  const Eigen::Quaterniond & q_raw() const { return q_raw_; }
  io::GimbalState state() const;

  // 双精度原值，避免 io::GimbalState 的 float 截断影响标定核对。
  double yaw() const { return yaw_; }
  double pitch() const { return pitch_; }
  double yaw_vel() const { return yaw_vel_; }
  double pitch_vel() const { return pitch_vel_; }

  // 同帧的其他 pose（世界系原点、枪口/相机相对云台的平移），供评估与外参核对。
  const Eigen::Vector3d & odom_position() const { return odom_position_; }
  const Eigen::Vector3d & muzzle_offset() const { return muzzle_offset_; }
  const Eigen::Vector3d & camera_offset() const { return camera_offset_; }

  // 控制输出。fire 只有在 allow_fire 且无任何 fault 时才可能真的置 1。
  // 返回值表示命令是否成功写入共享内存。
  bool send(bool control, bool fire, double yaw_rad, double pitch_rad, double distance_m);

  // distance_m = -1，fire = 0。仿真端见到 -1 会直接 return，不改动云台。
  bool send_safe_stop();

  // 空闲时定期重发安全停止，确保三缓冲里最新的命令永远是无害的：
  // 即使本进程崩溃，仿真端也不会消费到一条残留的开火命令。
  bool tick();

  void set_fault(std::uint32_t fault, bool active);
  void clear_faults(std::uint32_t mask);
  std::uint32_t faults() const;
  bool fire_allowed() const { return faults() == FAULT_NONE; }

  // 编码/解码，供 loopback 测试直接核对
  GimbalCmd encode(bool control, bool fire, double yaw_rad, double pitch_rad, double distance_m)
    const;
  // encode() 的严格逆变换，拿回下发的角（ROS 约定，与 yaw()/pitch() 同系）。
  double decode_yaw_rad(const GimbalCmd & cmd) const;
  double decode_pitch_rad(const GimbalCmd & cmd) const;

  // 仿真端 plugin.rs 会解出来的内部角（Bevy 的 YXZ 约定）。
  // pitch 与 decode_pitch_rad 相差一个负号，见文件头的推导。
  static double sim_internal_yaw_rad(const GimbalCmd & cmd);
  static double sim_internal_pitch_rad(const GimbalCmd & cmd);

  std::uint64_t sent_commands() const { return sent_commands_; }
  std::uint64_t fire_commands() const { return fire_commands_; }
  std::uint64_t suppressed_fires() const { return suppressed_fires_; }
  std::uint64_t safe_stops() const { return safe_stops_; }
  const SimGimbalConfig & config() const { return config_; }

private:
  SharedMemoryClient & client_;
  SimGimbalConfig config_;

  Eigen::Quaterniond q_{1.0, 0.0, 0.0, 0.0};
  Eigen::Quaterniond q_raw_{1.0, 0.0, 0.0, 0.0};
  Eigen::Quaterniond feedback_fix_{1.0, 0.0, 0.0, 0.0};
  Eigen::Vector3d odom_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d muzzle_offset_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d camera_offset_{Eigen::Vector3d::Zero()};

  double yaw_ = 0.0;
  double pitch_ = 0.0;
  double yaw_vel_ = 0.0;
  double pitch_vel_ = 0.0;
  bool has_state_ = false;
  bool has_prev_ = false;
  double prev_yaw_ = 0.0;
  double prev_pitch_ = 0.0;
  std::uint64_t prev_timestamp_ns_ = 0;
  std::uint64_t frame_seq_ = 0;
  std::chrono::steady_clock::time_point state_steady_{};

  std::uint32_t faults_ = FAULT_STARTUP;
  std::chrono::steady_clock::time_point last_send_{};
  bool has_sent_ = false;

  std::uint64_t sent_commands_ = 0;
  std::uint64_t fire_commands_ = 0;
  std::uint64_t suppressed_fires_ = 0;
  std::uint64_t safe_stops_ = 0;
  std::uint16_t bullet_count_ = 0;

  bool state_stale() const;
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_GIMBAL_HPP
