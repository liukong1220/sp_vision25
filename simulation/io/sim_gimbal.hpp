#ifndef SIMULATION_IO__SIM_GIMBAL_HPP
#define SIMULATION_IO__SIM_GIMBAL_HPP

// 仿真云台：把共享内存的 pose 通道当作云台反馈，把控制量编码成 GimbalCmd 发回。
//
// 两个关键点：
//   1. 状态必须来自与图像严格同帧的 pose（update(bundle) 只接受 SimCamera 的
//      last_bundle()），不能用“处理完成时的最新姿态”，否则解算用的旋转和图像
//      对不上，误差会被当成目标运动吃进 EKF。
//   2. 命令编码里 yaw/pitch 的正负与零点做成可配置仿射映射（scale/offset）。
//        这里编码: yaw_deg = deg(yaw), pitch_deg = deg(pitch)   —— identity
//
//      cmd.pitch_deg 就是算法侧的 ROS pitch 本身（正 = 低头，见 aimer.cpp:122
//      "世界坐标系下pitch向上为负"），所以正确映射是 identity。共享内存探针
//      实测（读回同帧发布的云台四元数，取 q_raw*(1,0,0) 当枪口指向）：
//        cmd 0 -> elev -0.01deg(水平)   cmd -20 -> elev +19.80deg(抬头)
//        cmd +20 -> elev -20.63deg      cmd -90 -> elev +90.00deg(朝天)
//      且 fb(raw) 的 ZYX pitch 逐点恒等于 cmd.pitch_deg。
//
//      标定判据必须是**枪口绝对指向**，不能只看"下发角 == 反馈角"：后者在任何
//      pitch_offset_deg 下都成立，只能验证自洽性。曾经据此把 offset 定成 -90，
//      结果算法瞄水平目标（ROS pitch≈0）时枪口被抬到 elev=+90 垂直朝天。

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
  FAULT_POSE_INVALID = 1u << 8,   // 同帧位姿不合法（非有限值 / 四元数模长异常 / 时间戳倒退）
  FAULT_REARM_PENDING = 1u << 9,  // 发布端换代/重连/时钟跳变之后尚未重新确认目标
};

// update() 对输入位姿的校验结果。Ok 之外的一律不更新云台状态：
// 上一帧的姿态原样保留（随后会因 state_stale 过期），而不是把 NaN 喂给 solver。
enum class PoseValidity
{
  Ok,
  NonFinite,       // position/quaternion 里有 NaN 或 Inf
  QuaternionNorm,  // 四元数模长偏离 1 太多（含全零）
  BadTimestamp,    // timestamp_ns 为 0 或相对上一帧倒退
};

const char * to_string(PoseValidity validity);

std::string describe_faults(std::uint32_t faults);

struct SimGimbalConfig
{
  // 四元数模长允许偏离 1 的量。仿真端发的是 f32，归一化误差量级 1e-7；
  // 放到 1e-3 只用来抓"根本不是单位四元数"（全零、未初始化、被别的结构覆盖）。
  double quaternion_norm_tol = 1e-3;

  // 命令映射：identity，cmd 的角就是 ROS 的角（推导见文件头）。
  double yaw_scale = 1.0;
  double yaw_offset_deg = 0.0;
  double pitch_scale = 1.0;
  double pitch_offset_deg = 0.0;

  // simulator 的 [projectile] speed。共享内存没有这个字段，必须与仿真端配置一致。
  double bullet_speed = 25.0;

  bool allow_fire = false;           // 默认禁止开火
  double state_timeout_ms = 200.0;   // 同帧姿态超过此龄视为过期
  double safe_stop_period_ms = 20.0; // 空闲时重发安全停止命令的周期
  double vel_lpf_alpha = 0.35;       // 角速度低通系数

  // 反馈四元数的坐标系修正，绕 ROS Y 轴右乘（单位：度）。默认 0 = 不修正。
  //
  // 上面那个 Rx(90°) 不是"多乘的"，它正是把 Bevy 的枪管前向对齐到 ROS +X 的
  // 那一步：Bevy 里枪管前向是该帧的局部 +Y，Rx(90°) 把 +Y 送到 -Z（Bevy 的前向
  // 约定），再过 to_ros_quat 之后 ROS 的 +X 就是出膛方向。所以仿真端发布的
  // q_raw 已经是标准 world<-gimbal（x 前 y 左 z 上），直接可用。
  //
  // 实测佐证：cmd pitch=0 时 q_raw*(1,0,0) = (1.0000, 0, -0.0001)，即水平，
  // 与 avian 真正用于 spawn 弹丸的方向一致；ZYX 分解是
  // yaw=0.000 pitch=0.000 roll=0.000，远离奇点。
  //
  // 反过来再右乘 Ry(+90°) 才会制造问题：那样 cmd pitch=0 会解出
  // fb pitch=90°、yaw/roll 一起退化，看着像万向锁，其实是被这次多余的修正推到
  // 奇点上的；而且这个四元数要交给 solver.set_R_gimbal2world()，会让 PnP 的
  // 世界坐标整体偏 90°。保持 0，除非仿真端改了发布式。
  double feedback_pitch_fix_deg = 0.0;
};

class SimGimbal
{
public:
  SimGimbal(SharedMemoryClient & client, SimGimbalConfig config = {});

  // 用与图像同帧的 pose 束更新状态。必须在 solver.set_R_gimbal2world() 之前调用。
  //
  // 返回校验结果。非 Ok 时**不更新任何状态**（q()/yaw()/pitch()/frame_seq() 仍是
  // 上一帧的值，state_steady_ 不推进，所以很快会 state_stale），并置起
  // FAULT_POSE_INVALID。调用方必须检查返回值：把 NaN 四元数交给
  // solver.set_R_gimbal2world() 会让整条 PnP 链路静默产出 NaN 目标位置。
  PoseValidity update(const FrameBundle & bundle, std::chrono::steady_clock::time_point steady_ts);

  // 最近一次 update() 的校验结果。
  PoseValidity last_validity() const { return last_validity_; }
  std::uint64_t invalid_poses() const { return invalid_poses_; }

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
  PoseValidity last_validity_ = PoseValidity::Ok;
  std::uint64_t invalid_poses_ = 0;

  bool state_stale() const;
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_GIMBAL_HPP
