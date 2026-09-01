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
  // 命令时刻的**世界观测年龄**（源帧龄 + 本帧全部处理耗时）超出预算。
  // 与 FAULT_STATE_STALE 是两件事：后者只管"本地状态放了多久"（喂数据的通道是否
  // 还活着），这一位管"我要据以开火的这份世界观测有多旧"。默认关闭
  // （max_command_age_ms = 0），必须由运行方按实测分布显式设定，见该配置项。
  FAULT_COMMAND_AGE = 1u << 10,
  // 发布端没有声明某个本模式必需的能力位，于是该通道的读数是"不可知"而不是"零"。
  // 由上层在识别出缺位时置起（例如 closed_loop 拿不到 CAP_RUNTIME_STATE，就无法
  // 判断仿真端到底有没有订阅云台命令）。缺位期间禁止开火——把"不可知"当成"正常"
  // 才是真正危险的那一种降级。
  FAULT_CAPABILITY_MISSING = 1u << 11,
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

  // 本地状态保有时长上限：**从本帧被取到算起**多久没有新帧就视为状态过期。
  // 这是"喂数据的通道死了"的看门狗，与源帧龄无关。
  //
  // 曾经它被拿源端采样时刻去比，于是量到的是"源帧龄 + 本帧全部处理耗时"：
  // 源帧龄 416ms（仿真端 GPU 回读 + 背压，本机固有）叠加检测 247ms 就是 663ms，
  // 对 200ms 的门限恒定超出 —— FAULT_STATE_STALE 永久点亮，closed_loop 里
  // fire 恒为 0 而 suppressed_fire 一路涨，且给出的理由（"姿态过期"）是错的：
  // 通道一直是活的。两个时间点必须分开，见 FrameStamps。
  double state_timeout_ms = 200.0;

  // 命令时刻的世界观测年龄上限（毫秒），0 = 不设限。超出即置 FAULT_COMMAND_AGE。
  //
  // 默认 0 是有意的：帧龄本身已经在**入口**被 SimCameraConfig::max_frame_age_ms
  // 拦过一道，这里再默认拦一次只会把刚修掉的"全局静默抑制"换个名字复活。
  // 报告里始终输出 command_age 的 p50/p95/p99，运行方按实测分布决定要不要设限。
  double max_command_age_ms = 0.0;

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
  // stamps.source = 源端采样时刻，stamps.arrival = 本帧在本进程被取到的时刻。
  // 直接传 SimCamera::last_stamps()。两者绝不能用同一个时间点：
  //   - arrival 决定 state_stale()（通道看门狗）
  //   - source  决定 command_age_ms()（世界观测年龄，用于 FAULT_COMMAND_AGE）
  PoseValidity update(const FrameBundle & bundle, const FrameStamps & stamps);

  // 丢弃时间戳水位线与速度历史，但保留 faults 与累计计数。
  //
  // 发布端换代/重连、以及本地 realtime<->steady 映射跳变之后必须调用。update() 用
  // `bundle.timestamp_ns <= prev_timestamp_ns_` 挡重复帧和乱序帧，而这条水位线跨
  // 换代是无意义的：发布端真的回拨了墙上时钟（重启后系统时间被 NTP 往回校，或换了
  // 一个时间靠后的发布端再换回来），新帧的时间戳会永远小于旧水位线，于是每一帧都
  // 判成 BadTimestamp，消费端被永久锁死在 FAULT_POSE_INVALID，只能重启进程。
  //
  // 速度历史同理：跨换代/跳变做差分得到的 yaw_vel_/pitch_vel_ 含有整个跳变量。
  void reset_history();

  // 最近一次 update() 的校验结果。
  PoseValidity last_validity() const { return last_validity_; }
  std::uint64_t invalid_poses() const { return invalid_poses_; }

  bool has_state() const { return has_state_; }
  std::uint64_t frame_seq() const { return frame_seq_; }

  // 当前状态在本地放了多久（now - arrival）。看门狗量的就是它。
  double state_age_ms() const;
  // 当前状态所描述的世界有多旧（now - source）= 源帧龄 + 至今的处理耗时。
  // 这是开火决策真正关心的量。从未取到状态时两者都返回 +inf。
  double command_age_ms() const;
  // command_age_ms() 超出 max_command_age_ms 的帧数（max_command_age_ms=0 时恒 0）。
  std::uint64_t command_age_violations() const { return command_age_violations_; }

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

  // 同帧的其他 pose，供评估与外参核对。参考系见 PoseIndex 的注释，各不相同：

  // 云台回转中心的**世界**位置（ROS：x 前 y 左 z 上）。
  const Eigen::Vector3d & odom_position() const { return odom_position_; }

  // 枪口的**世界**位置。协议 v3 起由发布端直接给出世界量。
  //
  // 原来这里叫 muzzle_offset()，拿到的是 reparented_to(gimbal) 之后的局部平移，
  // 而调用方最自然的用法 `odom_position() + muzzle_offset()` 是错的：那是把一个
  // 未经云台旋转的局部量加到世界坐标上。yaw=90° 时局部 +X 实际指向世界 +Y，
  // 误差等于偏移量全长。改名是为了让这种误用不再能编译通过。
  const Eigen::Vector3d & muzzle_position() const { return muzzle_position_; }

  // 相机相对云台的**局部**平移，用于与 t_camera2gimbal 外参自检。
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
  Eigen::Vector3d muzzle_position_{Eigen::Vector3d::Zero()};
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
  // 分开保存，语义见 FrameStamps。
  std::chrono::steady_clock::time_point state_source_{};
  std::chrono::steady_clock::time_point state_arrival_{};
  std::uint64_t command_age_violations_ = 0;

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

  // 通道看门狗：state_age_ms() > state_timeout_ms。只看本地保有时长。
  bool state_stale() const;
  // 世界观测年龄超预算。max_command_age_ms <= 0 时恒 false。
  bool command_age_exceeded() const;
};

}  // namespace sim_io

#endif  // SIMULATION_IO__SIM_GIMBAL_HPP
