#include "sim_gimbal.hpp"

#include <cmath>

#include "tools/math_tools.hpp"

namespace sim_io
{
std::string describe_faults(std::uint32_t faults)
{
  if (faults == FAULT_NONE) return "none";

  struct Entry
  {
    std::uint32_t bit;
    const char * name;
  };
  static const Entry entries[] = {
    {FAULT_STARTUP, "startup"},
    {FAULT_HEARTBEAT_LOST, "heartbeat_lost"},
    {FAULT_NO_NEW_FRAME, "no_new_frame"},
    {FAULT_TARGET_LOST, "target_lost"},
    {FAULT_CLOCK_JUMP, "clock_jump"},
    {FAULT_FRAME_FAULT, "frame_fault"},
    {FAULT_FIRE_DISABLED, "fire_disabled"},
    {FAULT_STATE_STALE, "state_stale"},
  };

  std::string out;
  for (const Entry & e : entries) {
    if ((faults & e.bit) == 0) continue;
    if (!out.empty()) out += "|";
    out += e.name;
  }
  return out;
}

SimGimbal::SimGimbal(SharedMemoryClient & client, SimGimbalConfig config)
: client_(client), config_(config)
{
  if (!config_.allow_fire) faults_ |= FAULT_FIRE_DISABLED;

  // 见 SimGimbalConfig::feedback_pitch_fix_deg 的推导：绕 ROS Y 轴右乘 +90°。
  feedback_fix_ = Eigen::Quaterniond(Eigen::AngleAxisd(
    config_.feedback_pitch_fix_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()));
}

void SimGimbal::update(
  const FrameBundle & bundle, std::chrono::steady_clock::time_point steady_ts)
{
  const PoseMeta & gimbal = bundle.gimbal();

  // Rust 侧发布顺序是 [w,x,y,z]，Eigen 构造函数也是 (w,x,y,z)。
  q_raw_ = Eigen::Quaterniond(
    static_cast<double>(gimbal.quaternion[0]), static_cast<double>(gimbal.quaternion[1]),
    static_cast<double>(gimbal.quaternion[2]), static_cast<double>(gimbal.quaternion[3]));
  q_raw_.normalize();

  // 剥掉仿真端多乘的 90° 滚转，得到真正的枪口姿态。
  // 不修正的话下发 pitch=0 会让欧拉角分解正好踩在万向锁上，yaw 直接失效。
  q_ = q_raw_ * feedback_fix_;
  q_.normalize();

  const PoseMeta & odom = bundle.odom();
  const PoseMeta & muzzle = bundle.muzzle();
  const PoseMeta & camera = bundle.camera();
  odom_position_ = Eigen::Vector3d(odom.position[0], odom.position[1], odom.position[2]);
  muzzle_offset_ = Eigen::Vector3d(muzzle.position[0], muzzle.position[1], muzzle.position[2]);
  camera_offset_ = Eigen::Vector3d(camera.position[0], camera.position[1], camera.position[2]);

  // 与 solver / planner 一致的欧拉角约定：intrinsic ZYX -> [yaw, pitch, roll]
  const Eigen::Vector3d ypr = tools::eulers(q_, 2, 1, 0);
  yaw_ = ypr[0];
  pitch_ = ypr[1];

  if (has_prev_ && bundle.timestamp_ns > prev_timestamp_ns_) {
    const double dt =
      static_cast<double>(bundle.timestamp_ns - prev_timestamp_ns_) * 1e-9;
    if (dt > 1e-6) {
      // yaw 跨 ±pi 需要解缠，否则一次跳变会产生巨大的假角速度。
      double dyaw = yaw_ - prev_yaw_;
      while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
      while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

      const double a = config_.vel_lpf_alpha;
      yaw_vel_ = (1.0 - a) * yaw_vel_ + a * (dyaw / dt);
      pitch_vel_ = (1.0 - a) * pitch_vel_ + a * ((pitch_ - prev_pitch_) / dt);
    }
  }

  prev_yaw_ = yaw_;
  prev_pitch_ = pitch_;
  prev_timestamp_ns_ = bundle.timestamp_ns;
  has_prev_ = true;

  frame_seq_ = bundle.frame_seq;
  state_steady_ = steady_ts;
  has_state_ = true;
  faults_ &= ~static_cast<std::uint32_t>(FAULT_STARTUP);
}

io::GimbalState SimGimbal::state() const
{
  io::GimbalState gs{};
  gs.yaw = static_cast<float>(yaw_);
  gs.yaw_vel = static_cast<float>(yaw_vel_);
  gs.pitch = static_cast<float>(pitch_);
  gs.pitch_vel = static_cast<float>(pitch_vel_);
  // 共享内存没有弹速字段，只能由配置与仿真端 [projectile] speed 对齐。
  gs.bullet_speed = static_cast<float>(config_.bullet_speed);
  gs.bullet_count = bullet_count_;
  return gs;
}

bool SimGimbal::state_stale() const
{
  if (!has_state_) return true;
  const double age_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state_steady_)
      .count();
  return age_ms > config_.state_timeout_ms;
}

std::uint32_t SimGimbal::faults() const
{
  std::uint32_t f = faults_;
  if (!has_state_) f |= FAULT_STARTUP;
  if (state_stale()) f |= FAULT_STATE_STALE;
  if (!config_.allow_fire) f |= FAULT_FIRE_DISABLED;
  return f;
}

void SimGimbal::set_fault(std::uint32_t fault, bool active)
{
  if (active) {
    faults_ |= fault;
  } else {
    faults_ &= ~fault;
  }
}

void SimGimbal::clear_faults(std::uint32_t mask) { faults_ &= ~mask; }

GimbalCmd SimGimbal::encode(
  bool control, bool fire, double yaw_rad, double pitch_rad, double distance_m) const
{
  GimbalCmd cmd{};
  cmd.timestamp_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch())
      .count());

  if (!control) {
    // 仿真端约定：distance_m == -1 表示无控制，直接 return，不改动云台也不开火。
    cmd.yaw_deg = 0.0f;
    cmd.pitch_deg = 0.0f;
    cmd.distance_m = -1.0f;
    cmd.fire_advice = 0;
    return cmd;
  }

  const double yaw_deg = config_.yaw_scale * (yaw_rad * 180.0 / M_PI) + config_.yaw_offset_deg;
  const double pitch_deg =
    config_.pitch_scale * (pitch_rad * 180.0 / M_PI) + config_.pitch_offset_deg;

  cmd.yaw_deg = static_cast<float>(yaw_deg);
  cmd.pitch_deg = static_cast<float>(pitch_deg);
  // 不能是 -1，否则会被仿真端当成无控制。
  cmd.distance_m = static_cast<float>(distance_m < 0.0 ? 0.0 : distance_m);
  cmd.fire_advice = fire ? 1 : 0;
  return cmd;
}

// decode_* 是 encode() 的严格逆变换：拿回我们下发的那个角（ROS 约定）。
// 注意必须走 config_ 里的 scale/offset，不能照抄仿真端公式硬编码——否则一旦
// 标定改了 offset，这两个函数就会静默地和 encode 脱钩。
double SimGimbal::decode_yaw_rad(const GimbalCmd & cmd) const
{
  const double scale = config_.yaw_scale != 0.0 ? config_.yaw_scale : 1.0;
  return (static_cast<double>(cmd.yaw_deg) - config_.yaw_offset_deg) / scale * M_PI / 180.0;
}

double SimGimbal::decode_pitch_rad(const GimbalCmd & cmd) const
{
  const double scale = config_.pitch_scale != 0.0 ? config_.pitch_scale : 1.0;
  return (static_cast<double>(cmd.pitch_deg) - config_.pitch_offset_deg) / scale * M_PI / 180.0;
}

// 仿真端 process_subscription 会解出来的内部角（Bevy 的 YXZ 约定），
// 直接抄 plugin.rs。它和 decode_pitch_rad 差一个负号，那正是两套约定的差异。
double SimGimbal::sim_internal_yaw_rad(const GimbalCmd & cmd)
{
  return static_cast<double>(cmd.yaw_deg) * M_PI / 180.0;
}

double SimGimbal::sim_internal_pitch_rad(const GimbalCmd & cmd)
{
  return (-static_cast<double>(cmd.pitch_deg) - 90.0) * M_PI / 180.0;
}

bool SimGimbal::send(
  bool control, bool fire, double yaw_rad, double pitch_rad, double distance_m)
{
  bool fire_out = fire;
  if (fire && !fire_allowed()) {
    ++suppressed_fires_;
    fire_out = false;
  }

  // 非有限值绝不能发出去：float 转换后会变成 NaN，仿真端 to_radians 之后
  // 会把云台旋转写成 NaN 并永久污染 Transform。
  if (control && (!std::isfinite(yaw_rad) || !std::isfinite(pitch_rad))) {
    control = false;
    fire_out = false;
  }

  const GimbalCmd cmd = encode(control, fire_out, yaw_rad, pitch_rad, distance_m);
  if (!client_.publish_gimbal_cmd(cmd)) return false;

  ++sent_commands_;
  if (cmd.fire_advice == 1) {
    ++fire_commands_;
    ++bullet_count_;
  }
  if (cmd.distance_m == -1.0f) ++safe_stops_;

  last_send_ = std::chrono::steady_clock::now();
  has_sent_ = true;
  return true;
}

bool SimGimbal::send_safe_stop() { return send(false, false, 0.0, 0.0, -1.0); }

bool SimGimbal::tick()
{
  if (has_sent_) {
    const double since_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - last_send_)
        .count();
    if (since_ms < config_.safe_stop_period_ms) return false;
  }
  return send_safe_stop();
}

}  // namespace sim_io
