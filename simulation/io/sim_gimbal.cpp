#include "sim_gimbal.hpp"

#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"

namespace sim_io
{
const char * to_string(PoseValidity validity)
{
  switch (validity) {
    case PoseValidity::Ok:
      return "Ok";
    case PoseValidity::NonFinite:
      return "NonFinite";
    case PoseValidity::QuaternionNorm:
      return "QuaternionNorm";
    case PoseValidity::BadTimestamp:
      return "BadTimestamp";
    case PoseValidity::FrameContract:
      return "FrameContract";
  }
  return "Unknown";
}

namespace
{
struct FaultEntry
{
  std::uint32_t bit;
  const char * name;
};
}  // namespace

std::string describe_faults(std::uint32_t faults)
{
  if (faults == FAULT_NONE) return "none";

  using Entry = FaultEntry;
  static const Entry entries[] = {
    {FAULT_STARTUP, "startup"},
    {FAULT_HEARTBEAT_LOST, "heartbeat_lost"},
    {FAULT_NO_NEW_FRAME, "no_new_frame"},
    {FAULT_TARGET_LOST, "target_lost"},
    {FAULT_CLOCK_JUMP, "clock_jump"},
    {FAULT_FRAME_FAULT, "frame_fault"},
    {FAULT_FIRE_DISABLED, "fire_disabled"},
    {FAULT_STATE_STALE, "state_stale"},
    {FAULT_POSE_INVALID, "pose_invalid"},
    {FAULT_REARM_PENDING, "rearm_pending"},
    {FAULT_COMMAND_AGE, "command_age"},
    {FAULT_CAPABILITY_MISSING, "capability_missing"},
    {FAULT_DYNAMIC_ERROR, "dynamic_error"},
    {FAULT_NOT_FOLLOWING, "not_following"},
  };

  std::string out;
  for (const Entry & e : entries) {
    if ((faults & e.bit) == 0) continue;
    if (!out.empty()) out += "|";
    out += e.name;
  }
  return out;
}

namespace
{
// 与 describe_faults 同一份顺序。放在这里而不是共用一个 static，是为了不改
// describe_faults 已被多处测试钉住的输出格式。
const FaultEntry kFaults[] = {
  {FAULT_STARTUP, "startup"},
  {FAULT_HEARTBEAT_LOST, "heartbeat_lost"},
  {FAULT_NO_NEW_FRAME, "no_new_frame"},
  {FAULT_TARGET_LOST, "target_lost"},
  {FAULT_CLOCK_JUMP, "clock_jump"},
  {FAULT_FRAME_FAULT, "frame_fault"},
  {FAULT_FIRE_DISABLED, "fire_disabled"},
  {FAULT_STATE_STALE, "state_stale"},
  {FAULT_POSE_INVALID, "pose_invalid"},
  {FAULT_REARM_PENDING, "rearm_pending"},
  {FAULT_COMMAND_AGE, "command_age"},
  {FAULT_CAPABILITY_MISSING, "capability_missing"},
  {FAULT_DYNAMIC_ERROR, "dynamic_error"},
  {FAULT_NOT_FOLLOWING, "not_following"},
};
constexpr std::size_t fault_table_size() { return sizeof(kFaults) / sizeof(kFaults[0]); }
}  // namespace

const std::vector<std::uint32_t> & fault_bits()
{
  static const std::vector<std::uint32_t> bits = [] {
    std::vector<std::uint32_t> v;
    for (std::size_t i = 0; i < fault_table_size(); ++i) v.push_back(kFaults[i].bit);
    return v;
  }();
  return bits;
}

const char * fault_name(std::uint32_t bit)
{
  for (std::size_t i = 0; i < fault_table_size(); ++i) {
    if (kFaults[i].bit == bit) return kFaults[i].name;
  }
  return "unknown";
}

SimGimbal::SimGimbal(SharedMemoryClient & client, SimGimbalConfig config)
: client_(client), config_(config)
{
  if (!config_.allow_fire) faults_ |= FAULT_FIRE_DISABLED;

  // 见 SimGimbalConfig::feedback_pitch_fix_deg 的推导：绕 ROS Y 轴右乘 +90°。
  feedback_fix_ = Eigen::Quaterniond(Eigen::AngleAxisd(
    config_.feedback_pitch_fix_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()));
  fault_history_.reserve(fault_bits().size());
  for (std::uint32_t bit : fault_bits()) {
    FaultHistory h;
    h.bit = bit;
    h.name = fault_name(bit);
    fault_history_.push_back(h);
  }
  // 构造时 faults_ 已是 FAULT_STARTUP（外加 allow_fire=false 时的 fire_disabled），
  // 采一次样，让它们的 first_seen_s = 0 而不是"第一次 set_fault 的时刻"。
  sample_faults_at(0.0);
}

namespace
{
bool finite3(const float v[3])
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool finite4(const float v[4])
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
         std::isfinite(v[3]);
}
}  // namespace

PoseValidity SimGimbal::update(const FrameBundle & bundle, const FrameStamps & stamps)
{
  const PoseMeta & gimbal = bundle.gimbal();

  // 输入校验先行。共享内存是外部进程写的，不能假定它一定合法：
  // 未初始化的槽位是全零（四元数模长 0），NaN 会一路穿过 normalize()、
  // tools::eulers()、solver.set_R_gimbal2world()，最后表现为目标位置 NaN、
  // planner 下发 NaN——只有 send() 里那道 isfinite 门能拦住，而那时 EKF
  // 已经被污染了。所以在这里就把不合法的帧挡掉，并且**不更新任何状态**。
  auto reject = [&](PoseValidity why) {
    last_validity_ = why;
    ++invalid_poses_;
    faults_ |= static_cast<std::uint32_t>(FAULT_POSE_INVALID);
    return why;
  };

  if (bundle.timestamp_ns == 0) return reject(PoseValidity::BadTimestamp);
  // 时间戳必须严格前进。相等或倒退意味着上游重复发布或换代，速度差分会得到
  // 无意义的结果（dt<=0）。第一帧（has_prev_ == false）没有比较基准，放过。
  if (has_prev_ && bundle.timestamp_ns <= prev_timestamp_ns_) {
    return reject(PoseValidity::BadTimestamp);
  }

  for (int i = 0; i <= static_cast<int>(PoseIndex::Camera); ++i) {
    if (
      !bundle.pose_present[i] || bundle.poses[i].frame_seq != bundle.frame_seq ||
      bundle.poses[i].timestamp_ns != bundle.timestamp_ns)
      return reject(PoseValidity::FrameContract);
  }

  for (int i = 0; i < static_cast<int>(POSE_CHANNEL_COUNT); ++i) {
    if (!finite3(bundle.poses[i].position) || !finite4(bundle.poses[i].quaternion)) {
      return reject(PoseValidity::NonFinite);
    }
  }

  {
    const double n = std::sqrt(
      static_cast<double>(gimbal.quaternion[0]) * gimbal.quaternion[0] +
      static_cast<double>(gimbal.quaternion[1]) * gimbal.quaternion[1] +
      static_cast<double>(gimbal.quaternion[2]) * gimbal.quaternion[2] +
      static_cast<double>(gimbal.quaternion[3]) * gimbal.quaternion[3]);
    if (std::abs(n - 1.0) > config_.quaternion_norm_tol) {
      return reject(PoseValidity::QuaternionNorm);
    }
  }

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
  // 协议 v3：Muzzle 通道就是枪口的世界位置，直接取用，不再与 odom 相加。
  muzzle_position_ = Eigen::Vector3d(muzzle.position[0], muzzle.position[1], muzzle.position[2]);
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
  state_source_ = stamps.source;
  state_arrival_ = stamps.arrival;
  has_state_ = true;
  last_validity_ = PoseValidity::Ok;
  faults_ &= ~static_cast<std::uint32_t>(FAULT_STARTUP);
  faults_ &= ~static_cast<std::uint32_t>(FAULT_POSE_INVALID);
  return PoseValidity::Ok;
}

void SimGimbal::reset_history()
{
  has_prev_ = false;
  prev_yaw_ = 0.0;
  prev_pitch_ = 0.0;
  prev_timestamp_ns_ = 0;
  yaw_vel_ = 0.0;
  pitch_vel_ = 0.0;
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

namespace
{
double age_ms_since(std::chrono::steady_clock::time_point t)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

double SimGimbal::state_age_ms() const
{
  if (!has_state_) return std::numeric_limits<double>::infinity();
  return age_ms_since(state_arrival_);
}

double SimGimbal::command_age_ms() const
{
  if (!has_state_) return std::numeric_limits<double>::infinity();
  return age_ms_since(state_source_);
}

bool SimGimbal::state_stale() const
{
  if (!has_state_) return true;
  // 必须用 arrival（本地接收时刻），不能用 source。用 source 量出来的是
  // "源帧龄 + 本帧处理耗时"，把入口那道 max_frame_age_ms 又算了一遍，还把
  // 检测耗时算成了"姿态过期"。见 SimGimbalConfig::state_timeout_ms。
  return state_age_ms() > config_.state_timeout_ms;
}

bool SimGimbal::command_age_exceeded() const
{
  if (config_.max_command_age_ms <= 0.0) return false;
  if (!has_state_) return false;  // 没状态由 FAULT_STARTUP 负责，不在这里重复
  const double age_ms = command_age_ms();
  // 负年龄意味着命令依据的观测来自未来。它不能因为数值上小于正预算而绕过年龄门。
  return age_ms < 0.0 || age_ms > config_.max_command_age_ms;
}

std::uint32_t SimGimbal::faults() const
{
  std::uint32_t f = faults_;
  if (!has_state_) f |= FAULT_STARTUP;
  if (state_stale()) f |= FAULT_STATE_STALE;
  if (command_age_exceeded()) f |= FAULT_COMMAND_AGE;
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
  sample_faults();
}

void SimGimbal::clear_faults(std::uint32_t mask)
{
  faults_ &= ~mask;
  sample_faults();
}

double SimGimbal::uptime_s() const
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - created_).count();
}

void SimGimbal::sample_faults() { sample_faults_at(uptime_s()); }

void SimGimbal::sample_faults_at(double now_s)
{
  // 时间必须单调：调用方混用注入时间与真实时钟时，宁可把这一段时长记为 0，
  // 也不能让 total_s 因为负的 dt 越记越少。
  if (now_s < fault_sampled_s_) now_s = fault_sampled_s_;
  const double dt = now_s - fault_sampled_s_;
  fault_sampled_s_ = now_s;

  const std::uint32_t now = faults();
  faults_seen_ |= now;

  for (FaultHistory & h : fault_history_) {
    const bool on = (now & h.bit) != 0;
    if (h.active) {
      // 上一段区间的时长归给这一位，无论它这一次是否熄灭。
      h.total_s += dt;
      const double episode = now_s - h.last_seen_s;
      if (episode > h.max_s) h.max_s = episode;
      if (!on) {
        h.active = false;
        h.last_cleared_s = now_s;
      }
    } else if (on) {
      h.active = true;
      ++h.episodes;
      h.last_seen_s = now_s;
      if (h.first_seen_s < 0.0) h.first_seen_s = now_s;
    }
  }
}

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
  // 只在真正下发控制的帧上计数。放在 command_age_exceeded() 里会变成"谓词被调用
  // 了多少次"（faults() 每帧要问好几遍），放在这里量的才是"多少条控制命令是基于
  // 超预算的世界观测发出的"。安全停止帧不计：它与世界观测无关。
  if (control && command_age_exceeded()) ++command_age_violations_;

  // 开火判据就是在这里生效的，采一次样保证"抑制开火的那一瞬间点亮了哪些位"
  // 一定进历史——即使它在本轮循环结束前就被清掉了。
  sample_faults();

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

  GimbalCmd cmd = encode(control, fire_out, yaw_rad, pitch_rad, distance_m);
  cmd.command_seq = ++last_command_seq_;
  if (!client_.publish_gimbal_cmd(cmd)) return false;
  last_command_ = cmd;

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
