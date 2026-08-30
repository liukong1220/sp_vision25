// 仿真 IPC 回环与故障注入测试。
//
// 生产者用 sim_io::testing::FakePublisher（Rust ShmPublisher 的 C++ 镜像），
// 因此本测试不需要启动 Bevy，可以在 CI 里无头运行。
// 覆盖：图像 RGB→BGR 回环、时间戳映射与帧龄、同帧姿态交接、背压握手、
//       帧号重复/回退/跳变、姿态与图像帧号不一致、心跳超时、过期帧、
//       云台命令与仿真端解码公式的往返一致性、开火抑制与安全停止。
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "simulation/io/sim_camera.hpp"
#include "simulation/io/sim_gimbal.hpp"
#include "simulation/io/testing/fake_publisher.hpp"

namespace
{
int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string & name, const std::string & detail = "")
{
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf(
    "%-58s %s%s%s\n", name.c_str(), ok ? "ok" : "FAIL", detail.empty() ? "" : "  ",
    detail.c_str());
}

void check_near(double got, double want, double tol, const std::string & name)
{
  char buf[128];
  std::snprintf(buf, sizeof(buf), "got=%.9g want=%.9g tol=%.3g", got, want, tol);
  check(std::abs(got - want) <= tol, name, buf);
}

std::uint64_t realtime_now_ns()
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch())
      .count());
}

// 生成可逐像素校验的 RGB8 图案：像素值只依赖坐标，通道之间互不相同，
// 这样一旦发生通道错序、行跨距错误或缓冲槽错位都能被采样点抓到。
struct RgbPattern
{
  std::vector<std::uint8_t> data;

  explicit RgbPattern(std::uint8_t salt) : data(sim_io::IMAGE_SIZE)
  {
    for (std::uint32_t y = 0; y < sim_io::IMAGE_HEIGHT; ++y) {
      for (std::uint32_t x = 0; x < sim_io::IMAGE_WIDTH; ++x) {
        const std::size_t i =
          (static_cast<std::size_t>(y) * sim_io::IMAGE_WIDTH + x) * sim_io::IMAGE_CHANNELS;
        data[i + 0] = static_cast<std::uint8_t>((x * 7u + salt) & 0xFFu);          // R
        data[i + 1] = static_cast<std::uint8_t>((y * 13u + salt * 3u) & 0xFFu);    // G
        data[i + 2] = static_cast<std::uint8_t>((x + y + salt * 5u) & 0xFFu);      // B
      }
    }
  }

  cv::Vec3b rgb_at(std::uint32_t x, std::uint32_t y) const
  {
    const std::size_t i =
      (static_cast<std::size_t>(y) * sim_io::IMAGE_WIDTH + x) * sim_io::IMAGE_CHANNELS;
    return cv::Vec3b(data[i + 0], data[i + 1], data[i + 2]);
  }
};

// 与 capture.rs 一致：gimbal 四元数以 [w,x,y,z] 发布。
//
// capture.rs 发布前会多乘一个绕枪口自身 X 轴的 90° 滚转，经 to_ros_quat 之后
// 等价于在 ROS 系里右乘 Ry(-90°)。这里必须照抄这个约定，否则回环测试测的是一个
// 现实中不存在的生产者，SimGimbal 的坐标系修正就永远测不到。
const Eigen::Quaterniond & simulator_feedback_roll()
{
  static const Eigen::Quaterniond q(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitY()));
  return q;
}

// 期望的枪口姿态（未加发布端滚转），SimGimbal::q() 应当还原出这个。
Eigen::Quaterniond muzzle_quat_from_yaw_pitch(double yaw, double pitch)
{
  return Eigen::Quaterniond(
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
}

void quat_wxyz_from_yaw_pitch(double yaw, double pitch, float out[4])
{
  Eigen::Quaterniond q = muzzle_quat_from_yaw_pitch(yaw, pitch) * simulator_feedback_roll();
  q.normalize();
  out[0] = static_cast<float>(q.w());
  out[1] = static_cast<float>(q.x());
  out[2] = static_cast<float>(q.y());
  out[3] = static_cast<float>(q.z());
}

// 仿真端 process_subscription 的原始解码公式，直接抄 plugin.rs，用来交叉验证编码。
double simulator_decode_yaw_rad(float yaw_deg) { return static_cast<double>(yaw_deg) * M_PI / 180.0; }
double simulator_decode_pitch_rad(float pitch_deg)
{
  return static_cast<double>(-pitch_deg - 90.0f) * M_PI / 180.0;
}
}  // namespace

int main()
{
  char dir_buf[128];
  std::snprintf(dir_buf, sizeof(dir_buf), "/tmp/sim_io_loopback_%d", static_cast<int>(::getpid()));
  const std::string dir = dir_buf;
  if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
    std::printf("无法创建临时目录 %s\n", dir.c_str());
    return 2;
  }

  sim_io::testing::FakePublisher::Options pub_opt;
  pub_opt.dir = dir;
  sim_io::testing::FakePublisher pub(pub_opt);

  std::string err;
  if (!pub.create(&err)) {
    std::printf("FakePublisher 创建失败: %s\n", err.c_str());
    return 2;
  }
  pub.update_heartbeat();

  sim_io::SimCameraConfig cam_cfg;
  cam_cfg.shm.dir = dir;
  cam_cfg.max_frame_age_ms = 200.0;
  cam_cfg.heartbeat_timeout_ms = 500.0;
  cam_cfg.read_timeout_ms = 200.0;

  sim_io::SimCamera cam(cam_cfg);
  if (!cam.open(&err)) {
    std::printf("SimCamera 打开失败: %s\n", err.c_str());
    pub.unlink_files();
    ::rmdir(dir.c_str());
    return 2;
  }
  check(cam.connected(), "SimCamera 连接成功");

  sim_io::SimGimbalConfig gim_cfg;
  gim_cfg.allow_fire = false;
  sim_io::SimGimbal gimbal(cam.client(), gim_cfg);

  cv::Mat img;
  std::chrono::steady_clock::time_point ts;

  // ---- 1. 图像 + 同帧姿态回环 -------------------------------------------------
  std::printf("--- 图像与姿态回环 --------------------------------------------------\n");
  const RgbPattern pattern(11);
  const double sent_yaw = 0.30, sent_pitch = -0.12;
  float quat[4];
  quat_wxyz_from_yaw_pitch(sent_yaw, sent_pitch, quat);
  const float odom_pos[3] = {1.25f, -2.5f, 0.375f};
  const float muzzle_rel[3] = {0.11f, 0.02f, -0.03f};
  const float camera_rel[3] = {0.06f, 0.01f, 0.09f};

  const std::uint64_t t0 = realtime_now_ns();
  check(
    pub.try_publish_synchronized_frame(
      pattern.data.data(), 1, t0, quat, odom_pos, muzzle_rel, camera_rel),
    "首帧同步发布成功");

  sim_io::ReadStatus st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok, std::string("首帧读取状态=") + sim_io::to_string(st));
  check(
    img.rows == static_cast<int>(sim_io::IMAGE_HEIGHT) &&
      img.cols == static_cast<int>(sim_io::IMAGE_WIDTH) && img.type() == CV_8UC3,
    "图像尺寸与类型正确");
  check(!img.empty() && img.isContinuous(), "图像为自有连续缓冲");

  // RGB→BGR：采样点必须是源 RGB 的通道反序。
  const std::uint32_t sample_x[5] = {0, 1, 719, 1439, 1000};
  const std::uint32_t sample_y[5] = {0, 1079, 540, 3, 777};
  bool bgr_ok = true;
  for (int i = 0; i < 5; ++i) {
    const cv::Vec3b src = pattern.rgb_at(sample_x[i], sample_y[i]);
    const cv::Vec3b got = img.at<cv::Vec3b>(
      static_cast<int>(sample_y[i]), static_cast<int>(sample_x[i]));
    if (got[0] != src[2] || got[1] != src[1] || got[2] != src[0]) bgr_ok = false;
  }
  check(bgr_ok, "采样点满足 BGR = 源 RGB 反序");

  check(cam.last_frame_seq() == 1, "帧号回环正确");
  check(cam.last_timestamp_ns() == t0, "源时间戳原样保留");
  const double age_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - ts).count();
  check(age_ms >= 0.0 && age_ms < 200.0, "首帧帧龄在合理范围", std::to_string(age_ms) + " ms");

  // 同帧姿态交接。
  gimbal.update(cam.last_bundle(), ts);
  check(gimbal.has_state(), "SimGimbal 取得姿态");
  check(gimbal.frame_seq() == cam.last_frame_seq(), "姿态与图像同帧");
  // 原始四元数必须逐位保真（不做任何修正）。
  check_near(gimbal.q_raw().w(), quat[0], 1e-6, "原始四元数 w 保真");
  check_near(gimbal.q_raw().x(), quat[1], 1e-6, "原始四元数 x 保真");
  check_near(gimbal.q_raw().y(), quat[2], 1e-6, "原始四元数 y 保真");
  check_near(gimbal.q_raw().z(), quat[3], 1e-6, "原始四元数 z 保真");

  // q() 必须已经剥掉发布端的 90° 滚转，逐分量对上期望的枪口姿态。
  // 四元数有 ±q 二义性，先统一符号再比。
  Eigen::Quaterniond expect = muzzle_quat_from_yaw_pitch(sent_yaw, sent_pitch);
  expect.normalize();
  Eigen::Quaterniond got = gimbal.q();
  if (got.dot(expect) < 0.0) got.coeffs() = -got.coeffs();
  check_near(got.w(), expect.w(), 1e-6, "修正后四元数 w 对上枪口姿态");
  check_near(got.x(), expect.x(), 1e-6, "修正后四元数 x 对上枪口姿态");
  check_near(got.y(), expect.y(), 1e-6, "修正后四元数 y 对上枪口姿态");
  check_near(got.z(), expect.z(), 1e-6, "修正后四元数 z 对上枪口姿态");

  // 这是本次修正要守住的核心性质：下发 pitch=0 时反馈 pitch 必须是 0，
  // 而不是踩在万向锁上的 -90°。
  check_near(gimbal.yaw(), sent_yaw, 1e-5, "yaw 还原");
  check_near(gimbal.pitch(), sent_pitch, 1e-5, "pitch 还原");
  check_near(gimbal.odom_position().x(), odom_pos[0], 1e-6, "odom x");
  check_near(gimbal.odom_position().z(), odom_pos[2], 1e-6, "odom z");
  check_near(gimbal.muzzle_offset().x(), muzzle_rel[0], 1e-6, "muzzle x");
  check_near(gimbal.camera_offset().z(), camera_rel[2], 1e-6, "camera z");

  // 万向锁回归：pitch=0 是实际闭环里最常见的下发值，而它恰好是发布端
  // 多乘 90° 滚转之后落在 ZYX 欧拉角奇点上的那个点。用独立实例做，避免
  // 污染上面那台 gimbal 的角速度状态。
  {
    sim_io::SimGimbal probe(cam.client(), gim_cfg);
    for (double probe_yaw : {0.0, 0.35, -0.80}) {
      float q_roll[4];
      quat_wxyz_from_yaw_pitch(probe_yaw, 0.0, q_roll);
      sim_io::FrameBundle fb{};
      fb.frame_seq = 424242;
      fb.timestamp_ns = realtime_now_ns();
      for (int c = 0; c < sim_io::POSE_CHANNEL_COUNT; ++c) fb.pose_present[c] = true;
      auto & gp = fb.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)];
      for (int c = 0; c < 4; ++c) gp.quaternion[c] = q_roll[c];
      probe.update(fb, std::chrono::steady_clock::now());
      check_near(probe.pitch(), 0.0, 1e-6, "pitch=0 不再落在万向锁上");
      check_near(probe.yaw(), probe_yaw, 1e-5, "奇点附近 yaw 仍然可解");
    }
  }

  // ---- 2. 背压握手 -----------------------------------------------------------
  std::printf("--- 背压握手 --------------------------------------------------------\n");
  const RgbPattern pattern2(77);
  check(
    pub.try_publish_synchronized_frame(pattern2.data.data(), 2, realtime_now_ns(), quat),
    "第 2 帧发布成功（上一帧已被消费）");
  check(
    !pub.try_publish_synchronized_frame(pattern2.data.data(), 3, realtime_now_ns(), quat),
    "未消费时第 3 帧被背压拒绝");
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok, "消费第 2 帧");
  check(cam.last_frame_seq() == 2, "读到的是第 2 帧");
  check(
    pub.try_publish_synchronized_frame(pattern2.data.data(), 3, realtime_now_ns(), quat),
    "消费后第 3 帧恢复发布");
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok && cam.last_frame_seq() == 3, "消费第 3 帧");
  check(
    img.at<cv::Vec3b>(5, 5)[2] == pattern2.rgb_at(5, 5)[0], "第 3 帧像素来自新图案");

  // 无新帧时必须报 Timeout（心跳仍在），而不是拿旧帧顶替。
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Timeout, "无新帧返回 Timeout");
  check(img.empty(), "Timeout 时不返回旧图像");

  // ---- 3. 帧号故障注入 -------------------------------------------------------
  std::printf("--- 帧号故障注入 ----------------------------------------------------\n");
  const std::uint64_t rejected_before = cam.rejected_frames();
  const std::uint64_t regressed_before = cam.client().regressed_frames();

  // 重复帧号：图像 meta 直接覆盖发布，绕过背压。
  pub.publish_pose_bundle(3, realtime_now_ns(), quat);
  pub.publish_image(pattern2.data.data(), 3, realtime_now_ns());
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "重复帧号被拒绝");
  check(cam.client().regressed_frames() == regressed_before + 1, "regressed_frames 计数 +1");

  // 回退帧号。
  pub.publish_pose_bundle(1, realtime_now_ns(), quat);
  pub.publish_image(pattern2.data.data(), 1, realtime_now_ns());
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "回退帧号被拒绝");
  check(cam.client().regressed_frames() == regressed_before + 2, "regressed_frames 计数 +2");
  check(cam.rejected_frames() == rejected_before + 2, "rejected_frames 累计正确");

  // 帧号跳变：last_seq 停在 3（回退帧不推进基准），跳到 10 应记 6 个丢帧。
  const std::uint64_t skipped_before = cam.client().skipped_frames();
  pub.publish_pose_bundle(10, realtime_now_ns(), quat);
  pub.publish_image(pattern2.data.data(), 10, realtime_now_ns());
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok, "跳号帧本身可用");
  check(
    cam.client().skipped_frames() == skipped_before + 6, "skipped_frames 记录 6 个缺口",
    std::to_string(cam.client().skipped_frames() - skipped_before));

  // 姿态帧号与图像不一致：只推进图像，不发新姿态。
  pub.publish_pose_bundle(11, realtime_now_ns(), quat);
  pub.publish_image(pattern2.data.data(), 12, realtime_now_ns());
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "姿态与图像帧号不一致被拒绝");

  // 姿态通道完全缺失。
  pub.publish_image(pattern2.data.data(), 13, realtime_now_ns());
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "姿态缺失被拒绝");

  // 分辨率不符（仿真端换了图像尺寸而 C++ 侧没跟上）。
  pub.publish_pose_bundle(14, realtime_now_ns(), quat);
  pub.publish_image(pattern2.data.data(), 14, realtime_now_ns(), 1280, 720);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "分辨率不符被拒绝");

  // 关键：上面这些被拒绝的帧必须已经把 pose 通道排空，否则背压会锁死仿真端。
  check(pub.synchronized_frame_consumed(), "被拒绝的帧同样完成了背压排空");
  check(
    pub.try_publish_synchronized_frame(pattern.data.data(), 20, realtime_now_ns(), quat),
    "故障注入后同步发布可以继续");
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok && cam.last_frame_seq() == 20, "恢复正常取流");

  // ---- 4. 时间故障注入 -------------------------------------------------------
  std::printf("--- 时间故障注入 ----------------------------------------------------\n");
  const std::uint64_t stale_before = cam.stale_frames();
  const std::uint64_t now_ns = realtime_now_ns();

  // 过期帧：时间戳落在 max_frame_age_ms 之前。
  pub.publish_pose_bundle(21, now_ns - 500'000'000ull, quat);
  pub.publish_image(pattern.data.data(), 21, now_ns - 500'000'000ull);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Stale, "500ms 前的帧被判为过期");
  check(cam.stale_frames() == stale_before + 1, "stale_frames 计数 +1");
  check(img.empty(), "过期帧不返回图像");

  // 未来帧：时间戳超前，帧龄为负，计入 future_frames 但不当过期丢弃。
  const std::uint64_t future_before = cam.future_frames();
  pub.publish_pose_bundle(22, realtime_now_ns() + 50'000'000ull, quat);
  pub.publish_image(pattern.data.data(), 22, realtime_now_ns() + 50'000'000ull);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok, "未来帧仍然可用");
  check(cam.future_frames() == future_before + 1, "future_frames 计数 +1");

  // 时钟桥：给定一个 40ms 前的源时间戳，算出的帧龄应当接近 40ms。
  const double bridged_ms = std::chrono::duration<double, std::milli>(
                              cam.clock().age_now(realtime_now_ns() - 40'000'000ull)).count();
  check_near(bridged_ms, 40.0, 5.0, "realtime→steady 映射帧龄");
  check(cam.clock().jump_count() == 0, "稳定环境下未误报时钟跳变");

  // 心跳超时：把心跳戳回 2 秒前。
  pub.set_heartbeat_ns(realtime_now_ns() - 2'000'000'000ull);
  check(!cam.heartbeat_alive(), "心跳停滞 2s 判为离线");
  check(cam.heartbeat_age_ms() > 1500.0, "heartbeat_age_ms 反映停滞时长");
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Disconnected, "心跳丢失且无新帧时报 Disconnected");
  pub.update_heartbeat();
  check(cam.heartbeat_alive(), "心跳恢复后重新判为在线");

  // ---- 5. 云台命令往返 -------------------------------------------------------
  std::printf("--- 云台命令往返 ----------------------------------------------------\n");
  const double probe_yaw[5] = {0.0, 0.5, -0.5, 1.2, -1.2};
  const double probe_pitch[5] = {0.0, 0.1, -0.1, 0.35, -0.35};
  bool roundtrip_ok = true, sim_formula_ok = true;
  for (int i = 0; i < 5; ++i) {
    check(gimbal.send(true, false, probe_yaw[i], probe_pitch[i], 4.2), "发送云台命令");
    sim_io::GimbalCmd cmd{};
    if (!pub.recv_gimbal_cmd(&cmd)) {
      roundtrip_ok = false;
      continue;
    }
    if (std::abs(gimbal.decode_yaw_rad(cmd) - probe_yaw[i]) > 1e-4) roundtrip_ok = false;
    if (std::abs(gimbal.decode_pitch_rad(cmd) - probe_pitch[i]) > 1e-4) roundtrip_ok = false;
    // 用 plugin.rs 的原始公式再算一遍，确认解码器没有跟着编码器一起写错。
    // yaw：两侧约定同向，仿真端解出来的就是我们下发的值。
    if (std::abs(simulator_decode_yaw_rad(cmd.yaw_deg) - probe_yaw[i]) > 1e-4)
      sim_formula_ok = false;
    // SimGimbal 自带的仿真端公式必须和这里手抄的 plugin.rs 公式逐位一致。
    if (sim_io::SimGimbal::sim_internal_yaw_rad(cmd) != simulator_decode_yaw_rad(cmd.yaw_deg))
      sim_formula_ok = false;
    if (
      sim_io::SimGimbal::sim_internal_pitch_rad(cmd) !=
      simulator_decode_pitch_rad(cmd.pitch_deg))
      sim_formula_ok = false;
    // pitch：仿真端把解码值塞进 Quat::from_euler(YXZ, yaw, pitch, 0)，绕 Bevy +X
    // 正转是抬头；而我们下发/反馈用的是 ROS ZYX 约定，绕 +Y 正转是低头。两个约定
    // 天生反向，所以仿真端内部的 pitch 必须是我们下发值的相反数——这正是
    // pitch_scale=+1 的由来（probe --axis=pitch 实测：scale=-1 时 fb=-cmd）。
    if (std::abs(simulator_decode_pitch_rad(cmd.pitch_deg) + probe_pitch[i]) > 1e-4)
      sim_formula_ok = false;
    if (cmd.distance_m < 0.0f) roundtrip_ok = false;
  }
  check(roundtrip_ok, "编码/解码往返一致（5 组角度）");
  check(sim_formula_ok, "与 plugin.rs 解码公式逐组一致");

  // 非有限角度必须降级为安全停止，不能把 NaN 写进仿真端 Transform。
  check(gimbal.send(true, false, std::nan(""), 0.0, 3.0), "NaN 角度命令已发出");
  sim_io::GimbalCmd nan_cmd{};
  check(pub.recv_gimbal_cmd(&nan_cmd), "收到 NaN 场景命令");
  check(nan_cmd.distance_m == -1.0f, "NaN 角度被降级为安全停止（distance=-1）");
  check(nan_cmd.fire_advice == 0, "NaN 场景不开火");

  // 安全停止与定期重发。
  check(gimbal.send_safe_stop(), "发送安全停止");
  sim_io::GimbalCmd stop_cmd{};
  check(pub.recv_gimbal_cmd(&stop_cmd), "收到安全停止命令");
  check(
    stop_cmd.distance_m == -1.0f && stop_cmd.yaw_deg == 0.0f && stop_cmd.pitch_deg == 0.0f,
    "安全停止命令内容正确");
  const std::uint64_t stops_before = gimbal.safe_stops();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  check(gimbal.tick(), "空闲超过周期后 tick 重发安全停止");
  check(gimbal.safe_stops() > stops_before, "safe_stops 计数增加");

  // ---- 6. 开火抑制 -----------------------------------------------------------
  std::printf("--- 开火抑制 --------------------------------------------------------\n");
  {
    // allow_fire=false：无论上层怎么请求，fire_advice 恒为 0。
    const std::uint64_t suppressed_before = gimbal.suppressed_fires();
    gimbal.send(true, true, 0.1, 0.05, 3.0);
    sim_io::GimbalCmd c{};
    check(pub.recv_gimbal_cmd(&c), "收到请求开火的命令");
    check(c.fire_advice == 0, "allow_fire=false 时开火被抑制");
    check(gimbal.suppressed_fires() > suppressed_before, "suppressed_fires 计数增加");
    check(!gimbal.fire_allowed(), "fire_allowed()=false（FAULT_FIRE_DISABLED）");
  }
  {
    // 打开 allow_fire，但保留故障位：仍然不允许开火。
    sim_io::SimGimbalConfig armed = gim_cfg;
    armed.allow_fire = true;
    sim_io::SimGimbal g2(cam.client(), armed);

    // 尚未拿到姿态 → FAULT_STARTUP。
    check(!g2.fire_allowed(), "未取得同帧姿态时禁止开火");
    g2.send(true, true, 0.0, 0.0, 3.0);
    sim_io::GimbalCmd c{};
    check(pub.recv_gimbal_cmd(&c) && c.fire_advice == 0, "启动阶段开火被抑制");

    // 用当前帧姿态喂进去，清空启动故障。
    g2.update(cam.last_bundle(), std::chrono::steady_clock::now());
    for (std::uint32_t f : {sim_io::FAULT_HEARTBEAT_LOST, sim_io::FAULT_NO_NEW_FRAME,
                            sim_io::FAULT_TARGET_LOST, sim_io::FAULT_CLOCK_JUMP,
                            sim_io::FAULT_FRAME_FAULT}) {
      g2.set_fault(f, true);
      g2.send(true, true, 0.0, 0.0, 3.0);
      sim_io::GimbalCmd cf{};
      const bool got = pub.recv_gimbal_cmd(&cf);
      check(
        got && cf.fire_advice == 0,
        std::string("故障 ") + sim_io::describe_faults(f) + " 下开火被抑制");
      g2.set_fault(f, false);
    }

    check(g2.fire_allowed(), "清空全部故障后允许开火");
    g2.send(true, true, 0.2, 0.03, 3.5);
    sim_io::GimbalCmd cy{};
    check(pub.recv_gimbal_cmd(&cy), "收到允许开火的命令");
    check(cy.fire_advice == 1, "无故障且 allow_fire=true 时确实开火");
    check(g2.fire_commands() == 1, "fire_commands 计数正确");
  }

  std::printf("\n检查项 %d，失败 %d\n", g_checks, g_failures);
  std::printf(
    "sim_io_loopback_test: %s\n", g_failures == 0 ? "全部 通过" : "存在 失败");

  cam.close();
  pub.destroy();
  pub.unlink_files();
  ::rmdir(dir.c_str());
  return g_failures == 0 ? 0 : 1;
}
