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
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "simulation/io/sim_camera.hpp"
#include "simulation/io/sim_gimbal.hpp"
#include "simulation/io/sim_ground_truth.hpp"
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
// capture.rs 里的 Rx(90°) 不是额外滚转，而正是把 Bevy 的枪管前向（局部 +Y）
// 对齐到 Bevy 前向（-Z）的那一步；经 to_ros_quat 之后，发布出来的四元数的
// ROS +X 就是出膛方向。已解析验证：对任意 R_muzzle_bevy，
//     A·(R·Rx(90°))·A⁻¹ · x_ros  ==  A·(R·y_bevy)
// （A = M_ALIGN_MAT3，右端是 projectile.rs 真正用于 spawn 弹丸的方向），
// 残差 <=5.6e-17。所以生产者发布的就是标准 world<-gimbal，这里不加任何额外
// 旋转；对应地 SimGimbal 的 feedback_pitch_fix_deg 必须是 0。
//
// 这个函数曾经右乘 Ry(-90°) 来模拟"发布端多转了 90°"，那是个现实中不存在的
// 生产者：它让测试与 feedback_pitch_fix_deg=90 互相自证，而真实链路两者都错。
const Eigen::Quaterniond & simulator_feedback_roll()
{
  static const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
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
// 仿真端 process_subscription: gimbal_data.pitch = (-cmd.pitch_deg - 90).to_radians()
// 这是 Bevy YXZ 内部量，不是 ROS pitch；两者关系由下面的 sim_formula 检查约束。
double simulator_decode_pitch_rad(float pitch_deg)
{
  // 先转 double 再做减法。写成 `-pitch_deg - 90.0f` 会在 float 里算，
  // 与 sim_internal_pitch_rad 的 double 运算差约 3e-8，逐位比较（==）必然失败，
  // 那道检查就只是在测浮点舍入，而不是在测两侧公式是否一致。
  return (-static_cast<double>(pitch_deg) - 90.0) * M_PI / 180.0;
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
    // pitch：仿真端把解码值塞进 Quat::from_euler(YXZ, yaw, pitch_bevy, 0)，绕
    // Bevy +X 正转是抬头，且它的零位是"垂直朝天"（因为解码里减了 90°）。我们
    // 下发/反馈用 ROS ZYX 约定，绕 +Y 正转是低头。两者的关系是
    //     pitch_bevy = -cmd_pitch_deg - 90    (deg)
    // 而 cmd_pitch_deg 就等于我们要的 ROS pitch（identity 映射），于是
    //     pitch_bevy = -ros_pitch - 90
    // 等价地 ros_pitch = -(pitch_bevy + 90)。检查这个恒等式，而不是检查
    // "内部量 == 下发值的相反数"（那个式子少了 90° 的零位，只在 0 附近碰巧接近）。
    const double pitch_bevy = simulator_decode_pitch_rad(cmd.pitch_deg);
    if (std::abs(-(pitch_bevy + M_PI / 2.0) - probe_pitch[i]) > 1e-4) sim_formula_ok = false;
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

  // ---- 7. 时钟跳变注入 -------------------------------------------------------
  std::printf("--- 时钟跳变 --------------------------------------------------------\n");
  {
    // 注入 500ms 的 realtime<->steady 偏移变化（等价于一次 NTP 跳表）。
    // 走的是与生产完全相同的判定路径：resample_if_due() 比较新采样与已记录
    // offset 之差，超过阈值即判跳变。
    const int jumps_before = cam.clock().jump_count();
    const std::uint64_t jump_frames_before = cam.clock_jump_frames();
    cam.clock_for_test().debug_shift_offset_ns(500000000);

    std::uint64_t nn = realtime_now_ns();
    pub.publish_pose_bundle(30, nn, quat);
    pub.publish_image(pattern.data.data(), 30, nn);
    st = cam.try_read(img, ts);
    check(st == sim_io::ReadStatus::ClockJump, "偏移跳变后首次读取报 ClockJump");
    check(cam.clock().jump_count() > jumps_before, "ClockBridge 记录到跳变");
    check(cam.clock_jump_frames() == jump_frames_before + 1, "clock_jump_frames 计数 +1");
    check(img.empty(), "跳变帧不返回图像");

    // 跳变必须进 fault 位，并因此禁止开火。这是评审指出的核心问题：
    // 原来入口里 set_fault(FAULT_CLOCK_JUMP, false) 是硬编码的，跳变只被计数。
    sim_io::SimGimbalConfig armed = gim_cfg;
    armed.allow_fire = true;
    sim_io::SimGimbal gj(cam.client(), armed);
    gj.update(cam.last_bundle(), std::chrono::steady_clock::now());
    gj.set_fault(sim_io::FAULT_CLOCK_JUMP, st == sim_io::ReadStatus::ClockJump);
    check(
      (gj.faults() & sim_io::FAULT_CLOCK_JUMP) != 0, "ClockJump 映射进 FAULT_CLOCK_JUMP");
    check(!gj.fire_allowed(), "时钟跳变帧禁止开火");
    gj.send(true, true, 0.1, 0.02, 3.0);
    sim_io::GimbalCmd cj{};
    check(pub.recv_gimbal_cmd(&cj) && cj.fire_advice == 0, "跳变帧的开火请求被抑制");

    // 跳变之后恢复：下一帧应当正常可用（跳变只丢一帧）。
    nn = realtime_now_ns();
    pub.publish_pose_bundle(31, nn, quat);
    pub.publish_image(pattern.data.data(), 31, nn);
    st = cam.try_read(img, ts);
    check(st == sim_io::ReadStatus::Ok, "跳变之后下一帧恢复可用");
    check(cam.clock_jump_frames() == jump_frames_before + 1, "恢复帧不再计入跳变");
  }

  // ---- 8. 位姿输入校验 -------------------------------------------------------
  std::printf("--- 位姿输入校验 ----------------------------------------------------\n");
  {
    sim_io::SimGimbal gv(cam.client(), gim_cfg);
    const auto now_sp = std::chrono::steady_clock::now();

    // 先喂一帧好的，建立基准（也验证正常路径返回 Ok）。
    sim_io::FrameBundle good = cam.last_bundle();
    good.timestamp_ns = realtime_now_ns();
    check(gv.update(good, now_sp) == sim_io::PoseValidity::Ok, "合法位姿返回 Ok");
    const double yaw_good = gv.yaw();
    const std::uint64_t invalid_before = gv.invalid_poses();

    // timestamp_ns == 0：未初始化槽位的典型表现。
    sim_io::FrameBundle b0 = good;
    b0.timestamp_ns = 0;
    check(
      gv.update(b0, now_sp) == sim_io::PoseValidity::BadTimestamp,
      "timestamp_ns=0 被拒 (BadTimestamp)");

    // 时间戳倒退。
    sim_io::FrameBundle bb = good;
    bb.timestamp_ns = good.timestamp_ns - 1000000ull;
    check(
      gv.update(bb, now_sp) == sim_io::PoseValidity::BadTimestamp,
      "时间戳倒退被拒 (BadTimestamp)");

    // NaN 位置。
    sim_io::FrameBundle bn = good;
    bn.timestamp_ns = good.timestamp_ns + 10000000ull;
    bn.poses[static_cast<int>(sim_io::PoseIndex::Odom)].position[1] = std::nanf("");
    check(
      gv.update(bn, now_sp) == sim_io::PoseValidity::NonFinite, "NaN 位置被拒 (NonFinite)");

    // inf 四元数分量。
    sim_io::FrameBundle bi = good;
    bi.timestamp_ns = good.timestamp_ns + 11000000ull;
    bi.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)].quaternion[2] =
      std::numeric_limits<float>::infinity();
    check(
      gv.update(bi, now_sp) == sim_io::PoseValidity::NonFinite, "inf 四元数被拒 (NonFinite)");

    // 全零四元数（未初始化槽位）：模长 0。
    sim_io::FrameBundle bz = good;
    bz.timestamp_ns = good.timestamp_ns + 12000000ull;
    for (int i = 0; i < 4; ++i)
      bz.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)].quaternion[i] = 0.0f;
    check(
      gv.update(bz, now_sp) == sim_io::PoseValidity::QuaternionNorm,
      "全零四元数被拒 (QuaternionNorm)");

    // 模长明显偏离 1。
    sim_io::FrameBundle bs = good;
    bs.timestamp_ns = good.timestamp_ns + 13000000ull;
    for (int i = 0; i < 4; ++i)
      bs.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)].quaternion[i] *= 3.0f;
    check(
      gv.update(bs, now_sp) == sim_io::PoseValidity::QuaternionNorm,
      "四元数模长异常被拒 (QuaternionNorm)");

    // 关键不变量：被拒的帧一律不得改动任何状态。
    check_near(gv.yaw(), yaw_good, 1e-12, "被拒帧未改动 yaw");
    check(gv.invalid_poses() == invalid_before + 6, "invalid_poses 累计 6 次拒绝");
    check((gv.faults() & sim_io::FAULT_POSE_INVALID) != 0, "被拒后置 FAULT_POSE_INVALID");
    check(!gv.fire_allowed(), "位姿不合法时禁止开火");

    // 恢复：合法帧应当清掉该故障位。
    sim_io::FrameBundle bok = good;
    bok.timestamp_ns = good.timestamp_ns + 20000000ull;
    check(gv.update(bok, now_sp) == sim_io::PoseValidity::Ok, "恢复帧返回 Ok");
    check((gv.faults() & sim_io::FAULT_POSE_INVALID) == 0, "合法帧清掉 FAULT_POSE_INVALID");
  }

  // ---- 9. 发布端重启（正常退出 / SIGKILL） -----------------------------------
  std::printf("--- 发布端重启 ------------------------------------------------------\n");
  {
    // 场景 A：**正常退出**。仿真端退出时会 unlink 掉 /tmp/talos_ipc_*，重启后是
    // 新 inode。消费端手里的旧映射既看不到新帧，也看不到新的 created_ns——旧页面
    // 还挂在已被删除的 inode 上，新发布端写的是另一个文件。所以这条路径只能靠
    // 复查路径身份 (dev,ino) 发现，不能靠 created_ns。
    pub.update_heartbeat();  // 心跳还在有效窗口内，走的正是"心跳未超时"那一路
    const std::uint64_t remaps_before = cam.client().remaps();
    const std::uint64_t reconnects_before = cam.reconnects();

    pub.destroy();
    pub.unlink_files();

    sim_io::testing::FakePublisher pub2(pub_opt);
    std::string e2;
    check(pub2.create(&e2), "正常退出后重建共享内存（新 inode）");
    check(cam.client().paths_changed(), "paths_changed() 检出文件身份变化");
    pub2.update_heartbeat();

    // 路径身份复查是按 remap_check_ms 周期做的（默认 200ms），不是每次
    // try_read 都查 —— stat() 在热路径上每帧跑一次是浪费。前面几个小节
    // 加起来远不到一个周期，所以这里必须等满，否则拿到的是 Timeout。
    // 等待量必须小于 heartbeat_timeout_ms(500)，否则会先走心跳超时那一路。
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    st = cam.try_read(img, ts);
    check(
      st == sim_io::ReadStatus::Reconnected, "正常退出重启后报 Reconnected",
      sim_io::to_string(st));
    check(cam.client().remaps() == remaps_before + 1, "remaps 计数 +1");
    check(cam.reconnects() == reconnects_before + 1, "reconnects 计数 +1");
    check(img.empty(), "重连的那一次不返回图像");
    check(!cam.has_bundle(), "重连后旧帧束已作废");

    // 重连必须重新挡住开火：入口把 Reconnected 映射进 FAULT_FRAME_FAULT，
    // 并置 FAULT_REARM_PENDING 直到连续确认若干帧目标。
    {
      sim_io::SimGimbalConfig armed = gim_cfg;
      armed.allow_fire = true;
      sim_io::SimGimbal gr(cam.client(), armed);
      gr.set_fault(sim_io::FAULT_FRAME_FAULT, st == sim_io::ReadStatus::Reconnected);
      gr.set_fault(sim_io::FAULT_REARM_PENDING, true);
      check(!gr.fire_allowed(), "重连后禁止开火 (frame_fault|rearm_pending)");
      check(
        sim_io::describe_faults(gr.faults()).find("rearm_pending") != std::string::npos,
        "故障描述里能看到 rearm_pending");
    }

    // 新纪元的帧号从 1 重新开始，不能再被旧水位线判成倒退。
    std::uint64_t nn = realtime_now_ns();
    pub2.publish_pose_bundle(1, nn, quat);
    pub2.publish_image(pattern.data.data(), 1, nn);
    st = cam.try_read(img, ts);
    check(st == sim_io::ReadStatus::Ok, "重连后新纪元第一帧可用");
    check(!img.empty() && cam.last_frame_seq() == 1, "新纪元帧号 1 被接受而非判倒退");

    // 场景 B：**SIGKILL 后重启**。文件没被 unlink（inode 不变），但新发布端
    // 会写入新的 created_ns。这条路径由 consume_frame 的 created_ns 比对捕获，
    // 与 paths_changed() 无关——两条路径都必须存在。
    const std::uint64_t restarts_before = cam.client().publisher_restarts();
    const std::uint64_t reconnects_b = cam.reconnects();
    const std::uint64_t remaps_b = cam.client().remaps();

    pub2.destroy();  // 相当于进程被 KILL：文件残留
    sim_io::testing::FakePublisher pub3(pub_opt);
    std::string e3;
    check(pub3.create(&e3), "SIGKILL 后原地重建共享内存（inode 不变）");
    check(!cam.client().paths_changed(), "inode 未变，paths_changed()=false");
    pub3.update_heartbeat();

    nn = realtime_now_ns();
    pub3.publish_pose_bundle(1, nn, quat);
    pub3.publish_image(pattern.data.data(), 1, nn);
    st = cam.try_read(img, ts);
    check(
      st == sim_io::ReadStatus::Reconnected, "created_ns 变化报 Reconnected",
      sim_io::to_string(st));
    check(cam.client().publisher_restarts() == restarts_before + 1, "publisher_restarts +1");
    check(cam.reconnects() == reconnects_b + 1, "reconnects 再 +1");
    check(cam.client().remaps() == remaps_b, "SIGKILL 路径不需要重映射，remaps 不变");
    check(!cam.has_bundle(), "换代后旧帧束已作废");

    // 被换代丢弃的那一帧仍留在三缓冲里，下一次读取应当正常拿到它。
    st = cam.try_read(img, ts);
    check(st == sim_io::ReadStatus::Ok, "换代后紧接着的一帧可用");
    check(cam.last_frame_seq() == 1, "换代后帧号从 1 重新计数且被接受");

    // ---- 10. 真值 seqlock 与同标签匹配 --------------------------------------
    std::printf("--- 真值 seqlock 与匹配 ---------------------------------------------\n");
    {
      // 红蓝三号步兵共用 armor label=3（场景里 INFANTRY_THREE_CONFIG 被两队复用）。
      // targets[0] 故意放红方，这样"只按 label 取第一个命中"必然配错车。
      sim_io::GroundTruthBatch b{};
      b.frame_seq = 1;
      b.timestamp_ns = nn;
      b.target_count = 2;
      b.targets[0].frame_seq = 1;
      b.targets[0].team = sim_io::GT_TEAM_RED;
      b.targets[0].armor_label = 3;
      b.targets[0].position[0] = 1.0f;
      b.targets[0].position[2] = 0.2f;
      b.targets[0].armor_position[0] = 1.2f;
      b.targets[0].armor_position[2] = 0.26f;
      b.targets[0].armor_position_valid = 1;
      b.targets[1].frame_seq = 1;
      b.targets[1].team = sim_io::GT_TEAM_BLUE;
      b.targets[1].armor_label = 3;
      b.targets[1].position[0] = 3.0f;
      b.targets[1].position[1] = 0.5f;
      b.targets[1].position[2] = 0.2f;
      b.targets[1].armor_position[0] = 2.8f;
      b.targets[1].armor_position[1] = 0.5f;
      b.targets[1].armor_position[2] = 0.26f;
      b.targets[1].armor_position_valid = 1;
      pub3.set_ground_truth(b);

      const Eigen::Vector3d est(2.9, 0.45, 0.2);

      sim_io::GroundTruthEvaluator ev_blue(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_blue.fetch(1), "seqlock 正常提交后真值可读");
      check(ev_blue.target_count() == 2, "读到 2 个目标");
      const auto err_blue = ev_blue.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_blue.valid, "蓝方评估命中");
      check(err_blue.team == sim_io::GT_TEAM_BLUE, "队伍过滤生效：匹配蓝方而非红方");
      check(!err_blue.matched_by_nearest, "按 (team,label) 命中，未退化成最近邻");
      check(!err_blue.ambiguous, "同队仅一个 label=3，无歧义");
      check(err_blue.has_armor_position, "板心真值随之带回");
      check_near(err_blue.gt_armor_position.x(), 2.8, 1e-5, "板心真值 x 正确");
      check_near(err_blue.gt_position.x(), 3.0, 1e-5, "整车中心真值 x 正确");

      // 反向验证：指定红方时必须匹配红方，即使估计值离蓝方近得多。
      // 这条能区分"真的按队伍过滤"和"恰好选了最近的一个"。
      sim_io::GroundTruthEvaluator ev_red(cam.client(), sim_io::GT_TEAM_RED);
      check(ev_red.fetch(1), "红方评估器读到同一批真值");
      const auto err_red = ev_red.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_red.valid && err_red.team == sim_io::GT_TEAM_RED, "指定红方时匹配红方");
      check_near(err_red.gt_position.x(), 1.0, 1e-5, "红方真值位置正确（未被蓝方抢走）");

      // 同队同标签两辆车：允许取最近的一个继续出数，但必须上报歧义。
      sim_io::GroundTruthBatch dup = b;
      dup.frame_seq = 2;
      dup.targets[0].frame_seq = 2;
      dup.targets[0].team = sim_io::GT_TEAM_BLUE;
      dup.targets[1].frame_seq = 2;
      pub3.set_ground_truth(dup);
      sim_io::GroundTruthEvaluator ev_dup(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_dup.fetch(2), "重复标签批次可读");
      const auto err_dup = ev_dup.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_dup.valid, "重复标签仍给出结果");
      check(err_dup.ambiguous, "同队重复 label 上报 ambiguous");
      check(ev_dup.ambiguous_matches() == 1, "ambiguous_matches 计数 +1");
      check_near(err_dup.gt_position.x(), 3.0, 1e-5, "歧义时取距估计值最近的那辆");

      // 帧号不一致必须拒绝：不能拿别的帧的真值评估这一帧的估计。
      check(!ev_dup.fetch(999), "帧号不匹配的真值被拒");
      check(ev_dup.seq_mismatches() >= 1, "seq_mismatches 计数增加");

      // 撕裂注入。这是评审第 4 条的核心：原来消费端靠"memcpy 前后 frame_seq
      // 相等"近似判断整块稳定，而同一帧号内重发时 frame_seq 根本不变、body 却在
      // 被改写，那种判据会把撕裂当成完好数据。现在区域停在"正在写"（seqlock 为
      // 奇数），读端必须拒绝，而不是返回半份。
      sim_io::GroundTruthBatch torn = b;
      torn.frame_seq = 3;
      torn.targets[0].frame_seq = 3;
      torn.targets[1].frame_seq = 3;
      torn.targets[1].position[0] = 9.0f;  // 与 b 同帧号语义不同的 body 内容
      pub3.begin_torn_ground_truth(torn);
      sim_io::GroundTruthBatch out{};
      check(!cam.client().read_ground_truth(&out), "seqlock 为奇数时拒绝读取");
      sim_io::GroundTruthEvaluator ev_torn(cam.client(), sim_io::GT_TEAM_BLUE);
      check(!ev_torn.fetch(3), "撕裂期间 fetch 失败而不是返回半份数据");

      // 收尾提交后恢复可读，且读回的是完整的新内容。
      pub3.set_ground_truth(torn);
      check(cam.client().read_ground_truth(&out), "收尾提交后恢复可读");
      check(out.frame_seq == 3, "读回第 3 帧真值");
      check_near(out.targets[1].position[0], 9.0, 1e-5, "读回的是撕裂后完整提交的新内容");

      // 板心缺失时必须显式告知，评估端据此不混口径（不拿整车中心充数）。
      sim_io::GroundTruthBatch noarmor = b;
      noarmor.frame_seq = 4;
      noarmor.targets[0].frame_seq = 4;
      noarmor.targets[1].frame_seq = 4;
      noarmor.targets[1].armor_position_valid = 0;
      pub3.set_ground_truth(noarmor);
      sim_io::GroundTruthEvaluator ev_na(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_na.fetch(4), "无板心批次可读");
      const auto err_na = ev_na.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_na.valid, "无板心时整车中心评估仍可用");
      check(!err_na.has_armor_position, "armor_position_valid=0 时不带回板心");

      // GT_TEAM_ANY 只允许诊断用：它会把自家车也纳入匹配，这里显式验证这一点，
      // 以免有人以为 any 是"更宽松但无害"。
      sim_io::GroundTruthEvaluator ev_any(cam.client(), sim_io::GT_TEAM_ANY);
      check(ev_any.fetch(4), "any 评估器可读");
      const auto err_any = ev_any.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_any.valid && err_any.ambiguous, "GT_TEAM_ANY 下红蓝同号互相污染，报歧义");
    }

    pub3.destroy();
    pub3.unlink_files();
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
