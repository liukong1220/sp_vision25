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

// 合成 bundle 的时间戳：源端采样与本地接收取同一时刻。
// 只用于"位姿字段校验"这类与时间语义无关的用例；凡是要检验帧龄/看门狗的地方
// 必须用 SimCamera::last_stamps()，两者绝不能混。
sim_io::FrameStamps stamps_now()
{
  const auto now = std::chrono::steady_clock::now();
  return sim_io::FrameStamps{now, now};
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
  // 协议 v3：Muzzle 通道是枪口的**世界**位置，不是相对云台的局部平移。
  // 取一个与 odom_pos 明显不同、且不等于 odom_pos+局部量的值，这样一旦
  // 有人又把它当成局部量去加 odom_position()，断言立刻不成立。
  const float muzzle_world[3] = {1.36f, -2.48f, 0.345f};
  const float camera_rel[3] = {0.06f, 0.01f, 0.09f};

  const std::uint64_t t0 = realtime_now_ns();
  check(
    pub.try_publish_synchronized_frame(
      pattern.data.data(), 1, t0, quat, odom_pos, muzzle_world, camera_rel),
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
  gimbal.update(cam.last_bundle(), cam.last_stamps());
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
  // 枪口世界位置必须逐分量原样透传，且**不**等于 odom+任何局部量的组合。
  check_near(gimbal.muzzle_position().x(), muzzle_world[0], 1e-6, "muzzle world x");
  check_near(gimbal.muzzle_position().y(), muzzle_world[1], 1e-6, "muzzle world y");
  check_near(gimbal.muzzle_position().z(), muzzle_world[2], 1e-6, "muzzle world z");
  check(
    (gimbal.muzzle_position() - gimbal.odom_position()).norm() > 1e-3,
    "枪口与云台原点不重合（视差项非零）");
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
      for (int c = 0; c < sim_io::POSE_CHANNEL_COUNT; ++c) {
        fb.pose_present[c] = true;
        fb.poses[c].frame_seq = fb.frame_seq;
        fb.poses[c].timestamp_ns = fb.timestamp_ns;
        fb.poses[c].quaternion[0] = 1.0f;
      }
      auto & gp = fb.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)];
      for (int c = 0; c < 4; ++c) gp.quaternion[c] = q_roll[c];
      probe.update(fb, stamps_now());
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
  const std::uint64_t seq10_timestamp = realtime_now_ns();
  pub.publish_pose_bundle(10, seq10_timestamp, quat);
  pub.publish_image(pattern2.data.data(), 10, seq10_timestamp);
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

  // frame_seq 相同但 timestamp 不同也必须拒绝，不能把相邻帧姿态拼到图像上。
  const std::uint64_t pose_timestamp = realtime_now_ns();
  pub.publish_pose_bundle(15, pose_timestamp, quat);
  pub.publish_image(pattern2.data.data(), 15, pose_timestamp + 1'000'000ull);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "姿态与图像 timestamp 不一致被拒绝");

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

  // 容差内的未来帧允许进入；超出有限容差必须拒绝。
  const std::uint64_t future_before = cam.future_frames();
  const std::uint64_t near_future = realtime_now_ns() + 4'000'000ull;
  pub.publish_pose_bundle(22, near_future, quat);
  pub.publish_image(pattern.data.data(), 22, near_future);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Ok, "容差内未来帧仍然可用");
  check(cam.future_frames() == future_before + 1, "future_frames 计数 +1");

  const std::uint64_t far_future = realtime_now_ns() + 50'000'000ull;
  pub.publish_pose_bundle(23, far_future, quat);
  pub.publish_image(pattern.data.data(), 23, far_future);
  st = cam.try_read(img, ts);
  check(st == sim_io::ReadStatus::Rejected, "超出容差的未来帧被拒绝");

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
    g2.update(cam.last_bundle(), cam.last_stamps());
    for (std::uint32_t f : {sim_io::FAULT_HEARTBEAT_LOST, sim_io::FAULT_NO_NEW_FRAME,
                            sim_io::FAULT_TARGET_LOST, sim_io::FAULT_CLOCK_JUMP,
                            sim_io::FAULT_FRAME_FAULT, sim_io::FAULT_NOT_FOLLOWING}) {
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
    gj.update(cam.last_bundle(), cam.last_stamps());
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
    const auto now_sp = stamps_now();

    // 先喂一帧好的，建立基准（也验证正常路径返回 Ok）。
    sim_io::FrameBundle good = cam.last_bundle();
    good.timestamp_ns = realtime_now_ns();
    for (int c = 0; c <= static_cast<int>(sim_io::PoseIndex::Camera); ++c) {
      good.pose_present[c] = true;
      good.poses[c].frame_seq = good.frame_seq;
      good.poses[c].timestamp_ns = good.timestamp_ns;
    }
    check(gv.update(good, now_sp) == sim_io::PoseValidity::Ok, "合法位姿返回 Ok");
    const double yaw_good = gv.yaw();
    const std::uint64_t invalid_before = gv.invalid_poses();
    auto set_pose_timestamp = [](sim_io::FrameBundle & bundle) {
      for (int c = 0; c <= static_cast<int>(sim_io::PoseIndex::Camera); ++c) {
        bundle.pose_present[c] = true;
        bundle.poses[c].frame_seq = bundle.frame_seq;
        bundle.poses[c].timestamp_ns = bundle.timestamp_ns;
      }
    };

    // timestamp_ns == 0：未初始化槽位的典型表现。
    sim_io::FrameBundle b0 = good;
    b0.timestamp_ns = 0;
    check(
      gv.update(b0, now_sp) == sim_io::PoseValidity::BadTimestamp,
      "timestamp_ns=0 被拒 (BadTimestamp)");

    // 时间戳倒退。
    sim_io::FrameBundle bb = good;
    bb.timestamp_ns = good.timestamp_ns - 1000000ull;
    set_pose_timestamp(bb);
    check(
      gv.update(bb, now_sp) == sim_io::PoseValidity::BadTimestamp,
      "时间戳倒退被拒 (BadTimestamp)");

    // NaN 位置。
    sim_io::FrameBundle bn = good;
    bn.timestamp_ns = good.timestamp_ns + 10000000ull;
    set_pose_timestamp(bn);
    bn.poses[static_cast<int>(sim_io::PoseIndex::Odom)].position[1] = std::nanf("");
    check(
      gv.update(bn, now_sp) == sim_io::PoseValidity::NonFinite, "NaN 位置被拒 (NonFinite)");

    // inf 四元数分量。
    sim_io::FrameBundle bi = good;
    bi.timestamp_ns = good.timestamp_ns + 11000000ull;
    set_pose_timestamp(bi);
    bi.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)].quaternion[2] =
      std::numeric_limits<float>::infinity();
    check(
      gv.update(bi, now_sp) == sim_io::PoseValidity::NonFinite, "inf 四元数被拒 (NonFinite)");

    // 全零四元数（未初始化槽位）：模长 0。
    sim_io::FrameBundle bz = good;
    bz.timestamp_ns = good.timestamp_ns + 12000000ull;
    set_pose_timestamp(bz);
    for (int i = 0; i < 4; ++i)
      bz.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)].quaternion[i] = 0.0f;
    check(
      gv.update(bz, now_sp) == sim_io::PoseValidity::QuaternionNorm,
      "全零四元数被拒 (QuaternionNorm)");

    // 模长明显偏离 1。
    sim_io::FrameBundle bs = good;
    bs.timestamp_ns = good.timestamp_ns + 13000000ull;
    set_pose_timestamp(bs);
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
    set_pose_timestamp(bok);
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
      // 真值只能随图像在**同一次发布事务**里提交，消费端也只在事务窗口里取它
      // （见 SharedMemoryClient::frame_ground_truth 与 consume_frame 的说明）。
      // 所以这一节的每一批真值都必须先跟一帧图像一起发出去、再读进来；以前那种
      // "set_ground_truth 之后让评估器现读槽位"的路径已经不存在了。
      // seq 从 2 起：上面第 9 节刚消费掉 seq=1，帧号必须严格前进。
      auto publish_and_consume = [&](const sim_io::GroundTruthBatch * gt, std::uint64_t seq) {
        const std::uint64_t frame_timestamp = realtime_now_ns();
        sim_io::GroundTruthBatch stamped{};
        const sim_io::GroundTruthBatch * published_gt = nullptr;
        if (gt != nullptr) {
          stamped = *gt;
          stamped.frame_seq = seq;
          stamped.timestamp_ns = frame_timestamp;
          for (std::uint32_t i = 0; i < stamped.target_count; ++i) {
            stamped.targets[i].frame_seq = seq;
            stamped.targets[i].timestamp_ns = frame_timestamp;
          }
          for (std::uint32_t i = 0; i < stamped.rune_count; ++i) {
            stamped.runes[i].frame_seq = seq;
            stamped.runes[i].timestamp_ns = frame_timestamp;
          }
          published_gt = &stamped;
        }
        const bool sent = pub3.try_publish_synchronized_frame(
          pattern.data.data(), seq, frame_timestamp, quat, nullptr, nullptr, nullptr, published_gt);
        const auto rs = cam.try_read(img, ts);
        return sent && rs == sim_io::ReadStatus::Ok && cam.last_frame_seq() == seq;
      };

      sim_io::GroundTruthBatch b{};
      b.frame_seq = 2;
      b.timestamp_ns = nn;
      b.target_count = 2;
      b.targets[0].frame_seq = 2;
      b.targets[0].team = sim_io::GT_TEAM_RED;
      b.targets[0].armor_label = 3;
      b.targets[0].position[0] = 1.0f;
      b.targets[0].position[2] = 0.2f;
      b.targets[0].armor_position[0] = 1.2f;
      b.targets[0].armor_position[2] = 0.26f;
      b.targets[0].armor_position_valid = 1;
      b.targets[1].frame_seq = 2;
      b.targets[1].team = sim_io::GT_TEAM_BLUE;
      b.targets[1].armor_label = 3;
      b.targets[1].position[0] = 3.0f;
      b.targets[1].position[1] = 0.5f;
      b.targets[1].position[2] = 0.2f;
      b.targets[1].armor_position[0] = 2.8f;
      b.targets[1].armor_position[1] = 0.5f;
      b.targets[1].armor_position[2] = 0.26f;
      b.targets[1].armor_position_valid = 1;
      check(publish_and_consume(&b, 2), "真值与图像同事务提交，消费端在事务窗口里取到");
      check(cam.client().ground_truth_captures() == 1, "ground_truth_captures 计 1");

      const Eigen::Vector3d est(2.9, 0.45, 0.2);

      sim_io::GroundTruthEvaluator ev_blue(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_blue.fetch(2, cam.last_timestamp_ns()), "同事务提交的真值可读且同帧");
      check(ev_blue.fetch_attempts() == 1, "fetch_attempts 记录同帧尝试");
      check(ev_blue.fetch_success() == 1, "fetch_success 记录同帧成功");
      check(ev_blue.fetch_missing() == 0, "同帧真值不存在 missing");
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
      check(ev_red.fetch(2, cam.last_timestamp_ns()), "红方评估器读到同一批真值");
      const auto err_red = ev_red.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_red.valid && err_red.team == sim_io::GT_TEAM_RED, "指定红方时匹配红方");
      check_near(err_red.gt_position.x(), 1.0, 1e-5, "红方真值位置正确（未被蓝方抢走）");

      // 同队同标签两辆车：允许取最近的一个继续出数，但必须上报歧义。
      sim_io::GroundTruthBatch dup = b;
      dup.frame_seq = 3;
      dup.targets[0].frame_seq = 3;
      dup.targets[0].team = sim_io::GT_TEAM_BLUE;
      dup.targets[1].frame_seq = 3;
      check(publish_and_consume(&dup, 3), "第 3 帧同事务提交");
      sim_io::GroundTruthEvaluator ev_dup(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_dup.fetch(3, cam.last_timestamp_ns()), "重复标签批次可读");
      const auto err_dup = ev_dup.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_dup.valid, "重复标签仍给出结果");
      check(err_dup.ambiguous, "同队重复 label 上报 ambiguous");
      check(ev_dup.ambiguous_matches() == 1, "ambiguous_matches 计数 +1");
      check_near(err_dup.gt_position.x(), 3.0, 1e-5, "歧义时取距估计值最近的那辆");

      // 帧号不一致必须拒绝：不能拿别的帧的真值评估这一帧的估计。
      check(!ev_dup.fetch(999, cam.last_timestamp_ns()), "帧号不匹配的真值被拒");
      check(ev_dup.seq_mismatches() >= 1, "seq_mismatches 计数增加");
      check(ev_dup.fetch_attempts() == 2, "mismatch 也计入 fetch_attempts");
      check(ev_dup.fetch_success() == 1, "mismatch 不计入 fetch_success");
      check(
        !ev_dup.fetch(3, cam.last_timestamp_ns() + 1),
        "帧号相同但 ground truth timestamp 不匹配时拒绝");
      check(ev_dup.timestamp_mismatches() == 1, "timestamp_mismatches 计数增加");

      // 撕裂注入。这是评审第 4 条的核心：原来消费端靠"memcpy 前后 frame_seq
      // 相等"近似判断整块稳定，而同一帧号内重发时 frame_seq 根本不变、body 却在
      // 被改写，那种判据会把撕裂当成完好数据。现在区域停在"正在写"（seqlock 为
      // 奇数），读端必须拒绝，而不是返回半份。
      sim_io::GroundTruthBatch torn = b;
      torn.frame_seq = 4;
      torn.targets[0].frame_seq = 4;
      torn.targets[1].frame_seq = 4;
      torn.targets[1].position[0] = 9.0f;  // 与 b 同帧号语义不同的 body 内容
      pub3.begin_torn_ground_truth(torn);
      // 让区域停在"正在写"的状态下走一次完整的事务窗口：本帧图像不带真值参数，
      // 否则 set_ground_truth 会把撕裂状态收尾掉。
      const auto captures_before_torn = cam.client().ground_truth_captures();
      check(publish_and_consume(nullptr, 4), "撕裂期间图像帧本身仍然正常");
      sim_io::GroundTruthBatch out{};
      check(!cam.client().read_ground_truth(&out), "seqlock 为奇数时拒绝读取");
      check(
        !cam.client().frame_ground_truth(&out),
        "撕裂帧不得留下同帧真值（宁可没有，也不能是半份）");
      check(
        cam.client().ground_truth_captures() == captures_before_torn,
        "撕裂帧不计入 ground_truth_captures");
      sim_io::GroundTruthEvaluator ev_torn(cam.client(), sim_io::GT_TEAM_BLUE);
      check(
        !ev_torn.fetch(4, cam.last_timestamp_ns()),
        "撕裂期间 fetch 失败而不是返回半份数据");

      // 收尾提交后恢复可读，且读回的是完整的新内容。
      pub3.set_ground_truth(torn);
      check(cam.client().read_ground_truth(&out), "收尾提交后恢复可读");
      check(out.frame_seq == 4, "读回第 4 帧真值");
      check_near(out.targets[1].position[0], 9.0, 1e-5, "读回的是撕裂后完整提交的新内容");

      // 板心缺失时必须显式告知，评估端据此不混口径（不拿整车中心充数）。
      sim_io::GroundTruthBatch noarmor = b;
      noarmor.frame_seq = 5;
      noarmor.targets[0].frame_seq = 5;
      noarmor.targets[1].frame_seq = 5;
      noarmor.targets[1].armor_position_valid = 0;
      check(publish_and_consume(&noarmor, 5), "第 5 帧同事务提交");
      sim_io::GroundTruthEvaluator ev_na(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_na.fetch(5, cam.last_timestamp_ns()), "无板心批次可读");
      const auto err_na = ev_na.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_na.valid, "无板心时整车中心评估仍可用");
      check(!err_na.has_armor_position, "armor_position_valid=0 时不带回板心");

      // GT_TEAM_ANY 只允许诊断用：它会把自家车也纳入匹配，这里显式验证这一点，
      // 以免有人以为 any 是"更宽松但无害"。
      sim_io::GroundTruthEvaluator ev_any(cam.client(), sim_io::GT_TEAM_ANY);
      check(ev_any.fetch(5, cam.last_timestamp_ns()), "any 评估器可读");
      const auto err_any = ev_any.evaluate(auto_aim::three, est, 0.0, 0.0);
      check(err_any.valid && err_any.ambiguous, "GT_TEAM_ANY 下红蓝同号互相污染，报歧义");

      // 前哨站真值匹配。仿真端 ArmorLabel::Outpost = 6，C++ 侧 auto_aim::outpost 也是
      // 6，但两套枚举其余项顺序不同（见 armor_label_to_name 的注释），所以这条要连
      // 换算一起验：发 label=6 进去，必须能用 auto_aim::outpost 取出来。
      //
      // 场景里红蓝各有一座前哨站，label 同为 6，和三号步兵一样构成"同 label 跨队"
      // 的情形，所以队伍过滤在这里同样是必须的而不是可选的。
      // is_outpost 只是信息位，匹配不靠它，这一节顺带把它读回来确认没被丢掉。
      sim_io::GroundTruthBatch op{};
      op.frame_seq = 6;
      op.timestamp_ns = realtime_now_ns();
      op.target_count = 2;
      // 红方（我方）前哨站，离估计值更近——用来证明队伍过滤不是"恰好最近"。
      op.targets[0].frame_seq = 6;
      op.targets[0].team = sim_io::GT_TEAM_RED;
      op.targets[0].armor_label = sim_io::armor_name_to_label(auto_aim::outpost);
      op.targets[0].is_outpost = 1;
      op.targets[0].position[0] = 5.5f;
      op.targets[0].position[1] = -1.1f;
      op.targets[0].position[2] = 1.14f;
      op.targets[0].yaw = 0.30f;
      op.targets[0].vyaw = 2.5133f;  // 顺时针，+0.8π rad/s
      op.targets[0].armor_position[0] = 5.32f;
      op.targets[0].armor_position[1] = -1.1f;
      op.targets[0].armor_position[2] = 1.14f;
      op.targets[0].armor_position_valid = 1;
      // 蓝方（敌方）前哨站。
      op.targets[1].frame_seq = 6;
      op.targets[1].team = sim_io::GT_TEAM_BLUE;
      op.targets[1].armor_label = sim_io::armor_name_to_label(auto_aim::outpost);
      op.targets[1].is_outpost = 1;
      op.targets[1].position[0] = -2.21f;
      op.targets[1].position[1] = 3.06f;
      op.targets[1].position[2] = 1.14f;
      op.targets[1].yaw = -0.70f;
      op.targets[1].vyaw = -2.5133f;  // 逆时针，-0.8π rad/s
      op.targets[1].armor_position[0] = -2.03f;
      op.targets[1].armor_position[1] = 3.06f;
      op.targets[1].armor_position[2] = 1.14f;
      op.targets[1].armor_position_valid = 1;
      check(publish_and_consume(&op, 6), "第 6 帧前哨站真值同事务提交");

      check(
        sim_io::armor_name_to_label(auto_aim::outpost) == 6,
        "auto_aim::outpost 对应仿真端 ArmorLabel::Outpost = 6");
      check(
        sim_io::armor_label_to_name(6) == auto_aim::outpost,
        "label 6 换算回 auto_aim::outpost");

      // 估计值故意放在红方前哨站附近（距红 0.2 m，距蓝 ~9 m）。
      const Eigen::Vector3d op_est(5.6, -1.25, 1.20);
      sim_io::GroundTruthEvaluator ev_op(cam.client(), sim_io::GT_TEAM_BLUE);
      check(ev_op.fetch(6, cam.last_timestamp_ns()), "前哨站真值批次可读且同帧");
      check(ev_op.target_count() == 2, "读到红蓝两座前哨站");
      const auto err_op = ev_op.evaluate(auto_aim::outpost, op_est, -0.70, -2.5133);
      check(err_op.valid, "前哨站评估命中");
      check(err_op.name == auto_aim::outpost, "命中的是 outpost 而不是别的标签");
      check(err_op.team == sim_io::GT_TEAM_BLUE, "队伍过滤生效：没有匹配到更近的我方前哨站");
      check(!err_op.matched_by_nearest, "按 (team,label) 命中，未退化成最近邻");
      check(!err_op.ambiguous, "同队仅一座前哨站，无歧义");
      check_near(err_op.gt_position.x(), -2.21, 1e-5, "前哨站回转中心真值 x 正确");
      check_near(err_op.gt_position.z(), 1.14, 1e-5, "前哨站回转中心真值 z 正确");
      check(err_op.has_armor_position, "前哨站板位真值随之带回");
      check_near(err_op.gt_armor_position.x(), -2.03, 1e-5, "前哨站板位真值 x 正确");
      check_near(err_op.yaw_err_rad, 0.0, 1e-5, "yaw 真值与估计一致时误差为 0");
      check_near(err_op.vyaw_err_radps, 0.0, 1e-4, "vyaw 真值与估计一致时误差为 0");
      check(ev_op.batch().targets[1].is_outpost == 1, "is_outpost 信息位原样带过来");

      // 反向：指定红方时必须拿到红方那座，且 vyaw 符号相反（顺时针 vs 逆时针）。
      // 这条同时把"vyaw 是有符号量、不是转速绝对值"钉在协议层。
      sim_io::GroundTruthEvaluator ev_op_red(cam.client(), sim_io::GT_TEAM_RED);
      check(
        ev_op_red.fetch(6, cam.last_timestamp_ns()),
        "红方评估器读到同一批前哨站真值");
      const auto err_op_red = ev_op_red.evaluate(auto_aim::outpost, op_est, 0.30, 2.5133);
      check(
        err_op_red.valid && err_op_red.team == sim_io::GT_TEAM_RED, "指定红方时匹配红方前哨站");
      check_near(err_op_red.gt_position.x(), 5.5, 1e-5, "红方前哨站真值位置正确");
      check(
        ev_op.batch().targets[0].vyaw * ev_op.batch().targets[1].vyaw < 0.0f,
        "红蓝前哨站旋向相反，vyaw 符号相反");

      // 用 label=3 去取前哨站必须取不到同一个点：否则说明 label 根本没参与过滤。
      const auto err_op_wrong = ev_op.evaluate(auto_aim::three, op_est, 0.0, 0.0);
      check(
        !err_op_wrong.valid || err_op_wrong.matched_by_nearest,
        "本帧没有 label=3 的目标，要么不命中，要么明确标注退化为最近邻");
    }

    pub3.destroy();
    pub3.unlink_files();
  }

  // ---- 11. 慢重启：旧心跳先超时，新文件后出现 ---------------------------------
  std::printf("--- 慢重启 ----------------------------------------------------------\n");
  {
    // 这一节和第 9 节的区别是**时序**：那里心跳还在有效窗口内，走的是
    // "心跳未超时 -> NoFrame -> 复查路径" 那一路；这里让心跳先彻底超时，之后才
    // 出现新文件。真实的慢重启就是这样：仿真器要几秒才起得来，而心跳超时只有
    // 几百毫秒，消费端一定会先进入心跳超时态。
    //
    // 修复前 try_read 的 NoFrame 分支是先判 `!heartbeat_alive()` 直接 return
    // Disconnected，复查路径的代码在它后面，永远走不到；连接又还"活着"
    // （connected()==true），于是所有 remap 入口都被跳过，消费端在心跳超时后
    // 永久失明，只能重启进程。
    sim_io::SimCameraConfig slow_cfg;
    slow_cfg.shm.dir = dir;
    slow_cfg.max_frame_age_ms = 200.0;
    slow_cfg.heartbeat_timeout_ms = 120.0;  // 故意小于下面的等待时间
    slow_cfg.read_timeout_ms = 50.0;
    slow_cfg.remap_check_ms = 30.0;

    sim_io::testing::FakePublisher::Options slow_opt = pub_opt;
    sim_io::testing::FakePublisher old_pub(slow_opt);
    std::string e_slow;
    check(old_pub.create(&e_slow), "慢重启：旧发布端建好");
    old_pub.update_heartbeat();

    sim_io::SimCamera slow_cam(slow_cfg);
    check(slow_cam.open(&e_slow), "慢重启：消费端连上旧发布端");

    std::uint64_t nn2 = realtime_now_ns();
    old_pub.publish_pose_bundle(1, nn2, quat);
    old_pub.publish_image(pattern.data.data(), 1, nn2);
    cv::Mat simg;
    std::chrono::steady_clock::time_point sts;
    check(slow_cam.try_read(simg, sts) == sim_io::ReadStatus::Ok, "慢重启：旧发布端首帧可用");

    // 旧发布端"崩了"：文件被 unlink，心跳停止。
    old_pub.destroy();
    old_pub.unlink_files();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    check(!slow_cam.heartbeat_alive(), "慢重启：旧心跳已超时");
    check(slow_cam.connected(), "慢重启：旧映射仍在（连接未断）");
    check(
      slow_cam.try_read(simg, sts) == sim_io::ReadStatus::Disconnected,
      "慢重启：无新文件时报 Disconnected");

    // 心跳超时之后才出现新文件——这正是修复前走不到的那条路。
    sim_io::testing::FakePublisher new_pub(slow_opt);
    check(new_pub.create(&e_slow), "慢重启：心跳超时后才创建新发布端");
    new_pub.update_heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));  // 满一个 remap_check 周期

    const std::uint64_t slow_remaps_before = slow_cam.client().remaps();
    const auto slow_st = slow_cam.try_read(simg, sts);
    check(
      slow_st == sim_io::ReadStatus::Reconnected, "慢重启：心跳超时后仍能检出新 inode 并重连",
      sim_io::to_string(slow_st));
    check(slow_cam.client().remaps() == slow_remaps_before + 1, "慢重启：remaps +1");

    nn2 = realtime_now_ns();
    new_pub.publish_pose_bundle(1, nn2, quat);
    new_pub.publish_image(pattern.data.data(), 1, nn2);
    check(slow_cam.try_read(simg, sts) == sim_io::ReadStatus::Ok, "慢重启：新纪元首帧可用");
    check(!simg.empty() && slow_cam.last_frame_seq() == 1, "慢重启：新纪元帧号 1 被接受");

    // ---- 12. meta 先出现、pool 后出现（remap 必须事务化） -------------------
    std::printf("--- 分步创建与事务化 remap ------------------------------------------\n");
    // 真实发布端的两个文件不可能同时出现，消费端一定会撞上中间态。要求是：
    //   (a) 这一次 remap 失败，但**旧映射必须原样保留**（connected() 仍为真）；
    //   (b) 失败可以反复重试，pool 补齐后能自己恢复。
    // 修复前 open() 的第一步就是 close()，一次失败就把活着的映射拆了，
    // connected() 变 false，而所有重试入口都以 connected() 为前提，于是重试
    // 永远进不去——一次中间态就是永久失明。
    new_pub.destroy();
    new_pub.unlink_files();

    sim_io::testing::FakePublisher staged(slow_opt);
    check(staged.create_meta_only(&e_slow), "分步创建：只建 meta，不建 pool");
    staged.update_heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));

    const std::uint64_t fail_before = slow_cam.client().remap_failures();
    check(slow_cam.connected(), "分步创建前：连接有效");
    const auto st_partial = slow_cam.try_read(simg, sts);
    check(
      st_partial == sim_io::ReadStatus::Disconnected, "缺 pool 时 remap 失败，报 Disconnected",
      sim_io::to_string(st_partial));
    check(
      slow_cam.client().remap_failures() > fail_before, "remap_failures 计数增加",
      std::to_string(slow_cam.client().remap_failures()));
    check(slow_cam.connected(), "remap 失败后旧映射仍在：事务化生效");

    // 反复重试都必须失败而不是崩溃或"假成功"，且每次都保住旧映射。
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      (void)slow_cam.try_read(simg, sts);
      check(slow_cam.connected(), "多次重试仍保住旧映射");
    }

    // pool 补齐 -> 同一个消费端必须自己恢复，不需要重启进程。
    check(staged.create_image_pool(&e_slow), "补齐 image pool");
    staged.update_heartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    const auto st_recover = slow_cam.try_read(simg, sts);
    check(
      st_recover == sim_io::ReadStatus::Reconnected, "pool 补齐后自行恢复为 Reconnected",
      sim_io::to_string(st_recover));

    nn2 = realtime_now_ns();
    staged.publish_pose_bundle(7, nn2, quat);
    staged.publish_image(pattern.data.data(), 7, nn2);
    check(slow_cam.try_read(simg, sts) == sim_io::ReadStatus::Ok, "恢复后可正常取帧");
    check(slow_cam.last_frame_seq() == 7, "恢复后帧号正确");

    // ---- 13. 发布端时间戳真回拨 ---------------------------------------------
    std::printf("--- 发布端时间戳回拨 ------------------------------------------------\n");
    // 这里改的是**发布端写进共享内存的 timestamp_ns 本身**，不是消费端的
    // realtime<->steady 偏移：重启后系统时间被 NTP 往回校、或换成一个时间更早的
    // 发布端，都会产生这种真回拨。
    //
    // SimGimbal::update() 用 `timestamp_ns <= prev_timestamp_ns_` 挡重复帧和乱序
    // 帧，这条水位线跨纪元没有意义：回拨之后每一帧都判 BadTimestamp，消费端被
    // 永久锁死在 FAULT_POSE_INVALID。reset_history() 就是为这条路径存在的。
    {
      sim_io::SimGimbal rb(slow_cam.client(), gim_cfg);
      const std::uint64_t t_late = realtime_now_ns();
      sim_io::FrameBundle b1{};
      b1.frame_seq = 100;
      b1.timestamp_ns = t_late;
      for (int c = 0; c < sim_io::POSE_CHANNEL_COUNT; ++c) {
        b1.pose_present[c] = true;
        b1.poses[c].frame_seq = b1.frame_seq;
        b1.poses[c].timestamp_ns = b1.timestamp_ns;
        b1.poses[c].quaternion[0] = 1.0f;
      }
      auto & g1 = b1.poses[static_cast<int>(sim_io::PoseIndex::Gimbal)];
      for (int c = 0; c < 4; ++c) g1.quaternion[c] = quat[c];
      check(
        rb.update(b1, stamps_now()) == sim_io::PoseValidity::Ok,
        "回拨前一帧正常");

      // 发布端时钟往回跳 5 秒后继续正常发帧。
      sim_io::FrameBundle b2 = b1;
      b2.frame_seq = 101;
      b2.timestamp_ns = t_late - 5000000000ull;
      for (int c = 0; c <= static_cast<int>(sim_io::PoseIndex::Camera); ++c) {
        b2.poses[c].frame_seq = b2.frame_seq;
        b2.poses[c].timestamp_ns = b2.timestamp_ns;
      }
      const auto v_back = rb.update(b2, stamps_now());
      check(
        v_back == sim_io::PoseValidity::BadTimestamp, "回拨帧先被判 BadTimestamp",
        sim_io::to_string(v_back));
      check((rb.faults() & sim_io::FAULT_POSE_INVALID) != 0, "置起 FAULT_POSE_INVALID");

      // 不 reset 的话，后续每一帧都还在旧水位线之下 —— 这就是"永久锁死"。
      sim_io::FrameBundle b3 = b2;
      b3.frame_seq = 102;
      b3.timestamp_ns = b2.timestamp_ns + 20000000ull;
      for (int c = 0; c <= static_cast<int>(sim_io::PoseIndex::Camera); ++c) {
        b3.poses[c].frame_seq = b3.frame_seq;
        b3.poses[c].timestamp_ns = b3.timestamp_ns;
      }
      check(
        rb.update(b3, stamps_now()) == sim_io::PoseValidity::BadTimestamp,
        "不 reset 时后续帧继续被拒（复现锁死）");

      // 消费端在 Reconnected/ClockJump 时会调 reset_history()，之后立刻恢复。
      rb.reset_history();
      const auto v_after = rb.update(b3, stamps_now());
      check(
        v_after == sim_io::PoseValidity::Ok, "reset_history() 后回拨纪元的帧被接受",
        sim_io::to_string(v_after));
      check((rb.faults() & sim_io::FAULT_POSE_INVALID) == 0, "FAULT_POSE_INVALID 被清掉");

      // 速度历史也必须一起丢：否则跨回拨的差分会算出一个含整个跳变量的假角速度。
      check(
        std::abs(rb.yaw_vel()) < 1e-9 && std::abs(rb.pitch_vel()) < 1e-9,
        "reset 后第一帧不产生跨纪元差分速度",
        std::to_string(rb.yaw_vel()) + "," + std::to_string(rb.pitch_vel()));
    }

    staged.destroy();
    staged.unlink_files();
    slow_cam.close();
  }

  // ---- 14. 能力位：发布端不写真值区时不能静默失去真值 -------------------------
  std::printf("--- 能力位 ----------------------------------------------------------\n");
  {
    // 布局里一直有 ground_truth，全零也是合法数据，所以"读出来是不是零"没法用来
    // 判断发布端到底写没写。没有能力位时，新消费端对着一个不写真值的发布端会拿到
    // 一份恒为零的真值，评估报告里所有误差项 count=0，看着像"跑通了但误差为零"。
    sim_io::testing::FakePublisher::Options nogt = pub_opt;
    nogt.capabilities = sim_io::SIMULATOR_CAPABILITIES & ~sim_io::CAP_GROUND_TRUTH;
    sim_io::testing::FakePublisher pub_nogt(nogt);
    std::string e4;
    check(pub_nogt.create(&e4), "创建不声明 CAP_GROUND_TRUTH 的发布端");
    pub_nogt.update_heartbeat();

    sim_io::SimCameraConfig c4;
    c4.shm.dir = dir;
    sim_io::SimCamera cam4(c4);
    check(cam4.open(&e4), "连接不声明真值的发布端");
    check(cam4.client().version() == sim_io::SHM_VERSION, "版本号仍然匹配（不是版本问题）");
    check(!cam4.client().has_capability(sim_io::CAP_GROUND_TRUTH), "has_capability 报告缺失");
    check(
      cam4.client().has_capability(sim_io::CAP_MUZZLE_WORLD_POSE),
      "其余能力位不受影响");

    // 即使真值区里真的有一份看起来合法的数据，也必须拒绝交出。
    sim_io::GroundTruthBatch fake{};
    fake.frame_seq = 1;
    fake.timestamp_ns = realtime_now_ns();
    fake.target_count = 1;
    fake.targets[0].frame_seq = 1;
    fake.targets[0].team = sim_io::GT_TEAM_BLUE;
    fake.targets[0].armor_label = 3;
    pub_nogt.set_ground_truth(fake);
    sim_io::GroundTruthBatch out4{};
    const std::uint64_t unsup_before = cam4.client().ground_truth_unsupported();
    check(!cam4.client().read_ground_truth(&out4), "未声明能力位时 read_ground_truth 返回 false");
    check(
      cam4.client().ground_truth_unsupported() == unsup_before + 1,
      "ground_truth_unsupported 计数 +1");

    // 声明之后同一份数据立刻可读 —— 证明拒绝的原因是能力位，不是数据本身有问题。
    pub_nogt.set_capabilities(sim_io::SIMULATOR_CAPABILITIES);
    check(cam4.client().read_ground_truth(&out4), "补上能力位后同一份数据可读");
    check(out4.frame_seq == 1 && out4.target_count == 1, "读回的正是那份数据");

    cam4.close();
    pub_nogt.destroy();
    pub_nogt.unlink_files();
  }

  // 14b. CAP_RUNTIME_STATE 缺位：read_runtime_state() 必须失败而不是返回恒零结构。
  {
    // 恒零的 RuntimeState 里 following==0，会被读成"仿真端没订阅云台命令"，
    // 于是排查方向被引到"按 F5 / 设 DAEDALUS_FORCE_AUTO_AIM"，而真实原因是发布端
    // 根本不报这个字段——照那条建议做多少次都不会有任何变化。
    sim_io::testing::FakePublisher::Options nort = pub_opt;
    nort.capabilities = sim_io::SIMULATOR_CAPABILITIES & ~sim_io::CAP_RUNTIME_STATE;
    sim_io::testing::FakePublisher pub_nort(nort);
    std::string e4b;
    check(pub_nort.create(&e4b), "创建不声明 CAP_RUNTIME_STATE 的发布端");
    pub_nort.update_heartbeat();

    // 真的把 following=1 写进去：证明拒绝的原因是能力位，不是那块内存里没数据。
    sim_io::RuntimeState rt_in{};
    rt_in.timestamp_ns = realtime_now_ns();
    rt_in.following = 1;
    pub_nort.set_runtime_state(rt_in);

    sim_io::SimCameraConfig c4b;
    c4b.shm.dir = dir;
    sim_io::SimCamera cam4b(c4b);
    check(cam4b.open(&e4b), "连接不声明 RuntimeState 的发布端");
    const std::uint64_t rt_unsup_before = cam4b.client().runtime_state_unsupported();
    sim_io::RuntimeState rt_out{};
    check(!cam4b.client().read_runtime_state(&rt_out), "缺位时 runtime state 快照读取失败");
    check(
      cam4b.client().runtime_state_unsupported() == rt_unsup_before + 1,
      "runtime_state_unsupported 计数 +1");

    pub_nort.set_capabilities(sim_io::SIMULATOR_CAPABILITIES);
    check(
      cam4b.client().read_runtime_state(&rt_out) && rt_out.following == 1,
      "补上能力位后读到 following=1");

    cam4b.close();
    pub_nort.destroy();
    pub_nort.unlink_files();
  }

  // 14c. CAP_CHASSIS_OBSERVATION 缺位：read_chassis_observation 必须返回 false。
  {
    // 全零的底盘观测是合法读数（车停着），所以"读出来是零"区分不了"确实静止"和
    // "发布端不填这个区"。缺位只能靠能力位识别。
    sim_io::testing::FakePublisher::Options noco = pub_opt;
    noco.capabilities = sim_io::SIMULATOR_CAPABILITIES & ~sim_io::CAP_CHASSIS_OBSERVATION;
    sim_io::testing::FakePublisher pub_noco(noco);
    std::string e4c;
    check(pub_noco.create(&e4c), "创建不声明 CAP_CHASSIS_OBSERVATION 的发布端");
    pub_noco.update_heartbeat();

    sim_io::ChassisObservation co_in{};
    co_in.frame_seq = 7;
    co_in.timestamp_ns = realtime_now_ns();
    co_in.v_body[0] = 1.5f;
    co_in.wz_radps = -0.25f;
    pub_noco.set_chassis_observation(co_in);

    sim_io::SimCameraConfig c4c;
    c4c.shm.dir = dir;
    sim_io::SimCamera cam4c(c4c);
    check(cam4c.open(&e4c), "连接不声明底盘观测的发布端");
    sim_io::ChassisObservation co_out{};
    const std::uint64_t co_unsup_before = cam4c.client().chassis_observation_unsupported();
    check(
      !cam4c.client().read_chassis_observation(&co_out),
      "缺位时 read_chassis_observation 返回 false");
    check(
      cam4c.client().chassis_observation_unsupported() == co_unsup_before + 1,
      "chassis_observation_unsupported 计数 +1");

    pub_noco.set_capabilities(sim_io::SIMULATOR_CAPABILITIES);
    check(cam4c.client().read_chassis_observation(&co_out), "补上能力位后同一份数据可读");
    check(
      co_out.frame_seq == 7 && std::abs(co_out.v_body[0] - 1.5f) < 1e-6f,
      "读回的正是那份数据");

    cam4c.close();
    pub_noco.destroy();
    pub_noco.unlink_files();
  }

  // 14d. CAP_MUZZLE_WORLD_POSE 缺位：必须在 open() 就拒绝，不允许降级运行。
  {
    // PoseIndex::Muzzle 在 v2 里是相对云台的局部平移、v3 起是枪口的世界位置，
    // 两者都是三个 float，形状上完全无法区分。缺这一位而放过去，消费端会把
    // (0,0,0) 当成"枪口就在世界原点"，muzzle_position() 一路进到弹道与误差分解，
    // 而不会有任何一项统计变成 0 去提示这件事。所以它只能拒绝连接。
    sim_io::testing::FakePublisher::Options nomz = pub_opt;
    nomz.capabilities = sim_io::SIMULATOR_CAPABILITIES & ~sim_io::CAP_MUZZLE_WORLD_POSE;
    sim_io::testing::FakePublisher pub_nomz(nomz);
    std::string e4d;
    check(pub_nomz.create(&e4d), "创建不声明 CAP_MUZZLE_WORLD_POSE 的发布端");
    pub_nomz.update_heartbeat();

    sim_io::SimCameraConfig c4d;
    c4d.shm.dir = dir;
    sim_io::SimCamera cam4d(c4d);
    std::string reason;
    check(!cam4d.open(&reason), "缺 CAP_MUZZLE_WORLD_POSE 时拒绝连接");
    check(
      reason.find("muzzle_world_pose") != std::string::npos,
      "拒绝理由点名缺失的能力位", reason);
    check(!cam4d.client().connected(), "拒绝之后确实没有连接");

    // 显式放宽要求时才允许连上——保留给"只看图像不看枪口"的诊断工具。
    sim_io::SimCameraConfig c4e;
    c4e.shm.dir = dir;
    c4e.shm.required_capabilities = 0u;
    sim_io::SimCamera cam4e(c4e);
    check(cam4e.open(&e4d), "显式放宽 required_capabilities 后可连接");
    cam4e.close();

    // 能力位补上之后默认要求即可通过。
    pub_nomz.set_capabilities(sim_io::SIMULATOR_CAPABILITIES);
    sim_io::SimCamera cam4f(c4d);
    check(cam4f.open(&e4d), "补上能力位后默认要求即可连接");
    cam4f.close();

    check(
      sim_io::describe_capabilities(sim_io::CAP_MUZZLE_WORLD_POSE) == "muzzle_world_pose",
      "describe_capabilities 单位名正确");
    check(sim_io::describe_capabilities(0u) == "none", "describe_capabilities(0) = none");

    pub_nomz.destroy();
    pub_nomz.unlink_files();
  }

  // ---- 15. 时间语义：源帧龄与本地保有时长必须分开 -----------------------------
  std::printf("--- 时间语义（源帧龄 vs 本地保有时长）-------------------------------\n");
  {
    // 复现被修掉的那个缺陷：源帧龄接近 max_frame_age_ms，再叠加非零的检测耗时，
    // 旧代码把两者之和拿去比 state_timeout_ms，于是 FAULT_STATE_STALE 恒亮、
    // closed_loop 里 fire 恒为 0，而给出的理由（"姿态过期"）是错的——喂数据的
    // 通道一直是活的，只是这一帧的世界观测本来就有点旧、算法又花了点时间。
    sim_io::testing::FakePublisher::Options o15 = pub_opt;
    sim_io::testing::FakePublisher pub15(o15);
    std::string e15;
    check(pub15.create(&e15), "创建时间语义测试发布端");
    pub15.update_heartbeat();

    sim_io::SimCameraConfig c15;
    c15.shm.dir = dir;
    c15.max_frame_age_ms = 200.0;   // 入口帧龄门限
    c15.read_timeout_ms = 200.0;
    sim_io::SimCamera cam15(c15);
    check(cam15.open(&e15), "连接时间语义测试发布端");

    sim_io::SimGimbalConfig g15;
    g15.allow_fire = true;
    g15.state_timeout_ms = 200.0;   // 本地保有时长看门狗
    g15.max_command_age_ms = 0.0;   // 默认不对世界观测年龄设限
    sim_io::SimGimbal gim15(cam15.client(), g15);

    // 源帧龄 180ms：贴着 200ms 的门限但仍然合法。
    const std::uint64_t src_ns = realtime_now_ns() - 180'000'000ull;
    check(
      pub15.try_publish_synchronized_frame(pattern.data.data(), 1, src_ns, quat),
      "发布一帧源帧龄 180ms 的同步帧");

    cv::Mat img15;
    std::chrono::steady_clock::time_point ts15;
    const auto st15 = cam15.try_read(img15, ts15);
    check(st15 == sim_io::ReadStatus::Ok, "贴阈值的帧仍然通过入口帧龄门限",
          sim_io::to_string(st15));
    check(cam15.has_stamps(), "两个时间点都已记录");

    const auto stamps = cam15.last_stamps();
    const double src_arr_gap_ms =
      std::chrono::duration<double, std::milli>(stamps.arrival - stamps.source).count();
    check(
      src_arr_gap_ms > 150.0 && src_arr_gap_ms < 250.0,
      "arrival 与 source 相差约等于源帧龄（两者确实不是同一个时间点）",
      std::to_string(src_arr_gap_ms) + " ms");

    check(gim15.update(cam15.last_bundle(), stamps) == sim_io::PoseValidity::Ok,
          "同帧姿态被接受");

    // 非零处理时间：模拟本机实测的检测耗时（p50 约 247ms，这里取 120ms 够用）。
    std::this_thread::sleep_for(std::chrono::milliseconds(120));

    const double state_age = gim15.state_age_ms();
    const double cmd_age = gim15.command_age_ms();
    check(
      state_age > 100.0 && state_age < g15.state_timeout_ms,
      "本地保有时长只含处理耗时，仍在看门狗阈值内",
      std::to_string(state_age) + " ms");
    check(
      cmd_age > g15.state_timeout_ms,
      "世界观测年龄 = 源帧龄 + 处理耗时，确实已超过 state_timeout_ms",
      std::to_string(cmd_age) + " ms");
    check(
      cmd_age - state_age > 150.0,
      "两个年龄的差就是源帧龄，绝不能被混成一个数",
      std::to_string(cmd_age - state_age) + " ms");

    check(
      (gim15.faults() & sim_io::FAULT_STATE_STALE) == 0,
      "源帧龄贴阈值 + 非零处理耗时不再触发 FAULT_STATE_STALE",
      sim_io::describe_faults(gim15.faults()));
    check(gim15.fire_allowed(), "此时开火不再被无理由抑制",
          sim_io::describe_faults(gim15.faults()));
    gim15.send(true, true, 0.0, 0.0, 3.0);
    sim_io::GimbalCmd cmd15{};
    check(pub15.recv_gimbal_cmd(&cmd15) && cmd15.fire_advice == 1, "开火命令真的发了出去");
    check(gim15.command_age_violations() == 0, "未设预算时不计违规");

    // 未来源时间戳不能以负数年龄绕过开火年龄门。
    sim_io::SimGimbalConfig g15_future = g15;
    g15_future.max_command_age_ms = 1000.0;
    sim_io::SimGimbal gim15_future(cam15.client(), g15_future);
    auto future_stamps = stamps;
    future_stamps.source = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    check(
      gim15_future.update(cam15.last_bundle(), future_stamps) == sim_io::PoseValidity::Ok,
      "未来 source 时间戳的位姿仍通过结构校验");
    check(gim15_future.command_age_ms() < 0.0, "未来 source 产生负 command_age");
    check(
      (gim15_future.faults() & sim_io::FAULT_COMMAND_AGE) != 0 &&
        !gim15_future.fire_allowed(),
      "负 command_age 不能绕过开火年龄门");

    // 设了预算就必须拦住，而且理由必须是 command_age 而不是 state_stale。
    sim_io::SimGimbalConfig g15b = g15;
    g15b.max_command_age_ms = 200.0;
    sim_io::SimGimbal gim15b(cam15.client(), g15b);
    check(gim15b.update(cam15.last_bundle(), stamps) == sim_io::PoseValidity::Ok,
          "同一帧喂进设了预算的云台");
    const std::uint32_t f15b = gim15b.faults();
    check(
      (f15b & sim_io::FAULT_COMMAND_AGE) != 0,
      "世界观测年龄超预算时置 FAULT_COMMAND_AGE", sim_io::describe_faults(f15b));
    check(
      (f15b & sim_io::FAULT_STATE_STALE) == 0,
      "超预算不得再冒用 state_stale 这个名义", sim_io::describe_faults(f15b));
    check(!gim15b.fire_allowed(), "超预算时禁止开火");
    gim15b.send(true, false, 0.0, 0.0, 3.0);
    check(gim15b.command_age_violations() == 1, "控制帧上的超预算被计数一次");
    gim15b.send_safe_stop();
    check(
      gim15b.command_age_violations() == 1,
      "安全停止帧不计入违规（它与世界观测无关）");

    // 看门狗本身必须还管用：不再喂新帧，超过 state_timeout_ms 就要过期。
    std::this_thread::sleep_for(
      std::chrono::milliseconds(static_cast<int>(g15.state_timeout_ms) + 40));
    check(
      (gim15.faults() & sim_io::FAULT_STATE_STALE) != 0,
      "长时间没有新帧时看门狗照常触发",
      sim_io::describe_faults(gim15.faults()));
    check(!gim15.fire_allowed(), "状态真过期时禁止开火");

    cam15.close();
    pub15.destroy();
    pub15.unlink_files();
  }

  // ---- 16. 全程故障历史 -------------------------------------------------------
  std::printf("--- 全程故障历史 ----------------------------------------------------\n");
  {
    // 为什么要有这一节：报告里的 `final_faults` 是**退出瞬间**的快照。实测出现过
    // final_faults="none" 与 suppressed_fire=3 并存（/tmp/closed_loop_v6.json）——
    // 开火确实被抑制过 3 次，说明运行中点亮过故障位，但退出时刚好都清了，报告里
    // 看不出是哪一位、点了多久。所以判据必须落在累计历史上。
    //
    // 时间用 sample_faults_at() 注入，不用真实时钟：真实时钟下 total_s/max_s 只能
    // 做区间断言，注入时间可以精确断言，也不会让测试随机变慢。
    // 注意 set_fault()/clear_faults()/send() 内部会用真实时钟再采一次样，
    // sample_faults_at 的单调钳位保证那一次的 dt 记为 0，不会污染注入的时间轴。
    sim_io::SimGimbalConfig g16;
    g16.allow_fire = false;  // 默认禁止开火：构造时就该点亮 fire_disabled
    sim_io::SimGimbal gim16(cam.client(), g16);

    auto hist = [&](std::uint32_t bit) {
      for (const auto & h : gim16.fault_history())
        if (h.bit == bit) return h;
      return sim_io::FaultHistory{};
    };

    check(gim16.fault_history().size() == sim_io::fault_bits().size(), "历史表覆盖全部故障位");
    check(
      (gim16.faults_seen() & sim_io::FAULT_STARTUP) != 0,
      "构造后 startup 已进入 faults_seen", sim_io::describe_faults(gim16.faults_seen()));
    check(
      (gim16.faults_seen() & sim_io::FAULT_FIRE_DISABLED) != 0,
      "allow_fire=false 时 fire_disabled 也进入 faults_seen");
    // startup / fire_disabled / state_stale 这三位是 faults() **算出来的**，不经过
    // set_fault，所以历史只能靠采样拿到。这条断言就是在钉这一点。
    check_near(hist(sim_io::FAULT_STARTUP).first_seen_s, 0.0, 1e-9, "startup 首次点亮记为 t=0");
    check(hist(sim_io::FAULT_STARTUP).episodes == 1, "startup 只算一次点亮");
    check(hist(sim_io::FAULT_STARTUP).active, "构造后 startup 处于点亮态");
    check(
      hist(sim_io::FAULT_TARGET_LOST).episodes == 0 &&
        hist(sim_io::FAULT_TARGET_LOST).first_seen_s < 0.0,
      "没点亮过的位保持 episodes=0 / first_seen_s<0（可与'点亮过但已清除'区分）");

    // t=0 -> 1.0：startup 一直亮着，累计时长应当就是 1.0。
    gim16.sample_faults_at(1.0);
    check_near(hist(sim_io::FAULT_STARTUP).total_s, 1.0, 1e-9, "startup 累计时长 1.0 s");
    check_near(hist(sim_io::FAULT_STARTUP).max_s, 1.0, 1e-9, "startup 单次最长 1.0 s");

    // 一段完整的"点亮 -> 清除"：t=1.0 亮，t=1.5 灭。
    gim16.set_fault(sim_io::FAULT_HEARTBEAT_LOST, true);
    gim16.sample_faults_at(1.0);
    check(hist(sim_io::FAULT_HEARTBEAT_LOST).episodes == 1, "heartbeat_lost 第 1 次点亮");
    check_near(
      hist(sim_io::FAULT_HEARTBEAT_LOST).first_seen_s, 1.0, 1e-9, "heartbeat_lost 首次点亮 t=1.0");
    gim16.sample_faults_at(1.5);
    gim16.set_fault(sim_io::FAULT_HEARTBEAT_LOST, false);
    gim16.sample_faults_at(1.5);
    check(!hist(sim_io::FAULT_HEARTBEAT_LOST).active, "清除后不再处于点亮态");
    check_near(
      hist(sim_io::FAULT_HEARTBEAT_LOST).last_cleared_s, 1.5, 1e-9, "记录清除时刻 t=1.5");
    check_near(hist(sim_io::FAULT_HEARTBEAT_LOST).total_s, 0.5, 1e-9, "第一段持续 0.5 s");
    check_near(hist(sim_io::FAULT_HEARTBEAT_LOST).max_s, 0.5, 1e-9, "单次最长 0.5 s");

    // 第二段更短：max_s 必须保留更长的那一段，total_s 累加。
    gim16.sample_faults_at(2.0);
    gim16.set_fault(sim_io::FAULT_HEARTBEAT_LOST, true);
    gim16.sample_faults_at(2.0);
    gim16.sample_faults_at(2.2);
    gim16.set_fault(sim_io::FAULT_HEARTBEAT_LOST, false);
    gim16.sample_faults_at(2.2);
    check(hist(sim_io::FAULT_HEARTBEAT_LOST).episodes == 2, "第 2 次点亮计数为 2");
    check_near(hist(sim_io::FAULT_HEARTBEAT_LOST).total_s, 0.7, 1e-9, "累计 0.5+0.2 s");
    check_near(hist(sim_io::FAULT_HEARTBEAT_LOST).max_s, 0.5, 1e-9, "max_s 保留更长的那一段");
    check_near(
      hist(sim_io::FAULT_HEARTBEAT_LOST).last_seen_s, 2.0, 1e-9, "last_seen_s 是最近一次点亮时刻");

    // 关键回归：在两次采样之间点亮又清除的故障位，仍然必须留在 faults_seen 里。
    // 这正是 final_faults="none" + suppressed_fire=3 那种组合的成因。
    const std::uint32_t seen_before = gim16.faults_seen();
    check(
      (seen_before & sim_io::FAULT_TARGET_LOST) == 0, "target_lost 此前从未点亮");
    gim16.set_fault(sim_io::FAULT_TARGET_LOST, true);
    gim16.set_fault(sim_io::FAULT_TARGET_LOST, false);
    check(
      (gim16.faults() & sim_io::FAULT_TARGET_LOST) == 0, "当前值里 target_lost 已经清了");
    check(
      (gim16.faults_seen() & sim_io::FAULT_TARGET_LOST) != 0,
      "点亮又立刻清除的位仍留在 faults_seen（final_faults 看不见它）",
      sim_io::describe_faults(gim16.faults_seen()));
    check(hist(sim_io::FAULT_TARGET_LOST).episodes == 1, "瞬时故障也计一次 episode");

    // clear_faults 不得抹掉历史：它只清当前值。
    gim16.clear_faults(0xFFFFFFFFu);
    check(
      (gim16.faults_seen() & sim_io::FAULT_HEARTBEAT_LOST) != 0,
      "clear_faults 不清历史（否则严格判据可以被一次 clear 洗白）");

    // 时间必须单调：倒退的时间戳只能被钳位，不能让 total_s 变小。
    const double total_before = hist(sim_io::FAULT_STARTUP).total_s;
    gim16.sample_faults_at(0.1);  // 故意回退
    check(
      hist(sim_io::FAULT_STARTUP).total_s >= total_before,
      "注入时间回退时 total_s 不减少（单调钳位生效）");

    // send() 里也要采样：开火被抑制的那一瞬间点亮了哪些位，必须进历史。
    check(!gim16.fire_allowed(), "allow_fire=false 时不允许开火");
    gim16.send(true, true, 0.0, 0.0, 3.0);
    check(gim16.suppressed_fires() >= 1, "被抑制的开火被计数");
    check(
      (gim16.faults_seen() & sim_io::FAULT_FIRE_DISABLED) != 0,
      "抑制开火时生效的 fire_disabled 在历史里");
    check(gim16.uptime_s() >= 0.0, "uptime_s 可读且非负");
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
