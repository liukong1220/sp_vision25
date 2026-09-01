// 单轴 yaw = 90° 的几何单测（阻塞项 4/5）。
//
// 这是驻留姿态 --park-yaw-deg=90 的场景，也是"把 reparented_to(gimbal) 的局部平移
// 直接加到世界坐标"这个错法误差最大的姿态：局部 +X 在世界里指向 +Y，两个方向正交，
// 错法给出的枪口位置与真实枪口位置相差 sqrt(2)*|水平偏移|，而不是一个小量。
//
// 之所以要一个纯几何单测而不是只看闭环报告：报告里的 aim_/parallax_/geom_ 都是
// 统计量，一个 2~5 度的系统性偏置在 p50/p95 里看起来完全像"跟踪误差偏大"。这里把
// 每一条公式单独钉住：
//   1. SimGimbal 在 yaw=90° 处的欧拉角分解不退化（不撞万向锁）；
//   2. Muzzle 通道被当作世界位置读取，不与 odom 相加；
//   3. 错法的误差量级是解析可算的，并被显式记录下来；
//   4. 视差角与"以枪口为原点的瞄准误差"之间的关系与 sim_auto_aim.cpp 的定义一致。
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <string>
#include <vector>

#include "simulation/io/shared_memory_client.hpp"
#include "simulation/io/sim_camera.hpp"
#include "simulation/io/sim_gimbal.hpp"
#include "simulation/io/testing/fake_publisher.hpp"
#include "tools/math_tools.hpp"

namespace
{
int g_checks = 0;
int g_failures = 0;
constexpr double RAD2DEG = 57.29577951308232;

void check(bool ok, const std::string & name, const std::string & detail = "")
{
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("%-58s %s", name.c_str(), ok ? "ok" : "失败");
  if (!detail.empty()) std::printf("  %s", detail.c_str());
  std::printf("\n");
}

void check_near(double got, double want, double tol, const std::string & name)
{
  char detail[96];
  std::snprintf(detail, sizeof(detail), "got=%.9g want=%.9g tol=%.3g", got, want, tol);
  check(std::abs(got - want) <= tol, name, detail);
}

// sim_auto_aim.cpp 里 angle_between 的同一公式。
double angle_between_deg(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0)) * RAD2DEG;
}

// 独立于 acos·dot 的第二种算法：用叉积/点积的 atan2。两者一致才说明数值上没问题
// （acos 在夹角接近 0 时精度差，小角度场景恰好全落在那一段）。
double angle_between_deg_atan2(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  return std::atan2(a.cross(b).norm(), a.dot(b)) * RAD2DEG;
}

// sim_auto_aim.cpp 里由下发角构造方向向量的同一公式（ROS：pitch 正 = 低头）。
Eigen::Vector3d cmd_dir(double yaw, double pitch)
{
  return Eigen::Vector3d(
    std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), -std::sin(pitch));
}
}  // namespace

int main()
{
  // ---- 第 1 节：纯解析部分，不需要共享内存 ----
  // 场景：云台回转中心在世界 (1.0, -2.0, 0.30)，yaw = +90°，pitch = 0。
  // 枪口相对云台的局部平移取 (0.11, 0.0, 0.02)：沿枪管前向 0.11m、抬高 0.02m。
  const Eigen::Vector3d pivot_world(1.0, -2.0, 0.30);
  const Eigen::Vector3d muzzle_local(0.11, 0.0, 0.02);
  const double yaw = M_PI / 2.0;
  const double pitch = 0.0;

  const Eigen::Quaterniond q_world_gimbal(
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));

  // 正确的世界枪口位置：局部平移先被云台姿态旋转，再加到世界原点上。
  const Eigen::Vector3d muzzle_world = pivot_world + q_world_gimbal * muzzle_local;
  // 错法（修复前 sim_auto_aim.cpp:771 的写法）：odom_position() + 未旋转的局部平移。
  const Eigen::Vector3d muzzle_world_wrong = pivot_world + muzzle_local;

  std::printf("== 第 1 节：yaw=90° 时局部平移必须先旋转 ==\n");
  // yaw=90° 时 Rz(90°) 把局部 +X 映射到世界 +Y、局部 +Y 映射到世界 -X，z 不变。
  check_near(muzzle_world.x(), pivot_world.x(), 1e-12, "正确枪口 x = 云台 x（局部+X 转到世界+Y）");
  check_near(muzzle_world.y(), pivot_world.y() + 0.11, 1e-12, "正确枪口 y = 云台 y + 0.11");
  check_near(muzzle_world.z(), pivot_world.z() + 0.02, 1e-12, "正确枪口 z = 云台 z + 0.02");

  // 错法的位置误差是解析可算的：两个水平分量各偏 0.11m，且方向正交。
  const Eigen::Vector3d pos_err = muzzle_world_wrong - muzzle_world;
  check_near(pos_err.x(), 0.11, 1e-12, "错法 x 偏 +0.11m");
  check_near(pos_err.y(), -0.11, 1e-12, "错法 y 偏 -0.11m");
  check_near(pos_err.z(), 0.0, 1e-12, "错法 z 不偏（旋转轴是 z）");
  check_near(pos_err.norm(), 0.11 * std::sqrt(2.0), 1e-12, "错法位置误差 = sqrt(2)*0.11m");
  // 关键量级判据：错法的位置误差不是"比偏移量小的高阶量"，而是比偏移量本身更大。
  check(
    pos_err.norm() > muzzle_local.norm(),
    "错法误差大于偏移量全长（不是可忽略的高阶项）");

  std::printf("\n== 第 2 节：错法造成的角度误差量级 ==\n");
  // 目标放在世界 (1.0, 1.0, 0.35)：正好在 yaw=+90° 方向上、距云台约 3m。
  const Eigen::Vector3d target_world(1.0, 1.0, 0.35);
  const Eigen::Vector3d from_pivot = target_world - pivot_world;
  const Eigen::Vector3d from_muzzle = target_world - muzzle_world;
  const Eigen::Vector3d from_muzzle_wrong = target_world - muzzle_world_wrong;

  // 真实视差：同一目标点，从云台原点和从枪口看过去的方向夹角。这是 sim_auto_aim.cpp
  // 里 muzzle_parallax_deg 的定义。3m 距离 + 0.11m 前向偏移，视差应当很小
  // （偏移几乎沿视线方向，只有 0.02m 的抬高产生横向分量）。
  const double parallax_true = angle_between_deg(from_pivot, from_muzzle);
  // 错法算出来的"视差"：0.11m 的横向偏移在 3m 上是 2 度量级。
  const double parallax_wrong = angle_between_deg(from_pivot, from_muzzle_wrong);
  char buf[160];
  std::snprintf(buf, sizeof(buf), "真实=%.6f° 错法=%.6f°", parallax_true, parallax_wrong);
  std::printf("视差对比: %s\n", buf);

  check(parallax_true < 0.5, "真实视差 < 0.5°（偏移基本沿视线）", buf);
  check(parallax_wrong > 2.0, "错法视差 > 2°（横向偏移被当成视线方向偏移）", buf);
  check(
    parallax_wrong > 4.0 * parallax_true,
    "错法视差比真实视差大 4 倍以上：这就是旧报告里那几度的来源", buf);

  // 两种夹角算法必须一致，否则上面的小角度结论是数值噪声。
  check_near(
    angle_between_deg_atan2(from_pivot, from_muzzle), parallax_true, 1e-6,
    "视差：acos·dot 与 atan2·cross 两种算法一致");

  std::printf("\n== 第 3 节：yaw=90° 处欧拉角分解不退化 ==\n");
  // 万向锁在 pitch=±90° 而不是 yaw=±90°，但 yaw=90 + pitch=0 这一组曾经因为
  // feedback_pitch_fix_deg=90 的错误修正被转到 pitch=±90 附近，yaw 随之失效。
  // 这里直接核对 ZYX 分解能把 (90°, 0°) 原样取回。
  {
    const Eigen::Vector3d ypr = tools::eulers(q_world_gimbal, 2, 1, 0);
    check_near(ypr[0] * RAD2DEG, 90.0, 1e-9, "分解回 yaw = 90°");
    check_near(ypr[1] * RAD2DEG, 0.0, 1e-9, "分解回 pitch = 0°");
    // 下发方向公式与四元数作用在机体 x 轴上的结果必须一致。
    const Eigen::Vector3d dir_from_q = q_world_gimbal * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d dir_from_angles = cmd_dir(ypr[0], ypr[1]);
    check_near((dir_from_q - dir_from_angles).norm(), 0.0, 1e-12, "方向公式与四元数一致");
    check_near(dir_from_q.y(), 1.0, 1e-12, "yaw=90° 时出膛方向就是世界 +Y");
  }

  std::printf("\n== 第 4 节：SimGimbal 必须把 Muzzle 当世界位置读 ==\n");
  // 上面的都是解析检查。这一节走真正的共享内存链路：发布端按协议 v3 发世界枪口位置，
  // 读端必须原样取回，而不能与 odom 相加。若哪天有人把 `odom + muzzle` 写回来，
  // 这里读到的就是 pivot + muzzle_world，误差 = pivot 全长（2 米量级），必然被抓住。
  char dir_buf[128];
  std::snprintf(dir_buf, sizeof(dir_buf), "/tmp/sim_geometry_%d", static_cast<int>(::getpid()));
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
    ::rmdir(dir.c_str());
    return 2;
  }
  pub.update_heartbeat();

  sim_io::SimCameraConfig cam_cfg;
  cam_cfg.shm.dir = dir;
  cam_cfg.max_frame_age_ms = 500.0;
  cam_cfg.heartbeat_timeout_ms = 2000.0;
  cam_cfg.read_timeout_ms = 500.0;
  sim_io::SimCamera cam(cam_cfg);
  if (!cam.open(&err)) {
    std::printf("SimCamera 打开失败: %s\n", err.c_str());
    pub.unlink_files();
    ::rmdir(dir.c_str());
    return 2;
  }

  sim_io::SimGimbalConfig gim_cfg;
  gim_cfg.allow_fire = false;
  sim_io::SimGimbal gimbal(cam.client(), gim_cfg);

  const float quat[4] = {
    static_cast<float>(q_world_gimbal.w()), static_cast<float>(q_world_gimbal.x()),
    static_cast<float>(q_world_gimbal.y()), static_cast<float>(q_world_gimbal.z())};
  const float odom_p[3] = {
    static_cast<float>(pivot_world.x()), static_cast<float>(pivot_world.y()),
    static_cast<float>(pivot_world.z())};
  const float muzzle_p[3] = {
    static_cast<float>(muzzle_world.x()), static_cast<float>(muzzle_world.y()),
    static_cast<float>(muzzle_world.z())};
  const float camera_rel[3] = {0.05f, 0.0f, 0.06f};

  const auto now_ns = []() {
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count());
  };

  std::vector<std::uint8_t> pixels(sim_io::IMAGE_SIZE, 0x20);
  check(
    pub.try_publish_synchronized_frame(
      pixels.data(), 1, now_ns(), quat, odom_p, muzzle_p, camera_rel),
    "同帧发布成功");

  cv::Mat img;
  std::chrono::steady_clock::time_point ts;
  const auto rs = cam.try_read(img, ts);
  check(rs == sim_io::ReadStatus::Ok, "SimCamera 读到该帧", sim_io::to_string(rs));

  const auto validity = gimbal.update(cam.last_bundle(), cam.last_stamps());
  check(
    validity == sim_io::PoseValidity::Ok, "SimGimbal 接受该 pose 束",
    sim_io::to_string(validity));

  // 世界枪口位置必须原样取回。float 往返，容差取 1e-6。
  check_near(gimbal.muzzle_position().x(), muzzle_world.x(), 1e-6, "读回枪口世界 x");
  check_near(gimbal.muzzle_position().y(), muzzle_world.y(), 1e-6, "读回枪口世界 y");
  check_near(gimbal.muzzle_position().z(), muzzle_world.z(), 1e-6, "读回枪口世界 z");
  check_near(gimbal.odom_position().x(), pivot_world.x(), 1e-6, "读回云台世界 x");
  check_near(gimbal.odom_position().y(), pivot_world.y(), 1e-6, "读回云台世界 y");
  check_near(gimbal.odom_position().z(), pivot_world.z(), 1e-6, "读回云台世界 z");

  // 反向判据：读回的枪口位置**不能**等于 pivot + muzzle_world（那是"又加了一次
  // odom"的回归），也不能等于错法的 pivot + 未旋转局部量。
  check(
    (gimbal.muzzle_position() - (pivot_world + muzzle_world)).norm() > 1.0,
    "读回值不是 pivot 与 Muzzle 通道相加的结果");
  check(
    (gimbal.muzzle_position() - muzzle_world_wrong).norm() > 0.1,
    "读回值不是未旋转局部量加世界原点的结果");

  // 局部平移可以由世界量反解出来，且与投进去的一致：这条把"世界量"的语义钉死。
  {
    const Eigen::Vector3d recovered =
      gimbal.q().conjugate() * (gimbal.muzzle_position() - gimbal.odom_position());
    check_near(recovered.x(), muzzle_local.x(), 1e-6, "反解局部平移 x");
    check_near(recovered.y(), muzzle_local.y(), 1e-6, "反解局部平移 y");
    check_near(recovered.z(), muzzle_local.z(), 1e-6, "反解局部平移 z");
  }

  // Camera 通道仍是**局部**平移（与 t_camera2gimbal 外参比对用），不能被一起改成世界量。
  check_near(gimbal.camera_offset().x(), 0.05, 1e-6, "Camera 通道仍是局部 x");
  check_near(gimbal.camera_offset().z(), 0.06, 1e-6, "Camera 通道仍是局部 z");
  check(
    (gimbal.camera_offset() - pivot_world).norm() > 1.0,
    "Camera 通道不是世界量（与云台世界位置差得远）");

  std::printf("\n== 第 5 节：以云台原点 / 枪口为原点的瞄准误差之差就是视差 ==\n");
  // sim_auto_aim.cpp 的三个量：aim_err_deg（云台原点）、aim_err_muzzle_deg（枪口）、
  // muzzle_parallax_deg。当下发方向恰好对准"从云台原点看目标"时，前者为 0，
  // 后者必须等于视差角——这条等式是那三列可以互相解释的依据。
  {
    const Eigen::Vector3d dir_pivot = (target_world - gimbal.odom_position()).normalized();
    const double yaw_cmd = std::atan2(dir_pivot.y(), dir_pivot.x());
    const double pitch_cmd = -std::asin(std::clamp(dir_pivot.z(), -1.0, 1.0));
    const Eigen::Vector3d dir = cmd_dir(yaw_cmd, pitch_cmd);

    const double err_pivot = angle_between_deg(dir, target_world - gimbal.odom_position());
    const double err_muzzle = angle_between_deg(dir, target_world - gimbal.muzzle_position());
    const double parallax =
      angle_between_deg(target_world - gimbal.odom_position(), target_world - gimbal.muzzle_position());
    std::snprintf(
      buf, sizeof(buf), "pivot=%.9f° muzzle=%.9f° parallax=%.9f°", err_pivot, err_muzzle, parallax);
    std::printf("三列关系: %s\n", buf);
    check_near(err_pivot, 0.0, 1e-6, "对准云台原点视线时 aim_err_deg = 0");
    check_near(err_muzzle, parallax, 1e-6, "此时 aim_err_muzzle_deg 恰等于 muzzle_parallax_deg");
  }

  cam.close();
  pub.destroy();
  pub.unlink_files();
  ::rmdir(dir.c_str());

  std::printf("\n共 %d 项检查，%d 项失败\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
