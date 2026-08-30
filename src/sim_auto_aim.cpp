// 仿真专用自瞄入口。
//
// 与 standard_mpc 的区别只有两处：
//   1. 相机是 sim_io::SimCamera（共享内存），不初始化 io::Camera；
//   2. 云台是 sim_io::SimGimbal（共享内存），不初始化 io::Gimbal，也就不开串口。
// YOLO / Solver / Tracker / Planner 全部复用真实机器上的同一份实现。
//
// 三种模式：
//   passive     只取流+感知，永不发控制（用来量帧龄、检测率）
//   probe       单轴扫描，不跑感知，用来确认 yaw/pitch 的符号与零位
//   closed_loop 完整闭环
// 默认禁止开火，必须显式 --allow-fire 才可能置 fire_advice=1。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "simulation/io/sim_camera.hpp"
#include "simulation/io/sim_gimbal.hpp"
#include "simulation/io/sim_ground_truth.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace
{
const std::string keys =
  "{help h usage ? |                       | 输出命令行参数说明}"
  "{@config-path   | configs/simulation.yaml | yaml配置文件路径}"
  "{mode           | closed_loop           | passive / probe / closed_loop}"
  "{axis           | yaw                   | probe 模式扫描的轴: yaw / pitch}"
  "{amplitude-deg  | 5.0                   | probe 模式扫描幅度(度)}"
  "{period-s       | 4.0                   | probe 模式扫描周期(秒)}"
  "{duration-s     | 0.0                   | 运行时长(秒)，0 表示直到 Ctrl-C}"
  "{allow-fire     |                       | 允许虚拟开火（默认禁止）}"
  "{eval           |                       | 打开真值评估（真值只进评估器）}"
  "{report         |                       | 指标 JSON 输出路径}"
  "{dump-frame     |                       | 把第一帧 Ok 图像存到该路径（排查视野/色序）}"
  "{dump-truth     |                       | 打印首个真值批次里的目标（排查视野里到底有没有目标）}"
  "{max-frame-age-ms | -1                   | 覆盖配置里的帧龄上限，<0 表示用配置值}"
  "{bias-yaw-deg   | 0.0                   | probe 模式的 yaw 固定偏置(度)，用来把云台指向某个方向}"
  "{bias-pitch-deg | 0.0                   | probe 模式的 pitch 固定偏置(度)，正=低头(ROS 约定)}"
  "{dump-detect    | 0                     | 打印前 N 帧的检测/跟踪明细(排查为什么检到了却跟不上)}";

double percentile(std::vector<double> v, double q)
{
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double pos = q * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) return v[lo];
  return v[lo] * (1.0 - (pos - static_cast<double>(lo))) +
         v[hi] * (pos - static_cast<double>(lo));
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const auto mode = cli.get<std::string>("mode");
  const auto axis = cli.get<std::string>("axis");
  const double amplitude_deg = cli.get<double>("amplitude-deg");
  const double period_s = cli.get<double>("period-s");
  const double bias_yaw_deg = cli.get<double>("bias-yaw-deg");
  const double bias_pitch_deg = cli.get<double>("bias-pitch-deg");
  const int dump_detect = cli.get<int>("dump-detect");
  const double duration_s = cli.get<double>("duration-s");
  const bool allow_fire = cli.has("allow-fire");
  const bool do_eval = cli.has("eval");
  const auto report_path = cli.get<std::string>("report");
  const auto dump_frame_path = cli.get<std::string>("dump-frame");
  const bool dump_truth = cli.has("dump-truth");

  if (mode != "passive" && mode != "probe" && mode != "closed_loop") {
    tools::logger()->error("[sim] 未知 mode: {}", mode);
    return 2;
  }
  if (axis != "yaw" && axis != "pitch") {
    tools::logger()->error("[sim] 未知 axis: {}", axis);
    return 2;
  }

  auto yaml = tools::load(config_path);
  const auto sim = yaml["sim"];

  sim_io::SimCameraConfig cam_cfg;
  cam_cfg.max_frame_age_ms = tools::read_or<double>(sim, "max_frame_age_ms", 100.0);
  // 标定/排查用的临时放宽：仿真端掉帧率高的时候，100ms 的默认上限会把绝大多数帧
  // 判成过期，probe 拿不到足够样本。只影响本次运行，不写回配置。
  const double max_age_override = cli.get<double>("max-frame-age-ms");
  if (max_age_override >= 0.0) {
    tools::logger()->warn(
      "[sim] 帧龄上限被命令行覆盖为 {:.1f}ms（配置值 {:.1f}ms）", max_age_override,
      cam_cfg.max_frame_age_ms);
    cam_cfg.max_frame_age_ms = max_age_override;
  }
  cam_cfg.heartbeat_timeout_ms = tools::read_or<double>(sim, "heartbeat_timeout_ms", 500.0);
  cam_cfg.read_timeout_ms = tools::read_or<double>(sim, "read_timeout_ms", 1000.0);
  // 共享内存位置可覆盖：便于对着 FakePublisher 或第二个仿真实例跑，不影响默认路径。
  cam_cfg.shm.dir = tools::read_or<std::string>(sim, "shm_dir", cam_cfg.shm.dir);

  sim_io::SimGimbalConfig gim_cfg;
  gim_cfg.yaw_scale = tools::read_or<double>(sim, "yaw_scale", 1.0);
  gim_cfg.yaw_offset_deg = tools::read_or<double>(sim, "yaw_offset_deg", 0.0);
  gim_cfg.pitch_scale = tools::read_or<double>(sim, "pitch_scale", 1.0);
  gim_cfg.pitch_offset_deg = tools::read_or<double>(sim, "pitch_offset_deg", -90.0);
  // 剥掉仿真端反馈里多乘的 90° 滚转。改动这个值等于改坐标系，正常不该动。
  gim_cfg.feedback_pitch_fix_deg =
    tools::read_or<double>(sim, "feedback_pitch_fix_deg", 90.0);
  gim_cfg.state_timeout_ms = tools::read_or<double>(sim, "state_timeout_ms", 200.0);
  gim_cfg.safe_stop_period_ms = tools::read_or<double>(sim, "safe_stop_period_ms", 20.0);
  gim_cfg.bullet_speed = tools::read_or<double>(yaml, "bullet_speed_fallback", 25.0);
  // probe 模式永远不开火：这是确认符号/零位的实验，弹道无关。
  gim_cfg.allow_fire = allow_fire && mode == "closed_loop";

  const double target_lost_ms = tools::read_or<double>(sim, "target_lost_ms", 300.0);
  // 合成图像上 YOLO 的颜色分类头不可靠（见 tracker.cpp 里的说明与实测），
  // 而 Tracker 会按 enemy_color 直接清空整帧检测结果。仿真默认关掉这道颜色门，
  // 只保留编号/几何校验；实车入口不受影响（Tracker::track 的默认值仍是 true）。
  const bool use_enemy_color = tools::read_or<bool>(sim, "use_enemy_color", false);
  const double extrinsic_tol_m = tools::read_or<double>(sim, "extrinsic_tol_m", 0.01);
  const double intrinsic_tol_px = tools::read_or<double>(sim, "intrinsic_tol_px", 0.5);

  tools::Exiter exiter;

  sim_io::SimCamera camera(cam_cfg);
  std::string err;
  if (!camera.open(&err)) {
    tools::logger()->error("[sim] 共享内存连接失败: {}", err);
    tools::logger()->error("[sim] 先启动 bevy_robomaster_simulator（默认 feature 带 talos）");
    return 3;
  }
  tools::logger()->info("[sim] 已连接共享内存，mode={} allow_fire={}", mode, gim_cfg.allow_fire);

  // ---- 内参自检 ---------------------------------------------------------------
  // 仿真端把 fov 反算出来的内参写进共享内存，配置里的 camera_matrix 必须与它一致，
  // 否则 PnP 会静默地整体偏。这里宁可拒绝启动也不要跑出一份看似合理的错误结果。
  const auto camera_matrix = tools::read<std::vector<double>>(yaml, "camera_matrix");
  if (camera_matrix.size() != 9) {
    tools::logger()->error("[sim] camera_matrix 长度应为 9，实际 {}", camera_matrix.size());
    return 3;
  }
  if (const auto * info = camera.camera_info()) {
    const double d_fx = std::abs(info->fx - camera_matrix[0]);
    const double d_fy = std::abs(info->fy - camera_matrix[4]);
    const double d_cx = std::abs(info->cx - camera_matrix[2]);
    const double d_cy = std::abs(info->cy - camera_matrix[5]);
    tools::logger()->info(
      "[sim] 共享内存内参 fx={:.4f} fy={:.4f} cx={:.2f} cy={:.2f} ({}x{})", info->fx, info->fy,
      info->cx, info->cy, info->width, info->height);
    const double worst = std::max(std::max(d_fx, d_fy), std::max(d_cx, d_cy));
    if (worst > intrinsic_tol_px) {
      tools::logger()->error(
        "[sim] 内参与配置不一致，最大偏差 {:.4f}px > {:.4f}px。请更新 configs/simulation.yaml "
        "的 camera_matrix（或确认仿真端 fov 未被改动）",
        worst, intrinsic_tol_px);
      return 3;
    }
    bool distortion_zero = true;
    for (int i = 0; i < 5; ++i)
      if (info->distortion[i] != 0.0) distortion_zero = false;
    if (!distortion_zero)
      tools::logger()->warn("[sim] 仿真端上报了非零畸变，配置里的 distort_coeffs 需要同步");
  } else {
    tools::logger()->warn("[sim] 共享内存没有 CameraInfo，跳过内参自检");
  }

  auto_aim::Solver solver(config_path);
  sim_io::SimGimbal gimbal(camera.client(), gim_cfg);
  sim_io::GroundTruthEvaluator evaluator(camera.client());

  // 感知链只在需要时才构造：probe 模式不加载 OpenVINO 模型，启动快且不依赖 assets。
  std::unique_ptr<auto_aim::YOLO> yolo;
  std::unique_ptr<auto_aim::Tracker> tracker;
  std::unique_ptr<auto_aim::Planner> planner;
  if (mode != "probe") {
    yolo = std::make_unique<auto_aim::YOLO>(config_path, false);
    tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
    planner = std::make_unique<auto_aim::Planner>(config_path);
  }

  // ---- 运行期统计 -------------------------------------------------------------
  std::vector<double> detect_ms, pipeline_ms;
  std::uint64_t frames = 0, detected_frames = 0, tracked_frames = 0;
  std::uint64_t control_cmds = 0, fire_cmds = 0;
  // 闭环几何瞄准误差：下发角所指方向 与 云台->真值目标方向 的夹角。
  //
  // 之所以要这个量：仿真端物理没有 CCD（fixed_hz=120、8 substep，弹丸每子步走
  // 约 26mm，而装甲板是薄 trimesh），弹丸会直接穿过板子，命中统计恒为 0，
  // 靠"打中几发"没法评价闭环。角度误差不依赖碰撞检测，能直接回答
  // "云台到底指对了没有"。
  //
  // 注意这里的系统偏差是有物理来源的、不该被当成算法误差：
  //   * 真值是**整车中心**，算法瞄的是被选中的**装甲板**（偏心半径 r≈0.2m，
  //     实测板心 z 比车心 z 高约 0.06m），1.5m 距离上就是 2 度量级；
  //   * planner 会做弹道抬枪补偿（25m/s、1.5m 时约 0.7 度）。
  // 所以真正说明闭环质量的是**离散度**（p50 与 p95/max 的差），系统偏移量另算。
  std::vector<double> aim_err_deg, aim_yaw_err_deg, aim_pitch_err_deg;
  std::uint64_t extrinsic_warnings = 0;
  double max_extrinsic_err = 0.0;
  bool extrinsic_checked = false;
  bool truth_dumped = false;

  const auto t_t0 = std::chrono::steady_clock::now();
  auto last_target_seen = t_t0;
  bool ever_seen_target = false;

  const auto t_camera2gimbal = tools::read<std::vector<double>>(yaml, "t_camera2gimbal");

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  // 故障态节流：read_blocking 在 Disconnected 时立刻返回，不加节流的话
  // 这个循环会以百万次/秒的速率刷共享内存命令通道和日志。
  bool last_read_ok = true;
  auto last_offline_warn = t_t0;
  bool warned_offline = false;

  while (!exiter.exit()) {
    if (duration_s > 0.0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_t0).count() > duration_s)
      break;

    const auto st = camera.read_blocking(img, t);

    // 心跳/取流故障统一映射到 fault 位，fault 非空即禁止开火。
    gimbal.set_fault(sim_io::FAULT_HEARTBEAT_LOST, !camera.heartbeat_alive());
    gimbal.set_fault(sim_io::FAULT_NO_NEW_FRAME, st == sim_io::ReadStatus::Timeout);
    gimbal.set_fault(
      sim_io::FAULT_FRAME_FAULT,
      st == sim_io::ReadStatus::Rejected || st == sim_io::ReadStatus::Stale);
    gimbal.set_fault(sim_io::FAULT_CLOCK_JUMP, false);

    if (st != sim_io::ReadStatus::Ok) {
      // 没有可用帧时必须主动重发安全停止，绝不能让旧命令留在三缓冲里被消费。
      // 进入故障态的第一次立刻发，之后按 safe_stop_period_ms 节流重发。
      if (last_read_ok) {
        gimbal.send_safe_stop();
      } else {
        gimbal.tick();
      }
      last_read_ok = false;

      if (st == sim_io::ReadStatus::Disconnected) {
        const auto now = std::chrono::steady_clock::now();
        if (
          !warned_offline ||
          std::chrono::duration<double>(now - last_offline_warn).count() >= 1.0) {
          tools::logger()->warn("[sim] 仿真端离线（心跳 {:.0f}ms）", camera.heartbeat_age_ms());
          last_offline_warn = now;
          warned_offline = true;
        }
        // Disconnected 不经过 read_blocking 的等待，必须自己让出 CPU。
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      continue;
    }

    last_read_ok = true;
    warned_offline = false;

    ++frames;
    const auto pipeline_start = std::chrono::steady_clock::now();

    // 落盘：确认相机视野里到底有什么，以及 RGB->BGR 色序是否正确。
    //
    // 只存第一帧不够用：闭环里检测会在第 4~5 帧突然全丢，要判断是"图像变了"还是
    // "检测器抖动"，就必须拿到丢失那几帧的原图。dump-detect>1 时按帧号分别落盘，
    // 与 detect#N 的日志一一对应。
    if (!dump_frame_path.empty() && static_cast<int>(frames) <= std::max(1, dump_detect)) {
      std::string path = dump_frame_path;
      if (dump_detect > 1) {
        const auto dot = path.find_last_of('.');
        const std::string stem = dot == std::string::npos ? path : path.substr(0, dot);
        const std::string ext = dot == std::string::npos ? ".png" : path.substr(dot);
        path = stem + "_" + std::to_string(frames) + ext;
      }
      if (cv::imwrite(path, img)) {
        tools::logger()->info("[sim] 第 {} 帧已存到 {}", frames, path);
      } else {
        tools::logger()->warn("[sim] 无法写入 {}", path);
      }
    }

    // 关键：用与这一帧图像严格同帧的姿态，而不是"处理完成时的最新姿态"。
    const auto & bundle = camera.last_bundle();
    gimbal.update(bundle, t);
    solver.set_R_gimbal2world(gimbal.q());

    // 打印首个真值批次：确认场景里到底有没有目标、在云台的哪个方向。
    // 真值只用于诊断输出，不进入任何算法输入。
    // 注意用的是 fetch_latest_diagnostic_only：真值通道只有一个"最新"槽位，
    // 且帧号按 Bevy 帧递增，而图像只在被消费时才发布，两者帧号永远对不上
    // （详见报告里的 seq skew 分析）。这里只为看一眼目标方位，不做评估。
    // 必须放在 gimbal.update(bundle, t) 之后：odom_position() 是 update() 里才从
    // 同帧 pose 束填的，放在前面打印会得到全 0，"相对云台"就退化成绝对坐标。
    if (dump_truth && !truth_dumped && evaluator.fetch_latest_diagnostic_only()) {
      truth_dumped = true;
      const auto & b = evaluator.batch();
      const Eigen::Vector3d gim = gimbal.odom_position();
      tools::logger()->info(
        "[sim] 真值批次 seq={}（图像 seq={}，差 {}）目标数={} 云台 odom=[{:.3f},{:.3f},{:.3f}]",
        b.frame_seq, camera.last_bundle().frame_seq,
        static_cast<long long>(b.frame_seq) -
          static_cast<long long>(camera.last_bundle().frame_seq),
        b.target_count, gim.x(), gim.y(), gim.z());
      for (std::uint32_t i = 0; i < b.target_count && i < sim_io::GROUND_TRUTH_MAX_TARGETS; ++i) {
        // pitch 取负号：这里按 ROS/solver 的约定，正 = 低头。
        // atan2(rel.z, horiz) 本身是"仰角"(正 = 目标在上方)，两者差一个负号，
        // 不取负的话打印值可以直接当 --bias-pitch-deg 用就会反向。
        const auto & tg = b.targets[i];
        const Eigen::Vector3d p(tg.position[0], tg.position[1], tg.position[2]);
        const Eigen::Vector3d rel = p - gim;
        tools::logger()->info(
          "[sim]   #{} team={} label={} outpost={} odom=[{:.3f},{:.3f},{:.3f}] "
          "相对云台=[{:.3f},{:.3f},{:.3f}] 距离={:.3f}m 方位yaw={:.2f}deg pitch={:.2f}deg",
          i, tg.team, tg.armor_label, tg.is_outpost, p.x(), p.y(), p.z(), rel.x(), rel.y(),
          rel.z(), rel.norm(), std::atan2(rel.y(), rel.x()) * 57.2957795130823,
          -std::atan2(rel.z(), rel.head<2>().norm()) * 57.2957795130823);
      }
    }

    // 外参自检：仿真端每帧发布 camera 相对云台的平移，和配置比一次就够。
    if (!extrinsic_checked && t_camera2gimbal.size() == 3) {
      extrinsic_checked = true;
      const auto & obs = gimbal.camera_offset();
      const double e = std::max(
        std::max(
          std::abs(obs.x() - t_camera2gimbal[0]), std::abs(obs.y() - t_camera2gimbal[1])),
        std::abs(obs.z() - t_camera2gimbal[2]));
      max_extrinsic_err = e;
      tools::logger()->info(
        "[sim] 观测 t_camera2gimbal = [{:.6f}, {:.6f}, {:.6f}]", obs.x(), obs.y(), obs.z());
      if (e > extrinsic_tol_m) {
        ++extrinsic_warnings;
        tools::logger()->warn(
          "[sim] t_camera2gimbal 与配置偏差 {:.4f}m > {:.4f}m，请把上面的观测值写回配置", e,
          extrinsic_tol_m);
      }
    }

    // ---- probe：单轴扫描，只验证符号与零位 ------------------------------------
    if (mode == "probe") {
      const double elapsed = std::chrono::duration<double>(t - t_t0).count();
      // bias 让 probe 也能当"把云台摆到某个方向"用（amplitude=0 即恒定指向）。
      const double sweep_deg =
        amplitude_deg * std::sin(2.0 * M_PI * elapsed / (period_s > 1e-3 ? period_s : 1.0));
      const double cmd_yaw_deg = bias_yaw_deg + (axis == "yaw" ? sweep_deg : 0.0);
      const double cmd_pitch_deg = bias_pitch_deg + (axis == "pitch" ? sweep_deg : 0.0);
      const double cmd_rad = (axis == "yaw" ? cmd_yaw_deg : cmd_pitch_deg) / 57.2957795130823;
      const double cmd_yaw = cmd_yaw_deg / 57.2957795130823;
      const double cmd_pitch = cmd_pitch_deg / 57.2957795130823;

      // probe 不需要目标，清掉 target_lost 以免它成为唯一的 fault 噪声源。
      gimbal.set_fault(sim_io::FAULT_TARGET_LOST, false);
      gimbal.send(true, false, cmd_yaw, cmd_pitch, 3.0);
      ++control_cmds;

      // 逐帧打印"下发角"与"下一帧反馈角"，符号错了会立刻表现为反向跟随。
      // 同时打印未修正的原始角：一眼就能看出坐标系修正有没有生效。
      const Eigen::Vector3d raw_ypr = tools::eulers(gimbal.q_raw(), 2, 1, 0);
      std::printf(
        "probe seq=%lu cmd_%s=%+8.4f deg  fb_yaw=%+8.4f deg  fb_pitch=%+8.4f deg  "
        "(raw_yaw=%+8.4f raw_pitch=%+8.4f)\n",
        static_cast<unsigned long>(bundle.frame_seq), axis.c_str(), cmd_rad * 57.2957795130823,
        gimbal.yaw() * 57.2957795130823, gimbal.pitch() * 57.2957795130823,
        raw_ypr[0] * 57.2957795130823, raw_ypr[1] * 57.2957795130823);
      pipeline_ms.push_back(
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pipeline_start).count());
      continue;
    }

    // ---- 感知 -----------------------------------------------------------------
    const auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo->detect(img, static_cast<int>(bundle.frame_seq));
    detect_ms.push_back(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - detect_start)
        .count());
    if (!armors.empty()) ++detected_frames;

    // 诊断快照必须在 track() 之前取：Tracker::track() 会就地改 armors
    // （按 enemy_color_ remove_if、排序、并在关联/NIS 校验里剔除装甲板），
    // 之后再读这个 list 会看到空表，误判成"检测失败"。这里顺手用 solver 解一遍
    // 单帧 PnP 世界坐标：静止目标的单帧解应当恒定，若它随云台反馈角跳变，
    // 就说明图像与同帧位姿没对齐（而不是检测或 EKF 的问题）。
    std::list<auto_aim::Armor> armors_dbg;
    if (dump_detect > 0 && static_cast<int>(frames) <= dump_detect) {
      armors_dbg = armors;
      for (auto & a : armors_dbg) solver.solve(a);
    }

    auto targets = tracker->track(armors, t, use_enemy_color);
    std::optional<auto_aim::Target> target;
    if (!targets.empty()) {
      target = targets.front();
      ++tracked_frames;
      last_target_seen = std::chrono::steady_clock::now();
      ever_seen_target = true;
    }

    // 检测/跟踪明细：检到了却 tracked_frames=0 时，用来区分是分类错(颜色/编号)
    // 还是 tracker 关联/几何校验拒了。只打印前 N 帧，纯诊断，不影响任何控制。
    if (dump_detect > 0 && static_cast<int>(frames) <= dump_detect) {
      std::string desc;
      for (const auto & a : armors_dbg) {
        desc += fmt::format(
          "[{} color={} conf={:.2f} ctr=({:.0f},{:.0f}) pnp=({:.3f},{:.3f},{:.3f})]",
          auto_aim::ARMOR_NAMES[a.name], static_cast<int>(a.color), a.confidence, a.center.x,
          a.center.y, a.xyz_in_world[0], a.xyz_in_world[1], a.xyz_in_world[2]);
      }
      if (desc.empty()) desc = "(无)";
      std::string tdesc = "(无)";
      if (target.has_value()) {
        const auto x = target->ekf_x();
        // 除了中心 xyz，还要打整车姿态量：planner 瞄的是"到弹丸落点时刻、被选中那块
        // 装甲板"的位置，而这块板绕中心以半径 r 转、转速 vyaw。静止靶的 vyaw 应当≈0，
        // 若它被 EKF 拟合出几百 deg/s，预测点就会绕着中心甩，下发角随之来回大摆——
        // 这与"中心 xyz 很稳但命令 ±20° 振荡"的现象吻合，所以必须能看到它。
        tdesc = fmt::format(
          "{} xyz=({:.3f},{:.3f},{:.3f}) yaw={:+.1f}deg vyaw={:+.1f}deg/s r=({:.3f},{:.3f}) h={:.3f}",
          auto_aim::ARMOR_NAMES[target->name], x[0], x[2], x[4],
          x[auto_aim::target_state::ROT_Z] * 57.2957795130823, x[auto_aim::target_state::VYAW] * 57.2957795130823,
          std::exp(x[auto_aim::target_state::LOG_R1]), std::exp(x[auto_aim::target_state::LOG_R2]),
          x[auto_aim::target_state::H]);
      }
      tools::logger()->info(
        "[sim] detect#{} seq={} fb=({:+.2f},{:+.2f})deg state={} armors={} -> track={}", frames,
        bundle.frame_seq, gimbal.yaw() * 57.2957795130823, gimbal.pitch() * 57.2957795130823,
        tracker->state(), desc, tdesc);
    }

    const double lost_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - last_target_seen).count();
    gimbal.set_fault(
      sim_io::FAULT_TARGET_LOST, !ever_seen_target || lost_ms > target_lost_ms);

    // ---- 真值评估（真值只进评估器，绝不进算法输入）-----------------------------
    std::optional<Eigen::Vector3d> gt_pos_this_frame;
    if (do_eval && target.has_value() && evaluator.fetch(bundle.frame_seq)) {
      const auto x = target->ekf_x();
      // Solver 的输出以云台为原点，真值以 odom 为原点，比较前必须加上同帧的云台平移。
      const Eigen::Vector3d in_odom =
        gimbal.odom_position() + Eigen::Vector3d(x[0], x[2], x[4]);
      const auto err = evaluator.evaluate(target->name, in_odom, x[6], x[7]);
      evaluator.record(err);
      if (err.valid) gt_pos_this_frame = err.gt_position;
    }

    // ---- 规划与下发 -----------------------------------------------------------
    if (mode == "closed_loop") {
      const auto plan = planner->plan(target, gimbal.state().bullet_speed);
      // 诊断：把规划下发的角和当前反馈角一起打出来。闭环丢目标时，先看是"规划把
      // 云台指飞了"还是"检测本身掉了"。纯诊断，不改变任何控制量。
      if (dump_detect > 0 && static_cast<int>(frames) <= dump_detect) {
        tools::logger()->info(
          "[sim] plan#{} control={} fire={} cmd=({:+.2f},{:+.2f})deg fb=({:+.2f},{:+.2f})deg",
          frames, plan.control, plan.fire, plan.yaw * 57.2957795130823,
          plan.pitch * 57.2957795130823, gimbal.yaw() * 57.2957795130823,
          gimbal.pitch() * 57.2957795130823);
      }
      const bool fired = gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.pitch,
        target.has_value() ? Eigen::Vector3d(
                               target->ekf_x()[0], target->ekf_x()[2], target->ekf_x()[4])
                               .norm()
                           : -1.0);
      (void)fired;
      // 几何瞄准误差。ROS 约定 R = Rz(yaw)Ry(pitch)，机体 x 轴前向，
      // 于是下发方向的单位向量是 (cos p cos y, cos p sin y, -sin p)。
      if (plan.control && gt_pos_this_frame.has_value()) {
        const Eigen::Vector3d cmd_dir(
          std::cos(plan.pitch) * std::cos(plan.yaw), std::cos(plan.pitch) * std::sin(plan.yaw),
          -std::sin(plan.pitch));
        const Eigen::Vector3d to_gt = *gt_pos_this_frame - gimbal.odom_position();
        if (to_gt.norm() > 1e-6) {
          const Eigen::Vector3d gt_dir = to_gt.normalized();
          const double dot = std::clamp(cmd_dir.dot(gt_dir), -1.0, 1.0);
          aim_err_deg.push_back(std::acos(dot) * 57.2957795130823);
          // 分轴看：pitch 那一路本来就带"板心 vs 车心 + 弹道补偿"的系统偏移，
          // 和 yaw 混在一起会掩盖 yaw 的真实精度。
          const double cmd_yaw = plan.yaw;
          const double gt_yaw = std::atan2(gt_dir.y(), gt_dir.x());
          double dyaw = cmd_yaw - gt_yaw;
          while (dyaw > M_PI) dyaw -= 2 * M_PI;
          while (dyaw < -M_PI) dyaw += 2 * M_PI;
          aim_yaw_err_deg.push_back(dyaw * 57.2957795130823);
          const double gt_pitch = -std::asin(std::clamp(gt_dir.z(), -1.0, 1.0));
          aim_pitch_err_deg.push_back((plan.pitch - gt_pitch) * 57.2957795130823);
        }
      }
      if (plan.control) ++control_cmds;
      if (plan.fire && gimbal.fire_allowed()) ++fire_cmds;
    } else {
      // passive：只看不动。安全停止让仿真端的 process_subscription 直接 return。
      gimbal.send_safe_stop();
    }

    pipeline_ms.push_back(
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pipeline_start).count());
  }

  // 退出前把最新命令固定成安全停止：即使本进程之后崩溃，
  // 仿真端消费到的最后一条命令也是无害的。
  gimbal.send_safe_stop();
  gimbal.send_safe_stop();

  // ---- 指标汇总 ---------------------------------------------------------------
  const auto age = camera.frame_age_stats();
  const auto & client = camera.client();

  std::ostringstream js;
  js << std::fixed << std::setprecision(4);
  js << "{\n";
  js << "  \"mode\": \"" << mode << "\",\n";
  js << "  \"allow_fire\": " << (gim_cfg.allow_fire ? "true" : "false") << ",\n";
  js << "  \"frames_ok\": " << frames << ",\n";
  js << "  \"frame_age_ms\": {\"count\": " << age.count << ", \"min\": " << age.min_ms
     << ", \"p50\": " << age.p50_ms << ", \"p95\": " << age.p95_ms << ", \"p99\": " << age.p99_ms
     << ", \"max\": " << age.max_ms << ", \"mean\": " << age.mean_ms << "},\n";
  js << "  \"ipc\": {\"consumed\": " << client.consumed_frames()
     << ", \"dropped\": " << client.dropped_frames()
     << ", \"skipped\": " << client.skipped_frames()
     << ", \"regressed\": " << client.regressed_frames()
     << ", \"corrupted\": " << client.corrupted_events()
     << ", \"publisher_restarts\": " << client.publisher_restarts()
     << ", \"last_seq\": " << client.last_seq() << "},\n";
  js << "  \"camera\": {\"stale\": " << camera.stale_frames()
     << ", \"rejected\": " << camera.rejected_frames()
     << ", \"future\": " << camera.future_frames()
     << ", \"clock_jumps\": " << camera.clock_jumps()
     << ", \"fps\": " << camera.camera_fps() << "},\n";
  js << "  \"detect_ms\": {\"p50\": " << percentile(detect_ms, 0.50)
     << ", \"p95\": " << percentile(detect_ms, 0.95) << ", \"p99\": " << percentile(detect_ms, 0.99)
     << "},\n";
  js << "  \"pipeline_ms\": {\"p50\": " << percentile(pipeline_ms, 0.50)
     << ", \"p95\": " << percentile(pipeline_ms, 0.95)
     << ", \"p99\": " << percentile(pipeline_ms, 0.99) << "},\n";
  js << "  \"perception\": {\"detected_frames\": " << detected_frames
     << ", \"tracked_frames\": " << tracked_frames << "},\n";
  js << "  \"gimbal\": {\"sent\": " << gimbal.sent_commands()
     << ", \"control\": " << control_cmds << ", \"plan_fire\": " << fire_cmds
     << ", \"fire\": " << gimbal.fire_commands()
     << ", \"suppressed_fire\": " << gimbal.suppressed_fires()
     << ", \"safe_stops\": " << gimbal.safe_stops()
     << ", \"final_faults\": \"" << sim_io::describe_faults(gimbal.faults()) << "\"},\n";
  js << "  \"extrinsic\": {\"max_err_m\": " << max_extrinsic_err
     << ", \"warnings\": " << extrinsic_warnings << "}";
  if (do_eval) {
    const auto gt = evaluator.stats();
    js << ",\n  \"ground_truth\": {\"count\": " << gt.count << ", \"pos_p50_m\": " << gt.pos_p50_m
       << ", \"pos_p95_m\": " << gt.pos_p95_m << ", \"pos_max_m\": " << gt.pos_max_m
       << ", \"pos_mean_m\": " << gt.pos_mean_m << ", \"xy_mean_m\": " << gt.xy_mean_m
       << ", \"z_mean_m\": " << gt.z_mean_m << ", \"yaw_p50_rad\": " << gt.yaw_p50_rad
       << ", \"yaw_p95_rad\": " << gt.yaw_p95_rad << ", \"vyaw_mean_radps\": " << gt.vyaw_mean_radps
       << ", \"seq_mismatches\": " << evaluator.seq_mismatches() << "}";
    // 闭环几何瞄准误差。mean 里含"板心 vs 车心 + 弹道补偿"的系统偏移，
    // 判闭环稳不稳看 p95/max 与 p50 的差以及分轴的取值范围。
    if (!aim_err_deg.empty()) {
      auto mean = [](const std::vector<double> & v) {
        double s = 0.0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
      };
      const auto yaw_mm = std::minmax_element(aim_yaw_err_deg.begin(), aim_yaw_err_deg.end());
      const auto pitch_mm =
        std::minmax_element(aim_pitch_err_deg.begin(), aim_pitch_err_deg.end());
      js << ",\n  \"aim_error_deg\": {\"count\": " << aim_err_deg.size()
         << ", \"p50\": " << percentile(aim_err_deg, 0.50)
         << ", \"p95\": " << percentile(aim_err_deg, 0.95)
         << ", \"max\": " << percentile(aim_err_deg, 1.0) << ", \"mean\": " << mean(aim_err_deg)
         << ", \"yaw_mean\": " << mean(aim_yaw_err_deg) << ", \"yaw_min\": " << *yaw_mm.first
         << ", \"yaw_max\": " << *yaw_mm.second << ", \"pitch_mean\": " << mean(aim_pitch_err_deg)
         << ", \"pitch_min\": " << *pitch_mm.first << ", \"pitch_max\": " << *pitch_mm.second
         << "}";
    }
  }
  js << "\n}\n";

  std::printf("%s", js.str().c_str());
  if (!report_path.empty()) {
    std::ofstream ofs(report_path);
    if (ofs) {
      ofs << js.str();
      tools::logger()->info("[sim] 指标已写入 {}", report_path);
    } else {
      tools::logger()->warn("[sim] 无法写入 {}", report_path);
    }
  }

  camera.close();
  return 0;
}
