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
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#include "simulation/io/cli_args.hpp"
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
  "{max-command-age-ms | -1                  | 覆盖配置里的世界观测年龄预算，<0 表示用配置值}"
  "{bias-yaw-deg   | 0.0                   | probe 模式的 yaw 固定偏置(度)，用来把云台指向某个方向}"
  "{bias-pitch-deg | 0.0                   | probe 模式的 pitch 固定偏置(度)，正=低头(ROS 约定)}"
  "{dump-detect    | 0                     | 打印前 N 帧的检测/跟踪明细(排查为什么检到了却跟不上)}"
  "{enemy-team     |                       | 覆盖真值评估的敌方队伍: red / blue / any}"
  // 这两个必须是**字符串**键。OpenCV 的 CommandLineParser 解析不了 "nan" 这个
  // 默认值：get<double>() 会静默返回 0.0（实测），而 has() 对有默认值的键恒为
  // true，两条路都拿不到"用户没给"这个信息。于是 isfinite(0.0)=true，驻留被当成
  // 已开启并把云台摆到 yaw=0——而目标在 yaw≈+90，表现是 detected_frames 从 447
  // 掉到 4，看着像感知坏了。空字符串默认值是这个 parser 唯一可靠的"未给定"标记
  // （report / dump-frame 也是这么用的）。
  "{park-yaw-deg   |                       | closed_loop 丢目标时把云台驻留到该 yaw(度)，缺省不驻留}"
  "{park-pitch-deg |                       | closed_loop 丢目标时的驻留 pitch(度)，正=低头(ROS 约定)}";

// 这两个解析函数放在 simulation/io/cli_args.hpp，以便单测覆盖边界取值。
using sim_io::parse_enemy_team;
using sim_io::parse_park_angle;

const char * enemy_team_name(std::uint8_t team)
{
  switch (team) {
    case sim_io::GT_TEAM_RED:
      return "red";
    case sim_io::GT_TEAM_BLUE:
      return "blue";
    default:
      return "any";
  }
}

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
  // 丢目标时的驻留指向。closed_loop 在 lost 状态下只发安全停止（不控制），
  // 云台会一直停在场景初始位姿；本机场景的初始位姿是 pitch≈+65°（低头看自己
  // 底盘），而两台敌方步兵在 yaw≈+90°，所以闭环根本无法自举——不是感知问题，
  // 而是"视野里从来没出现过目标"。真机上这一段由操作手或上层决策负责，仿真里
  // 没有那一层，于是提供一个显式的驻留指向。缺省 NaN=不驻留，保持原行为不变。
  // 两个都给且都能解析成有限数才算开启；只给一个视为用法错误，直接退出而不是
  // 悄悄按半个指向驻留。
  double park_yaw_deg = 0.0;
  double park_pitch_deg = 0.0;
  const bool park_enabled =
    parse_park_angle(cli.get<std::string>("park-yaw-deg"), &park_yaw_deg) &&
    parse_park_angle(cli.get<std::string>("park-pitch-deg"), &park_pitch_deg);
  if (!park_enabled && (!cli.get<std::string>("park-yaw-deg").empty() ||
                        !cli.get<std::string>("park-pitch-deg").empty())) {
    tools::logger()->error("[sim] --park-yaw-deg 与 --park-pitch-deg 必须同时给出有限角度");
    return 2;
  }
  const int dump_detect = cli.get<int>("dump-detect");
  const double duration_s = cli.get<double>("duration-s");
  const bool allow_fire = cli.has("allow-fire");
  const bool do_eval = cli.has("eval");
  const auto report_path = cli.get<std::string>("report");
  const auto dump_frame_path = cli.get<std::string>("dump-frame");
  const bool dump_truth = cli.has("dump-truth");
  const auto enemy_team_cli = cli.get<std::string>("enemy-team");

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
  // 断流判据与"重连探测"周期。open() 会把这两个值夹进 max_frame_age_ms /
  // read_timeout_ms 的可行区间，保证"无新帧"看门狗的实际触发时刻落在阈值之内，
  // 而不是被 read_timeout 拖到几百毫秒之后才报出来（见 SimCamera::open）。
  cam_cfg.no_new_frame_timeout_ms =
    tools::read_or<double>(sim, "no_new_frame_timeout_ms", cam_cfg.no_new_frame_timeout_ms);
  cam_cfg.remap_check_ms = tools::read_or<double>(sim, "remap_check_ms", cam_cfg.remap_check_ms);
  // 共享内存位置可覆盖：便于对着 FakePublisher 或第二个仿真实例跑，不影响默认路径。
  cam_cfg.shm.dir = tools::read_or<std::string>(sim, "shm_dir", cam_cfg.shm.dir);

  sim_io::SimGimbalConfig gim_cfg;
  gim_cfg.yaw_scale = tools::read_or<double>(sim, "yaw_scale", 1.0);
  gim_cfg.yaw_offset_deg = tools::read_or<double>(sim, "yaw_offset_deg", 0.0);
  gim_cfg.pitch_scale = tools::read_or<double>(sim, "pitch_scale", 1.0);
  // 回退默认值必须与 SimGimbalConfig 一致：cmd 的角就是 ROS 的角（identity）。
  // 这里曾经是 -90，配置漏键时会把水平瞄准编码成"垂直朝天"。
  gim_cfg.pitch_offset_deg = tools::read_or<double>(sim, "pitch_offset_deg", 0.0);
  // 仿真端发布的 q_raw 已经是 world<-gimbal，不需要修正。改这个值等于改坐标系。
  gim_cfg.feedback_pitch_fix_deg =
    tools::read_or<double>(sim, "feedback_pitch_fix_deg", 0.0);
  // 本地状态保有时长看门狗（通道活性），与源帧龄无关。
  gim_cfg.state_timeout_ms = tools::read_or<double>(sim, "state_timeout_ms", 200.0);
  // 命令时刻的世界观测年龄上限（开火决策关心的量）。0 = 不设限；报告始终输出实测
  // 分布。closed_loop + --allow-fire 时它是**必填**的，见下面那道门。
  // 见 SimGimbalConfig::max_command_age_ms。
  gim_cfg.max_command_age_ms = tools::read_or<double>(sim, "max_command_age_ms", 0.0);
  std::string command_age_budget_source = "config";
  // 命令行覆盖：让"本次运行用的是哪个预算"直接落在命令行与报告里，不必回溯当时
  // 的 yaml。语义与 --max-frame-age-ms 一致，<0 表示不覆盖。
  const double max_command_age_override = cli.get<double>("max-command-age-ms");
  if (max_command_age_override >= 0.0) {
    if (gim_cfg.max_command_age_ms > 0.0) {
      tools::logger()->warn(
        "[sim] 世界观测年龄预算被命令行覆盖为 {:.1f}ms（配置值 {:.1f}ms）",
        max_command_age_override, gim_cfg.max_command_age_ms);
    }
    gim_cfg.max_command_age_ms = max_command_age_override;
    command_age_budget_source = "cli";
  }
  if (gim_cfg.max_command_age_ms <= 0.0 || !std::isfinite(gim_cfg.max_command_age_ms)) {
    command_age_budget_source = "unset";
  }
  gim_cfg.safe_stop_period_ms = tools::read_or<double>(sim, "safe_stop_period_ms", 20.0);
  gim_cfg.bullet_speed = tools::read_or<double>(yaml, "bullet_speed_fallback", 25.0);
  // probe 模式永远不开火：这是确认符号/零位的实验，弹道无关。
  gim_cfg.allow_fire = allow_fire && mode == "closed_loop";

  // 开火必须有一个显式的世界观测年龄预算。
  //
  // 缺预算时 command_age_exceeded() 恒 false，FAULT_COMMAND_AGE 永不点亮，
  // fire_allowed() 就少了一条判据：只要姿态通道还活着（state_stale 看的是**本地
  // 到达时刻**），哪怕这条命令是基于几百毫秒前的世界观测算出来的，也照样发弹。
  // 这两个量在本机实测里差一整个检测耗时（detect p95 已到 241ms），不是同阶小量。
  //
  // 所以这里拒绝启用开火，而不是默默降级成"不设限地开火"：后者的运行记录看起来
  // 是一次正常的 --allow-fire 闭环，事后无法从报告里区分。
  // state_timeout_ms 是通道活性看门狗，不能拿来当这个预算——它量的是"上一条姿态
  // 到本地多久了"，与"这条命令基于多旧的世界观测"是两个不同的量。
  if (gim_cfg.allow_fire && command_age_budget_source == "unset") {
    tools::logger()->error(
      "[sim] closed_loop + --allow-fire 必须给出正的世界观测年龄预算 "
      "max_command_age_ms，当前为 {:.1f}（未配置或 <=0）", gim_cfg.max_command_age_ms);
    tools::logger()->error(
      "[sim] 用 --max-command-age-ms=<正数> 覆盖，或在 {} 的 sim 段里写 "
      "max_command_age_ms。不要用 state_timeout_ms（{:.1f}ms）代替：那是通道活性"
      "看门狗，量的不是世界观测年龄。", config_path, gim_cfg.state_timeout_ms);
    return 2;
  }
  if (gim_cfg.allow_fire) {
    tools::logger()->info(
      "[sim] 世界观测年龄预算 {:.1f}ms（来源 {}），超预算的帧会点亮 command_age 并"
      "抑制开火", gim_cfg.max_command_age_ms, command_age_budget_source);
  }

  const double target_lost_ms = tools::read_or<double>(sim, "target_lost_ms", 300.0);
  // 重新武装门限：换代/重连/时钟跳变之后，必须连续这么多帧稳定跟上目标才解除
  // FAULT_REARM_PENDING。1 帧不够——重连后的第一帧只证明"取到了一帧完整同步数据"，
  // 不证明 Tracker 在新纪元里已经建立起可信的目标状态。
  const int rearm_confirm_frames =
    std::max(1, tools::read_or<int>(sim, "rearm_confirm_frames", 5));
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

  // 真值评估的敌方队伍。优先级：命令行 > sim.enemy_team > 默认 blue。
  //
  // 为什么默认是 blue 而不是顶层的 enemy_color：场景 setup 里被 Controlled 标记的
  // 本车是红方，两台敌方步兵都是蓝方，而 configs/simulation.yaml 顶层
  // enemy_color: "red" 是给实车 Tracker 的颜色门用的，仿真里已经被
  // sim.use_enemy_color=false 关掉了。两者含义不同，不能互相顶替；下面会在它们
  // 不一致时打一条 warn，避免以后有人以为改顶层就能改评估对象。
  std::uint8_t enemy_team = sim_io::GT_TEAM_BLUE;
  {
    const auto from_yaml = tools::read_or<std::string>(sim, "enemy_team", "blue");
    if (!parse_enemy_team(from_yaml, &enemy_team)) {
      tools::logger()->error("[sim] sim.enemy_team 取值非法: {}（应为 red/blue/any）", from_yaml);
      return 2;
    }
    if (!enemy_team_cli.empty() && !parse_enemy_team(enemy_team_cli, &enemy_team)) {
      tools::logger()->error("[sim] --enemy-team 取值非法: {}（应为 red/blue/any）", enemy_team_cli);
      return 2;
    }
    const auto top_color = tools::read_or<std::string>(yaml, "enemy_color", std::string());
    if (!top_color.empty() && top_color != enemy_team_name(enemy_team)) {
      tools::logger()->warn(
        "[sim] 真值评估敌方队伍={}，与顶层 enemy_color={} 不一致。顶层键只作用于实车 "
        "Tracker 的颜色门（仿真已 use_enemy_color=false 关闭），评估对象只看 sim.enemy_team",
        enemy_team_name(enemy_team), top_color);
    }
    // 真值区必须由发布端明确声明它真的在写。协议 v2 里没有这个声明，新消费端
    // 对着一个不写真值区的发布端只会读到恒 0 的 seqlock，read_ground_truth()
    // 一律返回 false，于是所有 aim_/geom_/parallax_ 统计的 count 都是 0——报告
    // 看上去"跑通了"，只是没有任何一行误差数据，很容易被当成"这些指标为零"。
    // --eval 明确要求真值，这里直接拒绝启动，而不是静默降级。
    if (do_eval && !camera.client().has_capability(sim_io::CAP_GROUND_TRUTH)) {
      tools::logger()->error(
        "[sim] --eval 需要发布端提供真值区，但发布端未声明 CAP_GROUND_TRUTH"
        "（shm 版本={}，capabilities=0x{:08x}）。请确认仿真端为协议 v{} 及以上",
        camera.client().version(), camera.client().capabilities(), sim_io::SHM_VERSION);
      return 2;
    }
    if (do_eval && enemy_team == sim_io::GT_TEAM_ANY) {
      tools::logger()->warn(
        "[sim] enemy_team=any：红蓝同编号步兵会互相污染匹配，本次评估数据只能当诊断看");
    }
    tools::logger()->info("[sim] 真值评估敌方队伍 = {}", enemy_team_name(enemy_team));
  }

  auto_aim::Solver solver(config_path);
  sim_io::SimGimbal gimbal(camera.client(), gim_cfg);
  sim_io::GroundTruthEvaluator evaluator(camera.client(), enemy_team);

  // 感知链只在需要时才构造：probe 模式不加载 OpenVINO 模型，启动快且不依赖 assets。
  std::unique_ptr<auto_aim::YOLO> yolo;
  std::unique_ptr<auto_aim::Tracker> tracker;
  std::unique_ptr<auto_aim::Planner> planner;
  if (mode != "probe") {
    yolo = std::make_unique<auto_aim::YOLO>(config_path, false);
    tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
    planner = std::make_unique<auto_aim::Planner>(config_path);
  }

  // 重连/换代之后必须把估计状态全部丢掉重建。
  //
  // Tracker 和 Planner 都没有 reset() 接口，唯一干净的复位办法就是销毁重建：
  //   * Tracker 内部持有 EKF、装甲板关联历史和 lost/detecting 状态机，跨纪元
  //     沿用会把断流前后的位置当成同一条轨迹，差分出巨大的假速度；
  //   * Planner(TinyMPC) 持有参考轨迹与热启动解，同样会把旧解带进新纪元。
  // YOLO 是无状态推理，模型重载要几秒，不重建。
  auto rebuild_estimators = [&](const char * why) {
    if (mode == "probe") return;
    tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
    planner = std::make_unique<auto_aim::Planner>(config_path);
    tools::logger()->warn("[sim] {}：已重建 Tracker/Planner，重新武装前禁止控制与开火", why);
  };

  // ---- 运行期统计 -------------------------------------------------------------
  std::vector<double> detect_ms, pipeline_ms;
  // 下发控制那一刻的两个年龄，语义见 sim_io::FrameStamps：
  //   command_age = now - 源端采样时刻 = 源帧龄 + 至今的处理耗时（开火决策关心的量）
  //   state_age   = now - 本地接收时刻 = 本地状态保有时长（通道看门狗关心的量）
  // 分开采样是为了让报告能直接证明"state_stale 不再包含处理耗时"。
  std::vector<double> command_age_ms, state_age_ms;
  std::uint64_t frames = 0, detected_frames = 0, tracked_frames = 0;
  std::uint64_t park_frames = 0;
  std::uint64_t control_cmds = 0, fire_cmds = 0;
  // 闭环几何瞄准误差：下发角所指方向 与 云台->真值目标方向 的夹角。
  //
  // 之所以要这个量：仿真端物理没有 CCD（fixed_hz=120、8 substep，弹丸每子步走
  // 约 26mm，而装甲板是薄 trimesh），弹丸会直接穿过板子，命中统计恒为 0，
  // 靠"打中几发"没法评价闭环。角度误差不依赖碰撞检测，能直接回答
  // "云台到底指对了没有"。
  //
  // 参考原点必须明确写出来，并且与被比较的量一致：下发角是按**云台原点**解出来的
  // （tools::Trajectory 就是按这个原点解的），所以 aim_/geom_ 都以云台原点为原点；
  // 弹丸真正出膛的位置是枪口，相对云台有 0.1m 量级前伸/抬高，1.5m 距离上就是度
  // 量级，单列 aim_err_muzzle_deg 与 muzzle_parallax_deg，不把它混进瞄准误差。
  //
  // 各分量分开量，不做任何反推。此前那份"2.18°(板心-车心) + 0.68°(弹道) = 2.86°"
  // 的分解是拿几何估算去解释观测值，不是测量：板心偏移是按静态几何算的、弹道量是
  // 按标称弹速算的，都没有用运行时的真实取值核对过，所以也不能据此声称
  // "真正的闭环残差在 0.5° 以内"。现在改成：
  //   * 板心真值由仿真端直接发布（GroundTruthTarget::armor_position），
  //     瞄准误差直接对着它算，几何差不再进残差；
  //   * 弹道抬枪量直接取 planner 自己的 debug 量，不用弹速反算；
  //   * 保留对整车中心的旧口径（center_*），只为和历史数据对比。
  //
  // 各分量都要分开量，混在一起的"闭环残差"没有意义。**参考原点必须写清楚**：
  // 下发角是 tools::Trajectory 按**云台原点**解出来的，所以所有"下发方向 vs 真值
  // 方向"的口径都以云台原点为原点；弹丸真正出膛的位置是枪口，那一份单独列
  // aim_err_muzzle_deg，两者之差就是 muzzle_parallax_deg。
  //
  //   aim_*                下发方向 vs 云台原点->板心真值（与解算同原点）
  //   aim_err_muzzle_*     下发方向 vs 枪口->板心真值（物理出膛点口径）
  //   center_*             下发方向 vs 云台原点->整车中心真值（旧口径）
  //   geom_*               算法选中的瞄准点 vs 板心真值，同为云台原点
  //   ballistic_lift_*     重力抬枪量（planner 同一次解算里的 gravity_lift）
  //   ballistic_offset_*   标定 pitch_offset，与抬枪量分开
  //   ballistic_identity_* pitch-(geometric-lift-offset) 的残差，应恒为 0
  //   mpc_pitch_lag_*      plan.pitch 减同一时刻的瞬时解，属于 MPC 行为，不是弹道
  std::vector<double> aim_err_deg, aim_yaw_err_deg, aim_pitch_err_deg;
  std::vector<double> aim_err_muzzle_deg;
  std::vector<double> center_err_deg;
  std::vector<double> geom_err_deg;
  std::vector<double> ballistic_lift_deg;
  std::vector<double> ballistic_offset_deg;
  std::vector<double> ballistic_identity_deg;
  std::vector<double> mpc_pitch_lag_deg;
  //   parallax_*   云台原点->板心 与 枪口->板心 的夹角
  //
  // 为什么要单独量这一项：算法侧的目标坐标是相对**云台原点**的（PnP 出来的相机
  // 系坐标经 R/t_camera2gimbal 变换过去），tools::Trajectory 也就把弹丸当成从云台
  // 原点出膛来解；但仿真端真正 spawn 弹丸的位置是枪口（projectile.rs 用
  // launch_offset.translation），实测枪口相对云台原点 +0.110m（ROS z，见
  // PoseIndex::Muzzle）。两个原点差 0.11m，在近距离上就是可观的角度差。
  // 这是真机上同样存在的建模简化（真机靠 pitch_offset 标定吸收），不是本次接入
  // 引入的缺陷；但本场景目标只有 1.5~2.5m，视差比重力补偿还大，不单独列出来会
  // 让人把它误当成感知误差。
  //
  // 协议 v3 之前，这里的枪口世界位置是 `odom_position() + muzzle_offset()`，即世界
  // 平移加上一个未经云台旋转的局部平移，本身就是错的（yaw=90° 时误差等于偏移量
  // 全长）。因此改动前跑出来的 aim_/parallax_/geom_ 数值一律不可用于对比。
  std::vector<double> muzzle_parallax_deg;
  std::uint64_t aim_no_armor_gt = 0;
  std::uint64_t pose_invalid_frames = 0;
  std::uint64_t extrinsic_warnings = 0;
  double max_extrinsic_err = 0.0;
  bool extrinsic_checked = false;
  bool truth_dumped = false;

  const auto t_t0 = std::chrono::steady_clock::now();
  auto last_target_seen = t_t0;
  bool ever_seen_target = false;

  // 重新武装状态机。启动本身也算一次"待武装"：还没有任何一帧完整同步数据时
  // 就不该允许开火，这与 FAULT_STARTUP 的语义一致，这里只是把"必须连续确认
  // 目标"这一条补上。
  // 仿真端是否真的在消费我们的云台命令。
  //
  // 仿真端只在 auto-aim 订阅打开时才 apply gimbal_cmd（交互按 F5/RT，或
  // DAEDALUS_FORCE_AUTO_AIM=1），并把这个状态写进 RuntimeState::following。
  // 订阅没开时闭环在视觉侧完全看不出异常：命令照发、统计照涨，云台不动。
  // 所以 closed_loop 必须主动核对这一位，并在报告里留痕。
  bool following_seen = false;
  bool following_warned = false;
  std::uint64_t not_following_frames = 0;
  // 发布端没有声明 CAP_RUNTIME_STATE 的帧数。这与 not_following_frames 是两件事：
  // 前者是"我们无从知道仿真端有没有订阅"，后者是"仿真端明确说了没订阅"。
  bool runtime_state_warned = false;
  std::uint64_t runtime_state_missing_frames = 0;

  bool rearm_pending = true;
  int rearm_confirmed = 0;
  std::uint64_t rearm_events = 0;

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
    // 断流看门狗按"距上一帧到达多久"判定，而不是按 read_blocking 是否超时返回。
    // 后者的触发时刻受 read_timeout_ms 支配，会比 no_new_frame_timeout_ms 晚很多；
    // 且 Ok 帧之后如果长时间不再有新帧，只看返回值就永远报不出来。
    gimbal.set_fault(sim_io::FAULT_NO_NEW_FRAME, camera.no_new_frame());
    gimbal.set_fault(
      sim_io::FAULT_FRAME_FAULT,
      st == sim_io::ReadStatus::Rejected || st == sim_io::ReadStatus::Stale ||
        st == sim_io::ReadStatus::Reconnected);
    // 时钟跳变必须真的进 fault 位。跳变帧的帧龄不可信（realtime<->steady 偏移变了），
    // 用它做延迟补偿会让 planner 按错误的飞行时间提前量下发；原来这里硬编码 false，
    // 等于 ClockJump 只被计数、不影响控制与开火。
    gimbal.set_fault(sim_io::FAULT_CLOCK_JUMP, st == sim_io::ReadStatus::ClockJump);

    // 换代/重连/时钟跳变都要求重新武装：估计器状态作废，且必须重新连续确认目标
    // 才允许开火，避免旧纪元的目标状态"复活"成新纪元的开火依据。
    if (st == sim_io::ReadStatus::Reconnected || st == sim_io::ReadStatus::ClockJump) {
      rebuild_estimators(
        st == sim_io::ReadStatus::Reconnected ? "发布端换代/重连" : "时钟跳变");
      // 云台侧的时间戳水位线与角速度历史也必须一起丢掉，否则：
      //   * 发布端真的回拨了墙上时钟（重启后被 NTP 往回校、或换成一个时间更早的
      //     发布端），新帧 timestamp_ns 会永远小于旧水位线，update() 每帧都判
      //     BadTimestamp，消费端被永久锁死在 FAULT_POSE_INVALID，只能重启进程；
      //   * 跨换代/跳变做差分算出来的 yaw_vel_/pitch_vel_ 含整个跳变量。
      // 只清水位线和速度历史，faults 与累计计数保留（见 SimGimbal::reset_history）。
      gimbal.reset_history();
      rearm_pending = true;
      rearm_confirmed = 0;
      ++rearm_events;
      // 旧目标一律作废：ever_seen_target 置回 false，让 FAULT_TARGET_LOST 立刻生效，
      // 而不是靠 target_lost_ms 慢慢超时。
      ever_seen_target = false;
      gimbal.set_fault(sim_io::FAULT_TARGET_LOST, true);
    }
    gimbal.set_fault(sim_io::FAULT_REARM_PENDING, rearm_pending);

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
    // 位姿合法性必须在用它之前判。共享内存是外部进程写的：未初始化槽位是全零
    // （四元数模长 0），NaN 会一路穿过 normalize()/eulers()/set_R_gimbal2world()，
    // 最后污染 Solver 与 EKF，只在 send() 的 isfinite 门被拦住——那时已经太晚。
    // update() 在拒绝时不改动任何状态，所以这里直接跳过整帧处理即可。
    // 两个时间点必须分开传：t 是源端采样时刻（映射到本地 steady），
    // camera.last_stamps().arrival 是这一帧在本进程被取到的时刻。混成一个会让
    // state_stale 把"源帧龄 + 处理耗时"当成"姿态过期"，恒亮并抑制全部开火。
    const auto pose_ok = gimbal.update(bundle, camera.last_stamps());
    if (pose_ok != sim_io::PoseValidity::Ok) {
      ++pose_invalid_frames;
      --frames;  // frames 表示"真正进入感知的帧"，位姿被拒的帧不算
      gimbal.set_fault(sim_io::FAULT_POSE_INVALID, true);
      if (last_read_ok) {
        gimbal.send_safe_stop();
      } else {
        gimbal.tick();
      }
      last_read_ok = false;
      if (pose_invalid_frames <= 5 || pose_invalid_frames % 100 == 0) {
        tools::logger()->warn(
          "[sim] 同帧位姿不合法({})，已跳过第 {} 帧 (累计 {})", sim_io::to_string(pose_ok),
          bundle.frame_seq, pose_invalid_frames);
      }
      continue;
    }
    solver.set_R_gimbal2world(gimbal.q());

    if (mode == "closed_loop") {
      // runtime_state() 缺 CAP_RUNTIME_STATE 时返回 nullptr（而不是一块恒零的
      // RuntimeState），所以这里必须先把"不可知"和"明确没订阅"分开。
      //
      // 混在一起的代价是排查方向被带偏：曾经 following==0 一律打印"请按 F5 /
      // 设 DAEDALUS_FORCE_AUTO_AIM"，而当发布端根本不报这个字段时，这条建议怎么
      // 做都不会有任何变化。不可知期间置 FAULT_CAPABILITY_MISSING 禁止开火——
      // 闭环里"不知道命令有没有生效"就开火是最不该做的降级。
      const auto * rt = camera.client().runtime_state();
      if (rt == nullptr) {
        ++runtime_state_missing_frames;
        gimbal.set_fault(sim_io::FAULT_CAPABILITY_MISSING, true);
        if (!runtime_state_warned) {
          runtime_state_warned = true;
          tools::logger()->error(
            "[sim] 发布端未声明 CAP_RUNTIME_STATE（capabilities={}）：无法判断仿真端"
            "是否订阅云台命令，本次运行按不可知处理并置 capability_missing 禁止开火。"
            "这不是'仿真端没订阅'，请确认仿真端为协议 v{} 及以上",
            sim_io::describe_capabilities(camera.client().capabilities()), sim_io::SHM_VERSION);
        }
      } else {
        gimbal.set_fault(sim_io::FAULT_CAPABILITY_MISSING, false);
        const bool following = rt->following != 0;
        if (following) {
          if (!following_seen)
            tools::logger()->info("[sim] 仿真端已订阅云台命令 (RuntimeState.following=1)");
          following_seen = true;
        } else {
          ++not_following_frames;
          if (!following_warned) {
            following_warned = true;
            tools::logger()->warn(
              "[sim] 仿真端未订阅云台命令 (RuntimeState.following={})：本次下发的命令不会"
              "改变云台。请用 scripts/run.sh simulator-auto-aim 启动仿真端，或设"
              "DAEDALUS_FORCE_AUTO_AIM=1，或交互按 F5",
              static_cast<int>(rt->following));
          }
        }
      }
    }

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
      command_age_ms.push_back(gimbal.command_age_ms());
      state_age_ms.push_back(gimbal.state_age_ms());

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
    const bool target_lost = !ever_seen_target || lost_ms > target_lost_ms;
    gimbal.set_fault(sim_io::FAULT_TARGET_LOST, target_lost);

    // 重新武装：必须连续 rearm_confirm_frames 帧"这一帧有完整同步数据 + 跟上目标"
    // 才解除。中途只要断一次就从零重数，防止抖动式的一帧命中把开火放出去。
    if (rearm_pending) {
      if (target.has_value() && !target_lost) {
        if (++rearm_confirmed >= rearm_confirm_frames) {
          rearm_pending = false;
          tools::logger()->info(
            "[sim] 已重新武装（连续确认 {} 帧目标）", rearm_confirmed);
        }
      } else {
        rearm_confirmed = 0;
      }
      gimbal.set_fault(sim_io::FAULT_REARM_PENDING, rearm_pending);
    }

    // ---- 真值评估（真值只进评估器，绝不进算法输入）-----------------------------
    std::optional<Eigen::Vector3d> gt_pos_this_frame;
    std::optional<Eigen::Vector3d> gt_armor_this_frame;
    if (do_eval && target.has_value() && evaluator.fetch(bundle.frame_seq)) {
      const auto x = target->ekf_x();
      // Solver 的输出以云台为原点，真值以 odom 为原点，比较前必须加上同帧的云台平移。
      const Eigen::Vector3d in_odom =
        gimbal.odom_position() + Eigen::Vector3d(x[0], x[2], x[4]);
      const auto err = evaluator.evaluate(target->name, in_odom, x[6], x[7]);
      evaluator.record(err);
      if (err.valid) {
        gt_pos_this_frame = err.gt_position;
        if (err.has_armor_position) {
          gt_armor_this_frame = err.gt_armor_position;
        } else {
          ++aim_no_armor_gt;
        }
      }
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
      // 丢目标且开启驻留时，把云台摆到 park 指向而不是原地不动。
      // 这是**搜索**行为，不是瞄准：绝不带 fire（park_fire 恒 false），距离仍按
      // 无目标处理，plan.control 为真（有目标）时一律以规划结果为准，驻留只在
      // 无控制的空档里生效，因此不会与自瞄争夺控制权。
      const bool parking = park_enabled && !plan.control;
      const bool fired = parking
        ? gimbal.send(
            true, false, park_yaw_deg / 57.2957795130823,
            park_pitch_deg / 57.2957795130823, 0.0)
        : gimbal.send(
            plan.control, plan.fire, plan.yaw, plan.pitch,
            target.has_value() ? Eigen::Vector3d(
                                   target->ekf_x()[0], target->ekf_x()[2], target->ekf_x()[4])
                                   .norm()
                               : -1.0);
      if (parking) ++park_frames;
      (void)fired;
      command_age_ms.push_back(gimbal.command_age_ms());
      state_age_ms.push_back(gimbal.state_age_ms());
      // 几何瞄准误差。ROS 约定 R = Rz(yaw)Ry(pitch)，机体 x 轴前向，
      // 于是下发方向的单位向量是 (cos p cos y, cos p sin y, -sin p)。
      if (plan.control && gt_pos_this_frame.has_value()) {
        // 两个参考原点都取仿真端同帧发布的真值，不需要任何标定常数：
        //   pivot_world  = 云台回转中心世界位置，也是解算所用坐标系的原点；
        //   muzzle_world = 枪口世界位置，协议 v3 起由发布端直接给世界量。
        // 不要再写 `odom_position() + muzzle_*`：那是把局部量加到世界坐标上。
        const Eigen::Vector3d pivot_world = gimbal.odom_position();
        const Eigen::Vector3d muzzle_world = gimbal.muzzle_position();
        const Eigen::Vector3d cmd_dir(
          std::cos(plan.pitch) * std::cos(plan.yaw), std::cos(plan.pitch) * std::sin(plan.yaw),
          -std::sin(plan.pitch));

        // ROS 约定下"下发角 vs 目标方向"的分轴误差。pitch 正 = 低头。
        auto push_axis_errors = [&](const Eigen::Vector3d & dir) {
          const double gt_yaw = std::atan2(dir.y(), dir.x());
          double dyaw = plan.yaw - gt_yaw;
          while (dyaw > M_PI) dyaw -= 2 * M_PI;
          while (dyaw < -M_PI) dyaw += 2 * M_PI;
          aim_yaw_err_deg.push_back(dyaw * 57.2957795130823);
          const double gt_pitch = -std::asin(std::clamp(dir.z(), -1.0, 1.0));
          aim_pitch_err_deg.push_back((plan.pitch - gt_pitch) * 57.2957795130823);
        };
        auto angle_between = [](const Eigen::Vector3d & a, const Eigen::Vector3d & b) {
          return std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * 57.2957795130823;
        };

        // 主口径：**云台原点** -> 被选中装甲板板心真值。用云台原点而不是枪口，是
        // 因为下发角就是在这个原点下解出来的：拿枪口当原点会把 0.11m 视差算进
        // "瞄准误差"，而算法根本没有机会补偿它（它的输入坐标本身就是云台原点系）。
        // 物理出膛点那一份在下面单独记 aim_err_muzzle_deg。
        //
        // 板心真值不可用时（发布端没给，例如场景资产缺 CENTER 节点）不退化成整车
        // 中心充数，直接不记这一帧的 aim_*，只在 aim_no_armor_gt 里计数——否则
        // 统计里会混进两种口径。
        if (gt_armor_this_frame.has_value()) {
          const Eigen::Vector3d to_armor = *gt_armor_this_frame - pivot_world;
          if (to_armor.norm() > 1e-6) {
            const Eigen::Vector3d armor_dir = to_armor.normalized();
            aim_err_deg.push_back(angle_between(cmd_dir, armor_dir));
            push_axis_errors(armor_dir);
          }
          const Eigen::Vector3d to_armor_muzzle = *gt_armor_this_frame - muzzle_world;
          if (to_armor_muzzle.norm() > 1e-6)
            aim_err_muzzle_deg.push_back(
              angle_between(cmd_dir, to_armor_muzzle.normalized()));
        }

        // 旧口径：云台原点 -> 整车中心真值。它天然含板心-车心的几何差，不代表闭环
        // 精度。注意原点已从枪口改为云台原点，且改动前的枪口世界位置本身算错了，
        // 所以这一列与历史数据不可比。
        {
          const Eigen::Vector3d to_center = *gt_pos_this_frame - pivot_world;
          if (to_center.norm() > 1e-6)
            center_err_deg.push_back(angle_between(cmd_dir, to_center.normalized()));
        }

        // 枪口视差：同一块板心，分别从云台原点和枪口看过去的方向夹角。
        // 纯几何量，只用真值和仿真端同帧发布的位姿，不回灌算法。
        // 它同时解释了 aim_err_deg 与 aim_err_muzzle_deg 的差。
        if (gt_armor_this_frame.has_value()) {
          const Eigen::Vector3d from_pivot = *gt_armor_this_frame - pivot_world;
          const Eigen::Vector3d from_muzzle = *gt_armor_this_frame - muzzle_world;
          if (from_pivot.norm() > 1e-6 && from_muzzle.norm() > 1e-6) {
            muzzle_parallax_deg.push_back(
              angle_between(from_pivot.normalized(), from_muzzle.normalized()));
          }
        }

        // 几何/估计误差与弹道补偿分离，全部取自 planner **同一次**解算
        // （debug_aim_command，见 AimCommandDebug），不做任何反推：
        //
        //   geom_err   = 实际送进弹道解算的瞄准点方向 vs 板心真值方向，
        //                两个方向同以**云台原点**为原点 —— 纯几何+估计误差。
        //                注意用的是 debug_aim_command.aim_xyz 而不是 debug_xyza：
        //                后者是加 aim_z_compensation **之前**的点，不是真正被解算
        //                的那个点（planner.cpp resolve_aim_xyz）。
        //   ballistic_lift     = 重力抬枪量本身；
        //   ballistic_offset   = 标定 pitch_offset，与抬枪量分开报告；
        //   ballistic_identity = pitch-(geometric-lift-offset)，应恒等于 0，
        //                        用来证明这个分解不是拟合出来的。
        //   mpc_pitch_lag      = plan.pitch 减同一决策时刻的瞬时解。这一项属于 MPC
        //                        半时域输出与瞬时解的差，**不是**弹道补偿；旧代码
        //                        把它当成抬枪量，因此那些数无效。
        const auto & aim_dbg = planner->debug_aim_command;
        if (aim_dbg.valid) {
          mpc_pitch_lag_deg.push_back((plan.pitch - aim_dbg.pitch) * 57.2957795130823);
          ballistic_lift_deg.push_back(aim_dbg.gravity_lift * 57.2957795130823);
          ballistic_offset_deg.push_back(aim_dbg.pitch_offset * 57.2957795130823);
          ballistic_identity_deg.push_back(
            (aim_dbg.pitch -
             (aim_dbg.geometric_pitch - aim_dbg.gravity_lift - aim_dbg.pitch_offset)) *
            57.2957795130823);
        }
        if (gt_armor_this_frame.has_value() && aim_dbg.valid) {
          const Eigen::Vector3d aim_world = pivot_world + aim_dbg.aim_xyz;
          const Eigen::Vector3d to_aim = aim_world - pivot_world;
          const Eigen::Vector3d to_armor = *gt_armor_this_frame - pivot_world;
          if (to_aim.norm() > 1e-6 && to_armor.norm() > 1e-6) {
            geom_err_deg.push_back(
              angle_between(to_aim.normalized(), to_armor.normalized()));
          }
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
     << ", \"remaps\": " << client.remaps()
     << ", \"remap_failures\": " << client.remap_failures()
     << ", \"ground_truth_unsupported\": " << client.ground_truth_unsupported()
     << ", \"ground_truth_captures\": " << client.ground_truth_captures()
     << ", \"runtime_state_unsupported\": " << client.runtime_state_unsupported()
     << ", \"chassis_observation_unsupported\": " << client.chassis_observation_unsupported()
     << ", \"shm_version\": " << client.version()
     << ", \"capabilities\": " << client.capabilities()
     << ", \"capabilities_str\": \"" << sim_io::describe_capabilities(client.capabilities())
     << "\", \"last_seq\": " << client.last_seq()
     << "},\n";
  js << "  \"camera\": {\"stale\": " << camera.stale_frames()
     << ", \"rejected\": " << camera.rejected_frames()
     << ", \"future\": " << camera.future_frames()
     << ", \"clock_jumps\": " << camera.clock_jumps()
     << ", \"clock_jump_frames\": " << camera.clock_jump_frames()
     << ", \"reconnects\": " << camera.reconnects()
     << ", \"no_new_frame_timeout_ms\": " << camera.effective_no_new_frame_timeout_ms()
     << ", \"read_timeout_ms\": " << camera.effective_read_timeout_ms()
     << ", \"fps\": " << camera.camera_fps() << "},\n";
  js << "  \"detect_ms\": {\"p50\": " << percentile(detect_ms, 0.50)
     << ", \"p95\": " << percentile(detect_ms, 0.95) << ", \"p99\": " << percentile(detect_ms, 0.99)
     << "},\n";
  js << "  \"pipeline_ms\": {\"p50\": " << percentile(pipeline_ms, 0.50)
     << ", \"p95\": " << percentile(pipeline_ms, 0.95)
     << ", \"p99\": " << percentile(pipeline_ms, 0.99) << "},\n";
  // 两个年龄必须分开报告：state_age 就是 state_stale 的判据（本地保有时长），
  // command_age 是世界观测年龄。修复前它们是同一个数，state_stale 因此恒亮。
  js << "  \"command_age_ms\": {\"count\": " << command_age_ms.size()
     << ", \"p50\": " << percentile(command_age_ms, 0.50)
     << ", \"p95\": " << percentile(command_age_ms, 0.95)
     << ", \"p99\": " << percentile(command_age_ms, 0.99)
     << ", \"budget_ms\": " << gim_cfg.max_command_age_ms
     << ", \"budget_source\": \"" << command_age_budget_source << "\""
     << ", \"violations\": " << gimbal.command_age_violations() << "},\n";
  js << "  \"state_age_ms\": {\"count\": " << state_age_ms.size()
     << ", \"p50\": " << percentile(state_age_ms, 0.50)
     << ", \"p95\": " << percentile(state_age_ms, 0.95)
     << ", \"p99\": " << percentile(state_age_ms, 0.99)
     << ", \"timeout_ms\": " << gim_cfg.state_timeout_ms << "},\n";
  js << "  \"park\": {\"enabled\": " << (park_enabled ? "true" : "false")
     << ", \"frames\": " << park_frames << ", \"yaw_deg\": "
     << (park_enabled ? park_yaw_deg : 0.0) << ", \"pitch_deg\": "
     << (park_enabled ? park_pitch_deg : 0.0) << "},\n";
  js << "  \"perception\": {\"detected_frames\": " << detected_frames
     << ", \"tracked_frames\": " << tracked_frames << "},\n";
  js << "  \"gimbal\": {\"sent\": " << gimbal.sent_commands()
     << ", \"control\": " << control_cmds << ", \"plan_fire\": " << fire_cmds
     << ", \"fire\": " << gimbal.fire_commands()
     << ", \"suppressed_fire\": " << gimbal.suppressed_fires()
     << ", \"safe_stops\": " << gimbal.safe_stops()
     << ", \"invalid_poses\": " << gimbal.invalid_poses()
     << ", \"pose_invalid_frames\": " << pose_invalid_frames
     << ", \"rearm_events\": " << rearm_events
     << ", \"rearm_pending\": " << (rearm_pending ? "true" : "false")
     << ", \"sim_following_seen\": " << (following_seen ? "true" : "false")
     << ", \"sim_not_following_frames\": " << not_following_frames
     << ", \"runtime_state_missing_frames\": " << runtime_state_missing_frames
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
       << ", \"seq_mismatches\": " << evaluator.seq_mismatches()
       << ", \"seq_skew_mean\": " << evaluator.seq_skew_mean()
       << ", \"seq_skew_min\": " << evaluator.seq_skew_min()
       << ", \"seq_skew_max\": " << evaluator.seq_skew_max()
       << ", \"ambiguous_matches\": " << evaluator.ambiguous_matches()
       << ", \"enemy_team\": \"" << enemy_team_name(evaluator.enemy_team()) << "\"}";
    // 闭环瞄准误差。各口径分开发布，不做合并、不做反推：
    //   aim_*        枪口 -> 板心真值（主口径，仍含弹道抬枪量与预测提前量）
    //   center_*     枪口 -> 整车中心真值（旧口径，仅供与改动前对比）
    //   geom_*       planner 选中的瞄准点 -> 板心真值（几何+估计误差）
    //   ballistic_*  planner 实际加进 pitch 的抬枪量（含 pitch_offset_）
    // 无论哪个口径都不能单独当成"闭环精度"：aim_* 含弹道，geom_* 不含控制误差，
    // 而选板逻辑本身在评估端只是几何代理（取距相机最近的板），换板瞬间会不一致。
    auto mean = [](const std::vector<double> & v) {
      if (v.empty()) return 0.0;
      double s = 0.0;
      for (double x : v) s += x;
      return s / static_cast<double>(v.size());
    };
    if (!aim_err_deg.empty()) {
      const auto yaw_mm = std::minmax_element(aim_yaw_err_deg.begin(), aim_yaw_err_deg.end());
      const auto pitch_mm =
        std::minmax_element(aim_pitch_err_deg.begin(), aim_pitch_err_deg.end());
      js << ",\n  \"aim_error_deg\": {\"count\": " << aim_err_deg.size()
         << ", \"reference\": \"gimbal_pivot_world_to_gt_armor_center\""
         << ", \"p50\": " << percentile(aim_err_deg, 0.50)
         << ", \"p95\": " << percentile(aim_err_deg, 0.95)
         << ", \"max\": " << percentile(aim_err_deg, 1.0) << ", \"mean\": " << mean(aim_err_deg)
         << ", \"yaw_mean\": " << mean(aim_yaw_err_deg) << ", \"yaw_min\": " << *yaw_mm.first
         << ", \"yaw_max\": " << *yaw_mm.second << ", \"pitch_mean\": " << mean(aim_pitch_err_deg)
         << ", \"pitch_min\": " << *pitch_mm.first << ", \"pitch_max\": " << *pitch_mm.second
         << ", \"missing_armor_gt\": " << aim_no_armor_gt << "}";
    }
    if (!aim_err_muzzle_deg.empty()) {
      js << ",\n  \"aim_error_muzzle_deg\": {\"count\": " << aim_err_muzzle_deg.size()
         << ", \"reference\": \"muzzle_world_to_gt_armor_center\""
         << ", \"p50\": " << percentile(aim_err_muzzle_deg, 0.50)
         << ", \"p95\": " << percentile(aim_err_muzzle_deg, 0.95)
         << ", \"mean\": " << mean(aim_err_muzzle_deg) << "}";
    }
    if (!center_err_deg.empty()) {
      js << ",\n  \"aim_error_vehicle_center_deg\": {\"count\": " << center_err_deg.size()
         << ", \"reference\": \"gimbal_pivot_world_to_gt_vehicle_center\""
         << ", \"p50\": " << percentile(center_err_deg, 0.50)
         << ", \"p95\": " << percentile(center_err_deg, 0.95)
         << ", \"mean\": " << mean(center_err_deg) << "}";
    }
    if (!geom_err_deg.empty()) {
      js << ",\n  \"aim_point_geometry_err_deg\": {\"count\": " << geom_err_deg.size()
         << ", \"reference\": \"gimbal_pivot_world; point=solved_aim_xyz\""
         << ", \"p50\": " << percentile(geom_err_deg, 0.50)
         << ", \"p95\": " << percentile(geom_err_deg, 0.95)
         << ", \"mean\": " << mean(geom_err_deg) << "}";
    }
    if (!muzzle_parallax_deg.empty()) {
      js << ",\n  \"muzzle_parallax_deg\": {\"count\": " << muzzle_parallax_deg.size()
         << ", \"p50\": " << percentile(muzzle_parallax_deg, 0.50)
         << ", \"p95\": " << percentile(muzzle_parallax_deg, 0.95)
         << ", \"mean\": " << mean(muzzle_parallax_deg) << "}";
    }
    // 弹道抬枪量。定义写进 JSON 里，避免以后又被当成别的东西引用：同一次解算、
    // 同一预测时刻（命中时刻）、同一瞄准点、同一原点（云台原点）。
    if (!ballistic_lift_deg.empty()) {
      const auto mm = std::minmax_element(ballistic_lift_deg.begin(), ballistic_lift_deg.end());
      const auto id_mm =
        std::minmax_element(ballistic_identity_deg.begin(), ballistic_identity_deg.end());
      js << ",\n  \"ballistic_gravity_lift_deg\": {\"count\": " << ballistic_lift_deg.size()
         << ", \"definition\": \"trajectory_pitch_minus_geometric_elevation;"
            " same_solve_same_instant_same_point_origin=gimbal_pivot\""
         << ", \"mean\": " << mean(ballistic_lift_deg) << ", \"min\": " << *mm.first
         << ", \"max\": " << *mm.second
         << ", \"pitch_offset_deg\": " << mean(ballistic_offset_deg)
         << ", \"identity_residual_max_abs_deg\": "
         << std::max(std::abs(*id_mm.first), std::abs(*id_mm.second)) << "}";
    }
    // 与弹道无关：MPC 半时域状态减同一决策时刻的瞬时解。旧的 ballistic_pitch_deg
    // 算的就是这一项（还混了不同原点），所以那些数不能当抬枪量看。
    if (!mpc_pitch_lag_deg.empty()) {
      const auto mm = std::minmax_element(mpc_pitch_lag_deg.begin(), mpc_pitch_lag_deg.end());
      js << ",\n  \"mpc_pitch_lag_deg\": {\"count\": " << mpc_pitch_lag_deg.size()
         << ", \"definition\": \"plan_pitch_minus_instantaneous_solve_same_decision\""
         << ", \"mean\": " << mean(mpc_pitch_lag_deg) << ", \"min\": " << *mm.first
         << ", \"max\": " << *mm.second << "}";
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
