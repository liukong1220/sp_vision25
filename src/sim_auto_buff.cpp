// Software-only power-rune validation entry point.
//
// The detector/solver/target/aimer implementation is shared with the hardware path.  The
// only I/O in this binary is SimCamera + SimGimbal over Talos shared memory; truth is read
// after frame consumption and is never passed to the algorithm objects.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "simulation/io/sim_camera.hpp"
#include "simulation/io/dynamic_budget.hpp"
#include "simulation/io/sim_gimbal.hpp"
#include "simulation/io/report_metadata.hpp"
#include "simulation/io/rune_association.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_mode_classifier.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/path.hpp"
#include "tools/yaml.hpp"
#include "tools/math_tools.hpp"

namespace
{
const std::string keys =
  "{help h usage ? | | output command line help}"
  "{@config-path | configs/simulation.yaml | yaml configuration path}"
  "{mode | closed_loop | passive / closed_loop}"
  "{task | small_buff | small_buff / big_buff / all}"
  "{duration-s | 0.0 | run duration, zero means until Ctrl-C}"
  "{allow-fire | | allow virtual firing (default disabled)}"
  "{eval | | evaluate against same-frame rune truth}"
  "{report | | output JSON report}"
  "{max-command-age-ms | -1 | override sim.max_command_age_ms}";

double now_ms(const std::chrono::steady_clock::time_point & start)
{
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now() - start)
    .count();
}

template<typename T>
nlohmann::json sample_stats(const std::vector<T> & values)
{
  if (values.empty()) return nullptr;
  std::vector<double> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  auto q = [&](double p) {
    const double x = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(x));
    const auto hi = static_cast<std::size_t>(std::ceil(x));
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - static_cast<double>(lo));
  };
  return {{"count", sorted.size()}, {"p50", q(0.50)}, {"p95", q(0.95)}, {"max", q(1.0)}};
}

double vector_max(const std::vector<double> & values)
{
  if (values.empty()) return std::numeric_limits<double>::infinity();
  return *std::max_element(values.begin(), values.end());
}

struct RuneStats
{
  std::uint64_t attempts = 0;
  std::uint64_t same_frame = 0;
  std::uint64_t missing = 0;
  std::uint64_t mismatch = 0;
  std::uint64_t timestamp_mismatch = 0;
  std::uint64_t ambiguous = 0;
  std::uint64_t nearest = 0;
  std::uint64_t degraded = 0;
  std::uint64_t valid = 0;
  std::uint64_t activation_samples = 0;
  std::uint64_t blade_id_samples = 0;
  std::uint64_t direction_mismatch = 0;
  std::vector<double> radius;
  std::vector<double> angle;
  std::vector<double> speed;
  std::vector<double> center_error_m;
  std::vector<double> angle_error_rad;
  std::vector<double> speed_error_radps;
  std::vector<double> predicted_phase;
  std::vector<double> phase_error_rad;
  std::vector<double> commanded_direction;
  std::vector<double> truth_direction;
  std::vector<double> gimbal_error_deg;
  bool has_center_sample = false;
  std::uint64_t center_timestamp_ns = 0;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double max_center_translation_speed_mps = 0.0;
};

struct GroundTruthRuneEvaluator : RuneStats
{
};

struct RuneEvalSample
{
  std::optional<Eigen::Vector3d> estimated_center;
  std::optional<double> current_angle;
  std::optional<double> current_speed;
  std::optional<double> predicted_phase;
  std::optional<double> commanded_yaw;
  std::optional<double> commanded_pitch;
  std::optional<double> measured_yaw;
  std::optional<double> measured_pitch;
};

void evaluate_rune(
  sim_io::SharedMemoryClient & client, std::uint64_t frame_seq, std::uint64_t timestamp_ns,
  auto_buff::PowerRuneMode algorithm_mode, const RuneEvalSample & sample,
  GroundTruthRuneEvaluator & stats)
{
  ++stats.attempts;
  sim_io::GroundTruthBatch batch{};
  if (!client.frame_ground_truth(&batch)) {
    ++stats.missing;
    return;
  }
  if (batch.frame_seq != frame_seq) {
    ++stats.mismatch;
    return;
  }
  if (batch.timestamp_ns != timestamp_ns) {
    ++stats.timestamp_mismatch;
    return;
  }
  ++stats.same_frame;
  const int wanted = algorithm_mode == auto_buff::PowerRuneMode::Small
    ? 0
    : algorithm_mode == auto_buff::PowerRuneMode::Big ? 1 : -1;
  sim_io::GroundTruthRune same_frame[sim_io::GROUND_TRUTH_MAX_RUNES];
  std::uint32_t n_same = 0;
  for (std::uint32_t i = 0; i < batch.rune_count && i < sim_io::GROUND_TRUTH_MAX_RUNES; ++i) {
    const auto & rune = batch.runes[i];
    if (rune.frame_seq != frame_seq || rune.timestamp_ns != timestamp_ns) {
      ++stats.timestamp_mismatch;
      continue;
    }
    same_frame[n_same++] = rune;
  }
  const Eigen::Vector3d * estimate =
    sample.estimated_center.has_value() ? &*sample.estimated_center : nullptr;
  const auto assoc =
    sim_io::associate_rune_by_mode(same_frame, n_same, wanted, estimate);
  if (assoc.selected == nullptr) {
    if (wanted >= 0 && batch.rune_count > 0) ++stats.nearest;
    else if (wanted >= 0) ++stats.missing;
    return;
  }
  if (assoc.ambiguous) ++stats.ambiguous;
  const sim_io::GroundTruthRune * selected = assoc.selected;
  const double radius = selected->radius;
  const double speed = selected->v_roll;
  const bool finite = std::isfinite(radius) && std::isfinite(selected->current_angle) &&
    std::isfinite(speed) && std::isfinite(selected->r_center_odom[0]) &&
    std::isfinite(selected->r_center_odom[1]) && std::isfinite(selected->r_center_odom[2]);
  bool activations_valid = true;
  for (const auto activation : selected->target_activations)
    activations_valid = activations_valid && activation <= 3;
  const bool valid = finite && radius > 1e-4 && selected->blade_id >= 0 &&
    selected->blade_id < 5 && activations_valid;
  if (!valid) {
    ++stats.degraded;
    return;
  }
  ++stats.valid;
  ++stats.activation_samples;
  ++stats.blade_id_samples;
  stats.radius.push_back(radius);
  stats.angle.push_back(selected->current_angle);
  stats.speed.push_back(speed);
  const Eigen::Vector3d center(
    selected->r_center_odom[0], selected->r_center_odom[1], selected->r_center_odom[2]);
  if (stats.has_center_sample && selected->timestamp_ns > stats.center_timestamp_ns) {
    const double dt = static_cast<double>(selected->timestamp_ns - stats.center_timestamp_ns) * 1e-9;
    stats.max_center_translation_speed_mps =
      std::max(stats.max_center_translation_speed_mps, (center - stats.center).norm() / dt);
  }
  stats.center = center;
  stats.center_timestamp_ns = selected->timestamp_ns;
  stats.has_center_sample = true;
  if (sample.estimated_center.has_value())
    stats.center_error_m.push_back((*sample.estimated_center - center).norm());
  if (sample.current_angle.has_value())
    stats.angle_error_rad.push_back(
      std::abs(tools::limit_rad(*sample.current_angle - selected->current_angle)));
  if (sample.current_speed.has_value()) {
    stats.speed_error_radps.push_back(std::abs(*sample.current_speed - speed));
    const int estimated_dir =
      *sample.current_speed > 1e-9 ? 1 : *sample.current_speed < -1e-9 ? -1 : 0;
    const int truth_dir = selected->direction > 0 ? 1 : selected->direction < 0 ? -1 : 0;
    stats.commanded_direction.push_back(static_cast<double>(estimated_dir));
    stats.truth_direction.push_back(static_cast<double>(truth_dir));
    if (estimated_dir != 0 && truth_dir != 0 && estimated_dir != truth_dir)
      ++stats.direction_mismatch;
  }
  if (sample.predicted_phase.has_value()) {
    stats.predicted_phase.push_back(*sample.predicted_phase);
    stats.phase_error_rad.push_back(
      std::abs(tools::limit_rad(*sample.predicted_phase - selected->sin_phase)));
  }
  if (sample.commanded_yaw.has_value() && sample.commanded_pitch.has_value() &&
      sample.measured_yaw.has_value() && sample.measured_pitch.has_value()) {
    const double yaw_err = tools::limit_rad(*sample.commanded_yaw - *sample.measured_yaw);
    const double pitch_err = tools::limit_rad(*sample.commanded_pitch - *sample.measured_pitch);
    stats.gimbal_error_deg.push_back(std::hypot(yaw_err, pitch_err) * 180.0 / CV_PI);
  }
}

sim_io::SimCameraConfig camera_config(const YAML::Node & sim)
{
  sim_io::SimCameraConfig cfg;
  cfg.max_frame_age_ms = tools::read_or<double>(sim, "max_frame_age_ms", cfg.max_frame_age_ms);
  cfg.max_future_frame_ms =
    tools::read_or<double>(sim, "max_future_frame_ms", cfg.max_future_frame_ms);
  cfg.heartbeat_timeout_ms = tools::read_or<double>(sim, "heartbeat_timeout_ms", cfg.heartbeat_timeout_ms);
  cfg.read_timeout_ms = tools::read_or<double>(sim, "read_timeout_ms", cfg.read_timeout_ms);
  cfg.no_new_frame_timeout_ms = tools::read_or<double>(sim, "no_new_frame_timeout_ms", cfg.no_new_frame_timeout_ms);
  cfg.remap_check_ms = tools::read_or<double>(sim, "remap_check_ms", cfg.remap_check_ms);
  cfg.shm.dir = tools::read_or<std::string>(sim, "shm_dir", cfg.shm.dir);
  return cfg;
}

sim_io::SimGimbalConfig gimbal_config(
  const YAML::Node & root, const YAML::Node & sim, const cv::CommandLineParser & cli)
{
  sim_io::SimGimbalConfig cfg;
  cfg.yaw_scale = tools::read_or<double>(sim, "yaw_scale", cfg.yaw_scale);
  cfg.yaw_offset_deg = tools::read_or<double>(sim, "yaw_offset_deg", cfg.yaw_offset_deg);
  cfg.pitch_scale = tools::read_or<double>(sim, "pitch_scale", cfg.pitch_scale);
  cfg.pitch_offset_deg = tools::read_or<double>(sim, "pitch_offset_deg", cfg.pitch_offset_deg);
  cfg.feedback_pitch_fix_deg = tools::read_or<double>(sim, "feedback_pitch_fix_deg", cfg.feedback_pitch_fix_deg);
  cfg.state_timeout_ms = tools::read_or<double>(sim, "state_timeout_ms", cfg.state_timeout_ms);
  cfg.max_command_age_ms = tools::read_or<double>(sim, "max_command_age_ms", cfg.max_command_age_ms);
  const double override_ms = cli.get<double>("max-command-age-ms");
  if (override_ms >= 0.0) cfg.max_command_age_ms = override_ms;
  cfg.bullet_speed = tools::read_or<double>(root, "bullet_speed_fallback", cfg.bullet_speed);
  cfg.allow_fire = cli.has("allow-fire");
  return cfg;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const std::string config_path = tools::resolve_config_path_string(cli.get<std::string>("@config-path"));
  const std::string mode = cli.get<std::string>("mode");
  const std::string task = cli.get<std::string>("task");
  if (mode != "passive" && mode != "closed_loop") {
    tools::logger()->error("[sim-buff] invalid mode: {}", mode);
    return 2;
  }
  if (task != "small_buff" && task != "big_buff" && task != "all") {
    tools::logger()->error("[sim-buff] invalid task: {}", task);
    return 2;
  }
  auto yaml = tools::load(config_path);
  const auto sim = yaml["sim"];
  auto cam_cfg = camera_config(sim);
  auto gim_cfg = gimbal_config(yaml, sim, cli);
  if (mode != "closed_loop") gim_cfg.allow_fire = false;
  if (cli.has("allow-fire") && mode != "closed_loop") {
    tools::logger()->error("[sim-buff] --allow-fire is valid only in closed_loop mode");
    return 2;
  }
  if (gim_cfg.allow_fire && gim_cfg.max_command_age_ms <= 0.0) {
    tools::logger()->error("[sim-buff] --allow-fire requires positive sim.max_command_age_ms");
    return 2;
  }

  sim_io::SimCamera camera(cam_cfg);
  std::string error;
  if (!camera.open(&error)) {
    tools::logger()->error("[sim-buff] shared memory connection failed: {}", error);
    return 3;
  }
  sim_io::SimGimbal gimbal(camera.client(), gim_cfg);
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::RuneModeClassifier mode_classifier;
  std::unique_ptr<auto_buff::Target> target;
  std::unique_ptr<auto_buff::Aimer> aimer;
  auto_buff::PowerRuneMode active_mode = auto_buff::PowerRuneMode::Unknown;
  std::uint64_t small_estimator_instances = 0;
  std::uint64_t big_estimator_instances = 0;
  auto activate_mode = [&](auto_buff::PowerRuneMode selected) {
    target.reset();
    aimer.reset();
    active_mode = selected;
    if (selected == auto_buff::PowerRuneMode::Small) {
      target = std::make_unique<auto_buff::SmallTarget>();
      ++small_estimator_instances;
    } else if (selected == auto_buff::PowerRuneMode::Big) {
      target = std::make_unique<auto_buff::BigTarget>();
      ++big_estimator_instances;
    }
    if (target) aimer = std::make_unique<auto_buff::Aimer>(config_path);
  };
  const auto requested_mode = task == "small_buff"
    ? auto_buff::PowerRuneMode::Small
    : task == "big_buff" ? auto_buff::PowerRuneMode::Big : auto_buff::PowerRuneMode::Unknown;
  if (requested_mode != auto_buff::PowerRuneMode::Unknown) activate_mode(requested_mode);
  tools::Exiter exiter;

  const auto start = std::chrono::steady_clock::now();
  std::vector<double> source_stage, source_to_detection, detection_to_planning, planning_to_command;
  std::vector<double> detection_start_stage, detection_end_stage, planning_stage, command_publish_stage;
  std::uint64_t frames = 0, detector_found = 0, solver_valid = 0, target_valid = 0;
  std::uint64_t plan_control = 0, plan_fire = 0, sent_fire = 0;
  std::uint64_t command_published = 0;
  std::uint64_t routed_observations = 0, dual_feed_frames = 0, wrong_mode_routes = 0;
  std::uint64_t following_frames = 0, not_following_frames = 0;
  std::uint64_t runtime_missing_frames = 0, runtime_mismatch_frames = 0;
  std::uint64_t rearm_events = 0;
  const int rearm_confirm_frames =
    std::max(1, tools::read_or<int>(sim, "rearm_confirm_frames", 5));
  int rearm_confirmed = 0;
  bool rearm_pending = true;
  std::uint64_t pose_invalid_frames = 0;
  std::uint64_t dynamic_samples = 0, dynamic_violations = 0;
  const double dynamic_max_angle_error_deg =
    tools::read_or<double>(sim, "dynamic_max_angle_error_deg", 5.0);
  const double dynamic_max_position_error_m =
    tools::read_or<double>(sim, "dynamic_max_position_error_m", 0.25);
  const double controlled_max_yaw_rate_radps =
    tools::read_or<double>(sim, "controlled_max_yaw_rate_radps", 3.0);
  const double controlled_max_translation_speed_mps =
    tools::read_or<double>(sim, "controlled_max_translation_speed_mps", 2.0);
  const double controlled_max_target_rotation_radps =
    tools::read_or<double>(sim, "controlled_max_target_rotation_radps", 2.51);
  const double controlled_max_target_translation_speed_mps =
    tools::read_or<double>(sim, "controlled_max_target_translation_speed_mps", 0.5);
  const double command_to_consume_delay_s =
    tools::read_or<double>(sim, "command_to_consume_delay_s", 0.02);
  auto env_flag_on = [](const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "True";
  };
  const bool synthetic_enabled = env_flag_on("DAEDALUS_CONTROLLED_MOTION") ||
    env_flag_on("DAEDALUS_SYNTHETIC_OFFSETS");
  const char * synthetic_source = env_flag_on("DAEDALUS_CONTROLLED_MOTION")
    ? "DAEDALUS_CONTROLLED_MOTION"
    : (env_flag_on("DAEDALUS_SYNTHETIC_OFFSETS") ? "DAEDALUS_SYNTHETIC_OFFSETS" : "none");
  double max_yaw_rate = 0.0, max_chassis_translation = 0.0, max_target_rotation = 0.0;
  std::vector<double> source_to_consume, send_to_consume;
  struct SentCommand
  {
    std::uint64_t source_timestamp_ns = 0;
    std::uint64_t send_timestamp_ns = 0;
  };
  std::map<std::uint64_t, SentCommand> sent_commands;
  GroundTruthRuneEvaluator rune_stats;
  sim_io::RuntimeState initial_runtime{}, last_runtime{};
  bool has_initial_runtime = camera.client().read_runtime_state(&initial_runtime);
  bool has_runtime = has_initial_runtime;
  if (has_runtime) last_runtime = initial_runtime;

  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  while (!exiter.exit()) {
    const double duration_s = cli.get<double>("duration-s");
    if (duration_s > 0.0 && now_ms(start) >= duration_s * 1000.0) break;
    gimbal.sample_faults();
    const auto status = camera.read_blocking(image, timestamp);
    gimbal.set_fault(sim_io::FAULT_HEARTBEAT_LOST, !camera.heartbeat_alive());
    gimbal.set_fault(sim_io::FAULT_NO_NEW_FRAME, camera.no_new_frame());
    gimbal.set_fault(
      sim_io::FAULT_FRAME_FAULT,
      status == sim_io::ReadStatus::Rejected || status == sim_io::ReadStatus::Stale ||
        status == sim_io::ReadStatus::Reconnected);
    gimbal.set_fault(sim_io::FAULT_CLOCK_JUMP, status == sim_io::ReadStatus::ClockJump);
    if (
      status == sim_io::ReadStatus::Reconnected || status == sim_io::ReadStatus::ClockJump ||
      status == sim_io::ReadStatus::Rejected || status == sim_io::ReadStatus::Stale) {
      mode_classifier.reset();
      activate_mode(requested_mode);
      gimbal.reset_history();
      rearm_pending = true;
      rearm_confirmed = 0;
      ++rearm_events;
      gimbal.set_fault(sim_io::FAULT_TARGET_LOST, true);
    }
    gimbal.set_fault(sim_io::FAULT_REARM_PENDING, rearm_pending);
    if (status != sim_io::ReadStatus::Ok) {
      gimbal.send_safe_stop();
      continue;
    }
    const auto frame_start = std::chrono::steady_clock::now();
    const auto & bundle = camera.last_bundle();
    const auto pose_status = gimbal.update(bundle, camera.last_stamps());
    if (pose_status != sim_io::PoseValidity::Ok) {
      ++pose_invalid_frames;
      mode_classifier.reset();
      activate_mode(requested_mode);
      rearm_pending = true;
      rearm_confirmed = 0;
      ++rearm_events;
      gimbal.set_fault(sim_io::FAULT_POSE_INVALID, true);
      gimbal.send_safe_stop();
      continue;
    }
    ++frames;
    solver.set_R_gimbal2world(gimbal.q());

    sim_io::RuntimeState runtime{};
    const bool has_runtime_capability =
      camera.client().has_capability(sim_io::CAP_RUNTIME_STATE);
    const bool runtime_read = camera.client().read_runtime_state(&runtime);
    const bool runtime_same_frame = runtime_read && runtime.frame_seq == bundle.frame_seq &&
      runtime.timestamp_ns == bundle.timestamp_ns;
    const bool following = runtime_same_frame && runtime.following != 0;
    gimbal.set_fault(sim_io::FAULT_CAPABILITY_MISSING, !has_runtime_capability);
    gimbal.set_fault(
      sim_io::FAULT_FRAME_FAULT, has_runtime_capability && !runtime_same_frame);
    gimbal.set_fault(sim_io::FAULT_NOT_FOLLOWING, !following);
    if (!runtime_read) ++runtime_missing_frames;
    else if (!runtime_same_frame) ++runtime_mismatch_frames;
    if (following) ++following_frames;
    else ++not_following_frames;

    if (runtime_read) {
      has_runtime = true;
      last_runtime = runtime;
      if (!has_initial_runtime) {
        initial_runtime = runtime;
        has_initial_runtime = true;
      }
      const auto acknowledged = sent_commands.find(runtime.last_command_seq);
      if (acknowledged != sent_commands.end() && runtime.last_command_consume_timestamp_ns != 0) {
        if (runtime.last_command_consume_timestamp_ns >= acknowledged->second.source_timestamp_ns)
          source_to_consume.push_back(
            static_cast<double>(runtime.last_command_consume_timestamp_ns -
                                acknowledged->second.source_timestamp_ns) /
            1e6);
        if (runtime.last_command_consume_timestamp_ns >= acknowledged->second.send_timestamp_ns)
          send_to_consume.push_back(
            static_cast<double>(runtime.last_command_consume_timestamp_ns -
                                acknowledged->second.send_timestamp_ns) /
            1e6);
        sent_commands.erase(sent_commands.begin(), std::next(acknowledged));
      }
    }
    source_stage.push_back(
      std::chrono::duration<double, std::milli>(frame_start - timestamp).count());
    const auto detect_begin = std::chrono::steady_clock::now();
    detection_start_stage.push_back(
      std::chrono::duration<double, std::milli>(detect_begin - frame_start).count());
    auto rune = detector.detect(image, requested_mode);
    const auto detect_end = std::chrono::steady_clock::now();
    detection_end_stage.push_back(
      std::chrono::duration<double, std::milli>(detect_end - frame_start).count());
    // Compute this only after detector completion.  Measuring it at detection_start would
    // under-report the required source-to-detection latency by the detector runtime itself.
    source_to_detection.push_back(
      std::chrono::duration<double, std::milli>(detect_end - timestamp).count());
    if (rune.has_value()) ++detector_found;
    solver.solve(rune);
    if (rune.has_value() && !rune->is_unsolve()) ++solver_valid;
    if (task == "all" && rune.has_value() && !rune->is_unsolve()) {
      const auto classified = mode_classifier.observe(*rune, timestamp);
      if (classified != auto_buff::PowerRuneMode::Unknown && classified != active_mode) {
        activate_mode(classified);
        rearm_pending = true;
        rearm_confirmed = 0;
        gimbal.set_fault(sim_io::FAULT_REARM_PENDING, true);
      }
    }

    std::optional<auto_buff::PowerRune> routed;
    if (rune.has_value() && rune->mode() == active_mode && target) {
      routed = rune;
      ++routed_observations;
    } else if (
      rune.has_value() && rune->mode() != auto_buff::PowerRuneMode::Unknown && target) {
      ++wrong_mode_routes;
    }
    if (target) target->get_target(routed, timestamp);
    const bool has_target = target && !target->is_unsolve();
    if (has_target) ++target_valid;
    detection_to_planning.push_back(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - detect_begin).count());

    io::Command command{false, false, 0.0, 0.0};
    double target_distance = -1.0;
    if (mode == "closed_loop" && has_target && aimer) {
      auto target_copy = target->clone();
      command = aimer->aim(*target_copy, timestamp, gimbal.state().bullet_speed, true);
      const auto state = target->ekf_x();
      if (state.size() > 3) target_distance = state[3];
    }

    bool dynamic_ok = false;
    sim_io::ChassisObservation chassis{};
    const bool chassis_same_frame = camera.client().read_chassis_observation(&chassis) &&
      chassis.frame_seq == bundle.frame_seq && chassis.timestamp_ns == bundle.timestamp_ns;
    if (command.control && has_target && chassis_same_frame) {
      const auto state = target->ekf_x();
      const double age_s = gimbal.command_age_ms() / 1000.0;
      const double yaw_rate = std::abs(static_cast<double>(chassis.wz_radps));
      const double chassis_translation =
        std::hypot(static_cast<double>(chassis.v_body[0]), static_cast<double>(chassis.v_body[1]));
      const double target_rotation = state.size() > 6 ? std::abs(state[6]) :
        std::numeric_limits<double>::infinity();
      const double range = state.size() > 3 ? std::max(0.1, std::abs(state[3])) : 0.1;
      const double consume_delay_s = sim_io::resolve_command_to_consume_delay_s(
        command_to_consume_delay_s, send_to_consume);
      sim_io::DynamicMotionInput motion;
      motion.age_s = age_s;
      motion.command_to_consume_delay_s = consume_delay_s;
      motion.chassis_yaw_rate_radps = yaw_rate;
      motion.chassis_translation_speed_mps = chassis_translation;
      motion.target_rotation_radps = target_rotation;
      motion.target_translation_speed_mps = controlled_max_target_translation_speed_mps;
      motion.target_range_m = range;
      motion.target_radius_m = 0.7;
      const auto bound = sim_io::conservative_dynamic_bound(motion);
      const double angle_bound_deg = bound.angle_error_deg;
      const double position_bound = bound.position_error_m;
      dynamic_ok = bound.finite &&
        yaw_rate <= controlled_max_yaw_rate_radps &&
        chassis_translation <= controlled_max_translation_speed_mps &&
        target_rotation <= controlled_max_target_rotation_radps &&
        sim_io::observed_target_translation_within_assumed(
          rune_stats.max_center_translation_speed_mps,
          controlled_max_target_translation_speed_mps) &&
        angle_bound_deg <= dynamic_max_angle_error_deg &&
        position_bound <= dynamic_max_position_error_m;
      ++dynamic_samples;
      if (!dynamic_ok) ++dynamic_violations;
      max_yaw_rate = std::max(max_yaw_rate, yaw_rate);
      max_chassis_translation = std::max(max_chassis_translation, chassis_translation);
      max_target_rotation = std::max(max_target_rotation, target_rotation);
    }
    gimbal.set_fault(sim_io::FAULT_DYNAMIC_ERROR, command.control && !dynamic_ok);

    if (rearm_pending) {
      if (has_target && command.control && following && dynamic_ok) {
        if (++rearm_confirmed >= rearm_confirm_frames) rearm_pending = false;
      } else {
        rearm_confirmed = 0;
      }
      gimbal.set_fault(sim_io::FAULT_REARM_PENDING, rearm_pending);
    }
    const auto planning_done = std::chrono::steady_clock::now();
    planning_stage.push_back(
      std::chrono::duration<double, std::milli>(planning_done - frame_start).count());
    if (command.control) ++plan_control;
    if (command.shoot) ++plan_fire;
    gimbal.set_fault(sim_io::FAULT_TARGET_LOST, !command.control);
    bool published = false;
    if (mode == "passive" || !following) {
      published = gimbal.send_safe_stop();
    } else {
      published = gimbal.send(
        command.control, command.shoot && gim_cfg.allow_fire, command.yaw, command.pitch,
        target_distance);
    }
    if (published) {
      ++command_published;
      if (gimbal.last_command().fire_advice == 1) ++sent_fire;
      sent_commands[gimbal.last_command_seq()] =
        {bundle.timestamp_ns, gimbal.last_command().timestamp_ns};
    }
    command_publish_stage.push_back(now_ms(frame_start));
    planning_to_command.push_back(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_done).count());

    // Truth is evaluated only after routing, planning and command publication. No truth object is
    // reachable from the classifier, Target or Aimer paths above.
    if (cli.has("eval")) {
      RuneEvalSample sample;
      if (has_target) {
        const auto state = target->ekf_x();
        if (state.size() > 3 && std::isfinite(state[0]) && std::isfinite(state[2]) &&
            std::isfinite(state[3])) {
          sample.estimated_center = tools::ypd2xyz(Eigen::Vector3d(state[0], state[2], state[3]));
        }
        if (state.size() > 5 && std::isfinite(state[5])) sample.current_angle = state[5];
        if (state.size() > 6 && std::isfinite(state[6])) sample.current_speed = state[6];
      } else if (routed.has_value()) {
        sample.estimated_center = routed->xyz_in_world;
        sample.current_angle = routed->ypr_in_world[2];
      }
      if (command.control) {
        sample.commanded_yaw = command.yaw;
        sample.commanded_pitch = command.pitch;
      }
      if (gimbal.has_state()) {
        sample.measured_yaw = gimbal.yaw();
        sample.measured_pitch = gimbal.pitch();
      }
      evaluate_rune(
        camera.client(), bundle.frame_seq, bundle.timestamp_ns, active_mode, sample, rune_stats);
    }
    if (command.control &&
        !sim_io::observed_target_translation_within_assumed(
          rune_stats.max_center_translation_speed_mps,
          controlled_max_target_translation_speed_mps)) {
      if (dynamic_ok) ++dynamic_violations;
      dynamic_ok = false;
      gimbal.set_fault(sim_io::FAULT_DYNAMIC_ERROR, true);
    }
    (void)frame_start;
  }
  gimbal.send_safe_stop();

  nlohmann::json report;
  report["entry"] = "sim_auto_buff";
  report["mode"] = mode;
  report["task"] = task;
  report["allow_fire"] = gim_cfg.allow_fire;
  report["frames_ok"] = frames;
  report["detector_found"] = detector_found;
  report["solver_valid"] = solver_valid;
  report["target_valid"] = target_valid;
  report["plan_control"] = plan_control;
  report["plan_fire"] = plan_fire;
  const auto counter_delta = [&](std::uint32_t current, std::uint32_t initial) {
    return current >= initial ? static_cast<std::uint64_t>(current - initial) : 0u;
  };
  const std::uint64_t consumed_delta = has_runtime && has_initial_runtime
    ? counter_delta(last_runtime.consumed_commands, initial_runtime.consumed_commands)
    : 0;
  const std::uint64_t consumed_control_delta = has_runtime && has_initial_runtime
    ? counter_delta(
        last_runtime.consumed_control_commands, initial_runtime.consumed_control_commands)
    : 0;
  const std::uint64_t consumed_fire_delta = has_runtime && has_initial_runtime
    ? counter_delta(
        last_runtime.consumed_fire_commands, initial_runtime.consumed_fire_commands)
    : 0;
  const std::uint64_t launch_delta = has_runtime && has_initial_runtime
    ? counter_delta(last_runtime.projectile_launch, initial_runtime.projectile_launch)
    : 0;
  const std::uint64_t hit_delta = has_runtime && has_initial_runtime
    ? counter_delta(last_runtime.projectile_hit, initial_runtime.projectile_hit)
    : 0;
  const auto runtime_value = [&](std::uint64_t value) -> nlohmann::json {
    return has_runtime ? nlohmann::json(value) : nlohmann::json(nullptr);
  };
  report["gimbal_fire"] = consumed_fire_delta;
  report["sent_fire"] = sent_fire;
  report["arbitration"] = {
    {"single_command_arbiter", true}, {"single_gimbal_writer", true},
    {"commands_published", command_published}, {"dual_feed_frames", dual_feed_frames},
    {"wrong_mode_routes", wrong_mode_routes}, {"truth_available_to_router", false},
    {"routed_observations", routed_observations},
    {"active_mode", auto_buff::to_string(active_mode)},
    {"small_estimator_instances", small_estimator_instances},
    {"big_estimator_instances", big_estimator_instances}};
  report["runtime"] = {
    {"following_frames", following_frames}, {"not_following_frames", not_following_frames},
    {"missing_frames", runtime_missing_frames}, {"mismatch_frames", runtime_mismatch_frames},
    {"consumed_commands_delta", consumed_delta},
    {"consumed_control_commands_delta", consumed_control_delta},
    {"consumed_fire_commands_delta", consumed_fire_delta},
    {"last_command_seq", runtime_value(last_runtime.last_command_seq)},
    {"last_command_consume_timestamp_ns",
     runtime_value(last_runtime.last_command_consume_timestamp_ns)}};
  report["projectile_launch"] = runtime_value(last_runtime.projectile_launch);
  report["projectile_hit"] = runtime_value(last_runtime.projectile_hit);
  report["projectile_launch_delta"] = runtime_value(launch_delta);
  report["projectile_hit_delta"] = runtime_value(hit_delta);
  report["timing"] = {
    {"source", sample_stats(source_stage)},
    {"detection_start", sample_stats(detection_start_stage)},
    {"detection_end", sample_stats(detection_end_stage)},
    {"planning", sample_stats(planning_stage)},
    {"command_publish", sample_stats(command_publish_stage)},
    {"simulator_consume", sample_stats(source_to_consume)},
    {"send_to_simulator_consume_ms", sample_stats(send_to_consume)},
    {"projectile_launch", nullptr},
    {"source_to_detection_ms", sample_stats(source_to_detection)},
    {"detection_to_planning_ms", sample_stats(detection_to_planning)},
    {"planning_to_command_ms", sample_stats(planning_to_command)}};
  report["rune_evaluator"] = {
    {"attempts", rune_stats.attempts}, {"same_frame", rune_stats.same_frame},
    {"missing", rune_stats.missing}, {"mismatch", rune_stats.mismatch},
    {"timestamp_mismatch", rune_stats.timestamp_mismatch},
    {"ambiguous", rune_stats.ambiguous}, {"nearest", rune_stats.nearest},
    {"degraded", rune_stats.degraded}, {"valid", rune_stats.valid},
    {"activation_samples", rune_stats.activation_samples},
    {"blade_id_samples", rune_stats.blade_id_samples},
    {"direction_mismatch", rune_stats.direction_mismatch},
    {"radius_m", sample_stats(rune_stats.radius)}, {"current_angle_rad", sample_stats(rune_stats.angle)},
    {"v_roll_radps", sample_stats(rune_stats.speed)},
    {"center_error_m", sample_stats(rune_stats.center_error_m)},
    {"angle_error_rad", sample_stats(rune_stats.angle_error_rad)},
    {"speed_error_radps", sample_stats(rune_stats.speed_error_radps)},
    {"predicted_phase", sample_stats(rune_stats.predicted_phase)},
    {"phase_error_rad", sample_stats(rune_stats.phase_error_rad)},
    {"commanded_direction", sample_stats(rune_stats.commanded_direction)},
    {"truth_direction", sample_stats(rune_stats.truth_direction)},
    {"gimbal_error_deg", sample_stats(rune_stats.gimbal_error_deg)},
    {"max_center_translation_speed_mps", rune_stats.max_center_translation_speed_mps}};
  nlohmann::json fault_history = nlohmann::json::object();
  for (const auto & fault : gimbal.fault_history()) {
    fault_history[fault.name] = {
      {"episodes", fault.episodes}, {"total_s", fault.total_s}, {"max_s", fault.max_s},
      {"active_at_exit", fault.active}};
  }
  report["safety"] = {
    {"faults", sim_io::describe_faults(gimbal.faults())},
    {"faults_seen", sim_io::describe_faults(gimbal.faults_seen())}, {"by_bit", fault_history},
    {"suppressed_fires", gimbal.suppressed_fires()},
    {"pose_invalid_frames", pose_invalid_frames},
    {"rearm_events", rearm_events}, {"rearm_pending", rearm_pending},
    {"rearm_confirmed", rearm_confirmed}, {"rearm_confirm_frames", rearm_confirm_frames},
    {"color_gate", false},
    {"fire_evidence_eligible", consumed_fire_delta > 0 && launch_delta > 0}};

  report["dynamic_budget"] = {
    {"samples", dynamic_samples}, {"violations", dynamic_violations},
    {"limits",
     {{"max_angle_error_deg", dynamic_max_angle_error_deg},
      {"max_position_error_m", dynamic_max_position_error_m},
      {"max_chassis_yaw_rate_radps", controlled_max_yaw_rate_radps},
      {"max_chassis_translation_speed_mps", controlled_max_translation_speed_mps},
      {"max_target_rotation_radps", controlled_max_target_rotation_radps},
      {"assumed_max_target_translation_speed_mps",
       controlled_max_target_translation_speed_mps}}},
    {"observed_max",
     {{"chassis_yaw_rate_radps", max_yaw_rate},
      {"chassis_translation_speed_mps", max_chassis_translation},
      {"target_rotation_radps", max_target_rotation},
      {"target_translation_speed_mps", rune_stats.max_center_translation_speed_mps}}}};
  report["synthetic_offsets"] = {
    {"enabled", synthetic_enabled},
    {"source", synthetic_source},
    {"not_physical_dynamics", true}};

  const std::uint64_t strict_min_frames =
    std::max(1, tools::read_or<int>(sim, "strict_min_frames", 30));
  const std::uint64_t strict_min_samples =
    std::max(1, tools::read_or<int>(sim, "strict_min_matched_eval_samples", 10));
  const double strict_min_gt_coverage =
    tools::read_or<double>(sim, "strict_min_gt_coverage", 0.95);
  const double strict_center_error_m =
    tools::read_or<double>(sim, "strict_center_error_m", 0.15);
  const double strict_angle_error_rad =
    tools::read_or<double>(sim, "strict_angle_error_rad", 0.2);
  const double strict_speed_error_radps =
    tools::read_or<double>(sim, "strict_speed_error_radps", 0.4);
  const double strict_gimbal_error_deg =
    tools::read_or<double>(sim, "strict_gimbal_error_deg", 3.0);
  const double truth_coverage = frames == 0
    ? 0.0
    : static_cast<double>(rune_stats.same_frame) / static_cast<double>(frames);
  const bool eval_enabled = cli.has("eval");
  const bool attempts_cover_frames = rune_stats.attempts == frames;
  const bool no_gt_mismatch = rune_stats.mismatch == 0;
  const bool no_gt_timestamp_mismatch = rune_stats.timestamp_mismatch == 0;
  const bool enough_gt_coverage = truth_coverage >= strict_min_gt_coverage;
  const bool enough_valid_samples = rune_stats.valid >= strict_min_samples;
  const bool no_gt_ambiguous = rune_stats.ambiguous == 0;
  const bool no_gt_nearest = rune_stats.nearest == 0;
  const bool no_degraded = rune_stats.degraded == 0;
  const bool center_error_ok = !rune_stats.center_error_m.empty() &&
    vector_max(rune_stats.center_error_m) <= strict_center_error_m;
  const bool angle_error_ok = !rune_stats.angle_error_rad.empty() &&
    vector_max(rune_stats.angle_error_rad) <= strict_angle_error_rad;
  const bool speed_error_ok = !rune_stats.speed_error_radps.empty() &&
    vector_max(rune_stats.speed_error_radps) <= strict_speed_error_radps;
  const bool gimbal_error_ok = !rune_stats.gimbal_error_deg.empty() &&
    vector_max(rune_stats.gimbal_error_deg) <= strict_gimbal_error_deg;
  const bool no_direction_mismatch = rune_stats.direction_mismatch == 0;
  const bool truth_contract = eval_enabled && attempts_cover_frames && no_gt_mismatch &&
    no_gt_timestamp_mismatch && enough_gt_coverage && enough_valid_samples &&
    no_gt_ambiguous && no_gt_nearest && no_degraded && center_error_ok && angle_error_ok &&
    speed_error_ok && gimbal_error_ok && no_direction_mismatch;
  std::ostringstream criterion;
  criterion << "eval AND attempts==frames AND mismatch==0 AND timestamp_mismatch==0 AND "
               "coverage>=" << strict_min_gt_coverage << " AND valid>=" << strict_min_samples
            << " AND ambiguous==0 AND nearest==0 AND degraded==0 AND center_error_m<="
            << strict_center_error_m << " AND angle_error_rad<=" << strict_angle_error_rad
            << " AND speed_error_radps<=" << strict_speed_error_radps
            << " AND gimbal_error_deg<=" << strict_gimbal_error_deg
            << " AND direction_mismatch==0";
  report["truth_contract"] = {
    {"enabled", eval_enabled}, {"attempts_cover_frames", attempts_cover_frames},
    {"coverage", truth_coverage}, {"min_coverage", strict_min_gt_coverage},
    {"frame_mismatch_zero", no_gt_mismatch},
    {"timestamp_mismatch_zero", no_gt_timestamp_mismatch},
    {"enough_valid_samples", enough_valid_samples},
    {"no_gt_ambiguous", no_gt_ambiguous}, {"no_gt_nearest", no_gt_nearest},
    {"no_degraded", no_degraded}, {"center_error_ok", center_error_ok},
    {"angle_error_ok", angle_error_ok}, {"speed_error_ok", speed_error_ok},
    {"gimbal_error_ok", gimbal_error_ok}, {"no_direction_mismatch", no_direction_mismatch},
    {"center_error_m", sample_stats(rune_stats.center_error_m)},
    {"angle_error_rad", sample_stats(rune_stats.angle_error_rad)},
    {"speed_error_radps", sample_stats(rune_stats.speed_error_radps)},
    {"predicted_phase", sample_stats(rune_stats.predicted_phase)},
    {"commanded_direction", sample_stats(rune_stats.commanded_direction)},
    {"gimbal_error_deg", sample_stats(rune_stats.gimbal_error_deg)},
    {"thresholds",
     {{"center_error_m", strict_center_error_m},
      {"angle_error_rad", strict_angle_error_rad},
      {"speed_error_radps", strict_speed_error_radps},
      {"gimbal_error_deg", strict_gimbal_error_deg}}},
    {"criterion", criterion.str()},
    {"passed", truth_contract}};

  const bool perception_chain = detector_found > 0 && solver_valid > 0 && target_valid > 0;
  const bool planning_chain = plan_control > 0;
  const bool runtime_chain = following_frames > 0 && not_following_frames == 0 &&
    runtime_missing_frames == 0 && runtime_mismatch_frames == 0;
  const bool consume_chain = consumed_control_delta > 0 && !source_to_consume.empty();
  const bool shot_evidence = gim_cfg.allow_fire && plan_fire > 0 && sent_fire > 0 &&
    consumed_fire_delta > 0 && launch_delta > 0;
  const bool no_shot_aiming = !gim_cfg.allow_fire && plan_control > 0 &&
    consumed_control_delta > 0 && sent_fire == 0 && consumed_fire_delta == 0 && launch_delta == 0;
  const bool controlled_motion_observed = max_yaw_rate > 1e-3 &&
    max_chassis_translation > 1e-3 && max_target_rotation > 1e-3 &&
    rune_stats.max_center_translation_speed_mps > 1e-3;
  const bool target_translation_ok = sim_io::observed_target_translation_within_assumed(
    rune_stats.max_center_translation_speed_mps, controlled_max_target_translation_speed_mps);
  const bool dynamic_contract = dynamic_samples >= strict_min_samples &&
    dynamic_violations == 0 && controlled_motion_observed && target_translation_ok;
  const bool routing_contract = active_mode != auto_buff::PowerRuneMode::Unknown &&
    dual_feed_frames == 0 && wrong_mode_routes == 0 && routed_observations > 0;
  const std::uint32_t allowed_faults = sim_io::FAULT_STARTUP | sim_io::FAULT_REARM_PENDING |
    sim_io::FAULT_TARGET_LOST | (gim_cfg.allow_fire ? 0u : sim_io::FAULT_FIRE_DISABLED);
  // 严格判据看整段 faults_seen，而不是退出瞬间的 faults()；否则一个短暂的
  // command_age/clock_jump/pose_invalid 在退出前恰好清掉，就会被洗成假阳性。
  const std::uint32_t offending_faults = gimbal.faults_seen() & ~allowed_faults;
  const bool algorithm_closed_loop = mode == "closed_loop" && frames >= strict_min_frames &&
    perception_chain && planning_chain && runtime_chain && consume_chain && routing_contract &&
    dynamic_contract && !rearm_pending && rearm_events == 0 && offending_faults == 0 &&
    (shot_evidence || no_shot_aiming);
  report["algorithm_closed_loop"] = {
    {"closed_loop_mode", mode == "closed_loop"}, {"enough_frames", frames >= strict_min_frames},
    {"detector_found", detector_found}, {"solver_valid", solver_valid},
    {"target_valid", target_valid}, {"plan_control", plan_control}, {"plan_fire", plan_fire},
    {"perception_chain", perception_chain}, {"planning_chain", planning_chain},
    {"runtime_following", runtime_chain}, {"command_consumed", consume_chain},
    {"shot_evidence", shot_evidence}, {"no_shot_aiming_criterion", no_shot_aiming},
    {"routing_contract", routing_contract}, {"dynamic_contract", dynamic_contract},
    {"offending_faults", sim_io::describe_faults(offending_faults)},
    {"passed", algorithm_closed_loop}};
  const bool strict_met = truth_contract && algorithm_closed_loop;
  report["strict"] = {
    {"truth_contract", truth_contract}, {"algorithm_closed_loop", algorithm_closed_loop},
    {"verdict", strict_met ? "criteria_met_single_run_not_acceptance" : "criteria_not_met"}};
  report["ipc"] = {
    {"consumed", camera.client().consumed_frames()}, {"dropped", camera.client().dropped_frames()},
    {"ground_truth_captures", camera.client().ground_truth_captures()},
    {"runtime_state_snapshot_failures", camera.client().runtime_state_snapshot_failures()},
    {"shm_version", camera.client().version()}, {"capabilities", camera.client().capabilities()}};
  const std::string model_path = tools::resolve_path_from_config_string(
    config_path, yaml["model"].as<std::string>());
  report["metadata"] = sim_io::reproducibility_metadata(config_path, model_path);

  const std::string report_path = cli.get<std::string>("report");
  if (!report_path.empty()) {
    std::ofstream out(report_path);
    out << report.dump(2) << '\n';
  } else {
    std::printf("%s\n", report.dump(2).c_str());
  }
  return 0;
}
