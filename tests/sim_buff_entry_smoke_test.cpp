#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "simulation/io/testing/fake_publisher.hpp"
#include <Eigen/Dense>
#include "simulation/io/rune_association.hpp"
#include "tools/path.hpp"

namespace
{
int failures = 0;

void check(bool ok, const std::string & what)
{
  std::printf("%-68s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

std::string read_file(const std::string & path)
{
  std::ifstream in(path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

int run_entry(const std::vector<std::string> & args)
{
  std::vector<char *> raw;
  for (const auto & arg : args) raw.push_back(const_cast<char *>(arg.c_str()));
  raw.push_back(nullptr);
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::execv(raw[0], raw.data());
    std::_Exit(127);
  }
  if (pid < 0) return -1;
  int status = 0;
  ::waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

struct Producer
{
  sim_io::testing::FakePublisher pub;
  std::vector<std::uint8_t> rgb = std::vector<std::uint8_t>(sim_io::IMAGE_SIZE, 40);
  std::atomic<bool> running{true};
  std::atomic<bool> following{true};
  std::atomic<bool> reconnect{false};
  std::uint64_t seq = 1;

  explicit Producer(const sim_io::testing::FakePublisher::Options & options) : pub(options) {}

  bool create()
  {
    std::string error;
    if (!pub.create(&error)) {
      std::printf("producer create failed: %s\n", error.c_str());
      return false;
    }
    sim_io::CameraInfo info{};
    info.fx = 1303.6752833867;
    info.fy = 1303.6752833867;
    info.cx = 720.0;
    info.cy = 540.0;
    info.width = sim_io::IMAGE_WIDTH;
    info.height = sim_io::IMAGE_HEIGHT;
    pub.set_camera_info(info);
    pub.update_heartbeat();
    return true;
  }

  void run()
  {
    const float identity[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float odom[3] = {0.0f, 0.0f, 0.3f};
    const float muzzle[3] = {0.1f, 0.0f, 0.02f};
    const float camera[3] = {0.05f, 0.0f, 0.05f};
    while (running.load(std::memory_order_relaxed)) {
      if (reconnect.exchange(false, std::memory_order_acq_rel)) {
        pub.destroy();
        pub.unlink_files();
        create();
      }
      pub.update_heartbeat();
      const std::uint64_t timestamp = sim_io::testing::FakePublisher::now_ns();
      sim_io::RuntimeState runtime{};
      runtime.frame_seq = seq;
      runtime.timestamp_ns = timestamp;
      runtime.following = following.load(std::memory_order_relaxed) ? 1 : 0;
      pub.set_runtime_state(runtime);
      sim_io::ChassisObservation chassis{};
      chassis.frame_seq = seq;
      chassis.timestamp_ns = timestamp;
      pub.set_chassis_observation(chassis);
      pub.try_publish_synchronized_frame(
        rgb.data(), seq, timestamp, identity, odom, muzzle, camera);
      ++seq;
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
  }
};

std::string make_config(const std::string & dir)
{
  const std::string source = tools::resolve_config_path_string("configs/simulation.yaml");
  std::ifstream input(source);
  std::ostringstream text;
  text << input.rdbuf();
  const std::string needle = "shm_dir: \"/tmp\"";
  const auto pos = text.str().find(needle);
  std::string value = text.str();
  if (pos != std::string::npos) value.replace(pos, needle.size(), "shm_dir: \"" + dir + "\"");
  const std::string path = dir + "/simulation.yaml";
  std::ofstream output(path);
  output << value;
  return path;
}

bool json_bool(const nlohmann::json & json, const char * key)
{
  return json.contains(key) && json.at(key).is_boolean() && json.at(key).get<bool>();
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::printf("usage: sim_buff_entry_smoke_test <sim_auto_buff>\n");
    return 2;
  }
  const std::string dir = "/tmp/sim_buff_entry_" + std::to_string(::getpid());
  ::mkdir(dir.c_str(), 0700);
  const std::string config = make_config(dir);
  sim_io::testing::FakePublisher::Options options;
  options.dir = dir;
  Producer producer(options);
  if (!producer.create()) return 2;
  std::thread producer_thread([&] { producer.run(); });

  for (const char * task : {"small_buff", "big_buff", "all"}) {
    const std::string report_path = dir + "/" + task + ".json";
    const int code = run_entry(
      {argv[1], config, "--task=" + std::string(task), "--mode=passive", "--duration-s=1",
       "--eval", "--report=" + report_path});
    check(code == 0, std::string("入口 ") + task + " passive 正常退出");
    try {
      const auto report = nlohmann::json::parse(read_file(report_path));
      check(report.at("task") == task, std::string(task) + " 报告 task 正确");
      check(report.at("mode") == "passive", std::string(task) + " 报告 mode 正确");
      check(
        report.at("strict").at("verdict") == "criteria_not_met",
        std::string(task) + " 无闭环证据时 strict 不得误报");
      check(
        report.at("arbitration").at("truth_available_to_router") == false,
        std::string(task) + " 路由器没有 truth 输入");
      check(report.at("truth_contract").contains("thresholds"),
        std::string(task) + " 报告含误差阈值");
      check(report.at("truth_contract").contains("criterion"),
        std::string(task) + " 报告含 truth_contract criterion");
      {
        const auto criterion =
          report.at("truth_contract").at("criterion").get<std::string>();
        check(
          criterion.find("center_error_m") != std::string::npos &&
            criterion.find("angle_error_rad") != std::string::npos &&
            criterion.find("speed_error_radps") != std::string::npos &&
            criterion.find("gimbal_error_deg") != std::string::npos,
          std::string(task) + " criterion 由误差阈值 conjunct 生成");
      }
      check(report.at("rune_evaluator").contains("center_error_m"),
        std::string(task) + " 报告含 center_error_m");
      check(report.at("rune_evaluator").contains("angle_error_rad"),
        std::string(task) + " 报告含 angle_error_rad");
      check(report.at("rune_evaluator").contains("speed_error_radps"),
        std::string(task) + " 报告含 speed_error_radps");
      check(report.at("rune_evaluator").contains("predicted_phase"),
        std::string(task) + " 报告含 predicted_phase");
      check(report.at("rune_evaluator").contains("commanded_direction"),
        std::string(task) + " 报告含 commanded_direction");
      check(report.at("rune_evaluator").contains("gimbal_error_deg"),
        std::string(task) + " 报告含 gimbal_error_deg");
    } catch (const std::exception & error) {
      check(false, std::string(task) + " 报告是合法结构化 JSON: " + error.what());
    }
  }

  {
    sim_io::GroundTruthRune dual[2]{};
    dual[0].rune_mode = 0;
    dual[0].r_center_odom[0] = 0.0f;
    dual[0].r_center_odom[1] = 0.0f;
    dual[0].r_center_odom[2] = 0.0f;
    dual[1].rune_mode = 1;
    dual[1].r_center_odom[0] = 3.0f;
    dual[1].r_center_odom[1] = 0.0f;
    dual[1].r_center_odom[2] = 0.0f;
    const Eigen::Vector3d estimate(0.1, 0.0, 0.0);
    const auto dual_assoc = sim_io::associate_rune_by_mode(dual, 2, 0, &estimate);
    check(
      dual_assoc.hits == 1 && !dual_assoc.ambiguous && dual_assoc.selected == &dual[0],
      "dual-mode association helper is not inherently ambiguous");

    sim_io::GroundTruthRune same[2]{};
    same[0].rune_mode = 0;
    same[0].r_center_odom[0] = 0.0f;
    same[0].r_center_odom[1] = 0.0f;
    same[0].r_center_odom[2] = 0.0f;
    same[1].rune_mode = 0;
    same[1].r_center_odom[0] = 2.0f;
    same[1].r_center_odom[1] = 0.0f;
    same[1].r_center_odom[2] = 0.0f;
    const auto same_assoc = sim_io::associate_rune_by_mode(same, 2, 0, &estimate);
    check(
      same_assoc.hits == 2 && same_assoc.ambiguous && same_assoc.selected == &same[0],
      "same-mode duplicate association is ambiguous");
    const Eigen::Vector3d near_second(1.9, 0.0, 0.0);
    const auto same_near = sim_io::associate_rune_by_mode(same, 2, 0, &near_second);
    check(
      same_near.hits == 2 && same_near.ambiguous && same_near.selected == &same[1],
      "same-mode duplicate uses nearest-center, still ambiguous");
  }

  // following=0 必须禁止开火，并在报告里留下明确的 not_following 故障。
  producer.following = false;
  const std::string no_follow_report = dir + "/no_follow.json";
  const int no_follow_code = run_entry(
    {argv[1], config, "--task=all", "--mode=closed_loop", "--allow-fire",
     "--max-command-age-ms=5000", "--duration-s=1", "--report=" + no_follow_report});
  check(no_follow_code == 0, "following=0 入口仍安全退出");
  try {
    const auto report = nlohmann::json::parse(read_file(no_follow_report));
    check(
      report.at("gimbal_fire") == 0 && report.at("sent_fire") == 0,
      "following=0 时没有视觉/发送开火计数");
    check(
      report.at("safety").at("faults_seen").get<std::string>().find("not_following") !=
        std::string::npos,
      "following=0 记录 not_following 故障");
  } catch (const std::exception & error) {
    check(false, std::string("following=0 报告可解析: ") + error.what());
  }

  // 重建发布端文件时，入口必须走 Reconnected 并保持 rearm，不能复用旧状态开火。
  producer.following = true;
  const std::string reconnect_report = dir + "/reconnect.json";
  std::thread reconnect_trigger([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    producer.reconnect = true;
  });
  const int reconnect_code = run_entry(
    {argv[1], config, "--task=all", "--mode=closed_loop", "--allow-fire",
     "--max-command-age-ms=5000", "--duration-s=2", "--report=" + reconnect_report});
  reconnect_trigger.join();
  check(reconnect_code == 0, "重连场景入口安全退出");
  try {
    const auto report = nlohmann::json::parse(read_file(reconnect_report));
    check(report.at("safety").at("rearm_events") >= 1, "重连后记录 rearm_events");
    check(report.at("gimbal_fire") == 0, "重连无目标时没有实际开火");
  } catch (const std::exception & error) {
    check(false, std::string("重连报告可解析: ") + error.what());
  }

  producer.running = false;
  producer_thread.join();
  producer.pub.destroy();
  producer.pub.unlink_files();

  // 能力位缺失单独建发布端，入口不能将不可知状态当作 following=1。
  const std::string missing_dir = dir + "_missing";
  ::mkdir(missing_dir.c_str(), 0700);
  const std::string missing_config = make_config(missing_dir);
  auto missing_options = options;
  missing_options.dir = missing_dir;
  missing_options.capabilities &= ~sim_io::CAP_RUNTIME_STATE;
  Producer missing(missing_options);
  check(missing.create(), "创建缺 RuntimeState 能力位发布端");
  std::thread missing_thread([&] { missing.run(); });
  const std::string missing_report = missing_dir + "/missing.json";
  const int missing_code = run_entry(
    {argv[1], missing_config, "--task=all", "--mode=closed_loop", "--allow-fire",
     "--max-command-age-ms=5000", "--duration-s=1", "--report=" + missing_report});
  check(missing_code == 0, "缺能力位场景入口安全退出");
  try {
    const auto report = nlohmann::json::parse(read_file(missing_report));
    check(report.at("gimbal_fire") == 0, "缺 RuntimeState 时没有实际开火");
    check(
      report.at("safety").at("faults_seen").get<std::string>().find("capability_missing") !=
        std::string::npos,
      "缺 RuntimeState 记录 capability_missing");
    check(
      report.at("strict").at("verdict") == "criteria_not_met",
      "缺能力位时 strict 不得误报");
  } catch (const std::exception & error) {
    check(false, std::string("缺能力位报告可解析: ") + error.what());
  }
  missing.running = false;
  missing_thread.join();
  missing.pub.destroy();
  missing.pub.unlink_files();
  ::unlink(config.c_str());
  ::rmdir(dir.c_str());
  ::unlink(missing_config.c_str());
  ::rmdir(missing_dir.c_str());
  std::printf("sim_buff_entry_smoke_test: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
