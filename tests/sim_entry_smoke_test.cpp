// sim_auto_aim 入口的离线端到端冒烟测试。
//
// 用 FakePublisher 顶替 Bevy：本进程既是图像/姿态的生产者，也是云台命令的消费者，
// 于是整条入口链路（yaml 加载 -> 内参自检 -> YOLO/Tracker/Planner 构造 -> 主循环
// -> 指标汇总）可以在没有 GPU、没有仿真器、没有任何硬件的机器上跑完。
//
// 它验证三件在真机上代价很高、在这里几乎免费的事：
//   1. 入口能在完全干净的环境里跑起来并正常退出（不是崩在半路）；
//   2. 默认（不带 --allow-fire）时 fire_advice 恒为 0，且计入 suppressed；
//   3. closed_loop + --allow-fire 缺世界观测年龄预算、缺颜色门或颜色门配置错误时入口拒绝启动；
//   4. 关闭颜色门开火必须显式 --allow-colorblind-fire，并在报告中留下痕迹；
//   5. 心跳停掉之后入口不再发控制命令，只发安全停止。
//
// 刻意不追求检出率：合成图案里没有装甲板，detected_frames 预期为 0。

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "simulation/io/testing/fake_publisher.hpp"
#include "tools/path.hpp"

namespace
{
int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string & what, const std::string & detail = "")
{
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("%-52s %s%s%s\n", what.c_str(), ok ? "ok" : "FAIL", detail.empty() ? "" : "  ",
              detail.c_str());
}

std::string read_file(const std::string & path)
{
  std::ifstream ifs(path);
  if (!ifs) return {};
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// 极简 JSON 取数：报告是本程序自己生成的固定格式，不值得引入解析器。
bool json_number(const std::string & json, const std::string & key, double * out)
{
  const std::string needle = "\"" + key + "\":";
  const auto pos = json.find(needle);
  if (pos == std::string::npos) return false;
  return std::sscanf(json.c_str() + pos + needle.size(), " %lf", out) == 1;
}

std::string json_string(const std::string & json, const std::string & key)
{
  const std::string needle = "\"" + key + "\":";
  auto pos = json.find(needle);
  if (pos == std::string::npos) return {};
  pos = json.find('"', pos + needle.size());
  if (pos == std::string::npos) return {};
  const auto end = json.find('"', pos + 1);
  if (end == std::string::npos) return {};
  return json.substr(pos + 1, end - pos - 1);
}

bool json_bool(const std::string & json, const std::string & key, bool * out)
{
  const std::string needle = "\"" + key + "\":";
  const auto pos = json.find(needle);
  if (pos == std::string::npos) return false;
  const char * value = json.c_str() + pos + needle.size();
  if (std::strncmp(value, " true", 5) == 0) {
    *out = true;
    return true;
  }
  if (std::strncmp(value, " false", 6) == 0) {
    *out = false;
    return true;
  }
  return false;
}

// fork 出入口二进制并等它退出，返回退出码（异常终止返回 -1）。
// argv[0] 就是可执行文件路径本身，不要像 execl 那样再重复一次——重复的那一个会
// 顶掉位置参数 @config-path，入口会把二进制当 yaml 读，报的是"配置加载失败"。
int run_entry(const std::vector<std::string> & argv)
{
  std::vector<char *> raw;
  for (const auto & a : argv) raw.push_back(const_cast<char *>(a.c_str()));
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
}  // namespace

namespace
{
// 生产者线程：按固定节奏发布同步帧 + 心跳，并把云台命令排空。
// 与 simulator 一样受背压约束：上一帧没被消费就不发新帧。
struct Producer
{
  sim_io::testing::FakePublisher & pub;
  std::atomic<bool> running{true};
  std::atomic<bool> heartbeat{true};

  std::atomic<std::uint64_t> published{0};
  std::atomic<std::uint64_t> cmds{0};
  std::atomic<std::uint64_t> control_cmds{0};
  std::atomic<std::uint64_t> fire_cmds{0};
  std::atomic<std::uint64_t> safe_stops{0};
  std::atomic<std::uint64_t> cmds_after_heartbeat_off{0};
  std::atomic<std::uint64_t> control_after_heartbeat_off{0};

  std::vector<std::uint8_t> rgb;

  explicit Producer(sim_io::testing::FakePublisher & p) : pub(p), rgb(sim_io::IMAGE_SIZE)
  {
    // 灰底 + 缓变梯度。没有装甲板，YOLO 预期零检出，这里只关心链路是否通。
    for (std::size_t i = 0; i < rgb.size(); i += 3) {
      const auto x = static_cast<std::uint8_t>((i / 3) % 251);
      rgb[i + 0] = static_cast<std::uint8_t>(60 + x % 40);
      rgb[i + 1] = static_cast<std::uint8_t>(60 + (x / 2) % 40);
      rgb[i + 2] = static_cast<std::uint8_t>(60 + (x / 3) % 40);
    }
  }

  void drain_cmds()
  {
    sim_io::GimbalCmd cmd{};
    while (pub.recv_gimbal_cmd(&cmd)) {
      ++cmds;
      if (!heartbeat.load()) ++cmds_after_heartbeat_off;
      if (cmd.distance_m == -1.0f) {
        ++safe_stops;
      } else {
        ++control_cmds;
        if (!heartbeat.load()) ++control_after_heartbeat_off;
      }
      if (cmd.fire_advice == 1) ++fire_cmds;
    }
  }

  void run()
  {
    // 单位四元数 = yaw/pitch 皆为 0，probe/闭环都能从这里出发。
    const float identity[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float odom[3] = {0.0f, 0.0f, 0.3f};
    const float muzzle[3] = {0.1f, 0.0f, 0.02f};
    const float camera[3] = {0.05f, 0.0f, 0.05f};

    std::uint64_t seq = 1;
    while (running.load()) {
      if (heartbeat.load()) pub.update_heartbeat();
      drain_cmds();
      if (heartbeat.load()) {
        const std::uint64_t ts = sim_io::testing::FakePublisher::now_ns();
        if (pub.try_publish_synchronized_frame(
              rgb.data(), seq, ts, identity, odom, muzzle, camera)) {
          ++published;
          ++seq;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    drain_cmds();
  }
};
}  // namespace

int main(int argc, char * argv[])
{
  if (argc < 2) {
    std::printf("用法: sim_entry_smoke_test <sim_auto_aim 可执行文件路径>\n");
    return 2;
  }
  const std::string binary = argv[1];

  const std::string dir = "/tmp/sim_entry_smoke_" + std::to_string(::getpid());
  if (::mkdir(dir.c_str(), 0777) != 0 && errno != EEXIST) {
    std::printf("无法创建临时目录 %s: %s\n", dir.c_str(), std::strerror(errno));
    return 2;
  }

  // 用源码里的 configs/simulation.yaml 生成一份只改 shm_dir 的副本，
  // 这样内参、外参、模型路径等都与真实配置完全一致，测的是同一份配置。
  const std::string src_yaml = tools::resolve_config_path_string("configs/simulation.yaml");
  std::string yaml_text = read_file(src_yaml);
  if (yaml_text.empty()) {
    std::printf("无法读取 %s\n", src_yaml.c_str());
    return 2;
  }
  {
    const std::string needle = "shm_dir: \"/tmp\"";
    const auto pos = yaml_text.find(needle);
    if (pos == std::string::npos) {
      std::printf("configs/simulation.yaml 里找不到 shm_dir，测试无法重定向共享内存\n");
      return 2;
    }
    yaml_text.replace(pos, needle.size(), "shm_dir: \"" + dir + "\"");
  }
  const std::string yaml_path = dir + "/simulation.yaml";
  {
    std::ofstream ofs(yaml_path);
    ofs << yaml_text;
  }

  sim_io::testing::FakePublisher::Options opts;
  opts.dir = dir;
  sim_io::testing::FakePublisher pub(opts);
  std::string err;
  if (!pub.create(&err)) {
    std::printf("无法创建共享内存: %s\n", err.c_str());
    return 2;
  }

  // 与 simulator compute_camera_intrinsics(1440, 1080, 45deg) 的结果一致，
  // 否则入口的内参自检会直接拒绝启动（这本身也是被验证的行为之一）。
  sim_io::CameraInfo info{};
  info.fx = 1303.6752833867;
  info.fy = 1303.6752833867;
  info.cx = 720.0;
  info.cy = 540.0;
  info.width = sim_io::IMAGE_WIDTH;
  info.height = sim_io::IMAGE_HEIGHT;
  pub.set_camera_info(info);
  pub.update_heartbeat();

  Producer producer(pub);
  std::thread producer_thread([&producer] { producer.run(); });

  const std::string report = dir + "/report.json";

  std::printf("--- 闭环模式（默认禁止开火）-----------------------------------------\n");
  const pid_t pid = ::fork();
  if (pid == 0) {
    // 子进程：真正的入口二进制。stdout 留给它自己打印，便于失败时排查。
    ::execl(
      binary.c_str(), binary.c_str(), yaml_path.c_str(), "--mode=closed_loop", "--duration-s=4",
      ("--report=" + report).c_str(), static_cast<char *>(nullptr));
    std::_Exit(127);
  }
  if (pid < 0) {
    producer.running = false;
    producer_thread.join();
    std::printf("fork 失败: %s\n", std::strerror(errno));
    return 2;
  }

  int status = 0;
  ::waitpid(pid, &status, 0);
  const bool exited = WIFEXITED(status);
  const int code = exited ? WEXITSTATUS(status) : -1;
  check(exited && code == 0, "入口正常退出", "exit=" + std::to_string(code));

  const std::uint64_t published = producer.published.load();
  check(published > 0, "生产者发出了同步帧", std::to_string(published));

  const std::string json = read_file(report);
  check(!json.empty(), "指标报告已生成");

  double frames_ok = -1, fire = -1, suppressed = -1, plan_fire = -1, stale = -1, rejected = -1;
  double p50 = -1, p95 = -1, p99 = -1, dropped = -1, regressed = -1;
  json_number(json, "frames_ok", &frames_ok);
  json_number(json, "fire", &fire);
  json_number(json, "suppressed_fire", &suppressed);
  json_number(json, "plan_fire", &plan_fire);
  json_number(json, "stale", &stale);
  json_number(json, "rejected", &rejected);
  json_number(json, "p50", &p50);
  json_number(json, "p95", &p95);
  json_number(json, "p99", &p99);
  json_number(json, "dropped", &dropped);
  json_number(json, "regressed", &regressed);

  check(frames_ok > 0, "入口处理了至少一帧", "frames_ok=" + std::to_string((long)frames_ok));
  check(fire == 0, "默认配置下 fire_advice 恒为 0");
  check(producer.fire_cmds.load() == 0, "仿真侧没收到任何开火命令");
  check(regressed == 0, "没有帧号倒退");
  check(rejected == 0, "没有被拒绝的帧");
  check(p50 >= 0 && p95 >= p50 && p99 >= p95, "帧龄分位数单调",
        "p50=" + std::to_string(p50) + " p95=" + std::to_string(p95) + " p99=" +
          std::to_string(p99));

  std::printf("--- 开火必须带年龄预算与颜色安全门 ------------------------------------\n");
  // configs/simulation.yaml 没有 max_command_age_ms（默认 0 = 不设限），所以
  // --allow-fire 必须被拒。这里测的是"缺预算时不会悄悄跑成一次不设限的开火闭环"。
  {
    const std::uint64_t fire_before = producer.fire_cmds.load();
    const std::string no_budget_report = dir + "/no_budget.json";
    const int code_no_budget = run_entry(
      {binary, yaml_path, "--mode=closed_loop", "--allow-fire", "--duration-s=2",
       "--report=" + no_budget_report});
    check(code_no_budget == 2, "缺预算时 --allow-fire 被拒绝（exit 2）",
          "exit=" + std::to_string(code_no_budget));
    check(read_file(no_budget_report).empty(), "被拒的运行没有写出报告");
    check(producer.fire_cmds.load() == fire_before, "被拒的运行没有发出任何开火命令");
    ::unlink(no_budget_report.c_str());

    // 只有年龄预算还不够：默认配置关闭颜色门，必须拒绝，而不是冒着打己方目标的
    // 风险跑完。此前这里错误地期待 exit=0，等价于把新安全门当成回归。
    const std::string budget_report = dir + "/budget.json";
    const int code_missing_color_gate = run_entry(
      {binary, yaml_path, "--mode=closed_loop", "--allow-fire",
       "--max-command-age-ms=5000", "--duration-s=2", "--report=" + budget_report});
    check(code_missing_color_gate == 2, "缺颜色门时 --allow-fire 被拒绝（exit 2）",
          "exit=" + std::to_string(code_missing_color_gate));
    check(read_file(budget_report).empty(), "颜色门拒绝时不写报告");
    check(producer.fire_cmds.load() == fire_before, "颜色门拒绝时没有任何开火命令");

    // 无颜色门的开火只能走显式危险 opt-in；报告必须留下可机器读取的痕迹，避免
    // 事后把这次运行误当成带敌我约束的闭环证据。
    const int code_colorblind = run_entry(
      {binary, yaml_path, "--mode=closed_loop", "--allow-fire", "--allow-colorblind-fire",
       "--max-command-age-ms=5000", "--duration-s=2", "--report=" + budget_report});
    check(code_colorblind == 0, "显式色盲开火 opt-in 后能正常跑完",
          "exit=" + std::to_string(code_colorblind));
    const std::string budget_json = read_file(budget_report);
    double budget_ms = -1;
    json_number(budget_json, "budget_ms", &budget_ms);
    check(budget_ms == 5000, "报告记下了生效的预算", std::to_string(budget_ms));
    check(json_string(budget_json, "budget_source") == "cli", "报告记下了预算来源",
          json_string(budget_json, "budget_source"));
    bool colorblind_opt_in = false;
    bool fire_without_color_gate = false;
    check(
      json_bool(budget_json, "colorblind_fire_opt_in", &colorblind_opt_in) && colorblind_opt_in,
      "报告记下色盲开火 opt-in");
    check(
      json_bool(budget_json, "fire_without_color_gate", &fire_without_color_gate) &&
        fire_without_color_gate,
      "报告明确标记开火未受颜色门约束");
    ::unlink(budget_report.c_str());

    // 把颜色门打开但仍指向配置里的 red（评估敌队是 blue）也必须拒绝。这个分支
    // 防止未来有人只把 use_enemy_color 改成 true，就误以为开火已经安全。
    const std::string wrong_color_yaml = dir + "/wrong_color.yaml";
    std::string wrong_color_text = yaml_text;
    const std::string color_gate_off = "use_enemy_color: false";
    const auto color_gate_pos = wrong_color_text.find(color_gate_off);
    check(color_gate_pos != std::string::npos, "测试配置包含 sim.use_enemy_color");
    if (color_gate_pos != std::string::npos) {
      wrong_color_text.replace(color_gate_pos, color_gate_off.size(), "use_enemy_color: true");
      std::ofstream ofs(wrong_color_yaml);
      ofs << wrong_color_text;
      ofs.close();
      const int code_wrong_color = run_entry(
        {binary, wrong_color_yaml, "--mode=closed_loop", "--allow-fire",
         "--max-command-age-ms=5000", "--duration-s=2"});
      check(code_wrong_color == 2, "颜色门指向非敌方队伍时被拒绝（exit 2）",
            "exit=" + std::to_string(code_wrong_color));
    }
    ::unlink(wrong_color_yaml.c_str());

    // Tracker 对缺失 enemy_color 会默认蓝色；旧的一致性判断却把空字符串直接当作
    // "一致"。即使默认刚好是 blue，也不能让未声明的敌我约束解锁开火。
    const std::string missing_color_yaml = dir + "/missing_color.yaml";
    std::string missing_color_text = yaml_text;
    const std::string enemy_color_red = "enemy_color: \"red\"";
    const auto enemy_color_pos = missing_color_text.find(enemy_color_red);
    check(enemy_color_pos != std::string::npos, "测试配置包含顶层 enemy_color");
    if (enemy_color_pos != std::string::npos) {
      missing_color_text.replace(enemy_color_pos, enemy_color_red.size(), "enemy_color: \"\"");
      const auto missing_gate_pos = missing_color_text.find(color_gate_off);
      check(missing_gate_pos != std::string::npos, "缺颜色配置仍可打开 sim.use_enemy_color");
      if (missing_gate_pos != std::string::npos) {
        missing_color_text.replace(missing_gate_pos, color_gate_off.size(), "use_enemy_color: true");
        std::ofstream ofs(missing_color_yaml);
        ofs << missing_color_text;
        ofs.close();
        const int code_missing_color = run_entry(
          {binary, missing_color_yaml, "--mode=closed_loop", "--allow-fire",
           "--max-command-age-ms=5000", "--duration-s=2"});
        check(code_missing_color == 2, "缺 enemy_color 时开火被拒绝（exit 2）",
              "exit=" + std::to_string(code_missing_color));
      }
    }
    ::unlink(missing_color_yaml.c_str());
  }

  std::printf("--- 心跳丢失后不再发控制 --------------------------------------------\n");
  producer.heartbeat = false;
  const std::uint64_t control_before = producer.control_cmds.load();
  const pid_t pid2 = ::fork();
  if (pid2 == 0) {
    ::execl(
      binary.c_str(), binary.c_str(), yaml_path.c_str(), "--mode=closed_loop", "--duration-s=2",
      static_cast<char *>(nullptr));
    std::_Exit(127);
  }
  ::waitpid(pid2, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "心跳丢失下入口仍能正常退出");
  check(
    producer.control_after_heartbeat_off.load() == 0, "心跳丢失后没有任何控制命令",
    std::to_string(producer.control_after_heartbeat_off.load()));
  check(producer.safe_stops.load() > 0, "期间收到安全停止命令",
        std::to_string(producer.safe_stops.load()));
  (void)control_before;

  producer.running = false;
  producer_thread.join();

  pub.destroy();
  pub.unlink_files();
  ::unlink(yaml_path.c_str());
  ::unlink(report.c_str());
  ::rmdir(dir.c_str());

  std::printf("\n检查项 %d，失败 %d\n", g_checks, g_failures);
  std::printf(
    "sim_entry_smoke_test: %s\n", g_failures == 0 ? "全部 通过" : "存在 失败");
  return g_failures == 0 ? 0 : 1;
}
