// sim_auto_aim 入口的离线端到端冒烟测试。
//
// 用 FakePublisher 顶替 Bevy：本进程既是图像/姿态的生产者，也是云台命令的消费者，
// 于是整条入口链路（yaml 加载 -> 内参自检 -> YOLO/Tracker/Planner 构造 -> 主循环
// -> 指标汇总）可以在没有 GPU、没有仿真器、没有任何硬件的机器上跑完。
//
// 它验证三件在真机上代价很高、在这里几乎免费的事：
//   1. 入口能在完全干净的环境里跑起来并正常退出（不是崩在半路）；
//   2. 默认（不带 --allow-fire）时 fire_advice 恒为 0，且计入 suppressed；
//   3. 心跳停掉之后入口不再发控制命令，只发安全停止。
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
