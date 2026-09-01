// 真并发的 seqlock 压力测试：一个写线程持续覆盖真值区，一个读线程持续用
// SharedMemoryClient::read_ground_truth() 去读，验证两条性质：
//   1. 读端永远不会把奇数标记（"正在写"）当成一次成功读取交出去；
//   2. 读端拿到的每一批都是**同一代**数据，不存在半新半旧。
//
// 为什么必须真并发：单线程里"写一半 -> 读 -> 写完 -> 读"只能验证读端会拒绝一个
// **静止**的奇数标记。它验证不了真正的竞态——载荷正在被改写时读端恰好插进来。
// 那条路径需要的保证是两道 release/acquire 栅栏和"标记只做原子访问"，而这两点
// 在静态场景下即使全部去掉，测试依然会通过。
//
// 第 2 节（同帧真值事务）验证的是另一件事：真值必须与图像在同一次发布事务里提交，
// 消费端也必须在事务窗口里取它，于是 seq_mismatches 恒为 0。它同样必须真并发：
// 单线程"发一帧 -> 立刻取"永远不会暴露"取得太晚"这个问题，而流水线里检测一帧要
// ~250 ms，正是那段时间让发布端跑到了后面的帧上。
//
// 判据 2 靠"代数戳"实现：把同一个 generation 写进 batch 里每一个 target / rune 的
// 每一个数值字段。一批数据里只要有任何一个字段的代数与 frame_seq 不一致，就说明
// 读端看到的是两次写入的混合体。仅比较 frame_seq 前后相等是不够的：同一帧号内
// 重发时 frame_seq 根本不变，撕裂完全不会被察觉。
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simulation/io/shared_memory_client.hpp"
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
  std::printf("%-52s %s", name.c_str(), ok ? "ok" : "失败");
  if (!detail.empty()) std::printf("  %s", detail.c_str());
  std::printf("\n");
}

using sim_io::GROUND_TRUTH_MAX_RUNES;
using sim_io::GROUND_TRUTH_MAX_TARGETS;
using sim_io::GroundTruthBatch;

// 代数上限：float 能精确表示的整数只到 2^24，超过之后 static_cast<float>(gen) 会
// 舍入，两个相邻代数可能映射到同一个 float，撕裂就检测不出来了。到上限就停。
constexpr std::uint32_t GEN_LIMIT = 1u << 24;

// 把同一个代数写进 batch 里每一个数值字段。
//
// 8 位字段只能装下 gen 的低 8 位，所以"两代恰好相差 256 的整数倍"这一种混合它们
// 认不出来；float / u64 字段是全量程的，同一批里只要有一处混合就会被抓到。
void stamp(GroundTruthBatch * b, std::uint32_t gen)
{
  const float f = static_cast<float>(gen);
  const auto u8 = static_cast<std::uint8_t>(gen & 0xffu);

  std::memset(b, 0, sizeof(*b));
  b->frame_seq = gen;
  b->timestamp_ns = gen;
  // 计数字段固定打满，读端才会扫描全部 16 个目标 / 4 个符文。若跟着代数变，
  // 读端只检查前 count 个，后半段的撕裂就被漏掉了。
  b->target_count = GROUND_TRUTH_MAX_TARGETS;
  b->rune_count = GROUND_TRUTH_MAX_RUNES;

  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_TARGETS; ++i) {
    auto & t = b->targets[i];
    t.frame_seq = gen;
    t.timestamp_ns = gen;
    t.team = u8;
    t.armor_label = u8;
    t.is_outpost = u8;
    t.position[0] = t.position[1] = t.position[2] = f;
    t.vyaw = f;
    t.yaw = f;
    t.armor_position[0] = t.armor_position[1] = t.armor_position[2] = f;
    t.armor_position_valid = u8;
  }
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_RUNES; ++i) {
    auto & r = b->runes[i];
    r.frame_seq = gen;
    r.timestamp_ns = gen;
    r.team = u8;
    r.rune_mode = u8;
    r.mechanism_state = u8;
    r.r_center_odom[0] = r.r_center_odom[1] = r.r_center_odom[2] = f;
    r.radius = f;
    r.current_angle = f;
    r.v_roll = f;
    r.direction = static_cast<std::int32_t>(gen);
    r.sin_amplitude = f;
    r.sin_omega = f;
    r.sin_phase = f;
    r.sin_offset = f;
    r.relative_time = f;
    r.blade_id = static_cast<std::int32_t>(gen);
    for (auto & a : r.target_activations) a = u8;
  }
}

// 以 batch.frame_seq 为基准代数，返回第一个代数不符的字段名；整批自洽时返回空串。
std::string find_mixed_field(const GroundTruthBatch & b)
{
  const auto gen = static_cast<std::uint32_t>(b.frame_seq);
  const float f = static_cast<float>(gen);
  const auto u8 = static_cast<std::uint8_t>(gen & 0xffu);
  char buf[80];

  if (b.timestamp_ns != gen) return "batch.timestamp_ns";
  if (b.target_count != GROUND_TRUTH_MAX_TARGETS) return "batch.target_count";
  if (b.rune_count != GROUND_TRUTH_MAX_RUNES) return "batch.rune_count";

  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_TARGETS; ++i) {
    const auto & t = b.targets[i];
    const char * bad = nullptr;
    if (t.frame_seq != gen) bad = "frame_seq";
    else if (t.timestamp_ns != gen) bad = "timestamp_ns";
    else if (t.team != u8) bad = "team";
    else if (t.armor_label != u8) bad = "armor_label";
    else if (t.is_outpost != u8) bad = "is_outpost";
    else if (t.position[0] != f || t.position[1] != f || t.position[2] != f) bad = "position";
    else if (t.vyaw != f) bad = "vyaw";
    else if (t.yaw != f) bad = "yaw";
    else if (t.armor_position[0] != f || t.armor_position[1] != f || t.armor_position[2] != f)
      bad = "armor_position";
    else if (t.armor_position_valid != u8) bad = "armor_position_valid";
    if (bad != nullptr) {
      std::snprintf(buf, sizeof(buf), "targets[%zu].%s", i, bad);
      return buf;
    }
  }
  for (std::size_t i = 0; i < GROUND_TRUTH_MAX_RUNES; ++i) {
    const auto & r = b.runes[i];
    const char * bad = nullptr;
    if (r.frame_seq != gen) bad = "frame_seq";
    else if (r.timestamp_ns != gen) bad = "timestamp_ns";
    else if (r.team != u8) bad = "team";
    else if (r.rune_mode != u8) bad = "rune_mode";
    else if (r.mechanism_state != u8) bad = "mechanism_state";
    else if (r.r_center_odom[0] != f || r.r_center_odom[1] != f || r.r_center_odom[2] != f)
      bad = "r_center_odom";
    else if (r.radius != f) bad = "radius";
    else if (r.current_angle != f) bad = "current_angle";
    else if (r.v_roll != f) bad = "v_roll";
    else if (r.direction != static_cast<std::int32_t>(gen)) bad = "direction";
    else if (r.sin_amplitude != f) bad = "sin_amplitude";
    else if (r.sin_omega != f) bad = "sin_omega";
    else if (r.sin_phase != f) bad = "sin_phase";
    else if (r.sin_offset != f) bad = "sin_offset";
    else if (r.relative_time != f) bad = "relative_time";
    else if (r.blade_id != static_cast<std::int32_t>(gen)) bad = "blade_id";
    else {
      for (std::size_t k = 0; k < 5; ++k) {
        if (r.target_activations[k] != u8) {
          bad = "target_activations";
          break;
        }
      }
    }
    if (bad != nullptr) {
      std::snprintf(buf, sizeof(buf), "runes[%zu].%s", i, bad);
      return buf;
    }
  }
  return {};
}
// ---------------------------------------------------------------------------
// 第 2 节：同帧真值事务的长期一致性。
//
// 协议 v3 规定图像、同帧姿态、同帧真值三者的 frame_seq 严格相等，没有任何允许的
// 偏移；实现依据是发布端把真值写在图像 meta（唯一的提交标记）之前，消费端在
// consume_frame() 的"图像已消费、pose 未排空"窗口里把它拷走。所以本节的判据是
// seq_mismatches == 0 —— 不是"很小"，是 0。
//
// 关键在于消费端刻意**推迟**评估：拿到帧之后先睡一小会儿再 fetch()，复刻自瞄
// 流水线里检测 + 解算的 ~250 ms。这段时间背压已经放开（pose 通道排空了），发布端
// 会立刻推进到下一帧并覆盖真值槽位，于是"到评估时才去现读槽位"必然读到更新的
// 批次。旧实现就是这么做的，实测 seq_mismatches 占 frames_ok 的 10~17%。
void run_ground_truth_transaction_test()
{
  std::printf("\n--- 同帧真值事务（长期） --------------------------------------------\n");

  char dir_buf[128];
  std::snprintf(dir_buf, sizeof(dir_buf), "/tmp/sim_shm_gt_%d", static_cast<int>(::getpid()));
  const std::string dir = dir_buf;
  if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
    check(false, "创建临时目录", dir);
    return;
  }

  sim_io::testing::FakePublisher::Options pub_opt;
  pub_opt.dir = dir;
  sim_io::testing::FakePublisher pub(pub_opt);

  std::string err;
  if (!pub.create(&err)) {
    check(false, "FakePublisher 创建", err);
    ::rmdir(dir.c_str());
    return;
  }

  sim_io::SharedMemoryClient::Options cli_opt;
  cli_opt.dir = dir;
  sim_io::SharedMemoryClient client(cli_opt);
  if (!client.open(&err)) {
    check(false, "SharedMemoryClient 打开", err);
    pub.unlink_files();
    ::rmdir(dir.c_str());
    return;
  }

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> published{0};

  // 发布线程：完整走事务（判据 -> pose -> 真值 -> 图像 meta），帧号连续前进。
  // 图像像素传 nullptr：本节验证的是帧号一致性，4.6 MB/帧的拷贝只会拖慢采样。
  std::thread writer([&] {
    static const float quat[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    std::uint64_t seq = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      const std::uint64_t t = sim_io::testing::FakePublisher::now_ns();
      GroundTruthBatch gt{};
      gt.frame_seq = seq;
      gt.timestamp_ns = t;
      gt.target_count = 1;
      gt.targets[0].frame_seq = seq;
      gt.targets[0].timestamp_ns = t;
      gt.targets[0].team = sim_io::GT_TEAM_BLUE;
      gt.targets[0].armor_label = 3;
      gt.targets[0].position[0] = 3.0f;
      gt.targets[0].position[2] = 0.2f;
      if (pub.try_publish_synchronized_frame(
            nullptr, seq, t, quat, nullptr, nullptr, nullptr, &gt)) {
        published.store(seq, std::memory_order_relaxed);
        ++seq;
      } else {
        std::this_thread::yield();  // 背压未放行：上一帧还没被排空
      }
    }
  });

  sim_io::GroundTruthEvaluator ev(client, sim_io::GT_TEAM_BLUE);

  std::uint64_t frames_ok = 0;
  std::uint64_t fetched = 0;
  // 对照量：到评估时刻才去现读槽位，会读到哪一帧。它必须**经常**与图像帧号不等，
  // 否则说明这次运行里发布端根本没机会推进，判据 seq_mismatches==0 就是空的。
  std::uint64_t stale_slot_reads = 0;
  std::uint64_t direct_reads = 0;

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2500);
  sim_io::FrameBundle bundle;
  while (std::chrono::steady_clock::now() < deadline) {
    if (client.consume_frame(&bundle) != sim_io::ConsumeStatus::Ok) {
      std::this_thread::yield();
      continue;
    }
    ++frames_ok;

    // 复刻流水线耗时。真机上这里是 YOLO + 解算 + 规划，约 250 ms；这里只需要长到
    // 让发布端把下一帧发出去（背压刚被本次 consume_frame 放开）。
    std::this_thread::sleep_for(std::chrono::microseconds(500));

    GroundTruthBatch direct{};
    if (client.read_ground_truth(&direct)) {
      ++direct_reads;
      if (direct.frame_seq != bundle.frame_seq) ++stale_slot_reads;
    }

    if (ev.fetch(bundle.frame_seq)) ++fetched;
  }

  stop.store(true, std::memory_order_relaxed);
  writer.join();

  char detail[256];
  std::snprintf(
    detail, sizeof(detail), "发布 %llu 帧，消费 %llu 帧，评估取到 %llu 帧",
    static_cast<unsigned long long>(published.load()),
    static_cast<unsigned long long>(frames_ok), static_cast<unsigned long long>(fetched));
  std::printf("事务统计: %s\n", detail);

  // 前提：样本量够，否则下面的 ==0 判据没有意义。
  check(frames_ok >= 300, "长期运行：至少消费 300 帧", detail);

  std::snprintf(
    detail, sizeof(detail), "现读槽位 %llu 次，其中 %llu 次帧号不等",
    static_cast<unsigned long long>(direct_reads),
    static_cast<unsigned long long>(stale_slot_reads));
  check(stale_slot_reads > 0, "对照：延后现读槽位确实会读到别的帧（判据非空）", detail);

  std::snprintf(
    detail, sizeof(detail), "seq_mismatches=%llu skew[%lld, %lld] samples=%llu",
    static_cast<unsigned long long>(ev.seq_mismatches()),
    static_cast<long long>(ev.seq_skew_min()), static_cast<long long>(ev.seq_skew_max()),
    static_cast<unsigned long long>(ev.seq_skew_samples()));
  check(ev.seq_mismatches() == 0, "seq_mismatches == 0（协议不允许任何偏移）", detail);
  check(ev.seq_skew_samples() == 0, "没有任何一帧产生 skew 样本", detail);
  check(fetched == frames_ok, "每一个成功消费的帧都取到了同帧真值", detail);
  check(
    client.ground_truth_captures() == frames_ok, "ground_truth_captures 与消费帧数一致",
    detail);

  std::snprintf(
    detail, sizeof(detail), "regressed=%llu corrupted=%llu dropped=%llu restarts=%llu",
    static_cast<unsigned long long>(client.regressed_frames()),
    static_cast<unsigned long long>(client.corrupted_events()),
    static_cast<unsigned long long>(client.dropped_frames()),
    static_cast<unsigned long long>(client.publisher_restarts()));
  check(
    client.regressed_frames() == 0 && client.corrupted_events() == 0 &&
      client.dropped_frames() == 0 && client.publisher_restarts() == 0,
    "长期运行中没有回退/损坏/丢弃/换代", detail);

  client.close();
  pub.destroy();
  pub.unlink_files();
  ::rmdir(dir.c_str());
}
}  // namespace

int main()
{
  // 先自检探测器本身。如果 find_mixed_field 恒返回空串，下面的压力测试会 100%
  // 通过而什么都没验证——一个永真的判据比没有判据更糟。
  {
    GroundTruthBatch a{};
    stamp(&a, 1000);
    check(find_mixed_field(a).empty(), "自检：单代数据被判为自洽");
    GroundTruthBatch b{};
    stamp(&b, 2000);
    // 模拟"前半段是新数据、后半段还是旧数据"的撕裂：只覆盖 batch 头部与前 3 个目标。
    std::memcpy(&a, &b, offsetof(GroundTruthBatch, targets) + 3 * sizeof(a.targets[0]));
    const std::string mixed = find_mixed_field(a);
    check(!mixed.empty(), "自检：半新半旧数据被判为撕裂", mixed);
  }

  char dir_buf[128];
  std::snprintf(dir_buf, sizeof(dir_buf), "/tmp/sim_shm_stress_%d", static_cast<int>(::getpid()));
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

  sim_io::SharedMemoryClient::Options cli_opt;
  cli_opt.dir = dir;
  sim_io::SharedMemoryClient client(cli_opt);
  if (!client.open(&err)) {
    std::printf("SharedMemoryClient 打开失败: %s\n", err.c_str());
    pub.unlink_files();
    ::rmdir(dir.c_str());
    return 2;
  }

  // 先提交一代，保证读端一开始就有可读数据（seqlock != 0）。
  {
    GroundTruthBatch first{};
    stamp(&first, 1);
    pub.set_ground_truth(first);
  }

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> writes{0};

  std::thread writer([&] {
    GroundTruthBatch batch{};
    std::uint32_t gen = 2;
    while (!stop.load(std::memory_order_relaxed) && gen < GEN_LIMIT) {
      stamp(&batch, gen);
      pub.set_ground_truth(batch);
      writes.store(gen, std::memory_order_relaxed);
      ++gen;
    }
  });

  // 读端统计：成功读取次数、见到的不同代数个数、以及两类违约。
  std::uint64_t reads = 0;
  std::uint64_t read_failures = 0;
  std::uint64_t odd_markers = 0;
  std::uint64_t zero_markers = 0;
  std::string first_mixed;
  std::uint64_t mixed_count = 0;
  std::uint32_t min_gen = 0xffffffffu;
  std::uint32_t max_gen = 0;
  std::vector<std::uint32_t> distinct;

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  GroundTruthBatch got{};
  while (std::chrono::steady_clock::now() < deadline) {
    if (!client.read_ground_truth(&got)) {
      ++read_failures;  // 8 次重试都撞上写入窗口，属于允许的结果
      continue;
    }
    ++reads;
    if ((got.seqlock & 1u) != 0u) ++odd_markers;
    if (got.seqlock == 0u) ++zero_markers;
    const std::string mixed = find_mixed_field(got);
    if (!mixed.empty()) {
      ++mixed_count;
      if (first_mixed.empty()) {
        char buf[160];
        std::snprintf(
          buf, sizeof(buf), "gen=%u marker=%u 字段=%s", static_cast<std::uint32_t>(got.frame_seq),
          got.seqlock, mixed.c_str());
        first_mixed = buf;
      }
    }
    const auto gen = static_cast<std::uint32_t>(got.frame_seq);
    if (gen < min_gen) min_gen = gen;
    if (gen > max_gen) max_gen = gen;
    if (distinct.empty() || distinct.back() != gen) distinct.push_back(gen);
  }

  stop.store(true, std::memory_order_relaxed);
  writer.join();

  char detail[256];
  std::snprintf(
    detail, sizeof(detail), "成功读 %llu 次，读重试失败 %llu 次，写 %llu 代",
    static_cast<unsigned long long>(reads), static_cast<unsigned long long>(read_failures),
    static_cast<unsigned long long>(writes.load()));
  std::printf("并发统计: %s\n", detail);

  // 这三条是"测试确实跑起来了"的前提。少了它们，一个读不到任何数据的空转循环
  // 也会报全绿。
  check(reads > 0, "读端至少成功读到一次");
  check(writes.load() >= 100, "写端至少提交了 100 代", detail);
  check(distinct.size() >= 10, "读端观察到至少 10 个不同代数（读写确有交叠）");

  std::snprintf(
    detail, sizeof(detail), "代数区间 [%u, %u]，不同代数 %zu 个", min_gen, max_gen,
    distinct.size());
  check(max_gen > min_gen, "读端看到的代数在推进", detail);

  check(odd_markers == 0, "没有任何一次成功读取带奇数 marker");
  check(zero_markers == 0, "没有任何一次成功读取带 marker=0");
  check(mixed_count == 0, "没有任何一批是半新半旧", first_mixed);

  client.close();
  pub.destroy();
  pub.unlink_files();
  ::rmdir(dir.c_str());

  run_ground_truth_transaction_test();

  std::printf("\n共 %d 项检查，%d 项失败\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
