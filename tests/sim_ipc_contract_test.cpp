// 运行期 ABI 合同测试。
//
// shared_memory_layout.hpp 里的 static_assert 已经在编译期挡住了布局错误，
// 但那些期望值是我从 crates/talos-ipc/src/layout.rs 手工抄过来的。
// 这个测试把整张表在运行期打印出来，与 Rust 侧
// crates/talos-ipc/tests/layout_abi.rs 的输出逐行对照，
// 用来防止“两边各自自洽但互不一致”这种最难查的错误。

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "simulation/io/shared_memory_layout.hpp"

namespace
{
int failures = 0;

void check_u64(const char * what, unsigned long long actual, unsigned long long expected)
{
  const bool ok = actual == expected;
  if (!ok) ++failures;
  std::printf(
    "%-52s %10llu %10llu  %s\n", what, actual, expected, ok ? "ok" : "MISMATCH");
}
}  // namespace

#define CHECK_SIZE(type, expected) check_u64(#type " sizeof", sizeof(type), expected)
#define CHECK_ALIGN(type, expected) check_u64(#type " alignof", alignof(type), expected)
#define CHECK_OFFSET(type, field, expected) \
  check_u64(#type "::" #field " offset", offsetof(type, field), expected)

int main()
{
  using namespace sim_io;

  std::printf("%-52s %10s %10s\n", "item", "actual", "expected");
  std::printf("--- 常量 ---------------------------------------------------------------\n");
  check_u64("SHM_MAGIC", SHM_MAGIC, 0x54414C05ull);
  // v2 -> v3：从 ShmHeader::_pad 划出 capabilities，并把 poses[Muzzle] 由“云台局部
  // 平移”改为“枪口世界位姿”。两者都是**语义**变更，字节布局没动，所以只有版本号
  // 能拦住“旧发布端 + 新消费端”这种能 mmap 成功却静默读错语义的组合。
  check_u64("SHM_VERSION", SHM_VERSION, 3);
  check_u64("CAP_GROUND_TRUTH", CAP_GROUND_TRUTH, 1);
  check_u64("CAP_MUZZLE_WORLD_POSE", CAP_MUZZLE_WORLD_POSE, 2);
  check_u64("CAP_CHASSIS_OBSERVATION", CAP_CHASSIS_OBSERVATION, 4);
  check_u64("CAP_RUNTIME_STATE", CAP_RUNTIME_STATE, 8);
  check_u64("SIMULATOR_CAPABILITIES", SIMULATOR_CAPABILITIES, 0b1111);
  check_u64("GROUND_TRUTH_PAYLOAD_BYTES", GROUND_TRUTH_PAYLOAD_BYTES, 1600);
  check_u64("IMAGE_WIDTH", IMAGE_WIDTH, 1440);
  check_u64("IMAGE_HEIGHT", IMAGE_HEIGHT, 1080);
  check_u64("IMAGE_CHANNELS", IMAGE_CHANNELS, 3);
  check_u64("IMAGE_SIZE", IMAGE_SIZE, 4665600);
  check_u64("IMAGE_SLOT_COUNT", IMAGE_SLOT_COUNT, 3);
  check_u64("IMAGE_POOL_SIZE", IMAGE_POOL_SIZE, 13996800);
  check_u64("TRIPLE_SLOT_COUNT", TRIPLE_SLOT_COUNT, 3);
  check_u64("POSE_CHANNEL_COUNT", POSE_CHANNEL_COUNT, 5);
  check_u64("FLAG_NEW", FLAG_NEW, 0x80);
  check_u64("INDEX_MASK", INDEX_MASK, 0x03);
  check_u64("GROUND_TRUTH_MAX_TARGETS", GROUND_TRUTH_MAX_TARGETS, 16);
  check_u64("GROUND_TRUTH_MAX_RUNES", GROUND_TRUTH_MAX_RUNES, 4);
  check_u64("PoseIndex::Gimbal", static_cast<int>(PoseIndex::Gimbal), 0);
  check_u64("PoseIndex::Odom", static_cast<int>(PoseIndex::Odom), 1);
  check_u64("PoseIndex::Muzzle", static_cast<int>(PoseIndex::Muzzle), 2);
  check_u64("PoseIndex::Camera", static_cast<int>(PoseIndex::Camera), 3);
  check_u64("PoseIndex::ChassisObservation", static_cast<int>(PoseIndex::ChassisObservation), 4);

  if (std::strcmp(SHM_NAME_META, "talos_ipc_meta") != 0) {
    std::printf("SHM_NAME_META MISMATCH: %s\n", SHM_NAME_META);
    ++failures;
  }
  if (std::strcmp(SHM_NAME_IMAGE_POOL, "talos_ipc_image_pool") != 0) {
    std::printf("SHM_NAME_IMAGE_POOL MISMATCH: %s\n", SHM_NAME_IMAGE_POOL);
    ++failures;
  }

  std::printf("--- 结构体大小与对齐 ---------------------------------------------------\n");
  CHECK_SIZE(ImageMeta, 32);
  CHECK_ALIGN(ImageMeta, 32);
  CHECK_SIZE(PoseMeta, 64);
  CHECK_ALIGN(PoseMeta, 64);
  CHECK_SIZE(GimbalCmd, 32);
  CHECK_ALIGN(GimbalCmd, 32);
  CHECK_SIZE(CameraInfo, 128);
  CHECK_ALIGN(CameraInfo, 64);
  CHECK_SIZE(ChassisObservation, 128);
  CHECK_ALIGN(ChassisObservation, 64);
  CHECK_SIZE(ShmHeader, 64);
  CHECK_ALIGN(ShmHeader, 64);
  CHECK_SIZE(GroundTruthTarget, 64);
  CHECK_SIZE(GroundTruthRune, 128);
  CHECK_SIZE(GroundTruthBatch, 1664);
  CHECK_SIZE(RuntimeState, 64);
  CHECK_SIZE(ImageTripleBuffer, 192);
  CHECK_ALIGN(ImageTripleBuffer, 64);
  CHECK_SIZE(PoseTripleBuffer, 256);
  CHECK_ALIGN(PoseTripleBuffer, 64);
  CHECK_SIZE(GimbalTripleBuffer, 192);
  CHECK_ALIGN(GimbalTripleBuffer, 64);
  CHECK_SIZE(ShmMetaRegion, 3712);

  std::printf("--- 字段偏移 -----------------------------------------------------------\n");
  CHECK_OFFSET(ImageMeta, seq, 0);
  CHECK_OFFSET(ImageMeta, timestamp_ns, 8);
  CHECK_OFFSET(ImageMeta, width, 16);
  CHECK_OFFSET(ImageMeta, height, 20);
  CHECK_OFFSET(ImageMeta, buffer_id, 24);
  CHECK_OFFSET(ImageMeta, format, 25);

  CHECK_OFFSET(PoseMeta, frame_seq, 0);
  CHECK_OFFSET(PoseMeta, position, 8);
  CHECK_OFFSET(PoseMeta, quaternion, 20);
  CHECK_OFFSET(PoseMeta, timestamp_ns, 40);

  CHECK_OFFSET(GimbalCmd, timestamp_ns, 0);
  CHECK_OFFSET(GimbalCmd, yaw_deg, 8);
  CHECK_OFFSET(GimbalCmd, pitch_deg, 12);
  CHECK_OFFSET(GimbalCmd, distance_m, 16);
  CHECK_OFFSET(GimbalCmd, fire_advice, 20);

  CHECK_OFFSET(ShmHeader, magic, 0);
  CHECK_OFFSET(ShmHeader, version, 4);
  CHECK_OFFSET(ShmHeader, created_ns, 8);
  CHECK_OFFSET(ShmHeader, heartbeat_ns, 16);
  CHECK_OFFSET(ShmHeader, image_width, 24);
  CHECK_OFFSET(ShmHeader, image_height, 28);
  // capabilities 必须落在原 _pad 的起始处（32），否则 v2 的其余偏移会整体平移，
  // 与 Rust 侧手写镜像错开。
  CHECK_OFFSET(ShmHeader, capabilities, 32);

  CHECK_OFFSET(CameraInfo, timestamp_ns, 0);
  CHECK_OFFSET(CameraInfo, fx, 8);
  CHECK_OFFSET(CameraInfo, fy, 16);
  CHECK_OFFSET(CameraInfo, cx, 24);
  CHECK_OFFSET(CameraInfo, cy, 32);
  CHECK_OFFSET(CameraInfo, distortion, 40);
  CHECK_OFFSET(CameraInfo, width, 80);
  CHECK_OFFSET(CameraInfo, height, 84);

  CHECK_OFFSET(GroundTruthTarget, frame_seq, 0);
  CHECK_OFFSET(GroundTruthTarget, timestamp_ns, 8);
  CHECK_OFFSET(GroundTruthTarget, team, 16);
  CHECK_OFFSET(GroundTruthTarget, armor_label, 17);
  CHECK_OFFSET(GroundTruthTarget, is_outpost, 18);
  CHECK_OFFSET(GroundTruthTarget, position, 20);
  CHECK_OFFSET(GroundTruthTarget, vyaw, 32);
  CHECK_OFFSET(GroundTruthTarget, yaw, 36);

  CHECK_OFFSET(GroundTruthBatch, frame_seq, 0);
  CHECK_OFFSET(GroundTruthBatch, timestamp_ns, 8);
  CHECK_OFFSET(GroundTruthBatch, target_count, 16);
  CHECK_OFFSET(GroundTruthBatch, rune_count, 20);
  CHECK_OFFSET(GroundTruthBatch, targets, 32);
  CHECK_OFFSET(GroundTruthBatch, runes, 1088);
  // seqlock 的偏移就是 payload 长度：两端拷贝时都只拷这段前缀，标记本身只做原子访问。
  CHECK_OFFSET(GroundTruthBatch, seqlock, 1600);

  CHECK_OFFSET(RuntimeState, timestamp_ns, 0);
  CHECK_OFFSET(RuntimeState, following, 8);

  std::printf("--- 三缓冲字段偏移 -----------------------------------------------------\n");
  CHECK_OFFSET(ImageTripleBuffer, state, 0);
  CHECK_OFFSET(ImageTripleBuffer, write_idx, 1);
  CHECK_OFFSET(ImageTripleBuffer, read_idx, 2);
  CHECK_OFFSET(ImageTripleBuffer, slots, 64);
  CHECK_OFFSET(PoseTripleBuffer, slots, 64);
  CHECK_OFFSET(GimbalTripleBuffer, slots, 64);

  std::printf("--- ShmMetaRegion 布局 -------------------------------------------------\n");
  CHECK_OFFSET(ShmMetaRegion, header, 0);
  CHECK_OFFSET(ShmMetaRegion, image, 64);
  CHECK_OFFSET(ShmMetaRegion, poses, 256);
  CHECK_OFFSET(ShmMetaRegion, gimbal_cmd, 1536);
  CHECK_OFFSET(ShmMetaRegion, camera_info, 1728);
  CHECK_OFFSET(ShmMetaRegion, chassis_observation, 1856);
  CHECK_OFFSET(ShmMetaRegion, ground_truth, 1984);
  CHECK_OFFSET(ShmMetaRegion, runtime_state, 3648);

  std::printf("--- 三缓冲初始状态 -----------------------------------------------------\n");
  {
    // 零填充是非法初始状态；这里验证 triple_buffer_init 与 Rust 侧一致，
    // 并且刚初始化时消费端确实读不到数据。
    ImageTripleBuffer buf{};
    std::memset(&buf, 0, sizeof(buf));
    triple_buffer_init(buf);
    check_u64("init state", buf.state, 1);
    check_u64("init write_idx", buf.write_idx, 0);
    check_u64("init read_idx", buf.read_idx, 2);
    check_u64("init has_new_data", triple_buffer_has_new_data(buf) ? 1 : 0, 0);

    ImageMeta m{};
    m.seq = 7;
    triple_buffer_publish(buf, m);
    check_u64("after publish has_new_data", triple_buffer_has_new_data(buf) ? 1 : 0, 1);
    const ImageMeta * got = triple_buffer_consume(buf);
    check_u64("consume seq", got != nullptr ? got->seq : 0, 7);
    check_u64("consume twice returns null", triple_buffer_consume(buf) == nullptr ? 1 : 0, 1);
  }

  std::printf("\n");
  if (failures == 0) {
    std::printf("sim_ipc_contract_test: 全部 %s\n", "通过");
    return 0;
  }
  std::printf("sim_ipc_contract_test: %d 项不匹配\n", failures);
  return 1;
}
