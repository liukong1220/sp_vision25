#ifndef SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP
#define SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP

// 本文件是 bevy_robomaster_simulator `crates/talos-ipc/src/layout.rs` 里
// Rust `repr(C)` 结构体的 C++ 镜像。
//
// 字段顺序、类型、对齐和填充中的任何一项不一致都会造成静默的跨进程读错位，
// 而不是一个显式的错误。因此这里对每个结构体都做 sizeof / alignof / offsetof
// 编译期断言，不允许只靠“字段名看起来一样”来判定 ABI 兼容。
//
// 上游基线：SHM_MAGIC = 0x54414C05，SHM_VERSION = 2。
//
// 三缓冲的 `state` 在 Rust 侧是 `AtomicU8`。这里刻意用裸 `std::uint8_t` 保持
// 结构体是纯 POD（可 offsetof、可 static_assert 标准布局），原子语义由
// triple_buffer_* 函数用 `__atomic_*` builtin 显式提供，内存序与 Rust 侧一一对应。

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if !defined(__GNUC__) && !defined(__clang__)
#error "sim_io shared-memory layout relies on __atomic_* builtins (GCC/Clang)."
#endif

namespace sim_io
{
// ---------------------------------------------------------------- 协议常量 --

constexpr std::uint32_t SHM_MAGIC = 0x54414C05u;
constexpr std::uint32_t SHM_VERSION = 2u;

constexpr std::uint32_t IMAGE_WIDTH = 1440u;
constexpr std::uint32_t IMAGE_HEIGHT = 1080u;
constexpr std::uint32_t IMAGE_CHANNELS = 3u;
constexpr std::size_t IMAGE_SIZE =
  static_cast<std::size_t>(IMAGE_WIDTH) * IMAGE_HEIGHT * IMAGE_CHANNELS;
constexpr std::size_t IMAGE_SLOT_COUNT = 3;
constexpr std::size_t IMAGE_POOL_SIZE = IMAGE_SIZE * IMAGE_SLOT_COUNT;

// Rust 侧 ShmRegion 把名字当作 /tmp 下的文件名，而不是 POSIX shm 对象。
constexpr const char * SHM_NAME_META = "talos_ipc_meta";
constexpr const char * SHM_NAME_IMAGE_POOL = "talos_ipc_image_pool";
constexpr const char * SHM_DIR = "/tmp";

constexpr std::uint8_t FLAG_NEW = 0x80u;
constexpr std::uint8_t INDEX_MASK = 0x03u;
constexpr std::size_t TRIPLE_SLOT_COUNT = 3;

// ImageMeta.format：0 = RGB8。上游目前只发布这一种。
constexpr std::uint8_t IMAGE_FORMAT_RGB8 = 0u;

constexpr std::size_t POSE_CHANNEL_COUNT = 5;
constexpr std::size_t GROUND_TRUTH_MAX_TARGETS = 16;
constexpr std::size_t GROUND_TRUTH_MAX_RUNES = 4;

// PoseIndex，对应 Rust `#[repr(u8)] enum PoseIndex`。
enum class PoseIndex : std::uint8_t
{
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3,
  // 遗留通道：新接入应读 ShmMetaRegion::chassis_observation。
  ChassisObservation = 4,
};

// ---------------------------------------------------------------- 数据结构 --

struct alignas(32) ImageMeta
{
  std::uint64_t seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::uint8_t pad_[6];
};

struct alignas(64) PoseMeta
{
  std::uint64_t frame_seq;
  float position[3];
  // Rust 侧发布顺序是 [w, x, y, z]。
  float quaternion[4];
  std::uint64_t timestamp_ns;
  std::uint8_t pad_[16];
};

struct alignas(32) GimbalCmd
{
  std::uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  std::uint8_t fire_advice;
  std::uint8_t pad_[11];
};

struct alignas(64) CameraInfo
{
  std::uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t pad_[24];
};

struct alignas(64) ChassisObservation
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  float dt_s;
  float v_body[2];
  float wz_radps;
  float wheel_linear_mps[4];
  float wheel_angular_radps[4];
  float a_body[2];
  float alpha_z_radps2;
  float rpy_rad[3];
  float gyro_xyz_radps[3];
  float accel_xyz_mps2[3];
  std::uint8_t pad_[16];
};

template <typename Slot>
struct alignas(64) TripleBuffer
{
  std::uint8_t state;
  std::uint8_t write_idx;
  std::uint8_t read_idx;
  std::uint8_t pad1_[61];
  Slot slots[TRIPLE_SLOT_COUNT];
};

using ImageTripleBuffer = TripleBuffer<ImageMeta>;
using PoseTripleBuffer = TripleBuffer<PoseMeta>;
using GimbalTripleBuffer = TripleBuffer<GimbalCmd>;

struct alignas(64) ShmHeader
{
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t image_width;
  std::uint32_t image_height;
  std::uint8_t pad_[32];
};

struct alignas(32) GroundTruthTarget
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t is_outpost;
  std::uint8_t pad1_;
  float position[3];
  float vyaw;
  float yaw;
  // 被选中装甲板板心在 odom 系下的真值位置。整车中心（position）不是算法瞄的点：
  // 装甲板偏心半径 r 约 0.2m、板心比车心高约 0.06m，在 1.5m 距离上就是 2 度量级，
  // 拿整车中心算瞄准误差会把这段几何差当成闭环残差。发布端置 valid=0 时该字段
  // 无意义。占用原 pad_[24] 的前 16 字节，结构体大小和其余偏移不变。
  float armor_position[3];
  std::uint8_t armor_position_valid;
  std::uint8_t pad_[11];
};

struct alignas(64) GroundTruthRune
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t rune_mode;
  std::uint8_t mechanism_state;
  std::uint8_t pad1_;
  float r_center_odom[3];
  float radius;
  float current_angle;
  float v_roll;
  std::int32_t direction;
  float sin_amplitude;
  float sin_omega;
  float sin_phase;
  float sin_offset;
  float relative_time;
  std::int32_t blade_id;
  std::uint8_t target_activations[5];
  std::uint8_t pad_[20];
};

struct alignas(64) GroundTruthBatch
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t rune_count;
  GroundTruthTarget targets[GROUND_TRUTH_MAX_TARGETS];
  GroundTruthRune runes[GROUND_TRUTH_MAX_RUNES];
  // seqlock 序号。发布端写前置奇、写完置偶；读端读到奇数或前后不一致就重试。
  // 原来靠 "memcpy 前后读 frame_seq 相等" 近似判断整块稳定，这不是同步保证：
  // 覆盖写可能只改了 targets 的中段而 frame_seq 恰好没动（同一帧内重发），
  // 也可能 frame_seq 先写、body 后写，读端于是拿到半新半旧的一批目标。
  // 占用原 pad_[64] 的前 4 字节，结构体大小和其余偏移不变。
  std::uint32_t seqlock;
  std::uint8_t pad_[60];
};

struct alignas(64) RuntimeState
{
  std::uint64_t timestamp_ns;
  std::uint8_t following;
  std::uint8_t pad_[55];
};

struct ShmMetaRegion
{
  ShmHeader header;
  ImageTripleBuffer image;
  PoseTripleBuffer poses[POSE_CHANNEL_COUNT];
  GimbalTripleBuffer gimbal_cmd;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  GroundTruthBatch ground_truth;
  RuntimeState runtime_state;
};

// ------------------------------------------------------------ ABI 编译期断言 --
//
// 下面的期望值全部来自 talos-ipc `layout.rs` 中的 `const _: () = assert!(...)`，
// 以及按 Rust repr(C) 布局规则推导出的字段偏移。断言失败即表示 C++ 镜像和
// Rust 发布端不再是同一份协议，必须先修 ABI，不允许改断言绕过。

#define SIM_IO_ASSERT_LAYOUT(type, expected_size, expected_align)             \
  static_assert(sizeof(type) == (expected_size), #type " size mismatch");     \
  static_assert(alignof(type) == (expected_align), #type " align mismatch");  \
  static_assert(std::is_standard_layout<type>::value, #type " not standard layout")

#define SIM_IO_ASSERT_OFFSET(type, field, expected)                           \
  static_assert(offsetof(type, field) == (expected), #type "::" #field " offset mismatch")

SIM_IO_ASSERT_LAYOUT(ImageMeta, 32, 32);
SIM_IO_ASSERT_OFFSET(ImageMeta, seq, 0);
SIM_IO_ASSERT_OFFSET(ImageMeta, timestamp_ns, 8);
SIM_IO_ASSERT_OFFSET(ImageMeta, width, 16);
SIM_IO_ASSERT_OFFSET(ImageMeta, height, 20);
SIM_IO_ASSERT_OFFSET(ImageMeta, buffer_id, 24);
SIM_IO_ASSERT_OFFSET(ImageMeta, format, 25);

SIM_IO_ASSERT_LAYOUT(PoseMeta, 64, 64);
SIM_IO_ASSERT_OFFSET(PoseMeta, frame_seq, 0);
SIM_IO_ASSERT_OFFSET(PoseMeta, position, 8);
SIM_IO_ASSERT_OFFSET(PoseMeta, quaternion, 20);
SIM_IO_ASSERT_OFFSET(PoseMeta, timestamp_ns, 40);
SIM_IO_ASSERT_OFFSET(PoseMeta, pad_, 48);

SIM_IO_ASSERT_LAYOUT(GimbalCmd, 32, 32);
SIM_IO_ASSERT_OFFSET(GimbalCmd, timestamp_ns, 0);
SIM_IO_ASSERT_OFFSET(GimbalCmd, yaw_deg, 8);
SIM_IO_ASSERT_OFFSET(GimbalCmd, pitch_deg, 12);
SIM_IO_ASSERT_OFFSET(GimbalCmd, distance_m, 16);
SIM_IO_ASSERT_OFFSET(GimbalCmd, fire_advice, 20);

SIM_IO_ASSERT_LAYOUT(CameraInfo, 128, 64);
SIM_IO_ASSERT_OFFSET(CameraInfo, timestamp_ns, 0);
SIM_IO_ASSERT_OFFSET(CameraInfo, fx, 8);
SIM_IO_ASSERT_OFFSET(CameraInfo, fy, 16);
SIM_IO_ASSERT_OFFSET(CameraInfo, cx, 24);
SIM_IO_ASSERT_OFFSET(CameraInfo, cy, 32);
SIM_IO_ASSERT_OFFSET(CameraInfo, distortion, 40);
SIM_IO_ASSERT_OFFSET(CameraInfo, width, 80);
SIM_IO_ASSERT_OFFSET(CameraInfo, height, 84);

SIM_IO_ASSERT_LAYOUT(ChassisObservation, 128, 64);
SIM_IO_ASSERT_OFFSET(ChassisObservation, frame_seq, 0);
SIM_IO_ASSERT_OFFSET(ChassisObservation, timestamp_ns, 8);
SIM_IO_ASSERT_OFFSET(ChassisObservation, dt_s, 16);
SIM_IO_ASSERT_OFFSET(ChassisObservation, v_body, 20);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wz_radps, 28);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wheel_linear_mps, 32);
SIM_IO_ASSERT_OFFSET(ChassisObservation, wheel_angular_radps, 48);
SIM_IO_ASSERT_OFFSET(ChassisObservation, a_body, 64);
SIM_IO_ASSERT_OFFSET(ChassisObservation, alpha_z_radps2, 72);
SIM_IO_ASSERT_OFFSET(ChassisObservation, rpy_rad, 76);
SIM_IO_ASSERT_OFFSET(ChassisObservation, gyro_xyz_radps, 88);
SIM_IO_ASSERT_OFFSET(ChassisObservation, accel_xyz_mps2, 100);

SIM_IO_ASSERT_LAYOUT(ImageTripleBuffer, 192, 64);
SIM_IO_ASSERT_LAYOUT(PoseTripleBuffer, 256, 64);
SIM_IO_ASSERT_LAYOUT(GimbalTripleBuffer, 192, 64);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, state, 0);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, write_idx, 1);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, read_idx, 2);
SIM_IO_ASSERT_OFFSET(ImageTripleBuffer, slots, 64);
SIM_IO_ASSERT_OFFSET(PoseTripleBuffer, slots, 64);
SIM_IO_ASSERT_OFFSET(GimbalTripleBuffer, slots, 64);

SIM_IO_ASSERT_LAYOUT(ShmHeader, 64, 64);
SIM_IO_ASSERT_OFFSET(ShmHeader, magic, 0);
SIM_IO_ASSERT_OFFSET(ShmHeader, version, 4);
SIM_IO_ASSERT_OFFSET(ShmHeader, created_ns, 8);
SIM_IO_ASSERT_OFFSET(ShmHeader, heartbeat_ns, 16);
SIM_IO_ASSERT_OFFSET(ShmHeader, image_width, 24);
SIM_IO_ASSERT_OFFSET(ShmHeader, image_height, 28);

SIM_IO_ASSERT_LAYOUT(GroundTruthTarget, 64, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, team, 16);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_label, 17);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, is_outpost, 18);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, position, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, vyaw, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, yaw, 36);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_position, 40);
SIM_IO_ASSERT_OFFSET(GroundTruthTarget, armor_position_valid, 52);

SIM_IO_ASSERT_LAYOUT(GroundTruthRune, 128, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, r_center_odom, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, radius, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, current_angle, 36);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, v_roll, 40);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, direction, 44);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, sin_amplitude, 48);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, relative_time, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, blade_id, 68);
SIM_IO_ASSERT_OFFSET(GroundTruthRune, target_activations, 72);

SIM_IO_ASSERT_LAYOUT(GroundTruthBatch, 1664, 64);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, target_count, 16);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, rune_count, 20);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, targets, 32);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, runes, 1088);
SIM_IO_ASSERT_OFFSET(GroundTruthBatch, seqlock, 1600);

SIM_IO_ASSERT_LAYOUT(RuntimeState, 64, 64);
SIM_IO_ASSERT_OFFSET(RuntimeState, following, 8);

SIM_IO_ASSERT_LAYOUT(ShmMetaRegion, 3712, 64);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, header, 0);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, image, 64);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, poses, 256);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, gimbal_cmd, 1536);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, camera_info, 1728);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, chassis_observation, 1856);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, ground_truth, 1984);
SIM_IO_ASSERT_OFFSET(ShmMetaRegion, runtime_state, 3648);

static_assert(IMAGE_SIZE == 4665600, "image size mismatch");
static_assert(IMAGE_POOL_SIZE == 13996800, "image pool size mismatch");

// ------------------------------------------------------- 三缓冲原子操作原语 --
//
// 与 talos-ipc `triple_buffer.rs` 严格对应：
//   producer: old = state.swap(write_idx | FLAG_NEW, AcqRel); write_idx = old & INDEX_MASK
//   consumer: 最多两次 compare_exchange_weak(expected, read_idx, AcqRel, Acquire)
// 内存序映射：Acquire -> __ATOMIC_ACQUIRE，AcqRel -> __ATOMIC_ACQ_REL。
//
// state 的低两位可以表示 3，而槽位只有 3 个。Rust 侧靠数组越界 panic 兜底，
// 这里改成显式拒绝，避免共享内存被写坏时读到区域外的数据。

inline std::uint8_t triple_buffer_load_state(const std::uint8_t * state)
{
  return __atomic_load_n(state, __ATOMIC_ACQUIRE);
}

inline bool triple_buffer_has_new_data(const std::uint8_t * state)
{
  return (triple_buffer_load_state(state) & FLAG_NEW) != 0;
}

// 初始化到 Rust `init_triple_buffer` 的状态：state=1（就绪槽 1，无 FLAG_NEW）、
// write_idx=0、read_idx=2。零填充不是合法初始状态，仅测试用发布端需要它。
template <typename Slot>
inline void triple_buffer_init(TripleBuffer<Slot> & buf)
{
  buf.write_idx = 0;
  buf.read_idx = 2;
  __atomic_store_n(&buf.state, static_cast<std::uint8_t>(1), __ATOMIC_RELEASE);
}

// 消费一个槽位。返回 nullptr 表示没有新数据，或 state 已损坏。
// 返回的指针在下一次对同一通道调用本函数之前保持有效：CAS 成功后生产者的
// write_idx 会变成我们刚归还的槽位，不会是我们正在持有的 ready 槽位。
// 便捷重载：直接对整个三缓冲取状态字。
template <typename Slot>
inline std::uint8_t triple_buffer_load_state(const TripleBuffer<Slot> & buf)
{
  return triple_buffer_load_state(&buf.state);
}

template <typename Slot>
inline bool triple_buffer_has_new_data(const TripleBuffer<Slot> & buf)
{
  return triple_buffer_has_new_data(&buf.state);
}

template <typename Slot>
inline const Slot * triple_buffer_consume(TripleBuffer<Slot> & buf, bool * corrupted = nullptr)
{
  if (corrupted != nullptr) *corrupted = false;

  std::uint8_t expected = __atomic_load_n(&buf.state, __ATOMIC_ACQUIRE);

  for (int attempt = 0; attempt < 2; ++attempt) {
    if ((expected & FLAG_NEW) == 0) return nullptr;

    const std::uint8_t ready_idx = expected & INDEX_MASK;
    if (ready_idx >= TRIPLE_SLOT_COUNT || buf.read_idx >= TRIPLE_SLOT_COUNT) {
      if (corrupted != nullptr) *corrupted = true;
      return nullptr;
    }

    const std::uint8_t desired = buf.read_idx;
    if (__atomic_compare_exchange_n(
          &buf.state, &expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buf.read_idx = ready_idx;
      return &buf.slots[ready_idx];
    }
    // CAS 失败时 expected 已被更新为当前值，直接进入第二次尝试。
  }

  return nullptr;
}

// 发布一个槽位。先写槽位内容再翻转 state，state 是消费端观察到的提交点。
template <typename Slot>
inline bool triple_buffer_publish(TripleBuffer<Slot> & buf, const Slot & value)
{
  if (buf.write_idx >= TRIPLE_SLOT_COUNT) return false;

  const std::uint8_t idx = buf.write_idx;
  buf.slots[idx] = value;

  const std::uint8_t old = __atomic_exchange_n(
    &buf.state, static_cast<std::uint8_t>(idx | FLAG_NEW), __ATOMIC_ACQ_REL);
  buf.write_idx = old & INDEX_MASK;
  if (buf.write_idx >= TRIPLE_SLOT_COUNT) buf.write_idx = 0;
  return true;
}

#undef SIM_IO_ASSERT_LAYOUT
#undef SIM_IO_ASSERT_OFFSET

}  // namespace sim_io

#endif  // SIMULATION_IO__SHARED_MEMORY_LAYOUT_HPP
