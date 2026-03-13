# 步兵（standard）串口+MPC说明与`standard3.yaml`调参指南

## 1. `standard_mpc` vs `auto_aim_debug_mpc` (核心区别)

两者算法主链一致，均为 `YOLO -> Solver -> Tracker -> Planner -> Gimbal.send`，但工程目的不同：

- **`standard_mpc.cpp`**: 实战入口，支持自瞄与打符模式切换。
- **`auto_aim_debug_mpc.cpp`**: MPC专项调试入口，仅自瞄，但输出大量调试曲线用于分析。

两者在代码行为上的具体对比如下：

| 维度 | `standard_mpc.cpp` | `auto_aim_debug_mpc.cpp` |
|---|---|---|
| 模式支持 | AUTO_AIM + SMALL_BUFF + BIG_BUFF | 仅 AUTO_AIM |
| 通信 | 串口 `io::Gimbal` | 串口 `io::Gimbal` |
| 规划线程 | 有 `plan_thread` | 有 `plan_thread` |
| 目标输入 | `target_queue`，有目标则push，无目标push nullopt | 同样 `target_queue` |
| 调试数据 | 已补充 `plotter` 曲线（云台/目标/plan/发射） | 更偏自瞄专项调试 |
| **推荐用途** | **比赛/联调主程序** | **调参/定位问题** |

**关键代码位置:**
- `standard_mpc` 规划发送: `src/standard_mpc.cpp:69`
- `standard_mpc` plotter输出: `src/standard_mpc.cpp:74`
- `standard_mpc` 模式切换（含打符）: `src/standard_mpc.cpp:127`
- `auto_aim_debug_mpc` 调试绘图: `src/auto_aim_debug_mpc.cpp:73`

## 2. 关于 "standard_mpc看不到串口通信" 的解释

`standard_mpc.cpp` 里看不到底层 `serial.write/read` 是正常的，因为它通过 `io::Gimbal` 类封装了串口通信细节：

1. **构造时打开串口**: `io/gimbal/gimbal.cpp:17`
2. **后台线程持续读取下位机数据**（姿态、模式、弹速）: `io/gimbal/gimbal.cpp:196`
3. **控制指令通过串口发送**: `io/gimbal/gimbal.cpp:142`

因此，在 `standard_mpc.cpp` 的主逻辑中，只需调用高级接口即可，实现了上层调度与底层串口的解耦：
- **读状态**: `gimbal.mode()`, `gimbal.state()`, `gimbal.q(t)`
- **发控制**: `gimbal.send(...)`

## 3. `standard.cpp` 的改造：切换为串口+MPC

`src/standard.cpp` 文件已被修改，以适配串口+MPC的链路。主要改动包括：

1. 移除 `io::CBoard`（CAN通信）。
2. 接入 `io::Gimbal`（串口通信）。
3. 移除 `Aimer+Shooter`，改用 `Planner` 进行规划。
4. 仅在 `AUTO_AIM` 模式下执行检测、跟踪和规划，其余模式发送零指令。

**关键代码位置:**
- 串口对象: `src/standard.cpp:30`
- 模式读取: `src/standard.cpp:48`
- 目标规划: `src/standard.cpp:63`
- 指令发送: `src/standard.cpp:64`

**编译验证:**
- `cmake --build build -j4 --target standard` 已通过。

## 4. `configs/standard3.yaml` 参数调节指南

当前 `standard.cpp`（改造后）实际会用到的参数分组如下。

### 4.1 串口与模式 (Gimbal)
- **参数**: `com_port`
- **作用**: 定义视觉与下位机通信的串口端口。
- **调节建议**:
    1. 确保设备名稳定（如 `/dev/ttyACM0`）。
    2. 若偶发断连，优先排查系统层面的串口权限、供电或线缆问题，而不是立即调整算法参数。

### 4.2 相机输入 (Camera)
- **参数** (`camera_name: hikrobot` 时): `exposure_ms`, `gain`, `vid_pid`
- **作用**: 直接影响检测的输入图像质量与噪声水平。
- **调节顺序**:
    1. 先确定 `exposure_ms`，保证运动目标不过曝、不过暗。
    2. 再调整 `gain`，注意增益越大，图像噪声也越大。
    3. `vid_pid` 仅用于设备识别和重连。

### 4.3 检测 (YOLOV5)
- **参数**: `yolo_name`, `yolov5_model_path`, `device`, `min_confidence`, `use_roi`, `roi.*`
- **作用**: 控制检测的召回率与误检率；`use_roi` 可用于裁切图像以提速。
- **调节建议**:
    1. 先关闭 `use_roi` 进行全图稳定性验证。
    2. 验证通过后，再开启 `use_roi` 缩小搜索范围以提升帧率。
    3. `min_confidence` 从高到低调节，找到“误检可接受、漏检不致命”的平衡点。

### 4.4 几何解算 (Solver)
- **参数**: `R_gimbal2imubody`, `R_camera2gimbal`, `t_camera2gimbal`, `camera_matrix`, `distort_coeffs`
- **作用**: 决定世界坐标解算的精度，是后续跟踪和预测的基础。
- **调节建议**:
    1. 调节解算参数时，应固定检测参数，避免多变量干扰。
    2. 使用重投影误差和实弹射击偏差共同评估解算效果。
    3. 外参与内参必须成套更新，切勿混用不同标定版本的数据。

### 4.5 跟踪 (Tracker)
- **参数**: `enemy_color`, `min_detect_count`, `max_temp_lost_count`, `outpost_max_temp_lost_count`
- **作用**: 决定目标确认的速度与抗丢失的能力。
- **调节建议**:
    1. **若容易锁错目标**: 增大 `min_detect_count`。
    2. **若转头后立刻丢目标**: 增大 `max_temp_lost_count`。
    3. 前哨站的丢失阈值应独立调节，不与普通装甲板共用。

### 4.6 MPC规划 (Planner)
- **参数**:
    - `yaw_offset`, `pitch_offset`
    - `decision_speed`, `high_speed_delay_time`, `low_speed_delay_time`
    - `fire_thresh`
    - `max_yaw_acc`, `Q_yaw`, `R_yaw`
    - `max_pitch_acc`, `Q_pitch`, `R_pitch`
- **作用**: 决定规划的响应速度、平滑度和开火时机。
- **调节主线 (推荐顺序)**:
    1. **静态对准**: 调节 `yaw_offset`/`pitch_offset`。
    2. **动态提前量**: 调节 `high/low_speed_delay_time`。
    3. **响应与平滑**: 调节 `Q/R` 和 `max_*_acc` 以平衡“快”与“抖”。
    4. **开火窗口**: 调节 `fire_thresh` 控制开火的灵敏度。
- **常见问题与对策**:
    - **抖动大**: 增大 `R_*` 或降低 `max_*_acc`。
    - **跟随慢**: 增大 `Q_*` 或提高 `max_*_acc`。
    - **打点滞后**: 减小延时补偿的绝对值（或重新标定链路延时）。
    - **空放多**: 减小 `fire_thresh`。

### 4.7 不生效/不直接生效的参数
在当前 `standard.cpp`（串口+MPC）链路中，以下参数不会直接进入控制主链：
- `Shooter` 参数: `first_tolerance`, `second_tolerance`, `judge_distance`, `auto_fire`
- `Aimer` 参数: `comming_angle`, `leaving_angle`
- `CBoard` (CAN) 参数: `quaternion_canid`, `bullet_speed_canid`, `send_canid`, `can_interface`

> **注意**: 这些参数仍可能被项目中的其它程序（如 `mt_standard.cpp`, `uav.cpp`, `sentry.cpp`）使用。

## 5. 运行指南

- **运行修改后的 `standard` 程序**:
  ```bash
  ./build/standard configs/standard3.yaml
  ```

- **进行MPC专项调参**:
  ```bash
  ./build/auto_aim_debug_mpc configs/standard3.yaml
  ```

- **运行完整步兵模式（自瞄+打符）**:
  ```bash
  ./build/standard_mpc configs/standard3.yaml
  ```

## 6. 调参快速参考

### 6.1 图像输入
- **参数**: `exposure_ms`, `gain`
- **目标**: 稳定识别，为后续控制提供高质量输入。

### 6.2 坐标解算
- **参数**: `R_camera2gimbal`, `t_camera2gimbal`, `camera_matrix`, `distort_coeffs`
- **目标**: 保证重投影稳定，避免后续补偿参数“假调”。

### 6.3 跟踪稳定性
- **参数**: `min_detect_count`, `max_temp_lost_count`, `outpost_max_temp_lost_count`
- **现象与动作**:
    - **锁得慢**: 减小 `min_detect_count`。
    - **易丢锁**: 增大 `max_temp_lost_count`。

### 6.4 MPC 静态对准与延时补偿
- **参数**:
    - `yaw_offset`, `pitch_offset`
    - `high_speed_delay_time`, `low_speed_delay_time`, `decision_speed`
- **现象与动作**:
    - **静态偏左/右**: 调整 `yaw_offset`。
    - **静态偏高/低**: 调整 `pitch_offset`。
    - **动态跟不上**: 调整延时项（通常先改高速项）。

### 6.5 MPC 响应与平滑
- **参数**: `Q_yaw/R_yaw/max_yaw_acc`, `Q_pitch/R_pitch/max_pitch_acc`
- **现象与动作**:
    - **抖动大**: 增大 `R` 或降低 `max_*_acc`。
    - **响应慢**: 增大 `Q` 或提升 `max_*_acc`。

### 6.6 开火窗口
- **参数**: `fire_thresh`
- **现象与动作**:
    - **空放多**: 减小 `fire_thresh`。
    - **错过开火窗口**: 适当增大 `fire_thresh`。