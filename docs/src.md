# src 运行入口与文件关系清单

## 0. 相关调参文档

- `docs/MPC_DEBUG_DIFF_STANDARD3_TUNING.md`：`standard3.yaml` 与 `standard_mpc/auto_aim_debug_mpc` 的整体调参说明
- `docs/offset_delay_mpc.md`：配合 `Ballistic Debug` 窗口使用的 `offset / delay / MPC` 联调指南

## 1. 先回答你的核心问题：哪个是“最主要运行文件”

结论不是单一文件，而是按机器人形态分组：

1. 步兵/标准车（非 MPC，CAN CBoard 协议）主入口：`mt_standard.cpp`
2. 步兵/标准车（MPC，Gimbal 协议）主入口：`standard_mpc.cpp`
3. 哨兵（ROS2 + 全向感知）主入口：`sentry.cpp`（或其变体 `sentry_bp.cpp` / `sentry_multithread.cpp`）
4. 无人机链路主入口：`uav.cpp`

为什么 `mt_standard.cpp` 常被视为默认“主程序”：
- 同时集成自瞄 + 打符两套任务
- 使用模式切换（`auto_aim/small_buff/big_buff`）
- 使用多线程检测 + 多线程发指令，结构最接近实战部署
- 默认配置即 `configs/standard3.yaml`

## 2. 编译层面的入口事实（来自顶层 CMake）

`CMakeLists.txt` 会始终构建的 `src` 可执行程序：
- `standard`
- `mt_standard`
- `standard_mpc`
- `auto_aim_debug_mpc`
- `mt_auto_aim_debug`
- `auto_buff_debug`
- `auto_buff_debug_mpc`
- `uav`
- `uav_debug`

仅在 ROS2 依赖存在时才构建：
- `sentry`
- `sentry_bp`
- `sentry_debug`
- `sentry_multithread`

因此你当前环境如果未启 ROS2，哨兵系列不会参与实际运行。

## 3. src 全文件总览（作用 + 函数）

> 说明：`src` 中所有文件都只有一个顶层函数 `int main(int argc, char* argv[])`，  
> 另外部分文件在 `main` 内定义线程 lambda（不是独立函数定义）。

| 文件 | 作用定位 | 函数定义 | 线程/并发 |
|---|---|---|---|
| `standard.cpp` | 单线程基础自瞄主程序（CBoard） | `main` | 无 |
| `mt_standard.cpp` | 多线程标准车主程序（自瞄+打符） | `main` | `detect_thread` + `CommandGener`内部线程 |
| `standard_mpc.cpp` | 标准车 MPC 主程序（Gimbal） | `main` | `plan_thread` |
| `auto_aim_debug_mpc.cpp` | 自瞄 MPC 调试程序（曲线输出） | `main` | `plan_thread` |
| `mt_auto_aim_debug.cpp` | 多线程自瞄调试程序（重投影/EKF可视化） | `main` | `detect_thread` + `CommandGener`内部线程 |
| `uav.cpp` | 无人机模式主程序（传统Detector链） | `main` | 无 |
| `uav_debug.cpp` | 无人机调试程序（重投影+滤波指标） | `main` | 无 |
| `auto_buff_debug.cpp` | 打符调试（CBoard 通道） | `main` | 无 |
| `auto_buff_debug_mpc.cpp` | 打符 MPC 调试（Gimbal 通道） | `main` | 无 |
| `sentry.cpp` | 哨兵主程序（ROS2+全向感知） | `main` | 无（但调用多相机决策） |
| `sentry_bp.cpp` | 哨兵后相机/简化感知变体 | `main` | 无 |
| `sentry_debug.cpp` | 哨兵调试程序（可视化+EKF曲线） | `main` | 无 |
| `sentry_multithread.cpp` | 哨兵多摄并行感知版本 | `main` | 感知组件内部并行 |

## 4. 每个文件的具体逻辑关系

## 4.1 `standard.cpp`（基础版）
- 关键对象：`io::CBoard`、`io::Camera`、`YOLO`、`Solver`、`Tracker`、`Aimer`。
- 流程：
  1. 相机读图 + CBoard 获取 `imu_at(t-1ms)`
  2. `solver.set_R_gimbal2world(q)`
  3. `detector.detect(img)` -> `tracker.track(armors,t)` -> `aimer.aim(targets,t,bullet_speed)`
  4. `cboard.send(command)`
- 特征：简单直接，不含多线程检测，不含打符，不含显式 `Shooter::shoot` 判定。

## 4.2 `mt_standard.cpp`（标准车推荐）
- 关键对象：
  - 自瞄：`MultiThreadDetector + Solver + Tracker + Aimer + Shooter + CommandGener`
  - 打符：`Buff_Detector + Buff_Solver + Buff_Target + Buff_Aimer`
- 并发结构：
  - `detect_thread`：只在 `mode==auto_aim` 时采图并 `detector.push()`
  - 主线程：模式切换与任务分发
  - `CommandGener` 自带发送线程：做 `aim + shoot + send`
- 流程（auto_aim）：
  1. `debug_pop` 取推理结果
  2. `solver.set_R_gimbal2world`
  3. `tracker.track`
  4. `commandgener.push(targets,t,bullet_speed,gimbal_pos)`
- 流程（buff）：
  1. `buff_detector.detect` -> `buff_solver.solve`
  2. `small/big target.get_target`
  3. `buff_aimer.aim`
  4. `cboard.send`

## 4.3 `standard_mpc.cpp`
- 通信侧改用 `io::Gimbal`（不是 `io::CBoard`）。
- 自瞄侧改用 `Planner`（不是 `Aimer+Shooter`）。
- 并发结构：
  - `plan_thread` 周期读取 `target_queue`，执行 `planner.plan`，`gimbal.send(...)`
  - 主线程负责检测与跟踪，将目标推入队列
- 同时保留打符模式，并且打符也走 `gimbal.send(...)` 的 `Plan` 风格接口。

## 4.4 `auto_aim_debug_mpc.cpp`
- 基本链路：`YOLO -> Solver -> Tracker -> Planner -> Gimbal.send`
- 重点是调试数据：
  - 云台实际角速度、规划值、目标值、发弹事件、目标 z/vz/w
  - 通过 `tools::Plotter` 输出时序曲线
- 用途：验证 MPC 参数、相位延迟、开火窗口。

## 4.5 `mt_auto_aim_debug.cpp`
- 是 `mt_standard` 的自瞄调试强化版：
  - 多线程检测
  - `CommandGener` 异步发指令
  - 叠加重投影可视化（target 装甲板与 aim 点）
  - 输出 EKF 内部残差/NIS/NEES
- 用途：专门诊断识别-跟踪-瞄准链路。

## 4.6 `uav.cpp`
- 使用传统 `Detector`（非 YOLO）作为主检测源（YOLO行被注释）。
- 模式支持：`auto_aim/outpost/small_buff/big_buff`。
- 自瞄链路：`Detector -> Solver -> Tracker -> Aimer -> Shooter -> cboard.send`
- 用途：无人机场景和传统视觉方案。

## 4.7 `uav_debug.cpp`
- 基于 `uav.cpp` 的可视化调试版：
  - 画重投影装甲板、瞄准点
  - 输出 EKF 全状态及一致性检验指标
  - 输出云台响应与指令响应
- 用途：针对 UAV 链路做稳定性与滤波效果验证。

## 4.8 `auto_buff_debug.cpp`
- 打符专用（CBoard 协议）：
  - `Buff_Detector -> Buff_Solver -> SmallTarget.get_target -> Buff_Aimer.aim -> cboard.send`
- 调试输出：
  - rune 原始测量
  - 预测/当前重投影
  - EKF 内部状态
  - 云台与命令曲线

## 4.9 `auto_buff_debug_mpc.cpp`
- 打符专用（Gimbal 协议 + MPC 版）：
  - `Buff_Aimer.mpc_aim` 输出 `Plan`
  - `gimbal.send(plan...)`
- 调试输出与 `auto_buff_debug.cpp` 类似，但多了速度/加速度通道。

## 4.10 `sentry.cpp`
- 哨兵主链：
  1. 主相机检测：`yolo.detect`
  2. `decider` 结合 ROS2 无敌信息进行过滤与优先级
  3. `tracker.track`
  4. 若 `tracker.state()=="lost"`，走全向感知 `decider.decide(...)`
  5. 否则走 `aimer.aim`
  6. `shooter.shoot` + `cboard.send`
  7. 发布目标信息 `ros2.publish`
- 特征：全向感知与单相机跟踪之间有切换策略。

## 4.11 `sentry_bp.cpp`
- 与 `sentry.cpp` 类似，但全向分支使用 `decider.decide(yolo, gimbal_pos, back_camera)`，
- 特征是感知来源更简化（偏向后相机分支）。

## 4.12 `sentry_debug.cpp`
- 在 `sentry.cpp` 基础上叠加：
  - 自动目标选择接口 `get_auto_aim_target`
  - 重投影可视化与 EKF 指标曲线
- 用途：哨兵策略 + 跟踪稳定性联合调试。

## 4.13 `sentry_multithread.cpp`
- 引入 `omniperception::Perceptron`（4路 USB 相机）并行感知。
- 关键差异：
  - `detection_queue = perceptron.get_detection_queue()`
  - `tracker.track(detection_queue, armors, t)` 支持 `switching` 状态
  - `switching/lost/tracking` 三类状态分别给不同指令逻辑
- 用途：哨兵全向多摄并行版本，决策复杂度最高。

## 5. src 与其他目录的关系图（按调用层）

1. `src/*.cpp`（应用层 main）
2. `io/*`（硬件接口层）
   - `Camera/USBCamera/CBoard/Gimbal/ROS2`
3. `tasks/*`（算法功能层）
   - `auto_aim`：检测/解算/跟踪/瞄准/发射
   - `auto_buff`：符识别/解算/预测/发射
   - `omniperception`：多相机感知与优先级决策
4. `tools/*`（工具层）
   - `logger/plotter/math_tools/exiter/recorder/thread_safe_queue/...`

可理解为：`src` 只做调度编排，不做具体算法实现。

## 6. 实战上如何选入口（建议）

1. 你现在的工程（步兵 + 自瞄/打符 + CBoard）优先看 `mt_standard.cpp`。
2. 如果你要研究 MPC 云台控制，看 `standard_mpc.cpp` 和 `auto_aim_debug_mpc.cpp`。
3. 如果要重点研究哨兵与全向切换，看 `sentry.cpp` 与 `sentry_multithread.cpp`。
4. 如果你在查传统检测链路问题，看 `uav.cpp` / `uav_debug.cpp`。

## 7. 本目录函数清单（严格）

| 文件 | 顶层函数 | 局部线程 lambda |
|---|---|---|
| `standard.cpp` | `main` | 无 |
| `mt_standard.cpp` | `main` | `detect_thread` |
| `standard_mpc.cpp` | `main` | `plan_thread` |
| `auto_aim_debug_mpc.cpp` | `main` | `plan_thread` |
| `mt_auto_aim_debug.cpp` | `main` | `detect_thread` |
| `uav.cpp` | `main` | 无 |
| `uav_debug.cpp` | `main` | 无 |
| `auto_buff_debug.cpp` | `main` | 无 |
| `auto_buff_debug_mpc.cpp` | `main` | 无 |
| `sentry.cpp` | `main` | 无 |
| `sentry_bp.cpp` | `main` | 无 |
| `sentry_debug.cpp` | `main` | 无 |
| `sentry_multithread.cpp` | `main` | 无（并发由 `Perceptron` 内部实现） |

