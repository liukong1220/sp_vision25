# auto_aim/planner 分析

## 1. 目录职责
`planner` 是 `Aimer+Shooter` 的另一套控制路线：使用 MPC（TinyMPC）直接输出云台期望状态与开火信号。

它主要用于 `src/standard_mpc.cpp` / `src/auto_aim_debug_mpc.cpp` 这类程序入口。

## 2. 文件级说明

| 文件 | 核心类/内容 | 作用 | 关键依赖 |
|---|---|---|---|
| `planner.hpp` | `Planner`, `Plan`, 轨迹类型定义 | MPC控制接口定义 | `target.hpp`, `tinympc/tiny_api.hpp` |
| `planner.cpp` | `Planner` 实现 | 目标预测、参考轨迹构造、yaw/pitch双MPC求解、fire判定 | TinyMPC, trajectory |
| `tinympc/CMakeLists.txt` | 子库构建配置 | 编译静态库 `tinympcstatic` | admm/tiny_api/codegen |

## 3. 控制结算主链

1. 输入：`Target`（来自 `Tracker`）+ `bullet_speed`
2. `plan(std::optional<Target>)` 先做发射延时预测（高低速分支）
3. `plan(Target)`：
   - 先估飞行时间并外推目标
   - `aim()` 取当前最可能可击中的 yaw/pitch 参考
   - `get_trajectory()` 生成预测地平线参考轨迹（yaw, yaw_vel, pitch, pitch_vel）
   - yaw 与 pitch 分别调用 TinyMPC 求解（2状态1输入）
4. 输出 `Plan`：
   - 控制量：`yaw/yaw_vel/yaw_acc/pitch/pitch_vel/pitch_acc`
   - 目标参考：`target_yaw/target_pitch`
   - `fire`：跟踪误差低于 `fire_thresh_` 时触发

## 4. 关键算法点
- 参考轨迹地平线：`DT=0.01`, `HORIZON=100`, 控制中点索引 `HALF_HORIZON=50`。
- `aim()` 使用 `tools::Trajectory` 做弹道可解性判断，不可解抛异常。
- `get_trajectory()` 通过中心差分求速度，并对“剧烈加速度转折点”做速度衰减（平滑处理）。
- yaw/pitch 两套 solver 独立建模，约束各自最大角加速度。

## 5. 与 auto_aim 其余模块关系
- 输入完全复用 `Target`（来自 `Tracker`）。
- 输出不走 `io::Command`，而是直接提供连续状态给 `io::Gimbal::send(...)`。
- 因此此目录与 `aimer/shooter` 是并行方案，不是前后串联。

