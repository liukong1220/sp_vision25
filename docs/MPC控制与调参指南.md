# MPC 控制与调参指南

这份文档把原来分散的几份 MPC 文档收敛到一起，统一回答四件事：

1. `Planner` 在当前工程里到底负责什么
2. `standard_mpc.cpp` 和 `auto_aim_debug_mpc.cpp` 有什么区别
3. `offset / delay / MPC` 应该按什么顺序调
4. 小陀螺和前哨问题现在该优先看哪些内部量

## 1. 当前 MPC 链路是什么

当前项目里的 MPC 主链路是：

```text
YOLO
  -> Solver
  -> Tracker
  -> Planner
  -> Gimbal.send(...)
```

其中 `Planner` 负责：

1. 目标延迟预测
2. 命中时刻固定点迭代
3. 未来参考轨迹生成
4. yaw / pitch 双通道 TinyMPC 求解
5. `fire` 判定

也就是说，`Planner` 不是 `Aimer + Shooter` 的附加层，而是一条并行控制路线。

## 2. 哪些入口在用 MPC

### 2.1 `standard_mpc.cpp`

这是标准车 MPC 主程序。

特点：

1. 支持 `AUTO_AIM + SMALL_BUFF + BIG_BUFF`
2. 有 `plan_thread`
3. 更偏实战使用

### 2.2 `auto_aim_debug_mpc.cpp`

这是 MPC 专项调试入口。

特点：

1. 主链路和 `standard_mpc.cpp` 基本一致
2. 但增加了大量调试量
3. 带网页调试器和 Ballistic Debug 面板
4. 最适合做前哨与 MPC 验收

## 3. `Planner` 当前在代码里怎么工作

当前 `planner.cpp` 的主流程可以概括为：

1. `plan(std::optional<Target>)`
   先按延迟预测目标到当前决策时刻
2. `plan(Target)`
   再做命中时刻固定点迭代
3. 根据命中时刻目标生成未来参考轨迹
4. yaw / pitch 各自进入 `TinyMPC`
5. 输出：
   `yaw/pitch`
   `yaw_vel/pitch_vel`
   `yaw_acc/pitch_acc`
   `fire`

当前几项关键常量和结构：

1. `DT = 0.01`
2. `HORIZON = 100`
3. 中点控制索引 `HALF_HORIZON = 50`

## 4. 当前前哨在 MPC 里有哪些专用能力

这部分是当前工程最值得重视的地方。

前哨在 `Planner` 里已经支持：

1. `outpost_coming_angle`
2. `outpost_leaving_angle`
3. `outpost_delay_time`
4. `outpost_fire_z_compensation`
5. 命中时刻固定点迭代
6. 前哨专用开火相位门
7. fallback 参考连续、但 fallback 不允许开火

所以如果你现在要验证前哨，优先看 MPC 分支，而不是传统 `Aimer` 分支。

## 5. `standard_mpc` 和 `auto_aim_debug_mpc` 怎么选

| 维度 | `standard_mpc.cpp` | `auto_aim_debug_mpc.cpp` |
|---|---|---|
| 主要用途 | 实战/联调入口 | 调参与定位问题 |
| 模式支持 | 自瞄 + 打符 | 仅自瞄 |
| 通信 | `io::Gimbal` | `io::Gimbal` |
| 目标输入 | `target_queue` | `target_queue` |
| 调试量 | 基础 plotter | 最完整，可视化最强 |
| 前哨验收 | 可用 | 最推荐 |

一句话建议：

1. 想跑车用 `standard_mpc`
2. 想看问题用 `auto_aim_debug_mpc`

## 6. 当前最应该看的调试量

### 6.1 基础控制量

1. `plan_yaw / plan_pitch`
2. `plan_yaw_vel / plan_pitch_vel`
3. `plan_yaw_acc / plan_pitch_acc`
4. `target_yaw / target_pitch`

### 6.2 命中时刻与选板

1. `planner_selected_armor`
2. `planner_selected_physical_armor`
3. `planner_hit_fly_time_ms`
4. `planner_hit_iters`
5. `planner_hit_converged`

### 6.3 前哨相位与延迟

1. `planner_selected_delta_deg`
2. `planner_fire_phase_limit_deg`
3. `planner_fire_phase_ready`
4. `planner_spin_gate`
5. `planner_delay_ms`

### 6.4 tracker 配合量

1. `tracker_match_valid`
2. `tracker_match_score`
3. `tracker_reprojection_px`

## 7. `offset / delay / MPC` 的正确调参顺序

这一部分最重要，建议始终按下面顺序来。

### 7.1 第一步：先调 `offset`

对应参数：

1. `yaw_offset`
2. `pitch_offset`

目标：

1. 在低速、易跟踪目标下，先把静态左右偏和高低偏校准掉
2. 不要用 `delay` 去补静态偏差

典型现象：

1. 长期偏左/偏右：
   优先看 `yaw_offset`
2. 长期偏高/偏低：
   优先看 `pitch_offset`

### 7.2 第二步：再调 `delay`

对应参数：

1. `decision_speed`
2. `high_speed_delay_time`
3. `low_speed_delay_time`
4. 对前哨，还要看 `outpost_delay_time`

目标：

1. 修正高速目标或前哨切板时的整体超前/滞后

典型现象：

1. 静态能打中，动态总慢半拍：
   `delay` 可能偏小
2. 总体明显打到前面：
   `delay` 可能偏大

### 7.3 第三步：最后再调 MPC

对应参数：

1. `Q_yaw`
2. `R_yaw`
3. `max_yaw_acc`
4. `Q_pitch`
5. `R_pitch`
6. `max_pitch_acc`

目标：

1. 平衡“跟得上”和“不要抖”

经验判断：

1. 跟踪太慢：
   增大 `Q` 或增大 `max_*_acc`
2. 抖动太大：
   增大 `R` 或减小 `max_*_acc`

### 7.4 最后收口 `fire`

对应参数：

1. `fire_thresh`
2. 对前哨还要配合：
   `outpost_leaving_angle`
   `outpost_delay_time`

目标：

1. 让“理论能中”与“实际允许开火”尽量一致

## 8. Ballistic Debug 应该怎么看

当前 `auto_aim_debug_mpc` 里的 Ballistic Debug 可以帮你把问题拆成三类：

### 8.1 静态对准问题

表现：

1. 低速目标下长期固定偏左/偏右
2. 低速目标下长期固定偏高/偏低

优先看：

1. `yaw_offset`
2. `pitch_offset`

### 8.2 动态提前量问题

表现：

1. 目标速度上来后整体超前
2. 或整体落后

优先看：

1. `high_speed_delay_time`
2. `low_speed_delay_time`
3. `outpost_delay_time`

### 8.3 控制跟踪问题

表现：

1. 参考目标角和实际下发角差距明显
2. 曲线抖动大
3. 开火点不稳定

优先看：

1. `Q/R`
2. `max_*_acc`
3. `planner_fire_tracking_error_deg`

## 9. 当前工程里已经修过的小陀螺 / 前哨关键点

当前版本里，和 MPC 直接相关的关键改动可以总结成：

1. `Planner` 不再单纯选“最近板”，而是区分普通目标与 spin gate 目标
2. 命中时刻飞行时间求解会进入固定点迭代
3. 轨迹规划优先继承命中时刻求解出的板号
4. `get_trajectory()` 不再跨帧保留会污染当前参考的 `last_yaw_vel`
5. 新增了大量 `debug_*` 字段，便于现场确认到底是选板问题、延迟问题还是控制问题
6. 前哨 fallback 板只保控制连续，不保开火许可

## 10. 当前前哨验收时的优先观察顺序

建议按下面顺序看：

1. `tracker_match_valid`
   先确认跟踪是不是稳
2. `planner_selected_physical_armor`
   再确认打的是不是预期物理板
3. `planner_hit_converged`
   再确认命中时刻求解是不是稳
4. `planner_selected_delta_deg`
   看命中时刻相位是否合理
5. `planner_fire_phase_ready`
   看为什么不打或为什么空打
6. `planner_delay_ms`
   最后看是否是整体超前/滞后

## 11. 当前文档收敛说明

这份文档已经吸收了原来几份碎片化文档的主要内容：

1. `mpc.md`
2. `offset_delay_mpc.md`
3. `MPC_DEBUG_DIFF_STANDARD3_TUNING.md`
4. `MPC_SMALL_GYRO_DEBUG_FIX_SUMMARY.md`

后续如果继续补 MPC 相关内容，优先往这份文档追加，不再重复开同主题小文档。
