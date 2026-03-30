# `offset` / `delay` / `MPC` 联调指南

## 1. 目标

这份文档配合 `auto_aim_debug_mpc` 中新增的 `Ballistic Debug` 窗口使用，目标是把下面三类问题分开：

- 静态对准误差：`yaw_offset` / `pitch_offset`
- 动态提前量误差：`high_speed_delay_time` / `low_speed_delay_time` / `decision_speed`
- 跟踪控制误差：`Q_yaw` / `R_yaw` / `max_yaw_acc` / `Q_pitch` / `R_pitch` / `max_pitch_acc`

如果不分层，现场很容易出现“用 delay 补 offset”或者“用 MPC 掩盖解算偏差”的假调。

## 2. 运行方式

```bash
./build/auto_aim_debug_mpc configs/standard3.yaml
```

当前调试入口：

- 主画面：`Auto Aim Debug`
- 弹道诊断窗口：`Ballistic Debug`

`Ballistic Debug` 的实现位置：

- `src/auto_aim_debug_mpc.cpp`
- 命中诊断构造：`build_ballistic_diagnostic(...)`
- 自动判据生成：`build_tuning_judgement(...)`
- 独立窗口绘制：`draw_ballistic_panel(...)`

## 3. 窗口怎么看

### 3.1 侧视图 `d-z`

- 绿色曲线：几何理想弹道
- 青色曲线：当前 `plan.pitch` 对应的实际指令弹道
- 红点：目标点
- 青点：当前指令弹道在目标水平距离处的落点

判断：

- 青点长期高于红点：`pitch` 偏高
- 青点长期低于红点：`pitch` 偏低
- 绿色能打中但青色打不中：问题不在弹道模型，而在控制或预测

### 3.2 俯视图 `x-y`

- 绿色箭头：目标几何方向
- 青色箭头：当前 `plan.yaw` 射线
- 红点：目标在平面内的位置

判断：

- 青色长期在目标左边：`yaw` 偏左
- 青色长期在目标右边：`yaw` 偏右
- 绿色和青色方向差很大：优先查 `yaw_offset` 或 `yaw MPC`

### 3.3 文本区

- `Verdict: HIT / MISS`
- `plan.fire`
- `offset yaw/pitch`
- `geo yaw/pitch`
- `cmd-ref yaw/pitch`
- `plan yaw/pitch`
- `yaw/pitch residual`
- `lateral miss`
- `vertical miss`
- `total miss`

其中最关键的是：

- `yaw/pitch residual`：当前控制输出和参考值差多少
- `lateral miss / vertical miss`：当前指令几何上偏了多少

### 3.4 自动调参判据表

窗口下半部分新增 `Auto Hints` 判据表，分为四行：

- `Offset`
- `Delay`
- `MPC`
- `Fire`

每一行显示的是当前最值得优先检查的方向，不是绝对真理，而是基于当前目标、当前弹速和当前 `plan` 的保守工程判断。

各状态含义：

- `Offset = Good`
  说明在当前工况下没有明显静态偏差，先不要急着改 `yaw_offset / pitch_offset`。
- `Offset = Yaw offset ++ / --`
  说明在“低速且 MPC 基本跟上”的前提下，横向仍存在静态偏差，优先小步调整 `yaw_offset`。
- `Offset = Pitch offset ++ / --`
  说明在“低速且 MPC 基本跟上”的前提下，竖向仍存在静态偏差，优先小步调整 `pitch_offset`。
- `Offset = Check after MPC`
  说明当前控制没跟住，先别拿 `offset` 胡乱补偿。

- `Delay = Good`
  当前动态提前量没有明显异常。
- `Delay = Low speed`
  当前目标速度不高，暂时不建议优先动 `delay_time`。
- `Delay = Delay likely ++ / --`
  当前目标速度较高，且 `MPC` 已基本跟住，但横向仍表现出“落后/超前”趋势，这时优先检查 `high_speed_delay_time` 或 `low_speed_delay_time`。
- `Delay = Check after MPC`
  当前跟踪误差还比较明显，先不要急着判断 `delay`。

- `MPC = Good`
  参考跟踪基本正常。
- `MPC = Need fine-tune`
  还能继续优化，但还没到严重失真。
- `MPC = Track weak`
  当前参考跟踪偏差较大，优先改 `Q/R/max_acc`，不要先改 `offset` 和 `delay`。

- `Fire = Good`
  理论命中判据和 `plan.fire` 一致。
- `Fire = Thresh may ++`
  理论可中但不开火，说明 `fire_thresh` 可能偏严。
- `Fire = Thresh may --`
  理论未中却允许开火，说明 `fire_thresh` 可能偏松。
- `Fire = Hold fire`
  当前还不该开火。

建议使用方式：

- 先看 `MPC`
- 再看 `Offset`
- 最后看 `Delay`
- `Fire` 永远放在最后收口

## 4. 调参顺序

### 4.1 第一步：先调 `offset`

对应参数：

- `yaw_offset`
- `pitch_offset`

配置位置：

- `configs/standard3.yaml`

操作原则：

- 只看静止目标或极慢速目标
- 让 `Verdict` 稳定变成 `HIT`
- 让 `yaw/pitch residual` 靠近 0

现象和动作：

- 俯视图青色射线总偏左：调大 `yaw_offset`
- 俯视图青色射线总偏右：调小 `yaw_offset`
- 侧视图青色落点总偏高：调小 `pitch_offset`
- 侧视图青色落点总偏低：调大 `pitch_offset`

建议步长：

- `yaw_offset`：每次 `0.1 ~ 0.2 deg`
- `pitch_offset`：每次 `0.05 ~ 0.15 deg`

### 4.2 第二步：再调 `MPC`

对应参数：

- `max_yaw_acc`
- `Q_yaw`
- `R_yaw`
- `max_pitch_acc`
- `Q_pitch`
- `R_pitch`

代码入口：

- `tasks/auto_aim/planner/planner.cpp`

你可以把它理解为：

- `Q`：想不想更快跟上参考
- `R`：愿不愿意为了跟上参考付出更激进的控制量
- `max_*_acc`：允许的最大动态能力上限

现象和动作：

- 青线总追不上绿线，`yaw/pitch residual` 长期偏大：`MPC` 太软
- 青线频繁来回穿过目标，画面抖：`MPC` 太激进

如果 `MPC` 太软：

- 增大 `Q_yaw` / `Q_pitch`
- 减小 `R_yaw` / `R_pitch`
- 增大 `max_yaw_acc` / `max_pitch_acc`

如果 `MPC` 太激进：

- 减小 `Q_yaw` / `Q_pitch`
- 增大 `R_yaw` / `R_pitch`
- 降低 `max_yaw_acc` / `max_pitch_acc`

建议步长：

- `Q_*`：每次乘 `1.2 ~ 1.5`
- `R_*`：每次乘 `0.7 ~ 0.85` 或 `1.2 ~ 1.5`
- `max_*_acc`：每次改 `10 ~ 15`

### 4.3 第三步：最后调 `delay_time`

对应参数：

- `decision_speed`
- `high_speed_delay_time`
- `low_speed_delay_time`

代码位置：

- `tasks/auto_aim/planner/planner.cpp`
- `Planner::plan(std::optional<Target>, ...)`

注意：

- `delay_time` 是目标预测时间的一部分
- 它只应该在 `offset` 和 `MPC` 已经基本调对后再改

判断方法：

- 慢速目标基本准，高速目标总落后：增大 `high_speed_delay_time`
- 慢速目标就已经落后：增大 `low_speed_delay_time`
- 总是打在目标前方：减小对应 `delay_time`
- 某一速度附近突然表现变差：调整 `decision_speed`

建议步长：

- `high_speed_delay_time` / `low_speed_delay_time`：每次 `0.01 ~ 0.02 s`
- `decision_speed`：每次 `0.3 ~ 0.5`

## 5. 常见误判

### 5.1 用 `delay_time` 修静态偏差

现象：

- 静止目标都打不正，却一直改 `high/low_speed_delay_time`

后果：

- 静态不准
- 动态更乱

正确做法：

- 先把 `offset` 调对，再碰 `delay`

### 5.2 用 `offset` 修动态滞后

现象：

- 静止目标刚调准，运动目标仍总慢半拍
- 然后继续把 `yaw_offset` 往前拧

后果：

- 静止准星被破坏
- 运动和静止无法兼顾

正确做法：

- 这种情况优先查 `delay_time` 或 `yaw MPC`

### 5.3 用 `fire_thresh` 掩盖控制问题

现象：

- 控制跟踪还没调稳，就直接把 `fire_thresh` 改得很大

后果：

- `plan.fire` 很容易亮
- 实际空放变多

正确做法：

- `fire_thresh` 只在控制已经稳定后微调

## 6. 一套现场可执行流程

### 6.1 静态阶段

1. 固定一个静止装甲板
2. 跑 `auto_aim_debug_mpc`
3. 只调 `yaw_offset` / `pitch_offset`
4. 目标：`Verdict` 稳定 `HIT`

### 6.2 低速运动阶段

1. 让目标缓慢横移
2. 不动 `delay_time`
3. 先看青线是否能跟住绿线
4. 若跟不住，先调 `Q/R/max_acc`

### 6.3 高速运动阶段

1. 让目标中高速转动或平移
2. 若窗口里理论命中但实测总前后偏，开始调 `delay_time`
3. 先改 `high_speed_delay_time`
4. 再回头微调 `decision_speed`

### 6.4 开火窗口阶段

1. 跟踪稳定后再看 `plan.fire`
2. 若错过窗口太多：适当增大 `fire_thresh`
3. 若空放变多：减小 `fire_thresh`

## 7. 参数与现象速查表

| 现象 | 优先改 | 次要再看 |
|---|---|---|
| 静止目标左右偏 | `yaw_offset` | 解算外参 |
| 静止目标高低偏 | `pitch_offset` | 弹速是否异常 |
| 参考能到，控制跟不上 | `Q/R/max_acc` | 串口/执行延迟 |
| 控制过冲、来回抖 | `R_*` 增大 / `Q_*` 减小 | `max_*_acc` 降低 |
| 高速目标总落后 | `high_speed_delay_time` | `decision_speed` |
| 低速目标总落后 | `low_speed_delay_time` | `offset` 是否仍有残差 |
| `plan.fire` 很少亮 | `fire_thresh` 稍增 | 先确认控制已稳定 |
| `plan.fire` 常亮但空放多 | `fire_thresh` 降低 | 重新看 `delay` 与 `MPC` |

## 8. 最后提醒

`Ballistic Debug` 判断的是：

- 当前 `Planner` 选中的目标板
- 当前 `plan` 对这个预测目标板是否几何命中

它非常适合调：

- `offset`
- `MPC`
- `delay_time`

但它不等于实弹结果本身。最终仍要结合：

- 实际弹道散布
- 下位机执行延迟
- 实际目标运动状态

如果窗口里已经稳定 `HIT`，但实弹还总有系统性前后偏差，优先怀疑链路总延迟，而不是再次乱改 `offset`。
