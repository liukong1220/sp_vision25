# sp_vision25 自瞄与前哨站代码阅读指南

这份文档配合 [sp_vision25_outpost_and_autoaim_analysis.md](./sp_vision25_outpost_and_autoaim_analysis.md) 使用。

前一份文档更像“方案说明书”，这份文档更像“读代码导航图”，目标是帮你在真正打开源码时知道：

1. 先看哪个文件
2. 每个文件要解决什么问题
3. 读到哪里应该停下来想一想
4. 如果你只关心前哨站，最短路径是什么

## 1. 推荐的阅读顺序

如果你是第一次系统看这个项目，建议按下面顺序读：

1. 入口程序
2. 检测与数据结构
3. 解算
4. 跟踪
5. 火控
6. 调试与可视化

可以记成一句话：

```text
先看“谁在调谁”，再看“数据长什么样”，最后看“怎么算和怎么打”
```

## 2. 第一层：先看入口，搞清楚程序到底走哪条链

### 2.1 `src/standard.cpp`

这是最容易读懂的入口。

你读这个文件时，重点回答 4 个问题：

1. 输入从哪里来
2. `Armor` 是谁生成的
3. `Target` 是谁生成的
4. 命令最后是谁发出去的

这一层读完后，你应该知道：

```text
standard.cpp 走的是 Aimer + Shooter 传统火控链路
```

### 2.2 `src/standard_mpc.cpp`

这个文件的关键不是“比 standard 多了什么类”，而是：

```text
感知和规划被拆成了两个线程
```

你读的时候重点看：

1. 主线程什么时候往 `target_queue` 里塞数据
2. `plan_thread` 什么时候取数据
3. `Planner` 最终给 `Gimbal.send()` 的是什么量

这一层读完后，你应该知道：

```text
MPC 方案不是直接给一个 yaw/pitch，而是给一整组控制状态
```

### 2.3 `src/auto_aim_debug_mpc.cpp`

这个文件建议放在 `standard_mpc.cpp` 后面看。

不要一上来就看它，因为它会把主线淹没在调试逻辑里。

你在这个文件里重点关注：

1. 主链路是不是还和 `standard_mpc.cpp` 一样
2. 哪些 `debug_*` 字段是从 `Planner` 暴露出来的
3. Web 调试器里哪些信息和前哨站直接相关

这一层读完后，你应该知道：

```text
auto_aim_debug_mpc.cpp = MPC 主链路 + 调试可视化外壳
```

## 3. 第二层：看数据结构，弄清楚每一层在传什么

### 3.1 `tasks/auto_aim/armor.hpp`

这是你最先该读的数据结构文件。

重点看：

- `Color`
- `ArmorType`
- `ArmorName`
- `ArmorPriority`
- `Armor`

这里最重要的不是记住每个字段，而是理解：

```text
Armor 是“检测结果 + 空间解算结果”的统一容器
```

如果你把 `Armor` 理解清楚，后面 `Solver`、`Tracker`、`Aimer` 的输入就都不会乱。

### 3.2 `tasks/auto_aim/armor.cpp`

这里要看 3 件事：

1. 传统灯条配对生成的 `Armor`
2. YOLO keypoints 生成的 `Armor`
3. 类别和类型是如何落到 `ArmorName / ArmorType` 上的

建议特别留意：

- YOLOV5 的 `num_id -> ArmorName`
- `outpost` 在这里什么时候被认成小装甲板

这一层读完后，你应该知道：

```text
前哨站在检测层并没有“特殊结构体”，它只是 ArmorName::outpost
```

## 4. 第三层：看检测，理解前哨站和普通板是怎么被识别出来的

### 4.1 `tasks/auto_aim/yolo.cpp`

这是一个分发器。

它的意义很简单：

```text
根据 yaml 选择 YOLOV5 / YOLOV8 / YOLO11
```

读这个文件时不要停留太久，知道入口就够了。

### 4.2 `tasks/auto_aim/yolos/yolov5.cpp`

这是当前主配置最重要的检测实现。

你读的时候要重点看：

1. 网络输出怎么解析成颜色、编号、角点
2. NMS 后怎么变成 `Armor`
3. `use_traditional_` 打开时如何做二次角点矫正

建议带着两个问题去看：

1. 前哨站在这一层比普通板多了什么
2. 如果识别错了，会先错在颜色、编号还是角点

答案基本会是：

```text
前哨站在这一层几乎没有额外建模，主要只是类别不同
```

### 4.3 `tasks/auto_aim/detector.cpp`

如果你想完整理解“传统视觉那一支”，再读它。

读这个文件时重点看：

1. 灯条几何筛选
2. 灯条配对
3. 图案分类
4. 重叠装甲板消歧

这层的意义更多是：

- 理解旧链路
- 理解 YOLOV5 二次角点矫正依赖了什么

## 5. 第四层：看解算，弄清楚世界坐标是怎么出来的

### 5.1 `tasks/auto_aim/solver.cpp`

这是非常关键的文件。

建议你按下面顺序读：

1. 构造函数
2. `set_R_gimbal2world()`
3. `solve()`
4. `reproject_armor()`
5. `optimize_yaw()`

读这个文件时，你最好始终盯着一条线：

```text
像素角点 -> 相机坐标 -> 云台坐标 -> 世界坐标
```

这里你尤其要注意前哨站的特殊点：

- `reproject_armor()` 中前哨站用 `-15 deg`
- 普通装甲板用 `+15 deg`

如果这里没理解，后面的前哨站匹配代价函数就很难看顺。

## 6. 第五层：看跟踪，这是前哨站和普通板真正分叉的地方

### 6.1 `tasks/auto_aim/tracker.cpp`

建议你按下面顺序读：

1. `Tracker::track()`
2. `state_machine()`
3. `set_target()`
4. `update_target()`
5. `select_best_outpost_match()`

你读这个文件时，脑子里最好始终保持两个框：

```text
普通目标：直接更新
前哨站：专用匹配 + 板号偏移重映射
```

重点问题：

1. `set_target()` 为什么前哨站是 3 板，普通目标是 4 板
2. `update_target()` 为什么前哨站走专门分支
3. `outpost_armor_z_offsets` 到底在什么时候起作用

### 6.2 `tasks/auto_aim/target.cpp`

这是整个项目最值得慢读的文件之一。

建议阅读顺序：

1. 构造函数
2. `predict()`
3. `update()`
4. `update_ypda()`
5. `armor_xyza_list()`
6. `set_armor_id_offset()`
7. `h_armor_xyz()`
8. `h_jacobian()`

你读这个文件时，重点看 3 件事：

#### A. 普通目标怎么建模

- 4 板
- `r + l`
- `cz + h`

#### B. 前哨站怎么建模

- 3 板
- 统一 `r`
- `outpost_armor_z_offsets`

#### C. 前哨站为什么要有 `physical_armor_id`

这一步很关键，因为它直接关系到：

- 哪块板是低板
- 哪块板是高板
- 火控到底在补哪块板

如果你能把 `Target::armor_xyza_list()` 和 `Target::set_armor_id_offset()` 看懂，前哨站模型就基本吃透了一半。

## 7. 第六层：看火控，区分 Aimer 和 Planner 两套思路

### 7.1 `tasks/auto_aim/aimer.cpp`

建议重点看：

1. `aim()`
2. `choose_aim_point()`

读法建议：

先不要想着“每一行在算什么”，先抓住结构：

```text
先做延迟预测
再选板
再算飞行时间
再迭代
最后出 yaw/pitch
```

这个文件里，普通目标和前哨站的差异主要在：

- `use_spin_gate`
- `comming_angle / leaving_angle`

也要注意一个项目现状：

```text
Aimer 分支的前哨站窗口仍然是硬编码 70/30
```

### 7.2 `tasks/auto_aim/shooter.cpp`

这个文件不长，但很重要。

建议重点看：

1. 什么时候 `auto_fire_` 才可能生效
2. 为什么要比较 `last_command_`
3. 为什么还要比较当前云台角和上次命令角

这一层你读完应该明白：

```text
Shooter 不是瞄准器，它是最后一道安全门
```

### 7.3 `tasks/auto_aim/planner/planner.cpp`

如果你主要想学前哨站，应该重点啃这个文件。

建议阅读顺序：

1. `choose_aim_selection()`
2. `solve_hit_target()`
3. `plan(std::optional<Target>, ...)`
4. `plan(Target, ...)`
5. `solve_aim_command()`
6. `get_trajectory()`
7. `resolve_angle_window()`
8. `resolve_delay_time()`
9. `resolve_aim_z_compensation()`

读这个文件时，建议你始终追同一个问题：

```text
Planner 最终是根据哪块板、在什么时刻、带什么补偿去解 yaw/pitch 的
```

你只要把这个问题追清楚，前哨站“为什么会打高、打低、超前、滞后”就能自己定位很多了。

## 8. 如果你只想最快看懂前哨站，走这条最短路径

只关心前哨站的话，建议直接按下面顺序读：

1. `src/auto_aim_debug_mpc.cpp`
2. `tasks/auto_aim/tracker.cpp`
3. `tasks/auto_aim/target.cpp`
4. `tasks/auto_aim/planner/planner.cpp`
5. `tools/debug_visualization.cpp`
6. `assets/web_debugger/static/js/main.js`

这条路径对应的是：

```text
前哨站感知入口
-> 前哨站三板模型
-> 前哨站匹配与板号映射
-> 前哨站火控
-> 前哨站调试输出
```

如果你这样读，学习效率通常比从检测一路顺着读更高。

## 9. 如果你想系统吃透整个自瞄，再走这条完整路径

完整建议顺序如下：

1. `src/standard.cpp`
2. `src/standard_mpc.cpp`
3. `tasks/auto_aim/armor.hpp`
4. `tasks/auto_aim/armor.cpp`
5. `tasks/auto_aim/yolo.cpp`
6. `tasks/auto_aim/yolos/yolov5.cpp`
7. `tasks/auto_aim/solver.cpp`
8. `tasks/auto_aim/tracker.cpp`
9. `tasks/auto_aim/target.cpp`
10. `tasks/auto_aim/aimer.cpp`
11. `tasks/auto_aim/shooter.cpp`
12. `tasks/auto_aim/planner/planner.cpp`
13. `src/auto_aim_debug_mpc.cpp`

这条线适合做两件事：

1. 学项目结构
2. 为后续自己改算法做准备

## 10. 每读完一层，建议你自检这几个问题

### 10.1 读完入口后

- 我现在清楚程序走的是 `Aimer` 还是 `Planner` 吗？
- 我知道是谁最终 `send()` 给下位机吗？

### 10.2 读完检测后

- 我知道 `Armor` 是怎么构造出来的吗？
- 我知道前哨站在检测层和普通板到底差在哪里吗？

### 10.3 读完解算后

- 我能说清楚 `xyz_in_world` 是怎么来的吗？
- 我知道前哨站和普通板在重投影 pitch 假设上不同吗？

### 10.4 读完跟踪后

- 我能说清楚为什么普通目标是 4 板、前哨站是 3 板吗？
- 我能说清楚 `physical_armor_id` 为什么存在吗？

### 10.5 读完火控后

- 我能区分“选板问题”“延迟问题”“弹道补偿问题”吗？
- 我知道为什么 `Planner` 更适合前哨站精细调试吗？

## 11. 最后给你的一个阅读建议

读这个项目时，最容易卡住的不是公式，而是“层之间的职责混了”。

建议你始终把问题分成四层去看：

```text
检测有没有看见
解算有没有算对
跟踪有没有跟稳
火控有没有选对板、补对延迟、压对高度
```

这样即使后面你开始改代码，也不容易把问题改串。
