# `src/tasks/auto_aim/debug.cpp` 可视化绘制源码详解

## 1. 这份文档的目标

这份文档不是简单重复“画了什么”，而是结合 `src/tasks/auto_aim/debug.cpp` 的源码，解释它为什么这样画、每个图元对应哪一类系统状态、以及如果你要把自己的纯 C++ 项目做得更工程化，哪些设计值得保留、哪些地方值得重构。

适合阅读对象：

- 想学习 RoboMaster 自瞄调试可视化设计的人
- 想把自己项目里的调试图做得更系统的人
- 想把“算法状态”映射成“可视化信息”的人

## 2. `debug.cpp` 在整个系统中的位置

`debug.cpp` 里主要有两个函数：

1. `draw_auto_aim(cv::Mat& img, const AutoAimDebugCtx& ctx)`
   负责把各种调试信息直接画到图像上。

2. `write_debug_data(const AutoAimDebugCtx& ctx)`
   负责把时序调试量写成 JSON，供网页前端绘制曲线。

它们不是直接插在检测器或控制器内部，而是被 `src/runtime/auto_aim.cpp` 里的调试线程统一调用。实际调用链大致如下：

```text
检测 / 跟踪 / 控制主链
    ↓
更新 AutoAimDebugCtx
    ↓
debug 定时任务（60Hz）
    ↓
write_debug_data()
    ↓
draw_auto_aim()
    ↓
web::write_shm()
    ↓
网页调试器 /video、/data、/log
```

这样设计的核心好处是：

- 算法主链只负责产生状态，不直接关心页面
- 绘图逻辑统一放在一处，便于维护
- 调试输出是旁路，而不是业务主链的一部分

## 3. 调试数据从哪里来

`draw_auto_aim()` 并不自己计算跟踪结果和控制指令，而是从 `AutoAimDebugCtx` 中读取一份调试快照。

`AutoAimDebugCtx` 定义在 `src/tasks/auto_aim/debug.hpp`，它维护了这些关键字段：

- `camera_info_`
  相机内参与畸变参数，用于 3D 到 2D 的投影。
- `armors_buffer`
  当前检测到的装甲板集合。
- `armor_target_buffer`
  跟踪器输出的目标状态。
- `img_frame_buffer`
  当前图像帧。
- `expanded_buffer`
  当前 ROI 区域。
- `avg_latency_ms_buffer`
  平均处理延迟。
- `gimbal_cmd_buffer`
  云台控制器输出的指令。
- `fsm_state_buffer`
  自瞄状态机状态。
- `gimbal_yaw_pitch_buffer`
  当前云台姿态。
- `bullet_positions_buffer`
  子弹预测轨迹点。

你可以把它理解成“绘图层的共享上下文”。

从工程角度看，这种设计非常常见：

- 算法线程负责更新上下文
- 调试线程只消费上下文

这比在检测器、跟踪器、控制器里到处 `cv::line()`、`cv::putText()` 要干净很多。

## 4. `draw_auto_aim()` 的整体结构

函数入口在 `src/tasks/auto_aim/debug.cpp` 开头。整体结构很清晰，可以拆成下面 9 个阶段：

1. 读取调试快照
2. 画 ROI
3. 画原始检测装甲板
4. 画跟踪器预测出的目标装甲板
5. 画目标中心和角速度
6. 画瞄准点与控制方向
7. 画子弹轨迹
8. 画文字信息
9. 画中心参考点

伪代码可以概括成：

```cpp
if img empty:
    return

snapshot = ctx(...)

draw roi
draw raw armors

if tracking target valid:
    draw predicted armors
    draw target center + vyaw
    draw aim point + fire advice + velocity arrow

draw bullet positions
draw fire text
draw time + latency
draw cmd text
draw fsm + target class
draw image center
```

## 5. 第一部分：读取快照和基础保护

源码开头这几行很关键：

```cpp
if (img.empty()) {
    return;
}
auto armors = ctx.armors();
auto armor_target = ctx.armor_target();
auto camera_info = ctx.camera_info();
auto cmd = ctx.gimbal_cmd();
auto fsm = ctx.fsm_state();
auto bullet_poss = ctx.bullet_positions();
```

这里做了两件事：

1. 先检查图像是否为空。
2. 从 `ctx` 中读取当前快照。

为什么先把 `ctx` 里的内容读出来？

因为后面绘制阶段只操作局部变量，避免在绘图过程中反复加锁读取共享状态。这种方式虽然还不是严格的“整帧一致性快照”，但已经比边画边读强很多。

如果你做自己的项目，更工程化的做法通常是：

- 让调试线程一次性拿到完整快照结构
- 然后整个绘制过程只使用这份不可变快照

## 6. 第二部分：ROI 绘制

对应代码：

```cpp
const cv::Rect img_rect(0, 0, img.cols, img.rows);
const cv::Rect roi = ctx.expanded() & img_rect;
if (roi.width > 0 && roi.height > 0) {
    cv::rectangle(img, roi, cv::Scalar(255, 255, 255), 2);
}
```

这里不是直接画 `expanded`，而是先和整张图做一次相交：

```cpp
roi = expanded ∩ image_rect
```

这样做的作用是避免 ROI 越界。

这一步所表达的语义是：

- 当前系统并不一定在全图搜索
- 它可能根据上一帧的目标状态，只关注某个局部区域

从调试角度，ROI 是非常重要的，因为它直接反映：

- 当前是否进入局部跟踪模式
- 局部跟踪窗口是否过大或过小
- 跟踪器失锁是不是因为 ROI 偏了

如果你要改自己的项目，ROI 可视化几乎应该是必备项。

## 7. 第三部分：原始检测装甲板绘制

对应代码：

```cpp
armors.draw(img);
```

这里真正执行绘制的是 `src/tasks/auto_aim/type.hpp` 里的 `Armor::draw()` 和 `Armors::draw()`。

`Armor::draw()` 的主要逻辑是：

- 取出四个关键点
- 连成四边形
- 在底部写上 `color_class`

也就是说，这一层绘制的是“检测器看到了什么”，它属于原始感知结果，而不是预测结果。

这层图最重要的价值是帮助你判断：

- 检测框位置对不对
- 颜色识别是否正确
- 数字分类是否正确
- 检测是否抖动

所以调试图里一定要同时保留：

- 原始检测结果
- 跟踪器预测结果

这两层不能混为一层，否则你无法判断问题出在检测还是跟踪。

## 8. 第四部分：预测装甲板刚体姿态绘制

这部分是整个 `draw_auto_aim()` 最核心的内容。

对应代码：

```cpp
if (armor_target.check()) {
    auto target_state = armor_target.get_target_state();
    target_state.predict(Clock::now());
    auto armors_pose_in_camera_cv = target_state.get_armors_pose(armor_target.target_number);
    ...
}
```

这里的逻辑是：

1. 先确认当前目标仍然有效
2. 取出目标状态
3. 将状态预测到当前时刻
4. 根据目标类型，恢复出该目标上所有装甲板的 3D 位姿
5. 再投影回图像平面

这一步非常有代表性，因为它不是“拿检测框直接画”，而是：

```text
滤波状态 x
  → 恢复目标几何结构
  → 生成每块装甲板的 3D pose
  → 通过相机模型投影回 2D
```

这正是一个成熟视觉系统调试图的做法。

### 8.1 为什么要 `predict(Clock::now())`

`target_state` 是滤波器状态，它本身带时间戳。绘制前调用：

```cpp
target_state.predict(Clock::now());
```

目的是把状态推进到“当前显示时刻”，而不是停留在上次更新时刻。

这意味着当前画面更接近：

- 系统“现在认为目标在哪里”

而不是：

- 上一帧观测时目标在哪里

这对高速转动目标非常重要。

不过这里也有一个你可以思考的点：

- 当前调试图基底图像是某一帧图像
- 但预测时间用了 `Clock::now()`

所以这张图展示的是“旧图像 + 当前预测”。这对于在线控制调试很有意义，但如果你追求严格的“图像时间一致性”，也可以改成按图像时间戳预测。

### 8.2 `get_armors_pose()` 的意义

`get_armors_pose()` 定义在运动模型状态中，它会根据当前目标状态，恢复出整个目标上所有装甲板的 3D 位姿。

例如：

- 普通车通常有 4 块装甲板
- 前哨站有 3 块装甲板

函数内部大致做了这些事：

1. 根据 `armor_number` 确定装甲板数量
2. 根据目标状态计算每块装甲板中心位置
3. 根据 yaw 和固定 pitch 构造旋转矩阵
4. 输出每块装甲板在当前坐标系下的 `ISO3`

所以这里画出来的不只是一个目标中心，而是“目标刚体模型”。

### 8.3 `reprojection()` 的作用

对应调用：

```cpp
auto image_points = utils::reprojection(
    camera_info.camera_matrix,
    camera_info.distortion_coefficients,
    getArmorKeyPoints3D<cv::Point3f>(armor_target.target_number),
    armor_pose_in_camera_cv
);
```

这个函数做的是标准几何投影：

- 输入：3D 点、相机内参、畸变、目标位姿
- 输出：图像平面 2D 点

内部流程是：

1. 从 `ISO3` 取旋转和平移
2. 转成 OpenCV 需要的 `rvec/tvec`
3. 调 `cv::projectPoints()`

也就是说，这里不是凭经验画框，而是严格按 PnP 逆过程把 3D 模型投回图像。

### 8.4 为什么用不同颜色区分选中装甲板

对应代码：

```cpp
(i == cmd.select_id) ? cv::Scalar(255, 0, 255) : cv::Scalar(200, 255, 200)
```

语义是：

- 洋红色：当前控制器真正选择去打的那块板
- 淡绿色：同一目标上其它可推算出的装甲板

这层设计非常好，因为它回答了一个关键问题：

- 跟踪器知道整个目标在怎么转
- 控制器最后到底选了哪一块去打

如果你发现：

- 预测装甲板位置是对的
- 但选中的不是你预期那块

那么问题就很可能在目标选择策略，而不是跟踪器本身。

## 9. 第五部分：目标中心和角速度绘制

对应代码块：

```cpp
auto center_pose = ISO3::Identity();
center_pose.translation() = target_state.pos();
auto center_image_points = utils::reprojection(...);
cv::Point2f center = center_image_points[0];
const double scale = 50.0;
const double dy = scale * target_state.vyaw();
...
cv::arrowedLine(...)
cv::circle(...)
cv::putText(... "V_yaw: ...")
```

这部分做了三件事：

1. 在目标中心画一个点
2. 用一根竖直箭头表示角速度
3. 在旁边写 `V_yaw`

要注意，这里的箭头不是物理空间真实方向，而是一个“调试语义箭头”：

- 它只在图像上沿 y 方向上下变化
- 长度与 `vyaw` 成比例

它的目的不是精确表达三维旋转轴，而是用简单直观的方式告诉开发者：

- 当前目标在不在旋转
- 旋转速度大概多大
- 旋转方向有没有跳变

这是比赛调试里常见的做法：优先可读性，而不是严格物理可视化。

## 10. 第六部分：瞄准点和开火建议绘制

对应代码块：

```cpp
if (cmd.aim_point.pose.translation().z() > 0.1) {
    auto aim_point_img_points = utils::reprojection(...);
    constexpr double R = 0.02;
    cv::Point2f center = aim_point_img_points[0];
    double r = fx * R / z;
    cv::circle(img, center, r, cv::Scalar(255, 255, 255), 2);
    ...
}
```

### 10.1 这一步画的不是装甲板中心，而是“最终瞄准点”

`cmd.aim_point.pose` 来自控制器和弹道解算器，代表：

- 云台最终应该瞄准的空间点

它可能和目标中心不同，也可能和装甲板中心不同，因为里面已经包含了：

- 目标运动预测
- 飞行时间补偿
- 可能的偏置修正

所以这个白圈实际上是整套自瞄系统最关键的可视化元素之一。

### 10.2 半径公式为什么是 `r = fx * R / z`

这里定义了一个固定物理半径：

```cpp
constexpr double R = 0.02;
```

再根据针孔模型算出它在图像上的投影半径：

```text
r_image = fx * R_world / z
```

这说明作者想画的不是“屏幕固定大小圆圈”，而是“带深度感的空间参考圆”。

这样当目标远近变化时，瞄准圈大小也会变化，更符合真实投影关系。

这个细节很值得保留。

## 11. 第七部分：开火建议 `fire_advice`

对应代码：

```cpp
if (cmd.fire_advice) {
    int size = 50;
    cv::line(...);  // 斜杠
    cv::line(...);  // 反斜杠
}
...
if (cmd.fire_advice) {
    cv::putText(img, "Fire!", ...);
}
```

这里用了两层视觉提示：

1. 在瞄准点位置画一个红色 `X`
2. 在画面中央上方再写一个大号 `Fire!`

这是非常实战化的设计，因为：

- 瞄准圈上的 `X` 是局部提示
- 中央大字是全局提示

前者便于看空间位置，后者便于快速扫视整张图。

如果你以后做自己的项目，建议把“是否可开火”做成强提示，而不是只在日志里打印一个布尔值。

## 12. 第八部分：控制速度箭头

对应代码：

```cpp
const double scale = 10.0;
const double dx = -scale * v_yaw;
const double dy =  scale * v_pitch;
cv::arrowedLine(img, start_pt, end_pt, color_x, 4, cv::LINE_AA, 0, 0.2);
```

这根金黄色箭头表达的是：

- 当前控制器希望云台继续往哪个方向运动

这里的映射关系值得注意：

```text
v_yaw   → 图像 x 方向
v_pitch → 图像 y 方向
```

但 yaw 前面有个负号：

```cpp
dx = -scale * v_yaw
```

这是因为图像坐标系和控制坐标系的正方向定义不完全一致，所以需要做符号变换。

这也是你以后做自己项目时特别要注意的一类问题：

- 图像坐标系
- 相机物理坐标系
- 云台 yaw/pitch 正方向

这三者一旦约定不清，调试箭头就会“看起来反着动”。

## 13. 第九部分：子弹轨迹绘制

对应代码：

```cpp
for (auto& p: bullet_poss) {
    ISO3 pose = ISO3::Identity();
    pose.translation() = p;
    if (p.z() > 0.2) {
        auto bullet_img_points = utils::reprojection(...);
        constexpr double R = 0.017 / 2.0;
        ...
        cv::circle(img, center, r, cv::Scalar(100, 255, 100), 3);
    }
}
```

这部分画的是“在途子弹位置”。

它的意义很大，因为它把弹道模型从纯数字变成了可见对象。你可以通过这些绿色小圆点判断：

- 弹道模型是否偏高或偏低
- 预测出来的落点是否对准瞄点
- 飞行时间估计是否合理

这对调：

- 空气阻力模型
- 重力补偿
- 反陀螺提前量

都很有帮助。

不过这里有个你可以继续优化的细节：

```cpp
r = fx * R / pose.translation().norm();
```

当前用的是点到相机的欧氏距离 `norm()`，而不是更标准的 `z` 深度。对于小角度场景问题不大，但如果你追求更严格的投影一致性，可以改成按 `z` 计算。

## 14. 第十部分：时间与平均延迟文本

对应代码：

```cpp
const std::string latency_str =
    fmt::format("Avg Latency: {:.2f}ms", ctx.avg_latency_ms());
```

然后把：

- 当前系统时间
- 平均处理延迟

画在左上角。

这层信息看似简单，实际上非常重要，因为它回答的是：

- 当前画面是不是卡住了
- 系统当前处理压力大不大
- 你调参数后延迟有没有恶化

建议你在自己的系统里也至少保留：

- 当前时间
- 帧号
- 端到端延迟
- 推理耗时
- 控制耗时

如果再工程化一点，可以拆成：

- detector latency
- tracker latency
- controller latency
- render latency
- total latency

## 15. 第十一部分：底部控制指令文本

对应代码：

```cpp
const std::string yaw_cmd_str =
    "yaw:   p:" + format_col(cmd.yaw)
  + " v:" + format_col(cmd.v_yaw)
  + " a:" + format_col(cmd.a_yaw)
  + " enable:" + format_col(cmd.enable_yaw_diff);
```

`pitch` 也同理。

这一层表达的是控制器当前输出的“完整控制状态”，不是只有位置指令，而是：

- 位置
- 速度
- 加速度
- enable 标志

这非常适合调：

- 前馈
- MPC
- 五次多项式轨迹
- 带速度和加速度约束的控制器

因为你能直观看到：

- 目标在动，控制器给了多少速度
- 目标突变时，加速度是不是暴冲
- 某个轴是不是被 enable gating 限制住了

如果你的项目后面要升级控制器，这层信息强烈建议保留。

## 16. 第十二部分：状态机和攻击目标文本

对应代码：

```cpp
std::string state_str = string_by_auto_aim_fsm(fsm);
...
const std::string id_str =
    fmt::format("Attack: {}", string_by_armor_class(armor_target.target_number));
```

这里画了两种高层语义：

1. 当前自瞄状态机状态
   例如：
   - `AIM_SINGLE_ARMOR`
   - `AIM_WHOLE_CAR_ARMOR`
   - `AIM_WHOLE_CAR_PAIR`
   - `AIM_WHOLE_CAR_CENTER`

2. 当前攻击目标类别
   例如：
   - `sentry`
   - `outpost`
   - `no1`

这层信息非常重要，因为它让开发者看到的不只是几何，而是“策略层正在怎么思考”。

很多调试失败的原因，并不是检测错了，而是：

- 状态机切换太早
- 状态机切换太晚
- 选错打击目标

所以把高层决策状态明确画出来，是成熟调试图的一部分。

## 17. 第十三部分：中心参考点

最后这一句：

```cpp
cv::circle(img, cv::Point2i(img.cols / 2, img.rows / 2), 5, cv::Scalar(255, 255, 255), 2);
```

画的是图像中心参考点。

它作用很朴素，但很有用：

- 帮你快速看瞄点偏差
- 帮你判断机械中轴和相机中心偏差
- 调静态偏置时很直观

## 18. `write_debug_data()` 的作用

很多人只看 `draw_auto_aim()`，但 `write_debug_data()` 也非常值得学，因为它决定了网页曲线的数据质量。

这个函数主要做了四件事：

1. 生成相对时间轴 `t`
2. 读取云台角、控制指令、目标状态
3. 对 yaw 做“解缠绕”
4. 写入 `web::DebugDatas`

### 18.1 为什么要解缠绕 `yaw`

对应代码：

```cpp
auto un_warp = [&](double _yaw) {
    return last_yaw + angles::shortest_angular_distance_degrees(last_yaw, _yaw);
};
```

这个处理非常关键。

原因是 yaw 角天然会在：

- `179° -> -179°`

这种边界发生跳变。

如果直接画曲线，图表会出现一根巨大折线，看起来像系统突然暴走。解缠绕后，曲线会连续很多，更适合观察趋势。

这在你自己的调试系统里也几乎是必须处理的。

### 18.2 为什么 `cmd.appear == false` 时保留 `last_cmd`

对应代码：

```cpp
cmd = cmd.appear ? cmd : last_cmd;
last_cmd = cmd;
```

这一步是为了让图表更连续。

否则某些时刻目标短暂消失，控制指令会突变成空值，图表会出现明显断裂。当前实现选择在 `appear == false` 时暂时沿用上一条控制指令，便于观察整体趋势。

这是一种很典型的“为调试可读性服务”的设计。

## 19. 这份可视化最值得你学习的设计思想

如果你是要修改自己的项目，我认为最值得学的不是具体颜色或字体，而是下面这些设计思想：

### 19.1 把感知、跟踪、控制三层信息同时画在一张图上

这张调试图不是“检测图”，也不是“控制图”，而是一个完整系统图。

它同时告诉你：

- 检测器看到了什么
- 跟踪器预测了什么
- 控制器最终决定打哪里

这是非常正确的调试思路。

### 19.2 不只画 2D 框，而是画 3D 几何投影

这点非常关键。

作者不是直接沿用检测框，而是把目标几何结构恢复成 3D，再投影回图像。这样你看到的是：

- “模型预测的结果”

而不是：

- “检测器最后一次看到的结果”

如果你以后要做更强的预测显示，这种设计必须保留。

### 19.3 把高层策略状态也画出来

很多项目调试图只画框，不画状态机，这是不够的。

当前实现把：

- FSM 状态
- 当前攻击目标
- 开火建议

都直接画出来了，这让调试从“看几何”升级成“看系统决策”。

## 20. 如果你要改自己的项目，我建议优先这样改

## 20.1 先按“绘制层次”重构

建议你把可视化拆成几层，而不是写成一个巨大的 `draw()`：

1. 基础层
   原图、中心点、ROI

2. 感知层
   检测框、分类信息、关键点

3. 跟踪层
   目标状态、预测装甲板、中心、速度

4. 控制层
   瞄准点、控制箭头、可开火提示

5. 系统层
   时间、延迟、状态机、目标 ID

这样以后你想开关某一层、调整颜色、迁移到别的任务上都更方便。

## 20.2 把“绘制语义”做成数据结构

比起让业务代码直接调用 OpenCV，我更推荐把图元抽象出来，例如：

```cpp
struct DebugCircle { ... };
struct DebugLine   { ... };
struct DebugText   { ... };
struct DebugArrow  { ... };
```

业务层只负责发布：

- 我要画什么

渲染层负责决定：

- 怎么用 OpenCV / ImGui / WebGL 去画

这会让你的系统更工程化。

## 20.3 把常量参数配置化

当前代码里很多值是硬编码的，比如：

- `scale = 50.0`
- `scale = 10.0`
- `size = 50`
- 字体大小
- 颜色

如果你准备长期维护，建议把这些抽到配置或主题对象中。

## 20.4 把绘图时间和算法时间分离

当前代码里 `target_state.predict(Clock::now())` 更偏在线显示。

如果你想做：

- 录像回放复现
- 离线误差分析
- 严格对齐图像时间戳

那么建议引入两种模式：

1. 实时显示模式
   按当前时刻预测

2. 帧一致模式
   按图像时间戳预测

## 20.5 把网页曲线和图像绘制分离看待

`draw_auto_aim()` 是“空间调试”，`write_debug_data()` 是“时序调试”。

这两个维度要同时保留。

很多问题：

- 在图像上看不出来
- 但在曲线上很明显

例如：

- 控制振荡
- yaw 角跳变
- 延迟恶化
- 加速度尖峰

## 21. 当前实现中值得你继续优化的地方

下面这些是我结合代码后认为你可以继续优化的点：

### 21.1 `AutoAimDebugCtx` 不是强一致快照

当前每个字段单独加锁，绘图时逐个读取，所以严格来说可能读到“跨时刻混合状态”。如果你项目规模更大，建议改成单快照结构。

### 21.2 一些文本位置是硬编码

例如：

- `Fire!` 的位置
- `Attack:` 的位置
- 字体大小

后面如果你换分辨率，可能要重新调。

### 21.3 控制箭头是经验映射，不是严格物理映射

这不是坏事，但你要知道它主要服务“可读性”，不是严格几何一致性。

### 21.4 子弹绘制半径可以更规范

当前按 `norm()` 算半径，你如果要追求更精确显示，建议按深度 `z`。

### 21.5 预测时刻可以模式化

实时模式和回放模式最好分开。

## 22. 总结

`src/tasks/auto_aim/debug.cpp` 的价值不在于 OpenCV API 本身，而在于它把一个 RoboMaster 自瞄系统中最关键的三类信息统一映射到了图像上：

- 感知结果
- 状态估计结果
- 控制与决策结果

你可以把它理解成一个非常典型的“视觉系统调试仪表盘”。

如果你要把自己的项目做得更工程化，我建议：

1. 保留这种分层表达思路。
2. 保留 3D 预测投影回 2D 的做法。
3. 保留图像调试和曲线调试两条链路。
4. 把绘图层和业务层进一步解耦。
5. 把快照、图元、主题、显示模式做成正式架构的一部分。

这样你做出来的下一版系统，会既保留当前这种比赛中非常好用的直观性，又具备更强的可维护性和扩展性。
