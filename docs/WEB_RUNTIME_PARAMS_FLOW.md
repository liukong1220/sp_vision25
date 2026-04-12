# 网页热调参数链路说明

这份文档只说明一件事: 当前“网页改参但不回写 YAML, 同时自动落日志和快照, 并把新参数实时刷进击打链路”的实现到底分布在哪些文件里。

当前主要入口程序是:

- `src/auto_aim_debug_mpc.cpp`
- `tests/auto_aim_test_web.cpp`

## 1. 网页改参入口在哪

整条网页改参链路分成 4 层:

1. 前端参数面板
2. WebDebugger HTTP 接口
3. RuntimeParamSession 内存会话
4. 各算法模块按版本号热刷新

### 1.1 前端页面

前端面板在:

- `assets/web_debugger/index.html`
- `assets/web_debugger/static/js/main.js`
- `assets/web_debugger/static/css/style.css`

这里新增了一块“参数热调”面板, 位于 `Inspector` 视图。

前端真正负责改参的关键逻辑在 `assets/web_debugger/static/js/main.js`:

- `renderRuntimeParams(payload)`
  作用: 根据后端返回的参数描述动态渲染整张参数面板。
- `fetchRuntimeParams()`
  作用: 请求 `GET /api/params`, 拉取当前基线值、覆盖值、导出 YAML 片段、日志路径、快照路径。
- 单参数应用按钮
  作用: 向 `POST /api/params` 发送 `{ updates: { key: value } }`。
- 单参数恢复按钮
  作用: 向 `POST /api/params/reset` 发送 `{ keys: [key] }`。
- 全部恢复按钮
  作用: 向 `POST /api/params/reset` 发送 `{ keys: [] }`。

这里有一个很重要的设计点:

- 前端没有写死每个参数的控件布局。
- 参数分组、类型、单位、默认值、当前值，都是后端 `describe()` 返回什么，前端就渲染什么。

所以后面如果你要继续新增网页可调参数，前端通常不用再单独手搓一套新 UI，只要后端把参数放进描述结果里，它就会自动显示。

### 1.2 WebDebugger 后端接口

后端接口在:

- `tools/web_debugger.hpp`
- `tools/web_debugger.cpp`

关键新增点:

- `WebDebugger::set_runtime_config_path(const std::string & config_path)`
  作用: 把当前运行实例绑定到某个 YAML 配置路径，并注册运行时参数会话。

关键接口:

- `GET /api/params`
  读取当前运行时参数状态。
- `POST /api/params`
  应用网页改参，只改内存会话，不改 YAML。
- `POST /api/params/reset`
  将一个或全部参数恢复为 YAML 基线值。

### 1.3 哪两个程序真正把网页改参能力挂起来

程序入口里都加了:

- `src/auto_aim_debug_mpc.cpp`
- `tests/auto_aim_test_web.cpp`

关键调用都是:

```cpp
web_debugger->set_runtime_config_path(config_path);
```

这一步非常关键。没有这一步:

- `/api/params` 不会知道该绑定哪份配置
- 运行时参数会话不会建立
- 网页就只能看图, 不能热调参数

## 2. 参数状态实际存在哪

真正的参数热调核心不在前端，也不在 `WebDebugger`，而在:

- `tools/runtime_params.hpp`
- `tools/runtime_params.cpp`

这里实现的是一个“运行时参数会话”。

### 2.1 会话里保存了什么

`SessionState` 里主要有 3 份值:

- `base_values`
  从 YAML 读出来的基线值。
- `effective_values`
  当前真正给算法链路使用的值。
- `overrides`
  相比 YAML 基线发生了变化的那部分值。

另外还有:

- `version`
  每次网页改参或恢复时自增。
- `session_log_path`
  本次调参过程的 JSONL 日志文件。
- `snapshot_path`
  当前最新参数快照文件。

### 2.2 哪些参数允许网页改

白名单在 `tools/runtime_params.cpp` 的 `build_specs()` 里。

当前已经纳入热调的主要分组有:

- `YOLO筛选`
  `min_confidence`, `use_traditional`, `use_roi`, `roi.x`, `roi.y`, `roi.width`, `roi.height`
- `传统检测`
  `threshold`, `max_angle_error`, `min_lightbar_ratio`, `max_lightbar_ratio`, `min_lightbar_length`, `min_armor_ratio`, `max_armor_ratio`, `max_side_ratio`, `max_rectangular_error`
- `跟踪`
  `enemy_color`, `min_detect_count`, `max_temp_lost_count`, `outpost_max_temp_lost_count`, `outpost_radius`, `outpost_spin_speed_lock`, `outpost_fixed_center_rotation_model`, `outpost_armor_z_offsets`
- `规划/MPC`
  `yaw_offset`, `pitch_offset`, `comming_angle`, `leaving_angle`, `decision_speed`, `high_speed_delay_time`, `low_speed_delay_time`, `fire_thresh`, `max_yaw_acc`, `Q_yaw`, `R_yaw`, `max_pitch_acc`, `Q_pitch`, `R_pitch`

只有进入这个白名单的参数，网页上才会出现，也才能通过 API 修改。

## 3. 哪几处负责“不回写 YAML, 但要落日志和快照”

这部分也都在 `tools/runtime_params.cpp`。

### 3.1 不回写 YAML 的控制点

当前网页改参时调用的是:

- `tools::runtime_params::apply(...)`
- `tools::runtime_params::reset(...)`

它们只会改:

- `session.effective_values`
- `session.overrides`
- `session.version`

不会去写原始配置文件。

也就是说:

- YAML 仍然保持你启动程序时的原值
- 网页改过的值只存在于当前运行会话
- 后续由你自己把 `export_yaml` 手动复制回配置文件

### 3.2 自动日志落盘点

日志相关函数:

- `build_log_paths()`
  在 `logs/web_params/` 下生成日志路径和最新快照路径。
- `persist_change_event_locked()`
  把每次参数变更追加写入 `*.jsonl`。
- `persist_snapshot_locked()`
  把当前全部覆盖状态写入 `latest_*.runtime.json`。

底层持久化函数:

- `durable_write_append(...)`
- `durable_write_overwrite(...)`

这两个函数在写完之后都会调用 `fsync`，目的就是尽量降低突然断电时刚调好的参数完全丢失的概率。

当前落盘目录:

- `logs/web_params/<timestamp>_<config>.jsonl`
  本次调参过程的事件日志
- `logs/web_params/latest_<config>.runtime.json`
  当前最新快照

### 3.3 快照里会保存什么

快照文件里主要会有:

- `config_path`
- `version`
- `saved_unix_ms`
- `overrides`
- `flat_overrides`
- `effective`
- `export_yaml`

你最常用的字段一般是:

- `flat_overrides`
  直接看当前网页改过哪些 key。
- `export_yaml`
  直接复制回 YAML 用。

## 4. 哪几处真正把参数刷新进击打链路

这里只看“网页改了之后，算法什么时候真正用上新值”。

整体机制是:

1. 网页改参后, `runtime_params` 会把 `version` 加 1
2. 各模块在主流程里调用 `refresh_runtime_params_if_needed()`
3. 如果发现版本号变化，就把本模块内部缓存参数改成最新值

这套方式的优点是:

- 不需要重启程序
- 不需要重新加载整份 YAML
- 刷新开销很小
- 只影响当前模块关心的参数

### 4.1 Detector

文件:

- `tasks/auto_aim/detector.hpp`
- `tasks/auto_aim/detector.cpp`

关键函数:

- `Detector::refresh_runtime_params_if_needed()`

刷新进去的参数主要是:

- `threshold`
- `max_angle_error`
- `min_lightbar_ratio`
- `max_lightbar_ratio`
- `min_lightbar_length`
- `min_armor_ratio`
- `max_armor_ratio`
- `max_side_ratio`
- `min_confidence`
- `max_rectangular_error`

它在 `detect(...)` 等实际检测流程前会先检查版本号，所以改完网页参数后，后续帧就会直接按新阈值跑。

### 4.2 Tracker

文件:

- `tasks/auto_aim/tracker.hpp`
- `tasks/auto_aim/tracker.cpp`

关键函数:

- `Tracker::refresh_runtime_params_if_needed()`

刷新进去的参数主要是:

- `enemy_color`
- `min_detect_count`
- `max_temp_lost_count`
- `outpost_max_temp_lost_count`
- `outpost_radius`
- `outpost_spin_speed_lock`
- `outpost_fixed_center_rotation_model`
- `outpost_armor_z_offsets`

这里还有一个额外处理:

- 如果改的是会影响跟踪模型结构一致性的参数，比如前哨站半径、旋转模型、装甲板高度偏置、敌方颜色等，会主动把 tracker 状态重置为 `lost`。

这样做是为了避免旧状态继续沿用到新模型里，导致内部状态不一致。

### 4.3 Planner / MPC

文件:

- `tasks/auto_aim/planner/planner.hpp`
- `tasks/auto_aim/planner/planner.cpp`

关键函数:

- `Planner::refresh_runtime_params_if_needed()`
- `Planner::setup_yaw_solver()`
- `Planner::setup_pitch_solver()`

刷新进去的参数主要是:

- `yaw_offset`
- `pitch_offset`
- `comming_angle`
- `leaving_angle`
- `decision_speed`
- `high_speed_delay_time`
- `low_speed_delay_time`
- `fire_thresh`
- `max_yaw_acc`
- `Q_yaw`
- `R_yaw`
- `max_pitch_acc`
- `Q_pitch`
- `R_pitch`

这里是最接近“实际击打效果”的核心位置，因为这些参数会直接影响:

- 目标切板逻辑
- 预测延迟
- 云台零偏补偿
- MPC 约束和权重
- 开火判定阈值

也就是说，你在网页上调这些参数，后续实击效果变化最直接。

### 4.4 YOLOV5 / YOLOV8 / YOLO11

文件:

- `tasks/auto_aim/yolos/yolov5.cpp`
- `tasks/auto_aim/yolos/yolov8.cpp`
- `tasks/auto_aim/yolos/yolo11.cpp`

关键函数都是:

- `refresh_runtime_params_if_needed()`

刷新进去的参数主要是:

- `min_confidence`
- `use_roi`
- `roi.x`
- `roi.y`
- `roi.width`
- `roi.height`

其中 `YOLOV5` 额外还会刷新:

- `use_traditional`

这几处改完以后，下一帧推理就会按新的 ROI 和置信度门限执行。

## 5. 这三条线之间的实际调用关系

可以把当前结构理解成下面这条链:

```text
网页参数面板
  -> /api/params 或 /api/params/reset
  -> tools/web_debugger.cpp
  -> tools/runtime_params.cpp
  -> 更新 effective_values / overrides / version
  -> 追加 jsonl 日志 + 覆盖 latest 快照
  -> detector / tracker / yolo / planner 在下一轮主流程里检测到 version 变化
  -> 模块内部参数刷新
  -> 实际识别、跟踪、规划、开火判定马上体现新效果
```

## 6. 后面如果你要继续扩新的网页热调参数, 应该改哪

推荐按下面 4 步走。

### 第一步: 把参数加入白名单

改:

- `tools/runtime_params.cpp` 的 `build_specs()`

你需要补:

- key
- 分组 id / 分组名
- 中文标签
- 描述
- 单位
- 类型
- 默认值或枚举范围

### 第二步: 确保能从 YAML 基线读出来

当前基线读取也在:

- `tools/runtime_params.cpp`

会话初始化时会从配置文件中读取这些 key 作为 `base_values`。

如果你的新 key 是嵌套字段，比如 `xxx.yyy`，要确认它和现有 `roi.x` 一样能被正确解析。

### 第三步: 把它真正刷新进对应模块

如果你只把参数加入白名单，但模块里没有在 `refresh_runtime_params_if_needed()` 中读取它，那么:

- 网页会显示这个参数
- API 也能改
- 日志和快照也会记住
- 但算法不会真的用上新值

所以一定要去对应模块里补这一步。

### 第四步: 如果参数会改变状态机结构，要决定是否清状态

例如 `Tracker` 里已经做了一个例子:

- 某些关键参数变化后直接把状态置回 `lost`

如果你后面加的是:

- 目标模型尺寸
- 状态机阈值
- EKF 状态定义
- 装甲板布局

这类“改完以后旧内部状态不再可信”的参数，也建议同步做状态重置。

## 7. 当前实现的边界

当前方案已经满足:

- 网页可以直接改参数
- 改完立即作用到当前运行链路
- 不自动回写 YAML
- 自动保留日志和最新快照

但仍然要记住:

- 现在是“白名单参数热调”，不是“整份 YAML 任意字段热调”
- 只有已经接入 `refresh_runtime_params_if_needed()` 的模块，才会真正感知参数变化
- 断电保护依赖日志和快照落盘，不等于替代正式配置管理

## 8. 你后面最常看的文件

如果后面你要继续扩这套能力，最常打开的就是这几处:

- `tools/runtime_params.cpp`
- `tools/web_debugger.cpp`
- `assets/web_debugger/static/js/main.js`
- `src/auto_aim_debug_mpc.cpp`
- `tests/auto_aim_test_web.cpp`
- `tasks/auto_aim/detector.cpp`
- `tasks/auto_aim/tracker.cpp`
- `tasks/auto_aim/planner/planner.cpp`
- `tasks/auto_aim/yolos/yolov5.cpp`
- `tasks/auto_aim/yolos/yolov8.cpp`
- `tasks/auto_aim/yolos/yolo11.cpp`

如果只记一句话，可以记成:

> 网页负责发改参请求，`runtime_params` 负责保存运行时覆盖和落盘，Detector/Tracker/YOLO/Planner 负责在下一帧把新值真正用进击打链路。
