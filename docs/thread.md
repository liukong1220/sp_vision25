# auto_aim/multithread 分析

## 1. 目录职责
该目录做两件事：
- 把检测推理从主线程剥离（提升吞吐）
- 把瞄准与发指令剥离成独立线程（降低主循环阻塞）

## 2. 文件级说明

| 文件 | 核心类 | 作用 | 关键依赖 |
|---|---|---|---|
| `mt_detector.hpp` | `MultiThreadDetector` | 异步推理检测接口，提供 `push/pop` | `YOLO`, OpenVINO, ThreadSafeQueue |
| `mt_detector.cpp` | `MultiThreadDetector` 实现 | `push` 时 `start_async`，`pop` 时 `wait+postprocess` | `yolo_.postprocess` |
| `commandgener.hpp` | `CommandGener` | 异步命令生成/发送线程 | `Aimer`, `Shooter`, `io::CBoard` |
| `commandgener.cpp` | `CommandGener` 实现 | 读取最新目标并生成命令，做简单指令去抖后发送 | math_tools/plotter |

## 3. 核心逻辑关系

### 3.1 `MultiThreadDetector`
- `push(img,t)`：
  - 图像缩放到网络输入尺寸（当前硬编码 `640x640`）
  - 创建 infer request 并异步 `start_async()`
  - 把 `(原图,时间戳,request)` 放入线程安全队列
- `pop()` / `debug_pop()`：
  - 取队列头并 `wait()`
  - 取输出 tensor，调用 `yolo_.postprocess(scale, output, img, frame_count)` 还原 `Armor` 列表

关系上：`MultiThreadDetector` 不负责跟踪，只负责“异步推理 + 同步取结果”。

### 3.2 `CommandGener`
- 独立线程循环读取最近一条 `latest_` 输入（目标、时间戳、弹速、云台角）
- 调用：
  - `aimer_.aim(...)` 得到 yaw/pitch
  - `shooter_.shoot(...)` 得到 shoot
- 发送前额外做了简化去抖：
  - 与上次指令比较，若变化很小则按 `round_precision_` 四舍五入
  - 与上次完全相同时不重复发送

关系上：`CommandGener` 是“控制结算输出层”的线程化封装。

## 4. 控制侧重点
- 主线程只负责 `detector.pop -> tracker.track -> commandgener.push`，降低环路延迟波动。
- 实际“是否开火”仍由 `Shooter` 做最终门控。
- `CommandGener` 的阈值目前写死（`0.2rad`），未从 yaml 读取，后续可配置化。

