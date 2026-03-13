# auto_aim/yolos 分析

## 1. 目录职责
`yolos/` 提供不同检测模型版本的统一实现，输出统一 `Armor` 数据结构给后续 `Solver/Tracker`。

按照你的要求，这里不展开模型推理细节，只关注后处理、几何规整和控制相关输出一致性。

## 2. 文件级说明

| 文件 | 核心类 | 作用 | 与主链路关系 |
|---|---|---|---|
| `yolov5.hpp` | `YOLOV5` | YOLOv5 检测与后处理 | 输出 `Armor` 到 `Tracker` |
| `yolov5.cpp` | `YOLOV5` 实现 | 解析输出、NMS、类别映射、可选传统角点矫正 | 可调用 `Detector::detect(Armor&, img)` 二次修角点 |
| `yolov8.hpp` | `YOLOV8` | YOLOv8 检测与后处理 | 输出 `Armor` 到 `Tracker` |
| `yolov8.cpp` | `YOLOV8` 实现 | 输出转置、keypoints排序、`Classifier`补充识别、类型判定 | 兼容旧 `Armor` 接口 |
| `yolo11.hpp` | `YOLO11` | YOLO11 检测与后处理 | 输出 `Armor` 到 `Tracker` |
| `yolo11.cpp` | `YOLO11` 实现 | 多类别输出解析、NMS、keypoints排序、过滤 | 相比 v8，直接用 class 映射属性 |

## 3. 共同后处理链路
三者都遵循：

1. 前处理缩放到固定网络输入尺寸（416 或 640）
2. 解析输出得到 bbox + 角点 + 分类置信度
3. `cv::dnn::NMSBoxes` 去重
4. 构造 `Armor`（包含 keypoints、类别、颜色等）
5. 规则过滤：
   - `check_name()`
   - `check_type()`
   - 归一化中心 `center_norm`
6. 返回 `std::list<Armor>`

## 4. 各版本差异（算法/控制相关）
- `YOLOV5`：
  - 输出含颜色与编号独热，手动 `sigmoid + NMS`。
  - 可选 `use_traditional_`，用传统 `Detector` 对角点做二次矫正（对解算稳定性有帮助）。
- `YOLOV8`：
  - 类别较少，编号由 `Classifier` 二级分类补齐（先检装甲，再识别数字）。
  - `sort_keypoints()` 强制角点顺序，避免 PnP 输入顺序错乱。
- `YOLO11`：
  - 类别更完整，通常可直接从 `class_id` 恢复 `color/name/type`。
  - 也做 `sort_keypoints()` 保证几何一致。

## 5. 与上层关系
- `tasks/auto_aim/yolo.cpp` 只负责分发到 `YOLOV5/8/11`，不参与后处理细节。
- 该目录最终保证上游模型变化不会影响下游 `Solver/Tracker/Aimer` 接口。

