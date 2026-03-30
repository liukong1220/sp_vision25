# `auto_aim_debug_mpc` 网页调试器

## 目标

`auto_aim_debug_mpc` 现在默认启用内置网页调试器，替代原来的强依赖 `imshow` 本地窗口调试方式。

设计重点：

- 算法主链继续跑 `YOLO -> Solver -> Tracker -> Planner -> Gimbal.send`
- 网页只拉取当前状态 JSON 和两路 JPEG 快照
- 只有最近确实有人访问页面时，程序才会继续做重投影绘制、弹道面板绘制和 JPEG 编码
- 浏览器自己缓存曲线历史，嵌入式端不保存长时间序列

这比每帧都刷新本地 OpenCV 窗口更适合现场调参和远程查看。

## 启动

```bash
./build/auto_aim_debug_mpc configs/standard3.yaml
```

默认行为：

- 网页调试器开启
- 监听 `0.0.0.0:8090`
- 不再强制打开本地 OpenCV 窗口

程序启动后会在日志里打印类似：

```text
Web debugger listening on 0.0.0.0:8090 (open http://127.0.0.1:8090/)
```

本机直接打开：

```text
http://127.0.0.1:8090/
```

如果要从同网段平板或笔记本查看，把 `127.0.0.1` 换成机器人实际 IP。

## 配置文件

现在支持直接从 `configs/*.yaml` 读取网页调试参数，键名如下：

```yaml
show_local: false
disable_web: false
web_host: "0.0.0.0"
web_port: 8090
web_fps: 8.0
web_scale: 0.7
web_jpeg_quality: 70
web_client_ttl_ms: 2000
```

优先级：

- 命令行显式传参时，命令行覆盖 YAML
- 没传命令行时，使用 YAML 值

## 常用参数

```bash
./build/auto_aim_debug_mpc configs/standard3.yaml --web-port=8091 --web-fps=6 --web-scale=0.6
```

```bash
./build/auto_aim_debug_mpc configs/standard3.yaml --show-local
```

```bash
./build/auto_aim_debug_mpc configs/standard3.yaml --disable-web --show-local
```

参数说明：

- `--show-local`: 保留原来的本地 OpenCV 窗口
- `--disable-web`: 禁用网页调试器
- `--web-host`: 绑定地址，默认 `0.0.0.0`
- `--web-port`: 网页端口，默认 `8090`
- `--web-fps`: 网页图像刷新帧率，默认 `8`
- `--web-scale`: 网页图像缩放系数，默认 `0.7`
- `--web-jpeg-quality`: JPEG 质量，默认 `70`
- `--web-client-ttl-ms`: 页面最近访问后的继续渲染时间窗，默认 `2000ms`

## 页面内容

网页端主要包含：

- 主画面叠加层
- 弹道调试面板
- 预览目标/规划命令/弹道误差卡片
- 浏览器端实时曲线

其中曲线默认显示：

- 云台/预览/已发送 yaw
- 云台/预览/已发送 pitch
- 横向误差/竖向误差/总误差
- delay / `w`

## 性能策略

为了不让调试器反向拖慢主链，当前实现做了以下限制：

- 无网页访问时，不做网页帧渲染
- 网页帧单独限频，不跟主循环同频
- 只发送当前状态，不在服务端维护长曲线缓存
- 图像先缩放，再做 JPEG 编码

如果现场仍然觉得页面负担偏高，优先按下面顺序调：

1. 降低 `--web-fps`
2. 降低 `--web-scale`
3. 降低 `--web-jpeg-quality`

如果你感觉不是“负担偏高”，而是“页面太卡、太慢”，优先按下面顺序调：

1. 先把 `web_fps` 从 `8.0` 提到 `12.0` 或 `15.0`
2. 同时把 `web_scale` 从 `0.7` 降到 `0.6` 或 `0.5`
3. 把 `web_jpeg_quality` 从 `70` 降到 `55-65`
4. 确认没有开 `show_local`
5. 如果主链本身也慢，再去优化检测链路，例如 `use_roi`、曝光、模型负载和输入分辨率
