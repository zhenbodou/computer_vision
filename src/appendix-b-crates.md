# 附录 B · 常用 crate 速查表

> 本附录把全书用到的 crate 按用途归类，标出"在哪一章会用到"，并给出一份可以直接复制进项目的参考 `Cargo.toml`。需要选包、查版本或搭新工程时，翻这一页即可。

## B.1 按用途分类

### 图像读写与处理

| crate | 用途 | 本书章节 |
|-------|------|----------|
| `image` | 图片读写、缩放、裁剪、像素操作，Rust 图像处理的基础 crate | 第 2、3、5 章 |
| `imageproc` | 基于 image 的算法库：绘图、滤波、边缘检测、轮廓、画框 | 第 3、7、8、16、26 章 |
| `fast_image_resize` | 高性能缩放（SIMD 加速），比 image 自带 resize 快很多 | 第 9、10、29 章 |
| `ab_glyph` | 字体加载与文字绘制（在画框上写类别名、置信度） | 第 26 章 |

### 数组与张量

| crate | 用途 | 本书章节 |
|-------|------|----------|
| `ndarray` | N 维数组（张量），图像和模型数据的通用载体，Rust 版 NumPy | 第 3、10、11 章 |
| `ndarray-stats` | ndarray 的统计扩展（均值、方差、分位数） | 可选 |
| `nalgebra` | 线性代数（矩阵运算），卡尔曼滤波的状态预测用它 | 第 21 章 |

### 模型推理

| crate | 用途 | 本书章节 |
|-------|------|----------|
| `tract-onnx` | 纯 Rust 的 ONNX 推理引擎，无需外部依赖，最易上手 | 第 14 章 |
| `ort` | ONNX Runtime 的 Rust 绑定，性能更好、平台支持最全 | 第 14 章（GPU/NPU EP 见第 29 章） |
| `candle` | HuggingFace 出品的极简 ML 框架，适合 transformer 类模型 | 第 13、14 章 |
| `tflite` / `tract-tflite` | TensorFlow Lite 推理（移动端、边缘设备常见格式） | 第 13、14 章 |
| `openvino` | Intel OpenVINO 的 Rust 绑定（CPU/iGPU/VPU 加速） | 第 13、14 章 |

### 视频与采集

| crate | 用途 | 本书章节 |
|-------|------|----------|
| `opencv` | OpenCV 绑定：摄像头、视频文件、大量传统视觉算法，编译重、可选 | 第 6 章 |
| `ffmpeg-next` / `rsmpeg` | FFmpeg 绑定：解码、转码、RTSP 流媒体 | 第 6 章 |
| `gstreamer` | GStreamer 绑定，流媒体管线框架 | 第 6 章 |
| `nokhwa` | 跨平台摄像头采集，比 opencv 轻量 | 第 6 章 |

### 工程基础

| crate | 用途 | 本书章节 |
|-------|------|----------|
| `anyhow` | 错误处理：任何错误都能 `?` 传播，快速开发首选 | 全书 |
| `clap` | 命令行参数解析（derive 模式非常好用） | 第 26 章 |
| `rayon` | 数据并行：一行 `.par_iter()` 让循环多线程 | 第 3、29 章 |
| `crossbeam` | 并发原语：channel、scoped 线程，搭多线程流水线用 | 第 29 章 |
| `tracing` | 结构化日志，比 println 更适合正式项目 | 第 27、29 章 |
| `serde` / `serde_json` | 序列化/反序列化，配置文件和报警 API 必备 | 第 27、29 章 |

## B.2 参考 Cargo.toml

```toml
[package]
name = "my-vision-app"
version = "0.1.0"
edition = "2021"

[dependencies]
# --- 图像 ---
image = "0.25"                    # 图像读写与基础操作
imageproc = "0.25"                # 绘图、滤波、边缘、轮廓、画框
ab_glyph = "0.2"                  # 字体，给框写文字
fast_image_resize = "5"           # 快速缩放（letterbox 提速）

# --- 数组 / 张量 / 线代 ---
ndarray = { version = "0.16", features = ["rayon"] }
nalgebra = "0.33"                 # 卡尔曼滤波（第 21 章）

# --- 推理（二选一或都用）---
tract-onnx = "0.21"               # 纯 Rust，易上手
# ort = "2"                       # 性能更好（需 onnxruntime 动态库，支持 GPU/NPU）

# --- 视频（可选，编译较重）---
# opencv = "0.93"

# --- 工程 ---
anyhow = "1"
clap = { version = "4", features = ["derive"] }
rayon = "1"
crossbeam = "0.8"                 # 多线程流水线（第 29 章）
tracing = "0.1"                   # 结构化日志
serde = { version = "1", features = ["derive"] }
serde_json = "1"                  # 配置 / 报警 JSON
```

## B.3 版本说明

- 本书所有 crate 版本号都经过验证可用。Rust 生态更新快，如果读到本书时版本已过期，直接 `cargo add 包名`（不带版本号）让 cargo 自动选最新版，再对照官方迁移指南调整 API 差异。
- `opencv` 编译需要系统装有 OpenCV 库，新手建议先用 `image` + `ffmpeg` CLI（第 6 章的方案）处理视频，等确实有需求再装绑定。
- `tract-onnx` 支持的 ONNX opset 版本有限（本书用 opset 12 的模型验证），遇到不支持的算子会报 `unimplemented` 错误——第 12 章讲了用 Netron 查看模型算子与 shape 的方法，可先确认模型是否被支持。
- `ort`（ONNX Runtime 绑定）性能最强，但要链接 onnxruntime 动态库；想用 GPU/NPU 加速请配合对应的 Execution Provider（第 29 章讲部署与硬件加速时展开）。

## B.4 常用命令速查

```bash
cargo add image ndarray tract-onnx anyhow   # 添加依赖
cargo build --release                        # 发布模式编译（推理必须用！）
cargo run --release -- --input test.jpg      # 带参数运行
cargo doc --open                             # 生成本项目依赖的文档
```

## B.5 学习资源

- **The Rust Programming Language**（官方免费书）：rust-lang.org/book
- **Rust by Example**：边改边学
- **ndarray 文档**：docs.rs/ndarray，看 "ndarray for NumPy users" 一页就够入门
- **ONNX 模型库**：github.com/onnx/models，大量免费的开源模型，练手必备
- **Netron**（模型结构查看器）：netron.app，拖入 `.onnx` 即可可视化输入输出与算子
