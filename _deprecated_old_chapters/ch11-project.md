# 第 11 章 综合项目：一个完整的视觉应用

> **本章导读**：
> 这是全书的收官项目。前十章我们一块一块地学习了每个环节——这一章把它们全部组装起来，做一个真正完整的、可以跑在终端里的目标检测命令行工具。你会亲眼看到：之前学的每一块知识都不是孤立的，它们像流水线的工位一样首尾相接。

**学习目标**：

- 把前面各章的知识整合成一个能运行的完整程序
- 设计一个清晰的项目结构和模块划分
- 学会给程序的每个阶段计时，做性能分析
- 掌握把"玩具代码"变成"工程代码"的方法

## 11.1 我们要做什么

最终交付物：一个 Rust 命令行程序 `detect`，用法是：

```bash
# 对单张图片做目标检测
cargo run --release -- --model yolov8n.onnx --input test.jpg

# 输出：
# [INFO] 模型输入: 1×3×640×640
# [INFO] 检测到 5 个目标 (耗时 23ms)
# [INFO]  狗 (0.91) -> [123, 45, 456, 380]
# ...
# 生成 drawn.jpg（画了检测框的图片）
```

它内部完整地走完了我们在第 1 章看到的流水线：

```text
test.jpg ──解码──▶ RgbImage ──预处理──▶ Tensor [1,3,640,640] ─┐
                                                              ▼
drawn.jpg ◀──绘制── Vec<Bbox> ◀──后处理── 输出 [1,84,8400] ◀──推理
```

这个程序会用到：

- 第 3 章：image / ndarray 处理像素和张量
- 第 4 章：把 jpg 解码成图像
- 第 5 章：letterbox + 归一化 + CHW
- 第 8 章：导出的 ONNX 模型、查输入输出信息
- 第 9 章：tract 加载模型并推理
- 第 10 章：置信度过滤、坐标还原、NMS

## 11.2 项目结构

一个正式的 Rust 项目，应该把不同职责的代码放在不同模块：

```text
computer-vision-app/
├── Cargo.toml
├── test.jpg                    # 测试图片（用你自己的照片）
├── yolov8n.onnx                # 转换好的模型
├── coco.names                  # 80 个类别的名字
└── src/
    ├── main.rs                 # 程序入口：解析参数、调度流程
    ├── preprocess.rs           # 预处理（第 5 章的代码）
    ├── infer.rs                # 模型加载和推理（第 9 章）
    └── postprocess.rs          # 后处理（第 10 章的代码）
```

`Cargo.toml`：

```toml
[package]
name = "computer-vision-app"
version = "0.1.0"
edition = "2021"

[dependencies]
image = "0.25"
ndarray = { version = "0.16", features = ["rayon"] }
tract-onnx = "0.21"
imageproc = "0.25"
ab_glyph = "0.2"
anyhow = "1"
clap = { version = "4", features = ["derive"] }
```

几个新面孔：`clap` 是 Rust 事实标准的命令行参数解析库（derive 模式几行代码搞定）；`ab_glyph` 提供字体，用来在图上画类别文字。

`clap` 的用法（在 main.rs 顶部）：

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "detect", about = "YOLOv8 目标检测")]
struct Args {
    /// 模型文件路径
    #[arg(long, default_value = "yolov8n.onnx")]
    model: String,

    /// 输入图片
    #[arg(short, long)]
    input: String,

    /// 置信度阈值
    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    /// NMS 的 IoU 阈值
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
}
```

之后 `let args = Args::parse();` 一行，命令行参数就全在 `args` 结构体里了，还带 `--help`。

## 11.3 把各章代码放进模块

`preprocess.rs`——把第 5 章的 `letterbox`、`preprocess` 函数原样拷进来，`pub` 导出。

`postprocess.rs`——把第 10 章的 `Detection`、`filter_candidates`、`restore_and_convert`、`iou`、`nms`、`Bbox`、`postprocess` 原样拷进来。

`infer.rs`——封装推理为一个函数：

```rust
use anyhow::{Context, Result};
use ndarray::Array4;
use tract_onnx::prelude::*;

/// 加载模型（只做一次），返回可复用的 runnable 模型
pub fn load_model(path: &str) -> Result<TypedRunnableModel<TypedModel>> {
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

/// 对一张预处理好的图执行推理，返回原始输出张量
pub fn infer(
    model: &TypedRunnableModel<TypedModel>,
    input: &Array4<f32>,
) -> Result<tract_ndarray::ArrayD<f32>> {
    let input_tensor: Tensor = input.clone().into();
    let outputs = model.run(tvec!(input_tensor.into()))?;
    Ok(outputs[0].to_array_view::<f32>()?.to_owned())
}
```

注意我们把"加载模型"和"执行推理"分成两个函数——模型加载慢（几百毫秒），推理快（几十毫秒）。加载一次，复用多次（第 9 章的 Session 复用原则）。

## 11.4 主流程

`main.rs` 把所有环节串起来。这是全书最长的一段代码，但每一行你都学过：

```rust
mod infer;
mod postprocess;
mod preprocess;

use anyhow::{Context, Result};
use clap::Parser;
use imageproc::rect::Rect;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use ab_glyph::{FontRef, PxScale};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "detect", about = "YOLOv8 目标检测")]
struct Args {
    #[arg(long, default_value = "yolov8n.onnx")]
    model: String,
    #[arg(short, long)]
    input: String,
    #[arg(long, default_value_t = 0.25)]
    conf: f32,
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // ===== 第 0 步：加载类别名 =====
    let names = std::fs::read_to_string("coco.names")
        .context("读取 coco.names 失败")?;
    let names: Vec<&str> = names.lines().collect();

    // ===== 第 1 步：加载模型（耗时较长，只做一次）=====
    let t0 = Instant::now();
    let model = infer::load_model(&args.model)?;
    println!("[INFO] 模型加载耗时 {:?}\n", t0.elapsed());

    // ===== 第 2 步：读取并预处理图片 =====
    let t1 = Instant::now();
    let (input, r, (pad_x, pad_y)) = preprocess::preprocess(&args.input, 640)?;
    let prep_ms = t1.elapsed();

    // ===== 第 3 步：推理 =====
    // 热身一次（见第 9 章），不计时
    let _ = infer::infer(&model, &input)?;

    let t2 = Instant::now();
    let output = infer::infer(&model, &input)?;
    let infer_ms = t2.elapsed();

    // ===== 第 4 步：后处理 =====
    let t3 = Instant::now();
    let img = image::open(&args.input)?.to_rgb8();
    let (img_w, img_h) = img.dimensions();
    let output_view = output.view().into_dimensionality::<ndarray::Ix3>()?;
    let dets = postprocess::postprocess(
        &output_view, r, (pad_x as f32, pad_y as f32),
        img_w, img_h, args.conf, args.iou,
    )?;
    let post_ms = t3.elapsed();

    // ===== 第 5 步：打印结果 =====
    println!("[INFO] 预处理 {:?} | 推理 {:?} | 后处理 {:?}",
             prep_ms, infer_ms, post_ms);
    println!("[INFO] 检测到 {} 个目标：", dets.len());
    for d in &dets {
        let name = names.get(d.class_id).unwrap_or(&"?");
        println!("[INFO]  {} ({:.2}) -> [{:.0}, {:.0}, {:.0}, {:.0}]",
                 name, d.score, d.x1, d.y1, d.x2, d.y2);
    }

    // ===== 第 6 步：画框保存 =====
    let mut drawn = img;
    // 用系统自带的字体画文字；路径按你的系统调整
    let font_data = std::fs::read("/System/Library/Fonts/Helvetica.ttc")?;
    let font = FontRef::try_from_slice(&font_data[..])
        .context("字体解析失败")?;

    for d in &dets {
        let rect = Rect::at(d.x1 as i32, d.y1 as i32)
            .of_size((d.x2 - d.x1) as u32, (d.y2 - d.y1) as u32);
        draw_hollow_rect_mut(&mut drawn, rect, image::Rgb([255, 0, 0]));
        draw_text_mut(&mut drawn, image::Rgb([255, 255, 0]),
                      d.x1 as i32, (d.y1 - 14.0).max(0.0) as i32,
                      PxScale::from(14.0), &font,
                      &format!("{} {:.2}", names.get(d.class_id).unwrap_or(&"?"), d.score));
    }
    drawn.save("drawn.jpg")?;
    println!("\n[INFO] 结果已保存到 drawn.jpg");

    Ok(())
}
```

整个 main 函数清晰地带你走完六步，和我们在第 1 章画的流水线图一一对应。

## 11.5 准备模型和素材

```bash
# 1. 模型（第 8 章转换好的 yolov8n.onnx；如果没有，可直接下载现成的）
curl -L -o yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx

# 2. 类别名文件（COCO 80 类，一行一个）
curl -L -o coco.names \
  https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.names

# 3. 测试图片：放一张日常照片命名为 test.jpg
```

## 11.6 运行和调优

```bash
cargo run --release -- --input test.jpg
```

典型输出：

```text
[INFO] 模型加载耗时 312ms

[INFO] 预处理 4.2ms | 推理 21.5ms | 后处理 3.1ms
[INFO] 检测到 5 个目标：
[INFO]  person (0.93) -> [150, 32, 420, 640]
[INFO]  dog (0.91) -> [123, 245, 456, 580]
[INFO]  ...
[INFO] 结果已保存到 drawn.jpg
```

打开 drawn.jpg，你应该看到红框标出了图片里的人、动物、车等目标，框上方还有黄色的类别名和置信度。

**调优思路**（工程上很重要）：

- **漏检多**（该检出的没检出）→ 降低 `--conf`，如 0.15；或用更大的模型（yolov8s/yolov8m）
- **误检多**（不该出的框很多）→ 提高 `--conf`，如 0.4
- **同一个物体有很多重叠框** → 降低 `--iou` 阈值（NMS 更激进地删框）；反过来如果不同的物体被误删，提高它
- **推理太慢** → 看时间分布：如果推理占大头，考虑 INT8 量化或换更快的推理引擎（第 8 章）；如果预处理慢，用 `fast_image_resize`（第 3 章）

## 11.7 从这里出发：进阶方向

这个项目是个起点。真实世界里的视觉系统还有这些进阶方向，都建立在本书基础上：

- **视频流检测**：把第 4 章的 opencv 视频捕获接进来，逐帧调用本项目（注意复用已加载的模型，别每帧加载）
- **batch 推理**：一次传 N 张图提高吞吐（第 8 章的动态 batch）
- **部署到边缘设备**：模型转成 RKNN/NCNN 格式，跑在树莓派或 RK3588 开发板的 NPU 上（第 8 章）
- **Web 服务**：用 axum/actix-web 包一个 HTTP 接口，接收图片返回检测结果，变成真正的在线服务
- **更多任务**：分割、关键点、OCR——预处理和后处理变了，但整本书的框架（解码→预处理→推理→后处理）完全通用

工程上的建议：

- **先让代码正确，再让它快**：一开始用最好懂的写法，跑通了再优化
- **到处计时**：`Instant::now()` 是性能分析的起点，别凭感觉猜瓶颈
- **小步提交**：每加一个功能（如"支持视频输入"）单独测试
- **读错误信息**：Rust 的编译错误信息是业界最好的老师，耐心读完它让你少踩十倍坑

## 小结

- 一个完整的视觉应用 = 解码 + 预处理 + 推理 + 后处理 + 展示，每个环节对应本书一章
- 合理的模块划分让每个模块专注一件事，main 只做调度
- 模型只加载一次、推理前热身、每个阶段单独计时，是三个基本工程习惯
- 调参（conf/iou）要根据实际效果迭代，有明确的"多了降、少了升"的方向

至此，你已经从零开始，走完了一个真实机器视觉系统的完整链路。这不是终点——理解了这条流水线，任何新的视觉任务对你来说都是"换模型 + 换后处理"的组合题。

## 练习

1. 给程序加一个 `--output` 参数，让用户自定义输出图片的路径和格式。
2. 给每个检测框按类别画不同颜色（准备一个颜色表）。
3. 给程序加上每阶段的累计计时，跑 100 次取平均，输出性能报告。
4. （进阶）把本项目改成读视频文件，逐帧检测并把结果写成新的视频文件。
5. （进阶）用 axum 把这个检测器包成 HTTP 服务：POST 一张图片，返回 JSON 格式的检测结果列表。
