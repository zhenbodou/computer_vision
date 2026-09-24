# 第 26 章 · 实战一：端到端目标检测程序

> **本章导读**：前面 25 章我们一个零件一个零件地打磨——第 5 章教会程序把 JPG 解成像素，第 10 章的 letterbox 把图整理成模型爱吃的 `1×3×640×640` 张量，第 14 章让 ONNX 模型在 Rust 里真正跑了起来，第 16 章把那个坑人的 `[1,84,8400]` 输出用 NMS 收拾成干净的框。可它们一直是散落在各章的片段。本章第一次把这些零件**拼成一个真正能跑的命令行程序**：`detect --input dog.jpg --model yolov8n.onnx`，进去一张照片，出来一张画好框的图，外加控制台打印的检测清单。**本章不引入任何新算法**——解码、预处理、推理、后处理都是老朋友——重点全在**工程组织**：怎么划分模块、怎么用 clap 做命令行、怎么用 `anyhow` 把错误串起来、怎么批量处理一整个文件夹。学完你会有一个可以直接改造去接监控、接业务的检测程序骨架，也为第 27、28 章的完整系统打好地基。

---

## 26.1 目标与效果预览

我们要做的程序，一句话说清：**喂一张图，吐一张画好框的图，同时在控制台列出检测结果**。

```bash
# 基本用法：检测单张图，结果存到 out.jpg
$ detect --input dog.jpg --model yolov8n.onnx --output out.jpg

检测到 2 个目标：
  dog          91.2%  (55,60)-(210,320)
  car          88.4%  (300,120)-(540,290)
已保存到 out.jpg
```

打开 `out.jpg`，狗和车身上各套着一个彩色矩形，框顶上写着"dog 91%""car 88%"。就这么简单——但简单的背后，是前面各章那条完整的数据流在跑：

```text
  输入 input.jpg
  ┌───────────────┐
  │   一张照片      │
  │  (狗 + 汽车)    │
  └───────┬───────┘
          │
   ①解码  ▼  image::open → RgbImage（原图像素，RGB，第 5 章）
  ┌────────────────┐
  │  RgbImage       │  宽×高 任意，比如 1280×720
  └───────┬────────┘
          │
   ②预处理▼  preprocess()   第 10 章：letterbox + 归一化 + 转张量
  ┌────────────────┐
  │ Array4<f32>     │  形状 1×3×640×640，像素值缩到 0~1
  │ + LetterboxInfo │  记下 scale / pad，后处理还原坐标要用
  └───────┬────────┘
          │
   ③推理  ▼  session.run()  第 14 章：ONNX Runtime 前向一次
  ┌────────────────┐
  │ ArrayView3<f32> │  形状 [1, 84, 8400]，70 多万个浮点数
  └───────┬────────┘
          │
   ④后处理▼  postprocess()  第 16 章：解码 + 阈值 + 坐标还原 + NMS
  ┌────────────────┐
  │ Vec<Detection>  │  几个干净的框，已是原图像素坐标
  └───────┬────────┘
          │
   ⑤画框  ▼  draw_detections()   imageproc 画矩形 + ab_glyph 写字
  ┌────────────────┐        ┌──────────────────────────────┐
  │  output.jpg     │        │ 控制台同时打印：               │
  │ 画好框+标签的图  │        │ dog  91%  (55,60)-(210,320)   │
  └────────────────┘        │ car  88%  (300,120)-(540,290) │
                            └──────────────────────────────┘
```

这张图就是本章的"施工图"。五个环节，四个已经在前面章节造好了，我们的工作是**把它们收进模块、用命令行串起来、把边角料（错误处理、批量、画图）补齐**。

---

## 26.2 项目结构：把各章的代码"收进模块"

先建工程：

```bash
$ cargo new detect
$ cd detect
```

一个能长期维护的项目，不会把所有代码塞进 `main.rs`。我们按**数据流的环节**切模块，每个模块对应前面某一章的产物：

```text
detect/
├── Cargo.toml
├── yolov8n.onnx          # 模型文件（自己准备，第 13 章讲过怎么导出）
├── assets/
│   └── DejaVuSans.ttf    # 画标签用的字体（ab_glyph 要一个 ttf）
└── src/
    ├── main.rs           # 命令行解析 + run() 主流程 + 批量处理
    ├── preprocess.rs     # 第 10 章：letterbox / 归一化 / 转张量
    ├── model.rs          # 第 14 章：加载 Session、跑推理
    ├── postprocess.rs    # 第 16 章：解码 + NMS，产出 Vec<Detection>
    └── draw.rs           # 本章新增：画框 + 写标签 + COCO 类别名
```

`preprocess.rs` 和 `postprocess.rs` 基本就是把第 10、16 章写好的代码**原样搬进来**——我们在第 16 章已经把 `BBox`、`Detection`、`postprocess` 写成可编译的模块了，这里直接复用，不再重写。`main.rs` 和 `model.rs`、`draw.rs` 是本章要动手的部分。

完整的 `Cargo.toml`：

```toml
[package]
name = "detect"
version = "0.1.0"
edition = "2021"

[dependencies]
image = "0.25"                                    # 解码/编码 JPG/PNG，第 5 章
ndarray = "0.17"                                  # 与 ort rc.13 使用的 ndarray 主版本一致
ort = { version = "=2.0.0-rc.13", features = ["ndarray"] } # ONNX Runtime 绑定，第 14 章
anyhow = "1"                                      # 好用的错误处理
clap = { version = "4", features = ["derive"] }   # 命令行解析
imageproc = "0.25"                                # 画矩形、画文字
ab_glyph = "0.2"                                  # 字体光栅化（imageproc 写字要它）
rayon = "1"                                       # 批量处理时的数据并行（26.6）
```

几个工程提醒：

- **`ort` 默认会自动下载对应平台的 ONNX Runtime 动态库**，第一次编译会联网拉一份，无需自己装。若目标机器不能联网，要改用系统已装的库（见 `ort` 文档的 `load-dynamic`）。
- **`ndarray` 版本必须和 `ort` 依赖的那份对上**。这是新手常栽的隐坑：你的 `Array4<f32>` 和 `ort` 使用的 ndarray 类型若来自两个不同大版本，就会"看起来一样其实不兼容"。上面的依赖统一锁在 `ndarray 0.17`；升级 `ort` 时要重新检查它的依赖版本。
- `ab_glyph` 需要一个真实的 `.ttf` 字体文件。随便找一个开源字体（如 DejaVuSans）放进 `assets/` 即可；想画中文标签就换一个含中文字形的字体。

---

## 26.3 用 clap 定义命令行

一个像样的 CLI 工具，参数要能自解释、能 `--help`。**clap** 的 derive 模式让我们只写一个结构体，命令行解析、类型转换、帮助文档全自动生成。

```rust,ignore
use clap::Parser;

/// 端到端 YOLOv8 目标检测：输入一张图，输出画好框的图
#[derive(Parser, Debug)]
#[command(name = "detect", version, about = "端到端目标检测程序")]
struct Args {
    /// 输入图片路径，或一个文件夹（文件夹则批量处理，见 26.6）
    #[arg(short, long)]
    input: String,

    /// ONNX 模型路径
    #[arg(short, long, default_value = "yolov8n.onnx")]
    model: String,

    /// 置信度阈值：低于它的框直接丢弃（第 16 章）
    #[arg(long, default_value_t = 0.25)]
    conf: f32,

    /// NMS 的 IoU 阈值：重叠超过它的同类框只留分最高的（第 16 章）
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// 输出图片路径（批量模式下作为输出目录）
    #[arg(short, long, default_value = "out.jpg")]
    output: String,

    /// 只保留这些 COCO 类别号，逗号分隔，如 --classes 0,2,16（人/车/狗）
    /// 不填则保留所有类别
    #[arg(long, value_delimiter = ',')]
    classes: Option<Vec<usize>>,
}
```

几处值得说的：

- `#[arg(short, long)]` 让 `--input` 同时有短选项 `-i`；`default_value_t` 用于有 `Display` 的类型（数字），`default_value` 用于字符串。
- `--conf` / `--iou` 直接映射到第 16 章 `postprocess` 的两个阈值参数，**把调参能力开放到命令行**，不用改代码重编译就能试不同松紧。
- `--classes` 用 `value_delimiter = ','` 接受 `0,2,16` 这种逗号列表，解析成 `Vec<usize>`。这是**类别过滤**：比如做车辆检测就只留 `2`（car），把人、狗之类过滤掉。
- 光是这个结构体，`detect --help` 就会自动打印出每个参数的说明（就是上面那些 `///` 文档注释），这是 derive 模式的甜头。

---

## 26.4 主流程 `run()`：把四步串成一条线

程序的核心是一个 `run()` 函数，返回 `anyhow::Result<()>`。**`anyhow` 让我们用 `?` 一路上抛错误、用 `.with_context()` 给错误加上"当时在干嘛"的上下文**，出错时报的不是干巴巴的 `NotFound`，而是"打开图片失败: dog.jpg: No such file"。

先看 `model.rs`——它把 Session 的加载、预热、推理封装成一个 `Detector`：

```rust,ignore
// src/model.rs
use anyhow::{Context, Result};
use image::RgbImage;
use ndarray::{Array3, Array4, Ix3};
use ort::session::Session;
use ort::value::TensorRef;
use crate::postprocess::Detection; // 第 16 章的统一检测结果类型

/// 持有一个加载好的 ONNX Session。单线程推理流水线只需创建一个并反复复用。
pub struct Detector {
    session: Session,
}

impl Detector {
    /// 加载模型并预热一次
    pub fn load(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;   // 从 .onnx 文件建 Session
        let mut det = Detector { session };
        det.warmup()?;                        // 先空跑一次，见下方说明
        Ok(det)
    }

    /// 预热：拿一张全 0 假图先跑一次，触发内部的懒初始化和显存/内存分配。
    /// 这样正式计时或第一帧不会被"首次运行开销"拖慢——高帧率场景尤其明显。
    fn warmup(&mut self) -> Result<()> {
        let dummy = Array4::<f32>::zeros((1, 3, 640, 640));
        let _ = self.session.run(ort::inputs![
            "images" => TensorRef::from_array_view(&dummy)?
        ])?;
        Ok(())
    }

    /// 跑一次前向，返回原始输出张量 [1, 84, 8400]
    pub fn infer(&mut self, input: Array4<f32>) -> Result<Array3<f32>> {
        // 输入名 "images" 是 Ultralytics 导出的约定；不同导出可能不同（26.7）
        let outputs = self.session.run(ort::inputs![
            "images" => TensorRef::from_array_view(&input)?
        ])?;
        // 取出名为 "output0" 的输出，转成 f32 的三维视图 [1,84,8400]
        let view = outputs["output0"]
            .try_extract_array::<f32>()?          // → ArrayViewD<f32>
            .into_dimensionality::<Ix3>()?;       // → ArrayView3<f32>
        Ok(view.to_owned())                       // 拷成 owned，脱离 outputs 生命周期
    }

    /// 高层便捷方法：一帧原图 RgbImage → 一批**原图像素坐标**的 Detection。
    /// 把 26.4 的三步（preprocess → infer → postprocess）打包成一次调用，
    /// 用一组通用默认阈值（conf 0.25 / NMS IoU 0.45）。
    /// 第 27、28 章的视频系统只需要「喂一帧、拿一批框」，用它最省事；
    /// 要按命令行自定义阈值，仍可像 run_single 那样走 infer + postprocess 两步。
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Detection>> {
        let (tensor, info) = crate::preprocess::preprocess(img, 640);
        let output = self.infer(tensor)?;
        let flat = output.as_slice().context("模型输出不是连续内存")?;
        Ok(crate::postprocess::postprocess(
            flat,
            info.scale,
            (info.pad_x, info.pad_y),
            0.25, // 默认 conf：偏低，让上层按业务再过滤（如第 27 章的 score_thresh）
            0.45, // 默认 NMS IoU
        ))
    }
}
```

> 关于 `ort` 的两处版本差异（`ort` 2.x 还在迭代，以你 `cargo add ort` 拉到的版本文档为准）：一是喂输入，有的版本能直接把 `Array4` 传进 `ort::inputs!`，当前锁定版本可用 `TensorRef::from_array_view(&input)?` 借用连续数组；二是取输出，拿 ndarray 视图的方法名可能是 `try_extract_array` 或 `try_extract_tensor`。逻辑不变——**进去一个 `[1,3,640,640]`，出来一个 `[1,84,8400]`**。

`infer` 里 `to_owned()` 把输出拷成独立的 `Array3`，代价是复制约 84×8400≈70 万个 `f32`（约 2.7 MB），单张图完全无所谓；要跑高帧率视频再考虑把后处理直接放进这里、避免拷贝。

`infer` 之外还多给了一个 `detect(&img)`：它把"预处理 → 推理 → 后处理"三步打成一次调用，直接吐出原图坐标的 `Vec<Detection>`。本章的 `run_single` 因为要把命令行的 `--conf/--iou` 透传下去，仍然显式走三步；但**第 27、28 章的视频系统只关心"喂一帧、拿一批框"**，那里统一调 `detector.detect(frame)?` 就够了——这也是为什么把它做成 `Detector` 的方法。注意它返回 `Result`（推理可能因显存、形状不符等真实原因失败），所以**下游的 `process_frame` 也要返回 `Result`**，主循环里用 `if let Err(e) = ...` 跳过坏帧、别让一帧异常掀翻整条 7×24 的流水线。

再看 `main.rs` 的 `run()`——**这就是那张数据流图的代码化**：

```rust,ignore
// src/main.rs
mod preprocess;
mod model;
mod postprocess;
mod draw;

use anyhow::{Context, Result};
use clap::Parser;
use model::Detector;

fn main() -> Result<()> {
    let args = Args::parse();
    let mut detector = Detector::load(&args.model)
        .with_context(|| format!("加载模型失败: {}", args.model))?;
    run_single(&mut detector, &args.input, &args)
}

/// 处理单张图：解码 → 预处理 → 推理 → 后处理 → 画框保存
fn run_single(detector: &mut Detector, input: &str, args: &Args) -> Result<()> {
    // ① 解码：读图并转成 RGB8（第 5 章）
    let img = image::open(input)
        .with_context(|| format!("打开图片失败: {}", input))?
        .to_rgb8();

    // ② 预处理：letterbox + 归一化 + 转张量，同时拿回 letterbox 参数（第 10 章）
    let (tensor, info) = preprocess::preprocess(&img, 640);

    // ③ 推理：ONNX Runtime 前向一次，拿到 [1,84,8400]（第 14 章）
    let output = detector.infer(tensor)?;

    // ④ 后处理：解码 + 阈值 + 坐标还原 + 按类 NMS，得到原图坐标的干净框（第 16 章）
    let flat = output.as_slice().context("模型输出不是连续内存")?;
    let mut dets = postprocess::postprocess(
        flat,
        info.scale,
        (info.pad_x, info.pad_y),
        args.conf,
        args.iou,
    );

    // 可选：只保留命令行指定的类别
    if let Some(keep) = &args.classes {
        dets.retain(|d| keep.contains(&d.class_id));
    }

    // 控制台打印检测清单
    println!("检测到 {} 个目标：", dets.len());
    for d in &dets {
        let name = draw::COCO_NAMES.get(d.class_id).copied().unwrap_or("?");
        println!("  {:<12} {:>5.1}%  ({:.0},{:.0})-({:.0},{:.0})",
                 name, d.score * 100.0,
                 d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2);
    }

    // ⑤ 画框 + 保存
    let mut canvas = img;                       // 直接在原图副本上画
    let font = draw::load_font()?;
    draw::draw_detections(&mut canvas, &dets, &font);
    canvas.save(&args.output)
        .with_context(|| format!("保存失败: {}", args.output))?;
    println!("已保存到 {}", args.output);
    Ok(())
}
```

注意 `Detector::load` 在 `main` 里**只调用一次**，再把 `&mut Detector` 传给处理函数。加载模型（读文件、建计算图、分配算子）是整个流程里最贵的一步，几百毫秒起步——**每处理一张图就重建一次 Session，是新手最容易犯、也最拖性能的错**（26.6 会再强调）。模型加载一次、反复复用，是所有推理服务的铁律。

这里先用 `output.as_slice()` 取得第 16 章 `postprocess` 所需的连续切片，再从 `LetterboxInfo` 传入 `scale` 和 `(pad_x, pad_y)`。后处理据此把框从 640 letterbox 坐标还原回原图像素——**预处理和后处理靠 `LetterboxInfo` 这根扁担串起来**，一头记着"当初怎么缩放填充的"，一头据此"倒着还原回去"。

---

## 26.5 画框与标签

后处理吐出的 `Vec<Detection>` 每个都带着原图坐标、类别号、置信度。画图要做三件事：**画矩形框、写"类别名 置信度%"、给不同类别配不同颜色**。用 `imageproc` 画形状和文字，用 `ab_glyph` 加载字体。

先是 COCO 80 类的类别名。顺序不能乱——第几个名字对应类别号几，必须和模型训练时一致（Ultralytics 官方顺序）：

```rust,ignore
// src/draw.rs
/// COCO 80 类名（顺序必须与模型一致）。
pub const COCO_NAMES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
];
```

> 验证一下我们熟悉的几个：数组下标 0 是 `person`、2 是 `car`、15 是 `cat`、16 是 `dog`——和全书约定一致。少一个、错一位，画出来就会"狗标成猫"，而且不报错，极难发现。

配色：同一类别永远同一个颜色（这样一眼能分清"哪些是人、哪些是车"）。不必真去查表，用类别号**散列**出一个稳定颜色即可：

```rust,ignore
use image::Rgb;

/// 由类别号散列出一个稳定颜色：同类同色、异类多半异色
fn color_for_class(class_id: usize) -> Rgb<u8> {
    let r = (37 * (class_id + 1) % 200 + 40) as u8;
    let g = (91 * (class_id + 1) % 200 + 40) as u8;
    let b = (167 * (class_id + 1) % 200 + 40) as u8;
    Rgb([r, g, b])  // +40 是为了避开太暗的颜色，画在深色背景上也看得见
}
```

然后是画框和标签的主函数：

```rust,ignore
use ab_glyph::{FontArc, PxScale};
use anyhow::{anyhow, Result};
use image::RgbImage;
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use crate::postprocess::Detection;

/// 加载字体（ab_glyph 需要一个真实存在的 ttf 文件）。
pub fn load_font() -> Result<FontArc> {
    let bytes = std::fs::read("assets/DejaVuSans.ttf")?;
    FontArc::try_from_vec(bytes).map_err(|_| anyhow!("无效的字体文件: assets/DejaVuSans.ttf"))
}

/// 在图上画出所有检测框 + 标签
pub fn draw_detections(img: &mut RgbImage, dets: &[Detection], font: &FontArc) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    for d in dets {
        let color = color_for_class(d.class_id);

        // 框坐标转成整型，并 clamp 到图像范围内（贴边框可能有负坐标，第 16 章练习）
        let x1 = (d.bbox.x1 as i32).clamp(0, iw - 1);
        let y1 = (d.bbox.y1 as i32).clamp(0, ih - 1);
        let w = (d.bbox.width()  as i32).clamp(1, iw - x1) as u32;
        let h = (d.bbox.height() as i32).clamp(1, ih - y1) as u32;

        // 画框：画三层同心矩形当"加粗"，1px 的框在大图上太细看不清
        for t in 0..3 {
            let r = Rect::at(x1 - t, y1 - t).of_size(w + 2 * t as u32, h + 2 * t as u32);
            draw_hollow_rect_mut(img, r, color);
        }

        // 标签文字："dog 91%"
        let name = COCO_NAMES.get(d.class_id).copied().unwrap_or("?");
        let label = format!("{} {:.0}%", name, d.score * 100.0);
        let scale = PxScale::from(18.0);

        // 先在框顶画一个实心色条当底衬，白字压上去才看得清
        let bar_h = 20;
        let bar_y = (y1 - bar_h).max(0);
        let bar_w = (label.len() as u32 * 9).min(w.max(60));
        draw_filled_rect_mut(img, Rect::at(x1, bar_y).of_size(bar_w, bar_h as u32), color);
        draw_text_mut(img, Rgb([255, 255, 255]), x1 + 2, bar_y + 2, scale, font, &label);
    }
}
```

几个工程细节：

- **1 像素的框在高分辨率图上细得几乎看不见**，所以用三层同心矩形凑出 3px 粗细，这是最省事的"加粗"办法。
- **白字直接写在图上常常糊成一团**（背景可能也是浅色），所以先铺一条与框同色的实心底衬，再写白字，对比度立刻拉满。
- 示例从 `assets/DejaVuSans.ttf` 运行时加载字体，因此开始前要把字体文件放到该路径。若希望部署时只带一个二进制，可在项目确实包含该文件后改用 `include_bytes!`。

---

## 26.6 批量处理一整个文件夹

单张图跑通后，真实需求往往是"把这个文件夹里几百张图全检测一遍"。这里正好演示两件工程事：**遍历目录**和**用 `rayon` 并行**。

关键认知：本书锁定的 `ort 2.0.0-rc.13` 中，`Session::run` 需要 `&mut Session`，不能把同一个 `&Session` 交给多个线程同时 `run`。批处理若要并发推理，可以让**每个 Rayon 工作线程各自创建并长期复用一个 `Detector`**；代价是每个线程各占一份模型内存。显存或内存紧张时，更稳妥的方案是保留一个推理线程，用 batch 提高吞吐。

```rust,ignore
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// 处理一个目录：每个 Rayon 工作线程加载并复用一个模型
fn run_dir(dir: &Path, args: &Args) -> Result<()> {
    // 收集目录下所有 jpg/png
    let paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(
            p.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase).as_deref(),
            Some("jpg" | "jpeg" | "png")
        ))
        .collect();
    println!("共 {} 张图待处理", paths.len());
    std::fs::create_dir_all(&args.output)?;

    // for_each_init 的初始化器每个工作线程只执行一次，不会每张图重建 Session。
    paths.par_iter().for_each_init(
        || Detector::load(&args.model),
        |detector, path| match detector {
            Ok(detector) => match process_one(detector, path, args) {
                Ok(n) => println!("{}: {} 个目标", path.display(), n),
                Err(e) => eprintln!("{} 处理失败: {:#}", path.display(), e),
            },
            Err(e) => eprintln!("工作线程加载模型失败: {:#}", e),
        }
    );
    Ok(())
}

/// 处理单张并存盘，输出文件名在原名前加 "det_"
fn process_one(detector: &mut Detector, path: &Path, args: &Args) -> Result<usize> {
    let img = image::open(path)?.to_rgb8();
    let (tensor, info) = preprocess::preprocess(&img, 640);
    let output = detector.infer(tensor)?;
    let flat = output.as_slice().context("模型输出不是连续内存")?;
    let mut dets = postprocess::postprocess(
        flat,
        info.scale,
        (info.pad_x, info.pad_y),
        args.conf,
        args.iou,
    );
    if let Some(keep) = &args.classes {
        dets.retain(|d| keep.contains(&d.class_id));
    }

    let mut canvas = img;
    let font = draw::load_font()?;
    draw::draw_detections(&mut canvas, &dets, &font);

    // 输出到 args.output 目录，文件名前缀 det_
    let name = path.file_name().unwrap().to_string_lossy();
    let out = Path::new(&args.output).join(format!("det_{name}"));
    canvas.save(&out)?;
    Ok(dets.len())
}
```

`main` 里根据 `--input` 是文件还是目录分流即可：

```rust,ignore
fn main() -> Result<()> {
    let args = Args::parse();
    let input = Path::new(&args.input);
    if input.is_dir() {
        run_dir(input, &args)                                   // 批量
    } else {
        let mut detector = Detector::load(&args.model)?;
        run_single(&mut detector, &args.input, &args)           // 单张
    }
}
```

> **每张图重建 Session 是大坑。** 这是第 14、29 章反复强调的口径——"重活只做一次"。有人图省事，把 `Session::builder()...commit_from_file()` 写进图片循环体里，结果大量时间耗在反复加载模型上。上面的并行版是每个工作线程创建一次、随后反复复用；若一份模型已经很占内存，就只保留一个推理线程或改用 batch。

还有一层要留个心眼：**`rayon` 的数据并行和 ONNX Runtime 内部的算子并行会叠加**。ort 默认会开多个线程加速单次 `run`（intra-op 线程），你再用 `rayon` 同时跑好几张图，两边的线程数一乘可能远超 CPU 核数，线程互相抢占反而变慢。真要压满吞吐时，通常会把 ort 的 intra-op 线程数调小（甚至设 1），把并行度交给 `rayon` 那一层——这类调优是第 29 章的主题，这里先知道有这么回事。

---

## 26.7 常见错误与排查

端到端程序"跑不通"或"跑通了但结果不对"的地方就那么几处。按下表对号入座：

| 症状 | 可能原因 | 解法 |
| --- | --- | --- |
| 启动就报 `No such file` / `commit_from_file` 失败 | 模型路径不对，或相对路径相对的是错误的工作目录 | 用绝对路径，或确认在项目根目录运行；`--model` 传对 |
| `run` 报输入名无效 / 找不到 `"images"` | 输入张量名不是 `images`（有的导出叫 `input`、`input0`） | 用第 12 章方法打印 Session 的输入名，把代码里的名字改成实际值 |
| 取 `output0` 失败 / 提取张量报维度错 | 输出名不叫 `output0`，或形状其实是 `[1,8400,84]` | 打印输出名和 shape（第 12 章）；布局是 `[1,8400,84]` 时后处理取值方式要变（第 16 章） |
| 一个框都没有 | `--conf` 太高全被筛掉；或模型是"裸"导出、分数还是 logit 没过 sigmoid | 先把 `--conf` 降到 0.05 看有没有候选；确认导出是否已激活（第 16 章 16.4） |
| 框位置全错、缩在左上角 | 自己瞎解码时把 `cxcywh` 当 `xyxy`，或内存布局取反 | 别手写解码，交给第 16 章验证过的 `postprocess` |
| 框整体偏移、套不准 | 后处理没拿到正确的 `scale`/`pad` | 确认传给 `postprocess` 的参数来自同一张图的 `info.scale/info.pad_x/info.pad_y`（26.4） |
| 画出来颜色发蓝、发怪 | RGB/BGR 弄反了 | 全程用 RGB（`image` 默认就是 RGB）；只有 OpenCV 那套才是 BGR（第 4 章） |
| 框比物体偏了一截 | 原图带 EXIF 旋转信息，`image` 没自动转正 | 读图后按 EXIF orientation 先把图转正再检测 |
| 编译报一堆 `ndarray` 类型不匹配 | 你的 `ndarray` 版本和 `ort` 依赖的不是同一个大版本 | 按 26.2 将两者对齐；本书锁定为 `ndarray 0.17` + `ort 2.0.0-rc.13` |

排查的黄金顺序：**先确认"进得去"（输入名、shape 对不对），再确认"出得来"（输出名、布局），最后才怀疑阈值和画图**。绝大多数"框乱飞"的问题，根子都在输入输出的名字或布局，而不在算法。

---

## 26.8 小结

- 本章没有新算法，做的是**工程拼装**：把前 25 章的解码、letterbox 预处理、ONNX 推理、NMS 后处理，第一次拼成一个能跑的命令行程序 `detect`。
- **按数据流切模块**：`preprocess.rs`（第 10 章）/ `model.rs`（第 14 章）/ `postprocess.rs`（第 16 章）/ `draw.rs`（新增）/ `main.rs`（串流程），职责清晰、各章代码原样收纳。
- **clap derive** 把参数变成一个结构体，`--input/--model/--conf/--iou/--output/--classes` 一应俱全，还白送 `--help`；调阈值不用改代码。
- **`run()` 用 `anyhow::Result` 串起五步**，`?` 上抛、`.with_context()` 加上下文；模型 `load` 一次、`warmup` 一次，是性能的第一条铁律。
- **画框靠 `imageproc` + `ab_glyph`**：三层矩形加粗、色条底衬白字、按类别散列配色，同类同色。
- **批量用 `rayon` 并行**，每个工作线程各持有并复用一个 `Session`；不要每张图重建，也不要在 `ort 2.0.0-rc.13` 下共享同一个 Session 并发 `run`。线程数还会与 ORT 内部线程叠加，需要实测并限制。

下一章（第 27 章）会把这个"单张图检测器"升级成**周界入侵报警系统**：接入视频流逐帧检测、加空间过滤器（ROI/绊线，第 22、23 章）、把"检测到人"变成"该不该报警"。第 28 章再叠上跟踪与人脸抓拍。本章这个骨架，就是它们共同的地基。

## 26.9 练习

1. **加 `--save-crops`**：新增一个开关参数，打开后把每个检测框从原图里**抠出来**（`image::imageops::crop` 或按坐标取子图），按 `类别名_序号.jpg` 存进一个 `crops/` 目录。想想坐标要不要先 clamp 到图像范围内，否则抠图会 panic。
2. **按类别设不同置信度阈值**：现在 `--conf` 对所有类一刀切。改成支持 `--conf-person 0.5 --conf-car 0.3` 这类按类别的阈值（可以先做成一个 `HashMap<usize, f32>`，后处理后再按类过滤），体会"人要抓得准、车可以宽松"这种真实业务诉求。
3. **让 `--classes` 支持类别名**：现在只能传数字 `0,2,16`，改成也能传 `person,car,dog`——在 `COCO_NAMES` 里反查出类别号。顺手给个错误提示：传了不存在的类别名时友好报错，而不是默默忽略。
4. **改成读一段视频逐帧检测**（预告第 27、28 章）：用 `opencv` 或 `ffmpeg` 逐帧解码一个 mp4，对每一帧跑本章的检测流程，把画好框的帧再编码回一个视频。注意此时**模型更要只加载一次**、逐帧复用——这正是从"处理图片"迈向"处理视频流"的第一步，也是下一章报警系统的前提。
