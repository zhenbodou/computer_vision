# 第 5 章 · 模型预处理：让图片合模型的"胃口"

> **本章导读**：模型是个"挑食的惯宝宝"：它训练时吃什么尺寸、什么颜色顺序、什么数值范围，推理时就必须一模一样地喂，否则轻则精度暴跌、重则当场报错。本章是全书工程含量最高的章节之一，我们把预处理拆成五道工序逐一讲透：缩放、letterbox、色彩转换、归一化、布局转换，最后封装成可以直接进生产的预处理函数——它会原样出现在第 11 章的实战项目里。

**学习目标**：
- 说出预处理流水线五道工序各自解决什么问题
- 理解四种插值算法的取舍，会用 fast_image_resize 做高性能缩放
- 完整实现 letterbox（等比缩放 + 灰边填充），并知道为什么要记录缩放参数
- 熟练做 RGB↔BGR、归一化、HWC↔CHW、加 batch 维这些转换
- 背下"预处理常见坑清单"，能独立排查大部分 shape/颜色异常

## 5.1 模型为什么"挑食"

回忆第 7 章会展开的一个事实：模型的"经验"（权重）是在训练时，从**特定格式的数据**里学出来的。训练时所有图片都被处理成了统一的规格——比如 YOLO 的 640×640、RGB、[0,1] 数值范围——模型学到的"怎么看图"完全建立在这个规格上。

类比：你养了一只从小吃切好小块、拌好调料猫粮的猫，现在拿一整条生鱼给它，它不认识这是食物。**预处理就是那只"切菜拌料的手"**：无论哪来的图（相机拍的、视频截的、网上下的），都先加工成模型熟悉的"标准餐"。

挑食的三个维度：

| 维度 | 模型要求示例 | 原始图片 |
| --- | --- | --- |
| 尺寸 | 640×640 | 1920×1080 |
| 颜色 | RGB 顺序、f32 数值 | 可能 BGR、u8 数值 |
| 布局 | NCHW `[1,3,640,640]` | HWC `[1080,1920,3]` |

**预处理做错=一切白干**：后面推理、后处理再正确，喂错了就是垃圾进垃圾出。而且这类 bug 经常不报错——模型照样输出一坨数字，只是结果全错，排查起来非常痛苦。所以这一章的每个细节都值得较真。

## 5.2 尺寸变换：四种插值的取舍

把 1920×1080 缩到 640×360，意味着 200 万个像素要变成 23 万个——**新像素的颜色必须从旧像素"推算"出来**。怎么推算就是插值算法的选择：

| 算法 | 做法 | 质量 | 速度 | 适用 |
| --- | --- | --- | --- | --- |
| 最近邻 Nearest | 直接抄最近那个像素 | 有锯齿 | 最快 | 像素画、mask、要求快 |
| 双线性 Triangle | 周围 4 个像素加权平均 | 均衡 | 快 | **默认首选** |
| 双三次 CatmullRom | 周围 16 个像素拟合曲线 | 平滑 | 中 | 画质优先 |
| Lanczos3 | 更大窗口 + 窗函数 | 最锐利 | 慢 | 细节优先 |

直觉对比（放大看一条斜线被放大 3 倍的效果）：

```text
最近邻：   _--__        ← 明显阶梯锯齿
双线性：   _----_       ← 平滑但略糊
Lanczos：  _----_       ← 边缘最清晰
```

image crate 自带缩放：

```rust
use image::imageops::{resize, FilterType};

let small = resize(&rgb, 640, 360, FilterType::Triangle);
```

但工业界更常用 **`fast_image_resize`**：它用 SIMD（CPU 的向量指令）批量算插值，比纯 Rust 实现快几倍，是性能敏感场景（比如每帧都要缩放的视频检测）的事实标准：

```bash
cargo add fast_image_resize image ndarray anyhow
```

```rust
use fast_image_resize::{PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::RgbImage;

fn resize_fast(src: &RgbImage, dst_w: u32, dst_h: u32) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    let mut dst = RgbImage::new(dst_w, dst_h);
    let mut resizer = Resizer::new();
    resizer.resize(
        &fast_image_resize::Image::from_vec_u8(
            src_w, src_h, src.clone().into_raw(), PixelType::U8x3,
        ).unwrap(),
        &mut fast_image_resize::Image::from_vec_u8(
            dst_w, dst_h, std::mem::take(dst.as_mut()), PixelType::U8x3,
        ).unwrap(),
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3)),
    ).unwrap();
    dst
}
```

这个 API 略繁琐（它直接操作字节缓冲区，所以中间要 `from_vec_u8` 包一层），换来的是对任意布局图像的高性能处理。如果觉得重，**小项目直接用 image 的 resize 也完全没问题**——先正确，再快。

## 5.3 letterbox：本章的明星

### 为什么不能直接拉伸

上一节的 `resize_exact(640, 640)` 有个致命问题：**宽高比变了**。一张 1920×1080（16:9）的图被压成 640×640（1:1），人和车都会"长高"：

```text
原图 16:9          直接拉伸到 1:1        letterbox 到 1:1
┌──────────┐      ┌──────┐              ┌──────┐
│          │      │ 车   │              │▒▒▒▒▒▒│  ← 灰边 pad
│  车 人   │  →   │ 车 人 │     →       │ 车 人│
│          │      │ 车   │              │▒▒▒▒▒▒│
└──────────┘      └──────┘              └──────┘
 不变形              变形！               不变形，尺寸合规
```

拉伸对人是肉眼可见的怪，对模型是**分布偏移**——训练时模型见的是正常比例的目标，推理时目标都"变形"了，精度必掉。

### letterbox 的原理

**letterbox = 等比缩放 + 不够的地方填灰边**。分三步算：

```text
原图 W×H，目标 640×640：
1. 缩放比  r = min(640/W, 640/H)          （取长边适配，短边必然不足）
2. 新尺寸  new_w = round(W×r), new_h = round(H×r)
   例：1920×1080 → r = 640/1920 = 0.333 → new = 640×360
3. 灰边    pad_x = (640 - new_w) / 2 = 0
            pad_y = (640 - 360) / 2 = 140   （上下各垫 140 行）
```

灰边用什么颜色？YOLO 体系约定俗成用 **(114, 114, 114)** 的灰色——经验上这个值对模型的干扰最小。

### 完整实现

```rust
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{Rgb, RgbImage};

/// 返回：letterbox 后的图、缩放比 r、(pad_x, pad_y)
pub fn letterbox(img: &RgbImage, target: u32) -> (RgbImage, f32, (u32, u32)) {
    let (w, h) = img.dimensions();
    let r = (target as f32 / w as f32).min(target as f32 / h as f32);
    let new_w = (w as f32 * r).round() as u32;
    let new_h = (h as f32 * r).round() as u32;

    // 1. 等比缩放
    let mut resized = RgbImage::new(new_w, new_h);
    Resizer::new().resize(
        &fast_image_resize::Image::from_vec_u8(w, h, img.clone().into_raw(), PixelType::U8x3).unwrap(),
        &mut fast_image_resize::Image::from_vec_u8(new_w, new_h, std::mem::take(resized.as_mut()), PixelType::U8x3).unwrap(),
        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Triangle)),
    ).unwrap();

    // 2. 建一块 114 灰的新画布
    let mut canvas = RgbImage::from_pixel(target, target, Rgb([114, 114, 114]));

    // 3. 把缩放后的图贴到正中央
    let pad_x = (target - new_w) / 2;
    let pad_y = (target - new_h) / 2;
    image::imageops::replace(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    (canvas, r, (pad_x, pad_y))
}
```

### 为什么必须"记账"

注意函数返回了 `(r, (pad_x, pad_y))`。这不是多余——**模型输出的检测框坐标是在 640×640 这张画布上的**，画面上的"猫"坐标 (300, 200) 要变回原图的坐标，必须做逆变换：

```text
原图坐标 = (640画布坐标 - 灰边偏移) / 缩放比
```

忘了记录这两个参数，你就永远算不回原图坐标——检测结果画不到原图的正确位置上。这是初学者最经典的事故之一，第 10.5 节会用它做完整的坐标还原。

## 5.4 色彩与通道顺序

### RGB ↔ BGR

如果你用 OpenCV（第 4.7 节）读了帧，拿到的是 **BGR** 字节流。喂模型前必须换回 RGB：

```rust
fn bgr_to_rgb(img: &mut RgbImage) {
    for p in img.pixels_mut() {
        p.0.swap(0, 2);   // 交换 R 和 B 通道
    }
}
```

### RGB → 灰度

有的模型（早期 OCR、部分分类模型）要灰度输入。加权公式在 2.3.1 节已经讲过，Rust 实现：

```rust
let gray = img.to_luma8();   // image 内置，用的就是 0.299/0.587/0.114
```

## 5.5 数值归一化：从 0~255 到模型要的区间

像素是 `u8`（0~255），模型吃的是 `f32`，中间必须做一次"翻译"。**翻译规则不止一种，拿错规则是仅次于 BGR 的经典事故**：

| 规则 | 公式 | 谁在用 | 数值范围 |
| --- | --- | --- | --- |
| 除以 255 | `x / 255` | YOLO 全家、很多检测模型 | [0, 1] |
| ImageNet 标准化 | `(x/255 - mean) / std` | ResNet/MobileNet 等分类模型 | 大约 [-2, 2] |
| 原样保留 | `x as f32` | 部分 INT8 量化模型 | [0, 255] |

ImageNet 标准化的三个数（mean 和 std 各三个，对应 R/G/B 通道）：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

为什么分类模型要这么折腾？因为它们在 ImageNet 数据集上预训练时就是这么喂的，我们享受预训练权重就必须延续它的口味。

**逐通道标准化**用 ndarray 的广播最顺手：

```rust
use ndarray::{Array3, Axis};

/// hwc: [H, W, 3] 的 0~255 f32 数组 → ImageNet 标准化后的 CHW
fn imagenet_normalize(hwc: &Array3<f32>) -> Array3<f32> {
    let mean = ndarray::arr1(&[0.485f32, 0.456, 0.406]);
    let std  = ndarray::arr1(&[0.229f32, 0.224, 0.225]);
    let chw = hwc.permuted_axes([2, 0, 1]);          // [3, H, W]
    let mut out = chw.to_owned();
    for c in 0..3 {
        let mut ch = out.index_axis_mut(Axis(0), c);
        ch.mapv_inplace(|v| (v / 255.0 - mean[c]) / std[c]);
    }
    out
}
```

注意 `mean[c]` 这种标量索引配合 `Axis` 循环是"逐通道"操作的通用套路，预处理和后处理都会反复用到它。

## 5.6 内存布局：HWC 还是 CHW

同一个三维数组，"先排谁"有两种约定，这也是 3.8 节埋的伏笔：

```text
同一张 2×2 的 RGB 小图，元素在内存里的先后顺序：

HWC（像素优先，图像库/OpenCV/TFLite 常用）：
  (0,0)R (0,0)G (0,0)B | (1,0)R (1,0)G (1,0)B | ...

CHW（通道优先，PyTorch/ONNX/NCNN 常用）：
  所有 R 排完 → 所有 G 排完 → 所有 B 排完
```

**ONNX 模型几乎清一色要 NCHW**（N=batch，C=通道，H=高，W=宽），所以 HWC→CHW 是预处理的标准动作。3.8 节的 `permuted_axes([2, 0, 1])` 就是干这个的：

```rust
let chw = hwc.permuted_axes([2, 0, 1]);   // [H,W,3] → [3,H,W]
```

两个配套细节：

1. **`permuted_axes` 产生的是"视图"**，数据没搬家。有的引擎要求连续内存，保险起见转 `.to_owned()`。
2. **N 维（batch 维）最后加**：`insert_axis(Axis(0))`，把 `[3,H,W]` 变 `[1,3,H,W]`。

## 5.7 batch 维：一次喂几张图

[N, C, H, W] 的第一维 N 是 **batch size（批量）**：一次推理同时处理几张图。N=1 就是单张图。

为什么要有 batch？因为推理的大部分时间花在"把数据从内存搬进计算单元"。一次搬 4 张图做 4 张图的计算，比搬 4 次算 4 次划算得多——吞吐能提升 2~4 倍。代价是显存/内存占用成倍增加。

实践中：摄像头逐帧检测通常 N=1（等不起攒 batch）；离线批量处理（比如把一万张图过一遍质检）一定要开大 batch。第 9 章推理和第 11 章进阶都会用到。

## 5.8 完整预处理函数：串起所有工序

把前面所有工序封装起来——这个函数会原样进入第 11 章的实战项目：

```rust
use image::{RgbImage, open};
use ndarray::Array4;
use anyhow::Context;

/// 完整预处理：读图 → letterbox → RGB → 归一化 → CHW → batch
/// 返回：模型输入张量、缩放比 r、(pad_x, pad_y) —— 后处理还原坐标时要用！
pub fn preprocess(path: &str, target: u32) -> anyhow::Result<(Array4<f32>, f32, (u32, u32))> {
    // 工序 1：读图并统一成 RGB（自动处理 JPEG/PNG 等格式差异）
    let img = open(path).context("读图失败")?.to_rgb8();

    // 工序 2：letterbox 等比缩放（含记录 r 和 pad）
    let (canvas, r, pad) = letterbox(&img, target);

    // 工序 3：转张量 [1, 3, H, W]、f32、[0,1]（YOLO 规则：/255）
    let tensor = img_to_tensor(&canvas);   // 用 3.8 节的 img_to_tensor

    Ok((tensor, r, pad))
}

/// 逆变换：把模型输出坐标还原到原图（先睹为快，第 10.5 节详解）
pub fn restore_coords(
    x: f32, y: f32, w: f32, h: f32,
    r: f32, pad: (u32, u32),
) -> (f32, f32, f32, f32) {
    let px = pad.0 as f32;
    let py = pad.1 as f32;
    (
        (x - px) / r,   // 中心点 x 还原
        (y - py) / r,   // 中心点 y 还原
        w / r,          // 宽还原
        h / r,          // 高还原
    )
}
```

五道工序对应一个真实流水线：

```text
读图(第4章解码) → letterbox(5.3) → RGB 确认(5.4) → /255(5.5)
              → CHW+batch(5.6/5.7) → [1,3,640,640] 的 f32 张量
```

## 5.9 预处理常见坑清单

背下这张表，能救你无数个下午：

| 症状 | 病因 | 检查方法 |
| --- | --- | --- |
| 输出框位置整体偏移/镜像 | BGR 当 RGB 喂了 | 交换通道再跑，看是否恢复 |
| 输出全错但不报错 | 忘了归一化（u8 直接转 f32） | 确认是否除以 255 |
| 报 shape 错误 | HWC/CHW 搞反 | `println!("{:?}", tensor.shape())` |
| 检测框"画不到图上" | letterbox 的 r/pad 没传下去 | 检查逆变换公式 |
| 结果整体偏暗/偏色 | u8 运算溢出、或 save 前没乘回 255 | 检查类型转换链 |
| 精度莫名比论文低 5 个点 | mean/std 归一化用错了模型对应的规则 | 查模型文档确认是 /255 还是 ImageNet 标准化 |
| 大图检测不到小目标 | 先 resize 到很小再 letterbox，小目标被插值抹掉了 | 检查缩放顺序：先 letterbox 到 640，不要再提前缩 |

## 5.10 小结

- 预处理五道工序：读图 → letterbox → 颜色确认 → 归一化 → 布局+batch。
- 缩放四插值里双线性是默认首选；性能敏感用 fast_image_resize。
- letterbox = 等比缩放 + 灰边，**必须记录 r 和 pad 供后处理逆变换**。
- 归一化规则三种（/255、ImageNet 标准化、原样），拿错规则是经典事故。
- ONNX 系要 NCHW；`permuted_axes` 转布局，转完注意连续化。
- 调试预处理的第一原则：把处理后的张量存成图片看一眼。

## 5.11 练习

1. 把 5.8 的 `preprocess` 中间结果（letterbox 后的 canvas）存成 PNG，肉眼检查灰边和变形情况。
2. 写 `restore_coords` 的单元测试：构造 r=0.5、pad=(10, 20)，验证 (320, 320) 的框中心能被正确还原到原图坐标。
3. 对比实验：同一张图分别用 `resize_exact(640,640)` 和 letterbox 预处理，喂给同一个 YOLOv8n（第 9、10 章的方法），对比检测框精度差异。
4. 改造 `img_to_tensor` 支持批量：输入 `Vec<RgbImage>`，输出 `[N, 3, H, W]`。
