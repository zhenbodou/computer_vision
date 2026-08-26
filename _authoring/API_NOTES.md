# 已验证 API 速查（API_NOTES.md）

> 本文件不属于书稿。以下版本与签名均在本机 `cargo` 实测确认（2026-08，Rust 1.97.1）。
> 撰写正文代码时**必须**以此为准，不要臆造 API。

## 依赖版本（`code/Cargo.toml` 实际解析）

```toml
image = "0.25.10"       # 图像读写、像素、几何、颜色
imageproc = "0.27.0"    # 滤波、边缘、形态学、轮廓、霍夫、模板匹配、角点、绘制、几何变换
ndarray = "0.17.2"      # 多维数组/张量
nalgebra = "0.35.0"     # 线性代数、矩阵、几何
```
> 深度学习相关 crate（candle / burn / tract / ort）的版本与用法，在写到对应部分前会在此文件追加确认；未确认前，DL 章节代码需注明版本并以官方文档为准。

## 第六部分 传统机器学习 linfa（已核实版本，⚠️ 有版本冲突坑）

- linfa 全家桶当前均为 **0.8.1**（`linfa`、`linfa-clustering`、`linfa-trees`、`linfa-linear`、`linfa-logistic`、`linfa-nn`、`linfa-bayes`、`linfa-reduction`、`linfa-svm`、`linfa-datasets` …）。
- **🕳️ 关键坑：linfa 0.8.1 内部锁定 `ndarray 0.16.1`，而本书其余章节用的是 `ndarray 0.17.2`。** 若在同一个 `Cargo.toml` 里同时写 `ndarray="0.17"` 和 `linfa="0.8"`，cargo 会拉入**两个 ndarray 版本**，你用 0.17 造的 `Array2` 和 linfa（0.16）要的 `Array2` 是**不同类型**，编译报 "expected ArrayBase…found ArrayBase…"。
  - **对策（写 Part 6 时务必这么做并向读者讲清）**：在 linfa 相关示例工程里把 ndarray 也钉到 **0.16**（`ndarray = "0.16"`），与 linfa 内部一致；或干脆用 `linfa` 重新导出的 ndarray。基础 API（`array!`、`Array2::zeros`、`.dot`）在 0.16/0.17 基本一致，读者无感。这正是书里该诚实呈现的"真实生态摩擦"。
- KMeans API 形态（`linfa-clustering` 0.8.1，已核对源码）：
```rust
use linfa::prelude::*;                 // Fit / Predict trait
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::Array2;                   // 注意用 0.16
let model = KMeans::params(k)          // 或 params_with_rng(k, rng)
    .max_n_iterations(100)
    .tolerance(1e-4)
    .fit(&DatasetBase::from(data))?;   // data: Array2<f64>
let centroids = model.centroids();     // &Array2<f64>
let labels = model.predict(&dataset);  // 预测簇标签
```
  其余算法（决策树 `linfa-trees::DecisionTree`、逻辑回归 `linfa-logistic`、SVM `linfa-svm`）都遵循同样的 `Model::params(...).…fit(&dataset)` → `.predict(...)` 范式。写到 ch32-35 前再逐个核对具体签名。

**✅ ch34/ch35 已实测签名（linfa-svm / linfa-trees 0.8.1，均配 ndarray 0.16，独立工程 `cargo run` 通过）：**

```rust
use linfa::prelude::*;                 // Fit / Predict / confusion_matrix
use linfa::Dataset;
use ndarray::{Array1, Array2};         // 注意 0.16

// —— 二分类 SVM（linfa-svm 0.8.1）——
// 🕳️ 关键坑：bool 标签分类必须用 pos_neg_weights(c_pos, c_neg)；
//    c_eps/nu_eps 只适用于回归（SvmParams<F, F>），bool 用了会报 `bool: linfa::Float`。
use linfa_svm::Svm;
let ds = Dataset::new(x /*Array2<f64>*/, y /*Array1<bool>*/);
let model = Svm::<_, bool>::params()
    .linear_kernel()                   // 或 .gaussian_kernel(eps) / .polynomial_kernel(c, d)
    .pos_neg_weights(1.0, 1.0)         // 正/负类惩罚 C（bool 分类走这条，不是 c()）
    .fit(&ds)?;
let pred = model.predict(&valid);      // valid: &Dataset
let acc  = pred.confusion_matrix(&valid)?.accuracy();

// —— 决策树多分类（linfa-trees 0.8.1）——
use linfa_trees::{DecisionTree, SplitQuality};
let tree = DecisionTree::params()
    .split_quality(SplitQuality::Gini) // 或 SplitQuality::Entropy
    .max_depth(Some(10))
    .fit(&ds)?;                        // ds 的 y 是 Array1<usize>/整型标签
let acc = tree.predict(&valid).confusion_matrix(&valid)?.accuracy();
```

> 🕳️ 数据流打通：linfa 用 `ndarray 0.16` 的 `Array2<f64>`（特征）+ `Array1<label>`（标签）。把 image/imageproc 侧算出的特征先落成 `Vec<f64>`（或写 CSV），再 `Array2::from_shape_vec((n, d), v)?` 喂给 linfa，即可跨 0.17/0.16 边界打通（ch34 HOG→SVM、ch35 像素→KNN/树 均如此）。

## image 0.25 —— 核心 API（已验证）

```rust
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba};

// 打开 / 保存
let img: DynamicImage = image::open("in.png")?;      // 需要 ? 或 unwrap，返回 Result
let (w, h): (u32, u32) = img.dimensions();           // 来自 GenericImageView trait
img.save("out.png")?;                                // 按扩展名自动选格式

// 类型转换
let gray:  GrayImage = img.to_luma8();               // 8 位灰度  ImageBuffer<Luma<u8>, Vec<u8>>
let rgb:   RgbImage  = img.to_rgb8();                // 8 位 RGB  ImageBuffer<Rgb<u8>, Vec<u8>>
let rgba             = img.to_rgba8();
let g2               = img.grayscale();              // 返回 DynamicImage（仍是彩色容器里的灰度）

// 几何（DynamicImage 方法）
let s  = img.resize(nw, nh, image::imageops::FilterType::Lanczos3); // 保持比例
let se = img.resize_exact(nw, nh, image::imageops::FilterType::Triangle);
let c  = img.crop_imm(x, y, cw, ch);                 // 不可变裁剪，返回 DynamicImage
let r  = img.rotate90();  // rotate180() rotate270()
let f  = img.fliph();     // flipv()
// FilterType: Nearest | Triangle | CatmullRom | Gaussian | Lanczos3

// 逐像素（在具体 buffer 类型上，如 RgbImage / GrayImage）
let px: Rgb<u8> = *rgb.get_pixel(x, y);
let mut buf: RgbImage = ImageBuffer::from_fn(w, h, |x, y| Rgb([0u8, 0, 0]));
buf.put_pixel(x, y, Rgb([255, 0, 0]));
let one_color: RgbImage = ImageBuffer::from_pixel(w, h, Rgb([10, 20, 30]));
for (x, y, px) in rgb.enumerate_pixels() { /* &Rgb<u8> */ }
for (x, y, px) in buf.enumerate_pixels_mut() { /* &mut */ }

// imageops 模块常用：blur, resize, rotate90, overlay, crop, flip_horizontal ...
use image::imageops;
let blurred = imageops::blur(&rgb, 2.0);             // 高斯模糊，f32 sigma
```

## imageproc 0.27 —— 模块与已验证签名

滤波 `imageproc::filter`：
```rust
gaussian_blur_f32<P>(image: &Image<P>, sigma: f32) -> Image<P>
box_filter(image: &GrayImage, x_radius: u32, y_radius: u32) -> Image<Luma<u8>>
median_filter<P>(image: &Image<P>, x_radius: u32, y_radius: u32) -> Image<P>
bilateral_filter(...)                       // 见源码，双边滤波
sharpen3x3(image: &GrayImage) -> GrayImage
sharpen_gaussian(image: &GrayImage, sigma: f32, amount: f32) -> GrayImage
filter_clamped<P, K, S>(image: &Image<P>, kernel: Kernel<K>) -> Image<...>  // 自定义卷积核
separable_filter<P, K>(image, h_kernel: &[K], v_kernel: &[K]) -> Image<P>
```

边缘/梯度 `imageproc::edges`、`imageproc::gradients`：
```rust
edges::canny(image: &GrayImage, low_threshold: f32, high_threshold: f32) -> GrayImage
gradients::sobel_gradients(image: &GrayImage) -> Image<Luma<u16>>
gradients::horizontal_sobel(&GrayImage) -> Image<Luma<i16>>   // vertical_sobel 同理
gradients::{horizontal_scharr, vertical_scharr, horizontal_prewitt, vertical_prewitt}
```

对比度/阈值 `imageproc::contrast`：
```rust
otsu_level(image: &GrayImage) -> u8
threshold(image: &GrayImage, threshold: u8, ttype: ThresholdType) -> GrayImage
// ThresholdType: Binary | BinaryInverted | Truncate | ToZero | ToZeroInverted
adaptive_threshold(image: &GrayImage, block_radius: u32, delta: i32) -> GrayImage
equalize_histogram(image: &GrayImage) -> GrayImage
stretch_contrast(...) ; match_histogram(image, target) -> GrayImage
```

形态学 `imageproc::morphology`（配 `distance_transform::Norm`）：
```rust
use imageproc::distance_transform::Norm;   // Norm::{L1, L2, LInf}
dilate(image: &GrayImage, norm: Norm, k: u8) -> GrayImage    // + dilate_mut
erode / open / close 同签名
// 灰度形态学用 Mask：Mask::square(r) / diamond(r) / disk(r)
grayscale_dilate(image, mask: &Mask) -> GrayImage  // erode/open/close 同理
```

连通域/轮廓：
```rust
use imageproc::region_labelling::{connected_components, Connectivity}; // Four | Eight
connected_components<I>(image: &I, conn: Connectivity, background: I::Pixel) -> Image<Luma<u32>>

use imageproc::contours::{find_contours, find_contours_with_threshold, Contour, BorderType};
find_contours::<u32>(image: &GrayImage) -> Vec<Contour<u32>>   // 需指定点坐标类型 T
```

霍夫 `imageproc::hough`：
```rust
detect_lines(image: &GrayImage, options: LineDetectionOptions) -> Vec<PolarLine>
// LineDetectionOptions { vote_threshold: u32, suppression_radius: u32 }
// PolarLine { r: f32, angle_in_degrees: u32 }
draw_polar_lines_mut<P>(image: &mut Image<P>, lines: &[PolarLine], color: P)
```

模板匹配 `imageproc::template_matching`：
```rust
match_template(image: &GrayImage, template: &GrayImage, method: MatchTemplateMethod) -> Image<Luma<f32>>
// MatchTemplateMethod: SumOfSquaredErrors | SumOfSquaredErrorsNormalized
//                    | CrossCorrelation | CrossCorrelationNormalized
find_extremes<T>(image: &Image<Luma<T>>) -> Extremes<T>   // .max_value / .min_value / .max_value_location ...
```

角点 `imageproc::corners`：
```rust
corners_fast9(image: &GrayImage, threshold: u8) -> Vec<Corner>   // corners_fast12 同理
// Corner { x: u32, y: u32, score: f32 }
```

绘制 `imageproc::drawing`（在 mut 图上画，配 `rect::Rect`、`point::Point`）：
```rust
use imageproc::rect::Rect;
draw_filled_rect_mut(canvas, Rect::at(x, y).of_size(w, h), color);
draw_hollow_rect_mut(canvas, rect, color);
draw_line_segment_mut(canvas, (x0, y0), (x1, y1), color);   // f32 坐标
draw_hollow_circle_mut(canvas, (cx, cy), radius, color);    // draw_filled_circle_mut 同
draw_cross_mut(canvas, color, x, y);
draw_text_mut(canvas, color, x, y, scale, &font, "text");   // 需 ab_glyph 字体
```

几何变换 `imageproc::geometric_transformations`：
```rust
use imageproc::geometric_transformations::{
    rotate_about_center, warp, Border, Interpolation, Projection,
};
// Interpolation: Nearest | Bilinear | Bicubic
// Border<P>: Constant(P) | Replicate | Wrap    ← extend 参数用这个，不是裸颜色！
rotate_about_center(&img, theta_rad, Interpolation::Bilinear, Border::Constant(px)) -> Image<P>
let proj = Projection::translate(tx, ty)
    .and_then(Projection::scale(sx, sy))
    .and_then(Projection::rotate(theta));
warp(&img, proj, Interpolation::Bilinear, Border::Constant(px)) -> Image<P>  // proj 按值传！
// Projection::from_matrix([f32;9]) -> Option<Projection>  用于单应矩阵
```

> ⚠️ 易错点：`warp` 的第二参 `projection` **按值**传（`proj` 不是 `&proj`）；`rotate_about_center`/`warp` 的 `extend` 必须是 `Border::Constant(color)` 之类，**不能直接传颜色**。

## ndarray 0.17 / nalgebra 0.35（已验证基本用法）

```rust
use ndarray::{array, Array1, Array2, Array3, Axis};
let a: Array2<f32> = Array2::zeros((3, 4));
let b = array![[1.0f32, 2.0], [3.0, 4.0]];
let s = b.sum();  let m = b.mean();  let d = b.dot(&b);

use nalgebra::{Matrix3, Vector3, DMatrix};
let m = Matrix3::<f32>::identity();
let v = Vector3::new(1.0f32, 2.0, 3.0);
let mv = m * v;               // 矩阵×向量
let inv = m.try_inverse();    // Option
```

**🕳️ ndarray 0.17 reshape 坑（ch29/ch31 实测）**：`into_shape` 已 `#[deprecated(since=0.16)]`，别再用。改用：

- `a.to_shape((h, w))?` —— 借用、返回 `CowArray`（非连续时会拷贝）；
- `a.into_shape_with_order((h, w))?` —— 消耗所有权、显式指定顺序；
- HWC↔CHW 用 `a.permuted_axes([2, 0, 1])`（**零拷贝**换轴视图），换轴后内存**非连续**，喂给需要连续切片的接口前先 `.as_standard_layout()`。

**nalgebra 0.35 线代求解（ch30 实测签名）**：

```rust
let svd = a.clone().svd(true, true);       // 计算 U、V^T
let x = svd.solve(&b, 1e-12).unwrap();      // 最小二乘/超定方程组；eps 为奇异值截断阈值
let x2 = a.clone().lu().solve(&b);          // LU 解方阵方程组 -> Option<向量>
let d  = a.determinant();                   // 行列式（方阵）
let inv = a.try_inverse();                  // -> Option
```

## 常见类型别名

- `imageproc::definitions::Image<P> = ImageBuffer<P, Vec<P::Subpixel>>`
- `GrayImage = ImageBuffer<Luma<u8>, Vec<u8>>`，`RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>`
- 需要 `use image::GenericImageView;` 才能用 `.dimensions()`, `.get_pixel()`（在 DynamicImage 上）。

## 第四部分 特征相关 imageproc 0.27 API（已验证）

角点 `imageproc::corners`：
```rust
pub struct Corner { pub x: u32, pub y: u32, pub score: f32 }
corners_fast9(image: &GrayImage, threshold: u8) -> Vec<Corner>   // corners_fast12 同理
oriented_fast(image, threshold: Option<u8>, target: usize, ...) -> Vec<OrientedFastCorner> // ORB 式带方向 FAST，签名多，用前 grep 确认
fast_corner_score(image, threshold, x, y, variant: Fast) -> u8
```

非极大值抑制 `imageproc::suppress`：
```rust
suppress_non_maximum<I, C>(image: &I, radius: u32) -> Image<Luma<C>>   // 在响应图上抑制非局部极大
```

二进制描述子 `imageproc::binary_descriptors`（FAST + BRIEF + LSH 匹配，纯 Rust 全链路）：
```rust
use imageproc::point::Point;                 // Point::new(x, y)，Corner 可 .into() 成 Point<u32>
use imageproc::binary_descriptors::{brief::brief, match_binary_descriptors, BinaryDescriptor};
// BRIEF：length 是比特数（如 256）；keypoints 用 &[Point<u32>]
brief(image: &GrayImage, keypoints: &[Point<u32>], length: usize,
      override_test_pairs: Option<&Vec<TestPair>>)
    -> Result<(Vec<BriefDescriptor>, Vec<TestPair>), String>
// BriefDescriptor { bits: Vec<u128>, corner: Corner }，实现 BinaryDescriptor::hamming_distance
match_binary_descriptors<'a, T: BinaryDescriptor>(
    d1: &'a [T], d2: &'a [T], threshold: u32, seed: Option<u64>) -> Vec<(&'a T, &'a T)>
// 返回匹配对（基于 Hamming 距离 + 局部敏感哈希）。threshold 是最大 Hamming 距离。
```
> ⚠️ BRIEF 不抗旋转/缩放；要旋转不变可用 `oriented_fast`（ORB 思路）或纯 Rust 的 **`akaze`** crate（AKAZE 特征，需自行 cargo add 验证版本）。RANSAC 求单应：imageproc 无内置，**建议手写一个简单 RANSAC**（随机取 4 对点 → `Projection::from_control_points` 估计 → 统计内点）配 nalgebra，或用 rust-cv 的 `arrsac`/`sample-consensus` crate（用前验证）。
- **🎯 BRIEF 匹配两图必须共用同一套确定性测试点对（ch106 血泪坑）**：`imageproc` 的 `brief(...)` 若 `override_test_pairs=None` 会用**系统随机**生成测试点对 → ① 不可复现；② **更致命**：两张图各自随机生成 → 描述子比特位含义不同、**跨图汉明距离完全无意义**、匹配全错。**必做**：用固定 `Lcg` 生成一套 `Vec<TestPair>`，两图 `brief` 调用都传同一个 `Some(&pairs)`。
- **🎯 BRIEF 依赖"局部可区分纹理"，且纹理尺度有甜点（ch106 血泪坑，特征匹配/拼接通用）**：大矩形/大圆的直角角点在 31×31 描述子块内**自相似** → 误匹配爆炸；**最阴险**：RANSAC 会锁定一个**自洽但错误**的模型（内点重投影 0.4px 很漂亮、却与真值差 237px）。教训一：**RANSAC 内点数/内点误差漂亮 ≠ 模型正确**，必须拿**独立真值**（如已知单应的四角映射误差）复核，别只信内点统计（呼应 §15/§20"单级指标全绿≠系统可用"）。教训二：纹理**太细也不行**——高频斑点经 `warp` 双线性重采样被糊掉、真匹配消失；**中等尺度、局部各异**才是甜点（撒随机彩色斑点即可）。合成配准/拼接数据照此设计（同 §19"让对照显形"的合成家族）。

HOG 特征 `imageproc::hog`：
```rust
HogOptions::new(orientations: usize, signed: bool, cell_side: usize, block_side: usize, block_stride: usize)
hog(image: &GrayImage, options: HogOptions) -> Result<Vec<f32>, String>   // 返回特征向量
render_hist_grid(star_side: u32, grid: &View3d<f32>, signed: bool) -> Image<Luma<u8>>  // 可视化
```

局部二值模式 `imageproc::local_binary_patterns`：
```rust
local_binary_pattern<I>(image: &I, x: u32, y: u32) -> Option<u8>   // 单点 LBP 码
min_shift(byte: u8) -> u8          // 旋转不变（取最小循环移位）
count_transitions(byte: u8) -> u32 // 判断 uniform LBP（0/1 跳变次数）
```

噪声 `imageproc::noise`（合成含噪测试图很方便）：
```rust
salt_and_pepper_noise<P>(image: &Image<P>, rate: f64, seed: u64) -> Image<P>   // 椒盐噪声
gaussian_noise<P>(image: &Image<P>, mean: f64, stddev: f64, seed: u64) -> Image<P>
```

积分图 `imageproc::integral_image`、Haar 特征 `imageproc::haar`、感知哈希 `imageproc::image_hash` 也存在（分别用于快速区域求和 / 类 Viola-Jones 人脸 / 图像去重，写到相应章节前再确认签名）。

## 深度学习 / 推理 crate（第 7–16 部分，已核对版本）

> 版本号全部由**本机 `cargo add` 实测解析**（2026-08，Rust 1.97.1，macOS/darwin）。
> **candle 与 tract 的代码骨架已在 `/tmp` scratch 工程 `cargo run` / `cargo check` 编译通过**（下方标 ✅ 实测）。
> **ort 与 burn 未在本机编译**（ort 要拉 C++ onnxruntime 运行库、burn 依赖庞大）：其**版本号为本机 cargo 解析值**，但 **API 据 docs.rs 对应版本页查证**（标 📄 文档查证）。写正文时如需 ort/burn 的精确签名，务必再对官方文档复核。

### 版本速览与纯度

| crate | 实测版本 | 纯 Rust？ | 本机验证程度 |
|---|---|---|---|
| `candle-core` / `candle-nn` | **0.11.0** | ✅ 纯 Rust | `cargo run` 实测通过 |
| `tract-onnx` | **0.23.5** | ✅ 纯 Rust*（*构建期 cc 仅汇编自带 SIMD 内核，无外部 C/C++ 运行库，详见 §14） | `cargo check` 实测通过 |
| `ort` | **2.0.0-rc.13** | ❌ **底层链接 C++ onnxruntime** | 📄 文档查证，未编译 |
| `burn` | **0.21.0** | ✅ 纯 Rust | 📄 文档查证，未编译 |

---

### 1. candle 0.11.0（`candle-core` + `candle-nn`）✅ 纯 Rust —— 本机 `cargo run` 通过

```toml
candle-core = "0.11.0"
candle-nn   = "0.11.0"   # 默认 CPU；GPU 走 feature：cuda / metal / accelerate / mkl
```

张量运算（以下整段实测跑通，matmul 结果为实际输出 `[[22,28],[49,64]]`）：

```rust
use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu; // GPU（feature 开启后）：Device::new_cuda(0)? / Device::new_metal(0)?

    let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device)?; // 数据+形状+设备
    let b = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;           // 从嵌套数组字面量
    let r = Tensor::randn(0f32, 1f32, (2, 3), &device)?;                        // 均值,标准差,形状,设备

    println!("{:?} {:?} {:?}", a.shape(), a.dims(), a.dtype()); // [2, 3]  [2, 3]  F32

    let c  = a.matmul(&b)?;             // (2,3)x(3,2)=(2,2) → 实测 [[22.0,28.0],[49.0,64.0]]
    let _d = (&c + &c)?;                // 逐元素加法：&Tensor + &Tensor 返回 Result<Tensor>
    let _s = (&c + 1.0)?;               // 广播标量
    let _e = c.reshape((4,))?;          // reshape，形状 [4]
    let _f = a.to_dtype(DType::F64)?;   // dtype 转换
    let cv: Vec<Vec<f32>> = c.to_vec2()?; // 取回数据
    println!("{cv:?}");
    let _ = r;
    Ok(())
}
```

> 💡 `DType` 常用变体：`F16 / BF16 / F32 / F64 / U8 / U32 / I64`。`Device::Cpu` 已实测；`new_cuda/new_metal` 需对应 feature（本机未编译）。

**加载 safetensors 权重（本机 `cargo check` 通过——LLM/预训练模型加载惯用法）：**

```rust
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use std::collections::HashMap;

// A) 直接把 .safetensors 读成 name -> Tensor 字典
let weights: HashMap<String, Tensor> =
    candle_core::safetensors::load("model.safetensors", &device)?;

// B) 用 VarBuilder 按前缀取权重来搭网络（⚠️ from_mmaped_safetensors 是 unsafe）
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
};
let fc: Linear = candle_nn::linear(768, 10, vb.pp("classifier"))?; // vb.pp(前缀) 定位子权重
let y = fc.forward(&x)?; // 来自 candle_nn::Module trait
```

> 📖 candle 定位：Hugging Face 出品，safetensors 原生、生态偏 **LLM / Transformer / 手搓&微调模型**；纯 Rust，CPU/CUDA/Metal。
> 🕳️ candle-core 0.11 会拉入 `safetensors 0.8`、`memmap2`、`gemm`、`tokenizers 0.22` 等——首次编译较久（本机约 26s）。

---

### 2. tract-onnx 0.23.5 ✅ **纯 Rust（无外部 C/C++ 运行库；构建期 cc 仅作汇编器编译自带 SIMD 内核，精确边界见 §14）** —— 本机 `cargo check` 通过

```toml
tract-onnx = "0.23.5"   # Sonos 出品，纯 Rust ONNX/NNEF 推理引擎，交叉编译友好
```

加载 ONNX 并推理的骨架（整段实测编译通过）：

```rust
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")?                              // 读 .onnx
        .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?    // 指定输入 dtype+形状
        .into_optimized()?                                          // 图优化
        .into_runnable()?;                                          // 得到可执行模型

    let input: Tensor = Tensor::zero::<f32>(&[1, 3, 224, 224])?;
    let result = model.run(tvec!(input.into()))?;                   // 推理，输入用 tvec!(...)

    // ⚠️ 0.23 里取输出用 to_plain_array_view（不是旧版的 to_array_view！）
    let output = result[0].to_plain_array_view::<f32>()?;           // 得 ndarray view
    println!("output shape = {:?}", output.shape());
    Ok(())
}
```

> ⚠️ **易错点**：tract 0.23.5 从输出 `TValue` 取 ndarray 用 **`to_plain_array_view::<T>()`**，旧资料里的 `to_array_view` 已不存在（本机编译报错确认）。
> 💡 tract 纯 Rust、无外部运行库，是"**要纯 Rust 跑现成 ONNX**"的首选；算子覆盖广但非 100%，个别很新的模型/算子可能不支持——遇到再考虑 ort。

---

### 3. ort 2.0.0-rc.13 ⚠️ **非纯 Rust（底层是 C++ onnxruntime）** —— 📄 文档查证，未在本机编译

```toml
ort = "2.0.0-rc.13"   # ⚠️ 2.x 仍处 RC 阶段；底层链接 C++ onnxruntime（默认可下载预编译库）
```

推理大致形态（API 据 docs.rs 2.0.0-rc.13 页面，未本机编译）：

```rust
use ort::session::Session;

let mut session = Session::builder()?          // -> Result<SessionBuilder>
    .commit_from_file("model.onnx")?;          // SessionBuilder::commit_from_file 构建 Session

// 输入：ort::inputs! 宏（可用名字或下标），值来自 ndarray / Tensor
let outputs = session.run(ort::inputs!["input" => input_value]?)?; // run(&mut self, ...) -> SessionOutputs

// 输出：按名字/下标索引得到 DynValue，再 try_extract
let (shape, data) = outputs["output"].try_extract_tensor::<f32>()?; // -> (&Shape, &[f32])
// 或 outputs["output"].try_extract_array::<f32>()? -> ndarray::ArrayViewD<f32>（需 ndarray feature）
```

> ⚠️ **务必向读者讲清纯度取舍**：ort 只是 ONNX Runtime 的 Rust 绑定，**运行时依赖 C++ onnxruntime 动态/静态库**（`ort-sys` FFI），**不是纯 Rust**。换来的是**最强推理性能、最全算子覆盖、GPU/多种 Execution Provider 支持**——本书在"需要跑大模型的务实选择"处引用它。
> 🕳️ **版本坑**：ort 2.x 与 1.x API **差异极大**（1.x 是 `Environment` + `SessionBuilder::with_model_from_file` + `Value::from_array`）；且 2.x 各 rc 之间 `inputs!` 宏是否需 `?`、`try_extract_*` 方法名仍在变动。写正文时**认准 `2.0.0-rc.13` 并对照当时 docs.rs 复核**。已核实要点：`Session::builder() -> Result<SessionBuilder>`、`run(&mut self, …) -> Result<SessionOutputs>`、`try_extract_tensor::<T>() -> Result<(&Shape, &[T])>`。

---

### 4. burn 0.21.0 ✅ 纯 Rust —— 📄 文档查证（简要）

```toml
burn = { version = "0.21.0", features = ["ndarray"] } # 选后端：ndarray/wgpu/candle/tch/cuda…；训练加 "autodiff"
```

```rust
use burn::prelude::*;               // Tensor、Backend 等
use burn::backend::NdArray;         // 纯 Rust CPU 后端

type B = NdArray<f32>;
let device = Default::default();
let x: Tensor<B, 2> = Tensor::zeros([2, 3], &device); // 张量泛型见下
let _y = x.clone() + x;             // 算子对 Backend 泛型
```

- **Backend 概念**：绝大多数代码对 `Backend` trait 泛型，后端可换：`burn-ndarray`（纯 Rust CPU）、`burn-flex`（纯 Rust CPU，新，std/no_std/WASM）、`burn-wgpu`（跨平台 GPU/WebGPU）、`burn-candle`（复用 candle）、`burn-tch`（LibTorch）、`burn-cuda`。装饰器后端：**Autodiff**（给任意后端加反向传播）、**Fusion**（算子融合）。（后端 crate 名已在 docs.rs 0.21.0 依赖列表确认存在；各 feature flag 精确拼写写正文时再核。）
- **Tensor 泛型**（docs.rs 0.21.0 已核）：`Tensor<B, const D: usize, K = Float>`——`B` 后端、`D` 常量维度、`K` 种类默认 `Float`（可为 `Int` / `Bool`）。故常见写法 `Tensor<B, 2>` 即 `Tensor<B, 2, Float>`。
- **定位**：带自动微分、多后端的 **训练型** DL 框架（类 PyTorch 体验，一套代码切后端）；生态相对 candle 更新、更"框架化"。

---

### 5. 纯 Rust 下加载/推理 ONNX 的现状小结 + 选型建议

**ONNX 权重加载现状：**

- **tract**（纯 Rust）：自己用 `prost` 解析 `.onnx`（protobuf）并在纯 Rust 图上推理，**零 C/C++ 依赖**，交叉编译/静态链接友好——"纯 Rust 跑 ONNX"就选它。
- **ort**（C++ 后端）：不把权重解析进 Rust，直接交给 C++ onnxruntime；**算子最全、性能最强、支持 GPU/EP**，代价是失去纯 Rust 纯度与部分可移植性。
- **candle**（safetensors 为主）：主场是 **safetensors** + 自建/HF 模型；对 ONNX 支持有限（社区有 `candle-onnx` 子 crate，**覆盖范围/版本待核实**，用前验证）。

**选型建议表：**

| 需求场景 | 推荐 crate | 纯 Rust？ | 一句话理由 |
|---|---|---|---|
| 跑现成 ONNX 模型（分类/检测），要**纯 Rust、无 C++** | **tract-onnx** | ✅ | 直接加载 `.onnx` 推理，无外部运行库，交叉编译友好 |
| 跑**大模型 / 追求极致性能 / 要算子全 / GPU** | **ort** | ❌ (C++) | 性能与算子覆盖最强，代价是链接 C++ onnxruntime |
| **加载 safetensors / 手搓&微调模型（LLM、Transformer）** | **candle** | ✅ | HF 生态、safetensors 原生，CPU/CUDA/Metal |
| **从零训练、要自动微分 + 多后端、想要 PyTorch 式体验** | **burn** | ✅ | 泛型后端 + Autodiff，训练为主 |

> 💡 一句话决策：**只推理+要纯 Rust → tract；只推理+要最强性能/算子全（可接受 C++ 依赖）→ ort；训练或加载 safetensors 手搓模型 → candle；PyTorch 式多后端训练 → burn。**

---

### 6. nalgebra 0.35 —— SVD / 特征分解 / 动态矩阵（ch67-68 三维几何实测）

```rust
// SVD 取零空间（DLT 三角化、平面法向都靠它）：解 A x = 0 → 取最小奇异值对应的右奇异向量
let svd = a.svd(true, true);            // 计算 U、V^T
let vt = svd.v_t.unwrap();              // svd.v_t: Option<DMatrix>，即 V^T
// svd.singular_values 是 **降序** 排列 → 解 = v_t 的 **最后一行**
let x = vt.row(vt.nrows() - 1).transpose();

let m = DMatrix::from_row_slice(nrows, ncols, &data_f64);  // 按行填充动态矩阵
```

- `Matrix3x4::new(<12 个数>)` 按行填；`Matrix3 * Matrix3x4 -> Matrix3x4`；`Matrix3x4 * Vector4 -> Vector3`；元素索引用元组 `m[(i, j)]`。
- `Vector3` 几何：`.cross(&v)`、`.dot(&v)`、`.normalize()`、`.norm()`、`v[k]`；`sum += *p`（AddAssign 需先解引用）。
- ⚠️ **`Matrix3::symmetric_eigen() -> SymmetricEigen { eigenvalues, eigenvectors }` 的特征值不排序**（对角阵实测返回 `[2,5,1]`）。要最小特征向量得自己 argmin；**更省事的做法：把中心化数据矩阵做 SVD 取 `v_t` 末行**（天然降序）。PCA 平面拟合建议走 SVD 路线。
- **🎯 瘦 SVD 求最小齐次解 `Ax=0` 在"恰好最小定"系统上翻车（ch106 血泪坑，DLT/单应/本质矩阵务必牢记）**：`nalgebra` 的 `a.svd(true,true)` 是**瘦（thin/reduced）SVD**——对 `m×n` 矩阵 `V^T` 只有 `min(m,n)` 行。当 `m < n`（如 **4 点单应的 8×9** 矩阵），`V^T` 是 8×9、**只有 8 行**，零空间对应的第 9 个右奇异向量**根本不在这个瘦 `V^T` 里** → `vt.row(vt.nrows()-1)` 取到错向量、解出乱码单应（ch106 实测四角误差数百 px、症状极迷惑：RANSAC 多点精拟合明明正确，唯独 4 点真值单应算成乱码）。**为何多点没事**：≥5 点时 A 是 `2N×9`（≥10 行）超定，瘦 `V^T` 是 9×9、末行正是零空间解。**根治**（最小定/超定齐次都稳）：解 **`AᵀA`（恒 `n×n`，本例 9×9）的对称特征分解、取最小特征值的特征向量**——`let ata = a.transpose() * &a; let se = ata.symmetric_eigen();` 再对 `se.eigenvalues` argmin 取 `se.eigenvectors.column(idx)`（上一条已述 `symmetric_eigen` 特征值**不排序**，必须自己 argmin）。ch106 由此 513px→1.965px。**⚠️ 终检 spot-check**：三维部分（ch64-68 标定/三角化/PnP/单应）凡"取 `v_t` 末行解 `Ax=0`"处，确认系统超定（行数≥列数）才安全——最小定的一律改走 `AᵀA` 特征分解。
- **🕳️ `imageproc::geometric_transformations::warp_into` 内部对传入 projection 求逆（ch106 实测）**：故拼接/贴图须传"**源→目标**"方向的单应 `H`（源图坐标→画布坐标），而非目标→源。与 `warp`（按值传 projection、见上文 imageproc 段）配套记忆：warp 系列内部要的是"输出像素反查输入"的映射，`warp_into` 帮你把正向 `H` 求了逆。

---

### 7. 纯 Rust 读码 / OCR crate（ch81-82 实测 + ch98 QR 生成读回闭环 + ch99 受限 OCR 连通域/域偏移/诚实边界确认，与主工程 image 0.25.10 的兼容性已标注）

| crate | 版本 | 能力 | 纯 Rust | 与 image 0.25.10 | 备注 |
|---|---|---|---|---|---|
| `qrcode` | 0.14.1 | QR **生成** | ✅ | ✅ 兼容 | `QrCode::new(&[u8])?` → `.render::<Luma<u8>>().min_dimensions(w,h).build()` |
| `rqrr` | 0.10.1 | QR 定位+**解码** | ✅ | ✅ 兼容 | `PreparedImage::prepare(GrayImage).detect_grids()` → `grid.decode() -> (MetaData, String)`。qrcode+rqrr 是只读 QR 首选组合 |
| `rxing` | 0.9.2 | 1D/QR/DataMatrix **解码**（ZXing 移植） | ✅ | ❌ **锁死 image =0.25.8** | `rxing::helpers::detect_in_image(DynamicImage, Option<BarcodeFormat>)`；依赖重（拉 imageproc 0.25.1/nalgebra 0.32/chrono/regex）→ **必须独立工程**。无 `-sys`/`cc`/bindgen |
| `barcoders` | 2.0.0 | 一维码 **仅生成** | ✅ | ✅ | `Code39::new(s)?.encode() -> Vec<u8>`；`Image::ImageBuffer{...}.generate_buffer(&data)`。无解码 API |
| `datamatrix` | 0.3.3 | DataMatrix 编+解 | ✅ | — | `DataMatrix::encode(&[u8], SymbolList::default())?.bitmap()`；`decode(&[bool], width)` **只吃已二值化模块矩阵，不吃相机原图** |
| `ocrs` + `rten` | 0.12.2 / 0.25.0 | OCR（rten=纯 Rust ONNX 运行时） | ✅ | — | **仅解析版本、未编译验证**；需下载 ONNX 模型 |

> 🕳️ **纯 Rust 读码/OCR 的能力边界**（正文须诚实）：① **一维码解码**无轻量专用纯 Rust 库，只能上重量级 `rxing` 或自写条宽解析；② DataMatrix **从相机原图定位/检测**需 `rxing` 或自实现（`datamatrix` 只解已定位模块）；③ **高精度 OCR** 无"开箱即跑"纯 Rust 方案（`ocrs`/`rten` 需配 ONNX 模型）；Tesseract 系（`tesseract`/`leptess`）是 **C++ 绑定，非纯 Rust**。④ `rxing` 把 image 钉在 0.25.8，与主工程 0.25.10 冲突 → 独立工程隔离。
- **🕳️ `rqrr` 的 `MetaData.ecc_level` 是格式信息原始 2 bit，不是 L/M/Q/H 序号（ch98 实测）**：取值 **M=0b00 / L=0b01 / H=0b10 / Q=0b11**（QR 格式信息里纠错级的原始编码），**不是**按 L<M<Q<H 的序号。按序号查表会张冠李戴（把 M 报成 L 等）。要显示纠错级须按这张 2bit 映射转换。**ch98 确认 qrcode 0.14.1（生成）+ rqrr 0.10.1（读回）闭环端到端可复现**：生成 369×369px/33×33 模块/等级 M，旋转（直角 + 任意 5–45°）全成功、缩到 ≈90px（每模块 ≈2.7px）才失败；椒盐噪声下 **M+3×3 中值滤波全档 100%**——对症预处理 > 盲堆纠错等级（H 更密更脆，1% 噪声反而只 26.7%）。
- **🎯 ch99 确认：纯 Rust 无成熟通用 OCR 引擎（附录 A/D 与「诚实边界」据此措辞）**：Tesseract 系（`tesseract`/`leptess`）= C++/leptonica 绑定、PaddleOCR = Python/C++，均违反纯 Rust 约束。**务实替代**：① 通用/弯曲文字 → 用 PaddleOCR/CRNN **导出 ONNX** + `tract` 推理（**代码纯 Rust，但依赖外部训练好的模型、非自包含**）；② 手写/复杂版面 → 绑定或云 API（非纯 Rust）。**纯 Rust 能自己闭环的** = **受限字符集**（固定字体、数字 + 少量字母）的「连通域切分 + candle 小 CNN 分类」整串识别（ch99 实测 13 类闭环）。
- **🎯 受限 OCR 连通域切分的两条硬边界 + 预处理两难（ch99 实测数字）**：字符集 13 类 `0123456789OIS`；**分割成功率 95.5%（191/200，指分出的块数=真值字符数）**；NAIVE 逐字 69.6%(699/1005)/整串 20.5%(41/200)；PIPELINE 逐字 93.6%(941/1005)/整串 85.0%(170/200)。① **粘连击穿连通域**——字间距为负/笔画相连时「一个字 = 一个连通块」的前提崩塌，99.0% 的串被并成一块 → 整串 1.0%(1/100)（**纯连通域方案的天花板**，过此关须上滑窗/投影切分/CTC/CRNN）。② **模糊 + 前景阈值两难**——模糊过重 + 阈值过高（2× 模糊 + THR 0.5）会让细笔画掉到前景阈下、连通域碎裂（分割率一度崩到 6.5%）；最终定 **1× 模糊 + THR 0.35 + 最小面积 15**，在「连住同字笔画」与「分开相邻字」间平衡。
- **🎯 类别混淆的反直觉规律：最该防的不是「最像」的类，而是「唯一区别落在会被成像退化吃掉之处」的类（ch99 新发现，受限分类/OCR 通用）**：ch99 混淆几乎全压在 **0→O（21 次）**，而 **1/I、5/S 一次不混**。根因——0 与 O 只差「内部一个被暗背景包围的像素点」，**轻微模糊即抹平**；1/I、5/S 的判别特征在**字形边界、跨多像素**，模糊糊不掉。→ 设计受限字符集/类集时，别只按「视觉相似度」防混淆，要问**「这两类的唯一区别，是否落在成像退化（模糊/低分辨率/压缩）会最先吃掉的尺度上」**；若是，要么剔除其一，要么在训练数据里显式注入该退化，逼分类器去学其余判别特征。

---

### 8. 主工程 code/Cargo.toml 新增依赖（ch108/ch112 引入，均纯 Rust，可直接 use）

```toml
[dependencies]
rayon = "1.12.0"       # ch108 数据并行 par_iter/par_chunks
thiserror = "2.0.20"   # ch112 库层自定义错误枚举
anyhow = "1.0.104"     # ch112 应用层错误处理
[dev-dependencies]
criterion = "0.8.2"    # ch108 基准测试
[[bench]]
name = "ch108_bench"   # ⚠️ 对应 code/benches/ch108_bench.rs 必须存在，否则 manifest 解析失败→全工程 cargo 命令报错
harness = false
```

> 🕳️ **坑**：往主 code/Cargo.toml 加 `[[bench]]`/`[[example]]` 条目时，**对应源文件必须同时存在**，否则 `cargo metadata`/任何 cargo 命令因 manifest 解析失败而全线报错，会误伤同时在跑的其他章验证。（已踩坑：ch108 bench 条目一度先于文件提交。）

### 9. imageproc 梯度/绘图 + wasm/交叉编译（ch59-61 与 ch110-111 本机实测）

- 🕳️ **`imageproc::gradients::{horizontal_sobel, vertical_sobel}` 返回值是真导数的 8 倍**（3×3 Sobel 核系数绝对值和=8），边界按复制处理。凡把 Sobel 梯度与"原始帧差/像素差"放进同一方程（光流、亚像素边缘等）**必须先 /8 统一单位**，否则量纲差 8 倍。
- `imageproc::drawing::draw_line_segment_mut` 接受 **f32 坐标元组**，可直接画光流/轨迹折线；**无内置箭头**，需自己用两条短线段拼。
- nalgebra 0.35 混合尺寸矩阵别名齐全：`Matrix2x4/Matrix4x2/Matrix4/Matrix2/Vector4/Vector2` 可 `Matrix2x4::new(8 个行优先参数)` 构造；乘法类型推断顺畅；二次型标量 `(y.transpose()*s_inv*y)[(0,0)]`（1×1 用下标取值）。
- **WASM（本机 Rust 1.97.1 实测）**：`image 0.25.10` + `imageproc 0.27.0` **能编到 `wasm32-unknown-unknown`**（默认特性，cargo check 通过）；`wasm-bindgen` 0.2.127。**体积坑**：仅 wasm-bindgen 依赖树 13 crate，一加 image+imageproc 暴涨到 **149 crate**（拖入 png/gif/tiff/webp/avif/exr/qoi 全套编解码器）——浏览器 canvas 给的是解码后 RGBA，这些编解码器纯属负担，建议 `default-features=false` 或手写逐像素。wasm 默认无真线程（rayon 等于串行）；无 FS；`file://` 下 ES module+wasm 被拦，须走 HTTP。WASM demo 独立工程在 `code/wasm_demo/`。
- **交叉编译（macOS 本机实测）**：`rustup target add aarch64-unknown-linux-gnu` 可加，但在 macOS 上直接 `cargo build --target aarch64-unknown-linux-gnu` **链接期失败**（Apple `ld` 不认 GNU 参数 `--as-needed`/`-Bstatic`/`--gc-sections`，报 `ld: unknown options`）→ 用 **`cross`**（Docker）或 musl 静态目标。`tract-onnx 0.23.5` 纯 Rust 零 C++，交叉编译最省心；`ort` 链 C++ onnxruntime，边缘折腾。跑 Linux 的边缘盒子有 std，image/imageproc/tract 全链路可用；真正裸机 MCU（no_std）三者都编不过。

### 10. candle 0.11.0 手写训练/推理 实测签名与坑（ch44 多标签分类实测通过，供全部 DL 章复用）

- 手写训练脚手架（实测跑通）：`Var::from_tensor(&t)?` 建可训练参数 → `Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()))`、`Conv2d::new(w, Some(b), Conv2dConfig{ padding, ..Default::default() })`；优化器 `AdamW::new_lr(vars: Vec<Var>, lr: f64) -> Result<Self>`，`opt.backward_step(&loss)?`（trait `candle_nn::Optimizer`）。
- Tensor 常用方法实测可用：`.relu()/.abs()/.exp()/.log()/.neg()/.max_pool2d(2usize)/.flatten_from(1)/.ge(scalar)/.sum(0)/.sum(dim)/.to_vec2()/.to_vec1()`。
- 损失/激活：`candle_nn::ops::sigmoid(&Tensor)`、`candle_nn::ops::softmax::<D: Dim>(&Tensor, dim)`；`candle_nn::loss::cross_entropy(inp[N,C] logits, target u32[N])`（内部 log_softmax+nll）。
- 🕳️ **`binary_cross_entropy_with_logit(inp, target)` 两个坑**：① `inp` 传**原始 logits**，`target` 必须是**同形状 `[N,C]` 的 f32(0/1)** 张量（源码 doc 注释写 u32 是**错的**，实测须 f32）；② 它是 `sigmoid(x).log()` 直译版，**数值不稳定**，logit 变大后 `log(0)→NaN`（本机约 300 轮必崩）→ 多标签训练请自写数值稳定版 `max(x,0) - x*y + log(1+exp(-|x|))`。
- 🕳️ **最重要的复现坑**：candle 0.11.0 **CPU 后端 `Device::set_seed` 直接报错** `"cannot seed the CPU rng"`；`Tensor::rand_uniform/randn` 走 `rand::rng()` 取系统熵 → **默认随机初始化不可复现**。要让正文数字稳定，必须**自造确定性初始化**（如用固定种子的 LCG 生成权重再 `Tensor::from_vec`），不能依赖 candle 自带随机。
- dl_labs 隔离：candle 独立工程的 Cargo.toml 加 `[workspace]`（空表）即可与主 code/ 彻底隔离，避免被主工程 workspace 探测（ch44 已用）。
- **safetensors 存/取（ch42 实测闭环）**：`candle_core::safetensors::save(&HashMap<K,Tensor>, path)` / `::load(path, &Device) -> HashMap<String,Tensor>`；`VarMap::save(path)`/`VarMap::load(path)`（内部即前二者）；`Tensor::save_safetensors(name, path)`。建网加载：**`unsafe { VarBuilder::from_mmaped_safetensors(&[paths], DType::F32, &Device)? }`**（mmap，故 unsafe），或 `VarBuilder::from_varmap(&VarMap, DType, &Device)`；`vb.pp("conv1")` 决定键名 → `conv1.weight`[out,in,kh,kw]、`conv1.bias`；`linear(..,vb.pp("fc1"))`→`fc1.weight`[out,in]。**🎯 冻结/迁移利器**：`from_mmaped_safetensors` 加载出的是**常量 `Tensor`（非 `Var`）→ 天然不更新**；文件里**多余的键被静默忽略**（不 `.pp` 请求就不报错）→ 正好「只加载骨干、丢弃旧 head」；可训练新头单独放一个 `VarMap`，优化器只收 `head_vm.all_vars()`。指纹验证冻结：`Conv2d::weight()->&Tensor`、`Linear::weight()->&Tensor`、`Var::set(&Tensor)`、`VarMap::data()->&Mutex<HashMap<String,Var>>`。
- **AdamW 精确签名**：`AdamW::new(vars: Vec<Var>, ParamsAdamW{lr,beta1,beta2,eps,weight_decay})?`（Default: lr 0.001/β1 0.9/β2 0.999/eps 1e-8/wd 0.01）；`AdamW::new_lr(vars, lr)` 是只给 lr 的便捷版；`opt.backward_step(&loss)?` 或 `opt.step(&grads)`；`set_learning_rate(f64)`。`SGD::new(vars, lr)`（无 momentum）。`Tensor::max_pool2d(sz)`（kernel==stride）、`flatten_from(dim)`、`argmax(D::Minus1)`→U32。
- **上采样/解码器/图像复原（ch69 自编码器 + ch72 超分去噪 实测跑通）**：
  - 🎯 **candle 0.11 无转置卷积（ConvTranspose2d）算子** → 解码器/超分上采样用「**最近邻上采样 + 普通卷积**」：`Tensor::upsample_nearest2d(target_h, target_w)?`（实测 `[N,C,h,w]→[N,C,H,W]`）后接一层 `Conv2d` 抹平马赛克。转置卷积/PixelShuffle 都可用这套替代讲清。
  - `Tensor::avg_pool2d(sz)` / `max_pool2d(sz)`（kernel==stride，`[N,C,H,W]→[N,C,H/sz,W/sz]`）；`candle_nn::ops::sigmoid(&t)?` 实测可用。
  - **重建/回归任务稳妥组合**：末层 `sigmoid` 压到 `[0,1]` + **MSE**（`(pred - target)?.sqr()?.mean_all()?`，`&Tensor - &Tensor` 直接可用返回 Result）。**避开 BCE**（见本节数值不稳定坑）。输入值域必须与末层激活对齐（sigmoid 输出 [0,1] → 输入也 [0,1]，别再做 [-1,1] 标准化）。
  - **残差学习**易训易稳：SRCNN `out = x + &net(x)`（在双线性基线上学锐化残差）、DnCNN `clean = noisy - &net(noisy)`（学噪声再相减）。起点≈基线 → PSNR 只升不降。
  - **冻结编码器/骨干做表示学习**：与 ch43 同法——`unsafe { VarBuilder::from_mmaped_safetensors(&["ae.safetensors"], DType::F32, &dev)? }` 加载出常量张量（非 Var，天然冻结）；只 `.pp("enc")` 请求编码器键，文件里 `dec.*` 键静默忽略。实测：8 维 latent + 线性探针在少样本(每类8张)下 85.0% > 256 原始像素 75.8%。
  - **image 0.25 拼对照图**：`RgbImage::from_raw(w,h,vec)->Option`（interleaved RGB）、`GrayImage::from_fn`、`imageops::resize(&img,w,h,FilterType::Nearest)`（放大显示）、`imageops::overlay(&mut bottom, &top, x:i64, y:i64)`（坐标 i64）。PSNR 纯手写 `-10*mse.log10()`（值域 [0,1]，MAX=1）；比较前所有图先 `clamp(0,1)` 才公平。

### 11. 纯 Rust 视频能力边界与相关 crate（ch74-77 实测，2026-08）

- **`image` 0.25.10 多帧 GIF 编码**（gif 是**默认 feature**，无需额外开启；底层 `gif 0.14.2`）：`image::codecs::gif::{GifEncoder, Repeat}` + `image::{Frame, Delay}`。坑：① `Frame::from_parts(buffer, left, top, delay)` 的 `buffer` **必须是 `RgbaImage`**（`RgbImage` 先 `DynamicImage::ImageRgb8(rgb).to_rgba8()`）；② 帧延迟 `Delay::from_numer_denom_ms(ms, 1)`；③ `enc.set_repeat(Repeat::Infinite)?`。GIF/APNG 多帧是纯 Rust 唯一能顺畅读写的"视频"形态。
- **`retina` 0.4.20** — 纯 Rust RTSP/RTP 客户端（`Session::describe→setup→play→demuxed`，**异步、需 tokio**）。能收流并把 RTP 重组为帧，但 `CodecItem::VideoFrame::data()` 返回的是**编码后 H.264 NAL，不是像素**。
- **纯 Rust 能力边界（诚实核心）**：通用视频解码（H.264/H.265/mp4）**无成熟纯 Rust 方案**；`image::open` 不认 mp4；常用解码器 `openh264`(Cisco,C)/`dav1d`(AV1,C) 非纯 R；`symphonia`（纯 R）**只解音频不解视频**。务实路线：① `ffmpeg` 子进程（`std::process::Command`）抽帧成 PNG/裸 RGB 喂 Rust（最通用，你的 Rust 仍纯 R，但依赖系统 ffmpeg）；② `ffmpeg-next` FFI（进程内解码，非纯 R）。
- **实时管线（std 足够）**：`mpsc::channel`（无界）演示延迟累积；`sync_channel(cap)`+`try_send`（满则 `TrySendError::Full`）= 丢最新帧；`Arc<(Mutex<VecDeque>, Condvar)>` 手搓 = 丢最旧帧。工程红线：勿一次性 load 整段视频（1min 1080p 全 load 即 10GB+，要流式）；VFR/RTSP 信源头时间戳（PTS/RTP ts）而非"帧号÷标称 fps"；RTSP over TCP(稳)/UDP(低延迟)；断流自动重连+状态重置。ch76/77 用帧差运动能量+MHI、IoU 关联做跨帧 ID 的最小 tracker（复用 EVENTS_MODEL 的 BBox/Detection/Frame）。

### 12. axum/tokio 视觉服务栈（ch96 实测，Rust 1.97.1 macOS，2026-08；ch107 补充进程内测试 / 背压借用坑 / tract 模型共享）

- 版本（`cargo add` 解析 + `cargo build` 通过）：`axum 0.8`(实测0.8.9) / `tokio 1`(1.53.1，features `rt-multi-thread,macros,net,signal,time`，`spawn_blocking` 随 rt 提供) / `tower-http 0.7`(0.7.0，features `trace,timeout`) / `serde 1`/`serde_json 1`/`tracing 0.1`+`tracing-subscriber 0.3`(env-filter)/`image 0.25`(与主工程同版)。
- 启服务：`let listener = TcpListener::bind(addr).await?; axum::serve(listener, app).with_graceful_shutdown(sig).await?;`（0.8 用 **`axum::serve`**，非旧 `Server::bind`）。
- 状态共享：`.with_state(Arc::new(state))` + handler 参数 `State(s): State<Arc<AppState>>`；模型只 load 一次入 State。原始图片体用提取器 `body: axum::body::Bytes`（**放最后一个参数**）。
- CPU 密集必须 `tokio::task::spawn_blocking(move || …).await.map_err(|_|…)??`（**双 `??`**：外 JoinError→500，内业务错→400）。
- 体积上限：`axum::extract::DefaultBodyLimit::max(n)`（超限自动 413；默认 ~2MB）。超时：`tower_http::timeout::TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, dur)` —— ⚠️ `TimeoutLayer::new(dur)` 在 0.7.0 已 **deprecated**。
- 背压：`tokio::sync::Semaphore`+`try_acquire()`，抢不到→503（呼应 ch92）；⚠️ 已进入 `spawn_blocking` 的任务无法被超时/客户端取消中断。错误映射：自定义 enum impl `IntoResponse` → `(StatusCode, Json<Value>)`。隔离：独立工程 Cargo.toml **必须空 `[workspace]`**（否则被上层 code/ 的 cv_examples 吸入）。
- **进程内测试（ch107 实测，CI 防回归首选）**：axum 0.8 的 `Router` 本身即 `tower::Service` → `app.clone().oneshot(req).await?` 直接把请求打进路由拿到 `Response`，再 `axum::body::to_bytes(resp.into_body(), usize::MAX).await?` 读回体断言。**不占端口、不联网、进程内**，同一套断言让 `cargo run`（冒烟）与 `cargo test`（CI）复用。需给 `tower` 开 `util` feature（`oneshot` 在此）。ch107 实测冒烟 7/7（health / 红绿蓝三分类 conf 0.97–0.99 / 非法 base64→400 / 空 body→400 / 背压→503）。
- **🕳️ Semaphore 许可 vs 把 `Arc<AppState>` move 进 `spawn_blocking`（ch107 实测借用冲突）**：`let _permit = state.limiter.acquire().await?;` 借用了 `state.limiter`；若同一作用域又把整个 `state` **move** 进 `spawn_blocking(move || …)` 的 `'static` 闭包 → 借用检查器报冲突（`_permit` 的借用仍活着）。**修法**：handler 持 `&state`（保住 permit 存活到响应返回），在辅助函数里 `Arc::clone` 出模型句柄（`Arc<TypedRunnableModel>`，见 §14）给 `'static` 闭包，闭包只捕获 clone 出来的 `Arc`、不碰 `state` 整体。CPU 密集推理 `spawn_blocking` 里跑，外层 `??`（JoinError→500 / 业务错→400）。

### 13. 纯 Rust GPU（wgpu 30）与 数据/MLOps（serde COCO）（ch109/113 实测，Apple M1 Pro）

- GPU crate：`wgpu = "30.0.1"` + `pollster = "1.0.1"` + `bytemuck = "1.25.2"`(derive)。wgpu 默认开 metal/vulkan/dx12/gles/wgsl。放独立工程 `code/gpu_demo/`。
- **wgpu 30 API 变动（老资料会踩）**：`Instance::request_adapter(...)` 返回 **`Result`**（非 Option），headless→`Err` 分支优雅跳过（不 panic）；`adapter.request_device(&DeviceDescriptor)` **只收一个参数**（旧版第二个 trace 参数已去）；`device.poll(wgpu::PollType::wait_indefinitely())`（旧 `Maintain::Wait` 已改）；`buffer_slice.get_mapped_range()` 返回 **`Result`** 需 `unwrap()`；`ComputePipelineDescriptor.entry_point: Option<&str>`，新增 `compilation_options`/`cache` 字段。
- **GPU 能力边界**：纯 Rust 通用 GPU 计算＝wgpu 写 WGSL compute（跨 Metal/Vulkan/DX12/GL，可上 wasm）；训练/推理上 GPU＝`burn-wgpu` 一行换后端（`Autodiff<NdArray>`→`Autodiff<Wgpu>`，呼应 ch40）。**CUDA 路线（`cust`/candle cuda feature）非纯 R、绑 NVIDIA、依赖 C 运行库**；纯 R 缺 cuBLAS/cuDNN 对等物，极致性能仍属 CUDA。🎯 实测教训：① 低算术强度轻算子在**独显(PCIe)上常输 CPU**（搬运主导），统一内存(M1)才让灰度也险胜（GPU 暖 0.36ms vs 向量化 CPU 0.79ms）；重算子(每像素300次)GPU 11.6ms vs CPU 10270ms ≈ **884×**。② GPU 结果**必须与 CPU 参考对拍 + 容差**（f32 舍入误差随计算深度累积，本例最大误差 1→16）。③ benchmark **必须预热**（冷启 3.89ms vs 暖 0.36ms）。
- **数据/MLOps**：`serde = "1.0.229"`(derive) + `serde_json = "1.0.151"` 纯 R，**已入主 code/Cargo.toml**，与 image0.25/imageproc0.27 共存无冲突（ch93 规则引擎也将复用）。COCO 用 `#[derive(Deserialize)]` 三表(images/annotations/categories)+`serde_json::from_str` 一行解析；bbox 存 `[f64;4]`。**能力边界**：成熟 MLOps(MLflow/W&B/DVC/Airflow) 主力在 Python，**纯 R 无对标成熟替代，不宜硬造**；Rust 诚实定位＝高性能数据流水线/预处理/serving（呼应 ch96 axum），接入 Python 中枢。划分纪律(按 image_id/视频/主体/设备/时间分组防泄漏)、类别不平衡检测、坐标转换(xywh/xyxy/YOLO归一化) 均可纯 R 高效实测。

### 14. tract 0.23.5 ONNX 推理 + 纯 Rust 造 .onnx + 量化边界（ch94/95 实测，ch107 补充服务共享/内存加载/纯度星号）

- **tract 加载/推理流程**（实测跑通）：`tract_onnx::onnx().model_for_path(p)?.with_input_fact(0, f32::fact([1,4]).into())?.into_optimized()?.into_runnable()?`；输入 `Tensor::from_shape(&[1,4], &[..])?` → `model.run(tvec!(input.into()))?`；取输出 `result[0].to_plain_array_view::<f32>()?`（**证实：`to_array_view` 已移除 → 用 `to_plain_array_view`**）。
- **🎯 重要新发现：`tract_onnx::pb` 模块是公开的**，导出全套 ONNX protobuf 类型（`ModelProto/GraphProto/NodeProto/TensorProto/ValueInfoProto/TypeProto/OperatorSetIdProto`，prost 生成）。在 Cargo.toml 加 **`prost = "0.14"`**（须与 tract 内部 prost 0.14.4 统一，`Message::encode` 才匹配），即可**在纯 Rust 里构造 `ModelProto`→`.encode()` 成 `.onnx`→再用 tract 加载回来**——"零下载、可复现地造一个极小测试 .onnx"的正解（ch94 的真实输出即由此，无需任何外部模型）。关键字段：`TensorProto.float_data`(f32 权重)、`data_type = pb::tensor_proto::DataType::Float as i32`、`NodeProto{input,output,op_type}`、`opset_import = OperatorSetIdProto{domain:"".into(), version:13}`、`ir_version:7`。该独立工程用 edition 2021 + 空 `[workspace]`。
- **量化能力边界（诚实）**：**无成熟纯 Rust 量化工具链**（PTQ 校准 / QAT 训练都在 Python：PyTorch/ORT/TensorRT）。推理侧 tract 有一定量化 ONNX 支持、candle 能加载 GGUF int8/int4 权重。主流路径＝**训练侧量化好→导出量化 ONNX/GGUF→Rust(tract/candle)加载推理**。手写 int8（对称/非对称、per-tensor/per-channel、MSE/最大误差）是纯 std 教学实现，用于理解原理（实测：对称量化省 3/4 体积 MSE 6e-6；3σ 截断主体 MSE 降 44× 但离群最大误差飙到 0.94；per-channel 小尺度通道改善 8911×；非对称误差约对称一半）。
- **🎯 tract 模型共享到 Web 服务的类型 + 内存加载（ch107 实测）**：① `into_runnable()` 返回的是 **`Arc<TypedRunnableModel>`**（不是裸值），`run` 签名为 `fn run(self: &Arc<Self>, ...)` → 放进 `Arc<AppState>` 共享状态时字段类型写 `type M = Arc<TypedRunnableModel>;`，启动只 `load` 一次、各请求 clone `Arc` 即可。**`TypedRunnableModel` 是零泛型别名**（`= RunnableModel<TypedFact, Box<dyn TypedOp>>`），误写成 `TypedRunnableModel<TypedModel>` 会编译报错。② 服务里从**内存字节**直接加载 `.onnx`：`tract_onnx::onnx().model_for_read(&mut std::io::Cursor::new(&bytes))?`——无需像 ch94 那样先落盘再 `model_for_path`。
- **⚠️ 「tract 纯 Rust、零 C/C++」需加一个精确星号（ch107 实测更正，附录 A/D 须据此措辞）**：`tract-linalg 0.23.5` 有一个**构建期** `cc` 依赖，但它只当**汇编器**用——编译 tract **自带的 SIMD 汇编内核**（`arm64/*.S`、`x86_64/avx512*/*.S`，仓库里 **0 个 `.cpp`**）。**没有 `-sys` crate、不链接任何外部 C/C++ 运行库、产物是自包含单二进制**，与 ort 链接外部 C++ onnxruntime 是**性质完全不同**的两件事。所以对本书"纯 Rust / 无 OpenCV / 可交叉编译 / 自包含"的诉求，tract **依然成立、依然是首选**；星号只诚实交代"构建期会调系统汇编器编译它自己的 asm"。**为何早先没察觉**：§2/§14 当初用 `cargo check` 验证，而 `check` 不完整跑汇编内核的构建步骤，故未触发 cc（改用 `cargo build`/`run` 才会）。**交叉编译影响**：目标平台的汇编器需支持相应指令集，否则 tract **回退到通用纯 Rust 内核**（能跑，慢一些）——这反而印证它不依赖外部 C 库。ch107 已把这点作为核心 🕳️ 写进正文；ch94 若有"零 C++"绝对化措辞，终检时补一句同款星号。

### 15. candle 检测/多头/关键点热图 实测补充（ch46/47 确认，供 ch50/53/56 等复用）

- **多通道输出头切分/组装**：`Tensor::narrow(dim, start, len)` 沿通道切出 obj/box/cls（或每个关键点热图）三段；`Tensor::cat(&[&a, &b, ..], dim)` 拼回；`Tensor::i(k)`（需 `use candle_core::IndexOp`）取 batch 第 k 张；`Tensor::to_vec3::<f32>()` 取回三维数据在纯 Rust 里解码峰值/框；`argmax(D::Minus1)`→U32 取每像素/每格类别。
- **1×1 卷积输出头模式**（检测头、关键点热图头通用）：`conv2d(C_in, A_out, 1, Conv2dConfig{padding:0, ..Default::default()}, vb.pp("head"))` → 权重键 `head.weight` 形状 `[A_out, C_in, 1, 1]`。检测 `A_out = 每格(obj+4框+类别数)`；关键点 `A_out = 关键点数`（每点一张热图通道）。
- **标量/加权运算**：`Tensor::affine(mul, add)`（= `mul*x + add`，对 0 维标量张量也работает）用于 `+1`、按标量缩放、把 obj/box/cls 三项**加权成一个标量 loss** 反向；`Tensor::broadcast_add/sub/mul`（含与标量张量广播）。
- **训练技巧**：`AdamW::set_learning_rate(f64)` 可**训练中途改 lr**（做 lr 衰减，实测明显提升 objectness 锐度、加快收敛）。检测 objectness/类别损失、比对二分类都用**自写数值稳定 BCE** `max(x,0)-x*y+log(1+exp(-|x|))`（复用 §10 结论，**别用** `binary_cross_entropy_with_logit`）；框回归用 L1/MSE。
- **🎯 合成数据划分泄漏的隐蔽陷阱**（ch47 实测）：卷积网络**平移等变（translation-equivariant）**，因此"只改目标位置"的近似重复帧几乎**不会**让随机划分的指标虚高——这会掩盖泄漏问题。要让"随机划分 vs 按场景分组划分"的差距**显形**，需**减少场景数 + 每场景放多个近似重复帧 + 引入难样本（小目标/细条/随机亮度）**；ch47 调到 16 场景×12 帧才得到 mAP 0.945→0.705（24 点）的清晰泄漏差距。做泄漏教学实验时务必这样构造数据，否则会误以为"随机划分没问题"。

### 16. candle 人脸检测/关键点回归/ArcFace 度量学习 实测补充（ch56/57 确认，ch54/55 补充 FC头/热图分离/Umeyama镜像，供 ch58 复用）

- **🎯 1×1 卷积头直接回归"绝对坐标"学不动（平移不变陷阱，ch56 实测，务必牢记）**：关键点/人脸五点 landmark 若让 1×1 卷积头直接输出**绝对像素坐标**，损失会**死卡**（ch56 实测卡在 lmk loss 0.12 不降）——因为卷积**平移不变**，同样的局部特征在图像任何位置都给同样输出，无法表达"这个点在图像哪个绝对位置"。**正解**：像检测框中心那样**显式注入格子坐标**——每个格子输出**相对该格中心的偏移量**，再 `+ 格中心绝对坐标` 还原（grid-center + 线性 offset）。ch56 改此法后 lmk loss 0.12 → **0.0066（≈1.26px）**。这是 §15 ch47 平移等变洞见的又一次落地。**替代方案**：用热图回归（§15，天然平移等变，ch52/53 已验证），则无此问题。
- **🎯 ArcFace `sinθ=√(1-cos²θ)` 浮点负数开根出 NaN（ch57 实测）**：`cosθ` 由归一化点积得来，理论 ∈[-1,1]，但浮点误差可使 `1-cos²θ` **略小于 0**，`sqrt` 直接出 NaN 污染整个训练。**正解**：开根前先 `.relu()` 夹到 ≥0——`(1.0 - cos.sqr()?)?.relu()?.sqrt()?`。同理任何"理论非负、浮点可能微负"的开根/对数都要先 clamp/relu。
- **ArcFace 角度间隔（additive angular margin）实现要点**（ch57 实测跑通）：embedding 与类中心权重**都 L2 归一化**（`x / x.sqr()?.sum_keepdim(D)?.sqrt()?` 广播相除），点积即 `cosθ`；对**真实类**那一项把角度加间隔 `m`：`cos(θ+m)=cosθ·cos m − sinθ·sin m`（`sinθ` 用上面 relu-sqrt 求），非真实类保持 `cosθ`；整体 `×s`（尺度，实测 s≈16~30）再送 `cross_entropy`（target U32）。消融 m=0 vs m=0.5：开集未见身份 TAR@FAR=1% **49.1%→69.3%**、验证准确率 94.7%→96.3%——**间隔是把类内压紧、类间推开的关键**，单变量对照很有说服力。
- **确认可用的 candle 算子（ch56/57 新增）**：`Tensor::sqr()`、`sqrt()`、`sum_keepdim(dim)`、`clamp(min,max)`、`maximum(&other)`（逐元素取大，可配 BCE）、`affine(mul,add)`、`broadcast_add/sub/mul/div`；L2 归一化、点积（`matmul` 或 `broadcast_mul`+`sum`）全部纯 candle 搭出，无需额外库。
- **多任务分类头**（ch58 人脸属性将复用）：一个共享骨干 + 多个 `linear`/1×1 头，各头各算 `cross_entropy` 后**加权求和成一个标量 loss** 反向（`affine`/`broadcast` 加权，同 §15 检测多项 loss 合并法）。
- **可复现与训练规模**（ch56/57 实测）：确定性 `Lcg` + 固定数据种子 → 完全可复现；**48×48 是纯 CPU candle 的可训练"甜点"尺寸**（ch57 48×48/288 图/200 epoch ≈ 2:48），64×64 起训练明显变慢，教学示例宜守在 48×48 上下。
- **🎯 FC（全连接）头能学"绝对坐标"，1×1 卷积头不能（ch54 实测，与上条互补收口）**：ch56 证明 **1×1 卷积头**回归绝对坐标死卡（平移不变）；ch54 反向验证——把关键点回归头换成**全连接 `Linear`**（先 `flatten` 空间维再全连接），**可以**直接回归绝对坐标（人脸五点 MSE 0.283831 → **0.000074**，定位 **1.113px**），因为全连接权重对每个空间位置各不相同、天然打破平移不变。**决策**：绝对坐标回归 → 用 FC 头或注入格子坐标；任意位置/多目标 → 用热图（§15）。三条路各有适用，别再拿 1×1 卷积头硬回归绝对坐标。
- **🎯 关键点热图目标点必须"空间分离"（ch55 血泪坑，供所有热图关键点章防）**：手部 11 点若把**弯曲手指的指尖都折向掌心/手腕**，多个目标点在图上**堆叠重合** → 热图互相污染 → 定位崩（ch55 v1 实测 2.29px、手势仅 47%）。**修法**：弯曲指尖折向**各自的指根**（分散开）+ 加数据(300)/轮次(250) → **1.26px、手势 89%**。合成关键点数据时务必让目标点彼此拉开距离；这是 ch53"减场景/加难样本"合成陷阱的姊妹坑。
- **🎯 nalgebra 0.35 二维相似变换（Umeyama/仿射对齐）必须做镜像校正（ch54 实测）**：由匹配点对求旋转+缩放+平移把人脸摆正时，`Matrix2::svd(true, true)` 拿到 `u / v_t / singular_values`，旋转 `R = U·Vᵀ`。但 **SVD 不保证 `det(R)>0`——`det(U·Vᵀ)<0` 时 `R` 是"反射"（对齐结果是镜像脸）**。必须校正：构造 `S = Matrix2::new(1,0, 0, d)`（`d = if det(U·Vᵀ) < 0 { -1 } else { 1 }`），令 `R = U·S·Vᵀ`（等价于翻转最后一个奇异方向）。ch54 加此校正后对齐残差 0.887px，脸方向正确；不加则整脸镜像翻转。**凡用 SVD 求刚体/相似变换（点云配准、手眼、拼接）都要这一步。**
- **应用性质章可不重推理论**（ch54 对齐、ch55 手势判定）：几何规则（关节角度阈值、五点仿射对齐）用纯 `nalgebra`/std 实现即可；教学重点转向**"感知 vs 逻辑"的瓶颈定位**——ch55 用「GT 关键点跑规则 100% vs 预测关键点跑规则 89%」两条曲线，清楚指出误差来自感知而非规则，是应用章的好范式。
- **🎯 关节角在"接近伸直"处病态（ill-conditioned，ch105 实测，凡由近共线三点算夹角都要防）**：动作计数用关节角（膝/肘/髋）+ 滞回状态机时，腿/臂**快伸直**时三关键点近共线，关键点微小垂直噪声→夹角**剧烈跳变**（ch105 实测 σ=2px 时滞回计数从 97.5% 崩到 2.5%）。**两条对策**：① 角度信号**先时间平滑再送状态机**（把 σ=2px 救回 65%，呼应 ch85 去抖）；② **上阈值别贴满伸直**（设 150° 而非 175°，给病态区留余量）。抬手、伸肘、任何"三点夹角"判定同理。配套教学法：量出**感知噪声**落在哪个"容噪档"、**逻辑容噪区间**多宽，一比即知系统被感知还是逻辑卡住（σ≥3px 全崩＝感知是硬瓶颈）。

### 17. candle 逐像素语义/实例分割 实测补充（ch49/51 确认，ch50/ch102 复用并补充类别不平衡加权 CE）

- **逐像素分割交叉熵配方**（实测）：logits `[N,C,H,W]` → `.permute((0,2,3,1))?.contiguous()?.reshape((n*h*w, c))?`；target `[N,H,W]` U32 → `.reshape((n*h*w,))?`；再 `candle_nn::loss::cross_entropy`。**⚠️ permute 后必须 `.contiguous()`**，否则 reshape 因内存不连续报错。
- **🎯 类别不平衡→自写加权逐像素 CE（ch102 血泪坑，缺陷/小目标分割必看）**：candle 的 `cross_entropy` / `nll` **没有 class-weight 参数**（PyTorch `CrossEntropyLoss(weight=…)` 在 candle 无对应）。当正类稀少（ch102 缺陷仅占 **7.3%** 像素）时普通 CE **"躺平判全负"**——缺陷召回卡在 89.5% 且 loss 假性很低。**自写加权逐像素 CE**：① `let logp = candle_nn::ops::log_softmax(&logits2d, 1)?;`（`logits2d`=上面 reshape 成 `[n*h*w, C]` 的）② 取真类 log 概率 `let picked = logp.gather(&target_u32.unsqueeze(1)?, 1)?.squeeze(1)?;`（`gather` 的 index 必须 **U32** 且多一维）③ 造**逐像素权重** `let w = target_f32.affine(W-1.0, 1.0)?;`（二分类：正类权重 `W`、负类 `1`；`target_f32` 是 0/1 掩码，`affine` 得 `(W-1)*t+1`）④ 加权归一 `picked.neg()?.mul(&w)?.sum_all()? / w.sum_all()?`（**除以权重之和**而非像素数，才是真正的加权平均）。ch102 实测 `W=4`：缺陷级召回 **89.5%→99.1%（113/114）、0 误报**。多类可把 `w` 换成按 `target` 查表的权重张量（`index_select` 一张 `[C]` 权重表）。
- **解码器上采样**（candle 无 ConvTranspose2d）：`Tensor::upsample_nearest2d(h,w)`（最近邻）与 `Tensor::interpolate2d(h,w)` 前向均可用作替代；U-Net 解码器 = 上采样 + `Tensor::cat(&[&up, &skip], 1)` 通道维 skip-concat + Conv2d。
- **🎯 上采样的反向传播强约束（ch50 血泪坑，训练解码器/超分/GAN/扩散务必牢记）**：`upsample_nearest2d` 的**反向只支持整数且均匀的放大倍数**（非整数会 `backward not supported for non integer upscaling factors`）；`upsample_bilinear2d` **完全没有反向**（`bail!("backward not supported for upsample_bilinear2d")`）。**后果**：① 训练里的上采样只能走**整数倍最近邻**（如每级 2×：7→14→28、8→16→32），别用双线性、别用任意目标尺寸；② **真正的特征级 RoIAlign**（把任意大小特征框 resize 到固定尺寸）**训练时反向必炸**。**绕过 RoIAlign**：把 RoI 的 resize 放到 **Rust 图像侧**做（`image::imageops::crop_imm` + `resize`，不进 autograd），网络内部只做整数倍上采样（代价：掩码头不共享骨干特征——教学取舍，正文诚实说明）。（前向随便用；只有**参与训练反向**的上采样受此限制。ch49/51 之所以没踩到，是因其 U-Net 恰好都是 2× 整数上采样。）
- **空洞/膨胀卷积（DeepLab）**：`Conv2dConfig{ padding, dilation, ..Default::default() }`；dilation=2+padding=2 使 3×3 卷积**保持分辨率、感受野扩到 5×5、参数量不变**（实测输出 `[1,8,8,8]`、576 权重）。感受野公式 `RF = 1 + (k-1)*d`。
- **skip 消融的干净做法**：`Tensor::zeros_like()` 把 skip 支路送零 → 有/无 skip 两模型权重形状一致、可公平对比（实测 mIoU 0.970 vs 0.414，+0.556）。
- **其它确认可用**：`dims4()->(usize,usize,usize,usize)`；`argmax(0)` 对 `[C,H,W]`→`[H,W]` U32（逐像素类别图）；`to_vec2::<u32>()`/`to_vec3::<u32>()` 读掩码；`Conv2d::weight().elem_count()` 数参数量。
- **交互式分割（迷你 SAM）**：输入拼**提示通道**（RGB + 点热力图 = 4 通道）→ 复用 U-Net 输出 2 类前景/背景；同图不同点→不同掩码（点条件化，实测前景 IoU 0.951）。经典**种子区域生长**（纯 `image`、无需网络）作对照：点/框提示 + 阈值敏感性（阈值过松→漏进背景，框提示可救回）。
- **诚实边界**：真实 SAM 用数百 MB~GB ViT 编码器，纯 Rust CPU 一键跑不动；务实路线＝candle 官方 `segment-anything` example（需下载权重）。

### 18. ⚠️ 全书通用：自造 LCG 取离散随机务必用高位（ch58 血泪坑）

- **现象**（ch58 实测）：用线性同余 `Lcg` 时 `next() % k` / `next() & 1` 取离散选择会**退化**——LCG 的**最低位周期极短**（严格 0/1 交替），叠加"每样本固定步长采样"，导致**某一整类（"屏幕翻拍"）一个样本都没生成**、布尔属性非随机。这是最隐蔽的合成数据坑，**凡用自造确定性 RNG 造数据的章节都要防**（不限 candle，std 合成数据同样中招）。
- **正解**：离散随机一律**从高位派生**——先取 `unit()`（内部用 `(state >> 11)` 高位转 `[0,1)` 浮点），再 `boolean() = unit() < 0.5`、`below(n) = (unit() * n as f64) as usize`。**别直接对 state 低位做 `% k` / `& 1`**。
- **自查**：造完数据先打印**每类/每标签的计数直方图**，确认无空类、分布大致均匀，再开训——比训练后才发现某类缺失省事得多。
- 顺带（ch58）：手工高频/纹理特征要在**平滑区（如脸颊 patch）**测，别在整脸测（眼/嘴强边缘的拉普拉斯会淹没摩尔纹等弱信号）；`image` 坐标 y 向下，画"微笑 ∪ 形"用 `my + amp`（`my - amp` 画出的是皱眉）。

### 19. candle CLIP / 多模态 / 对比学习 InfoNCE 实测补充（ch73 确认，供 ch100 以图搜图 及检索/嵌入类章复用）

- **可微词嵌入查表**：`Tensor::index_select(&flat_u32_ids, 0)` 从一张当作 `Var` 的词表张量取词向量，**梯度正常回流**（训练收敛即证），无需 `candle_nn::Embedding` 层——纯 candle 最透明写法。
- **🎯 对称 InfoNCE 的转置坑**：图→文用 `logits`，文→图用 `logits.t()?`——但 **`.t()` 结果非连续**，直接送 `cross_entropy`（内部 `log_softmax`）会报错，须接 `.contiguous()?`（同 §17 permute 后规则）。对角线 target 用 `Tensor::arange(0u32, n as u32, dev)?`（**必须写 `0u32` 才是 U32**）；温度缩放用 `.affine(1.0/τ, 0.0)`；两方向 loss 取平均。
- **`Tensor::mean(dim)` 会移除该维**（`[N,L,E]→[N,E]`），正好做序列/空间**均值池化**得句/图嵌入。
- **L2 归一化防除零**：eps 加在 **sqrt 之前**——`x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, 1e-12)?.sqrt()?` 再 `broadcast_div`；嵌入检索余弦相似度 = 两个 L2 归一化向量点积（`broadcast_mul`+`sum` 或 `matmul`）。
- **🎯 方法学坑（非 API，检索/对比学习通用）**：词表/描述极小时，InfoNCE 要求 **batch 内文本互不相同**——否则同类样本共享同一描述会制造"对角线外的隐藏正样本"，把它们当负样本推开 → 损失自相矛盾、学不动。**对策**：每步每类只取一张，**batch 大小 = 类别数**（ch73 全训 9、留出后 6）。造检索评测数据同理：类内要多样、类间要可分，才能让"嵌入 vs 原始像素"的对照显形（ch73 组合泛化："颜色 100% 迁移、形状 55% 部分迁移"是最有说服力的诚实教学点）。
- **🎯 "嵌入 vs 原始像素"对照数据设计是成败关键（ch100 实测，检索/表示学习章通用）**：要让"学习嵌入"明显胜过"原始像素"，合成图库必须同时 (a) **让位置成为主要干扰**（紧凑小物体 + 大画布 + 随机 2D 位置，使原始像素按"位置重叠"乱配）、(b) **堵掉颜色正交后门**（三种颜色都点亮**全部通道**如 `[1.0,0.35,0.35]`，而非各占一个 R/G/B 通道——否则颜色免费线性可分，原始像素也刷满分，对照失效）。ch100 初版（颜色各占一通道 + 全长条 + 稠密图库）→ 原始像素也 99.4%/100% 与嵌入并列；改造后才拉开 **嵌入 32 维 98.6% vs 原始像素 1728 维 45.3%**。同 ch47/ch53/ch57 "让泄漏/差距显形"的合成设计家族。
- **🎯 检索类务必报 precision@K，别只报 top-1（ch100 实测）**：原始像素 top-1=84.7% 看着尚可，precision@5 却崩到 45.3%——只看 top-1 会掩盖"前几名里混入大量错检"。检索/推荐一律同时报 precision@K / recall@K。
- **合成集索引小坑**：每类样本连续排布时，类 `c` 的第 1 张实例在下标 `c*per_class`（不是 `c`）；可视化选代表图时易错。检索余弦矩阵一把出：`query @ gallery.t()`（两侧先 L2 归一化），再逐行 argsort 取 top-K。

### 20. P20 集成/级联项目 实测教训（ch101 人脸考勤确立 + ch97/103/104 事件集成 + ch99 OCR 域偏移补充；供附录 D）

- **🎯 级联域偏移（cascade domain gap）——集成项目头号杀手（ch101 血泪坑）**：多模型串联（检测→对齐→活体→识别）时，**下游模型必须在上游模型的真实输出上训练/注册/验证/推理**，四个环节走**同一条链**。踩坑现象：用**真值关键点**裁剪训练识别头、却用**检测器输出关键点**裁剪推理 → 员工放行率 **0/29**、本人相似度从验证 0.84 崩到 0.3，**且不报任何错**（最阴险：静默失效）。修法：训练/注册/验证/推理裁剪全部走 `detect→align` 同一链 → 放行率 0%→40.7%。**推广**：凡"A 模型输出喂给 B 模型"，B 的训练分布必须 = A 的真实输出分布，不能用干净的真值代替（呼应 §17 ch102 缺陷分割、OCR 字符分类等一切"分割/检测→分类"的两段式管线）。
- **🎯 上游误差被下游放大 → 选鲁棒中间表示（ch101）**：按"双眼瞳距"（~7px 短基线）定人脸裁剪尺度，会把检测器 **1.3px** 关键点误差放大成 **~25%** 尺度抖动 → 识别崩。改用**多点相似变换对齐**（5 点最小二乘，平均掉单点噪声，即 ch54 做法）→ 放行率 40.7%→**83.3%**。教训：级联里中间表示要选**对上游噪声不敏感**的（多点最小二乘 > 两点基线；框中心 > 框角点）。
- **端到端按最终决策评估，别只看单级指标（ch101）**：检测 IoU 0.82、关键点 1.3px 单看都不错，但端到端放行率一度 0%——**单级指标全绿 ≠ 系统可用**。集成章一律报**最终业务决策**的混淆（放行/拒止 × 真人/陌生/假体），单级指标仅作体检。
- **安全类阈值主动往安全侧偏（APCER/BPCER 代价不对称，ch101）**：门禁/入侵/缺陷等场景，"漏"（放行假体/漏报缺陷）代价 ≫ "误"（拒真人/误报）。ch101 取阈值使陌生人/假体拒止 100%、0 事故，牺牲便利到 83.3%——正文应显式讲这个**主动偏置**的取舍，而非追求对称准确率。
- **流水线顺序即工程（ch101）**：把**便宜且能早拒**的环节前置（活体闸门放在昂贵的识别之前 → 既安全又省算力）；注册库 gallery 用**多张模板取平均**、支持增删员工不重训、规模化上向量检索（呼应 ch100）。
- **集成章诚实边界模板**：合成/受限演示打通"感知→逻辑→决策"闭环即达教学目的；生产还需真实传感（深度/红外活体等）、大规模底库、姿态/质量/遮挡处理、现场标定，以及**隐私合规红线**（人脸等生物特征）。复用 §16"感知 vs 逻辑瓶颈"叙事。
- **🎯 事件层线段几何两坑（ch103/104 实战调出，附录 D 收录）**：① **脚点恰好压在警戒线/绊线上→浮点退化**：严格线段相交判据 `d1·d2 < 0` 在端点压线时得 `0`（非 `<0`）→ 漏判越线（ch103 翻墙者"进院 0 次"被漏报）。对策：造数据/布线避开边界、放宽为 `≤` 或加迟滞带。② **绊线是线段不是无限直线**：目标在绊线**延长线外**竖直穿过应计 0（ch104 track#7 x=600>端点560）；必须用"**两组叉积同时严格异号**"的线段相交判据，用"点在直线哪侧"会误计线外穿越。
- **🎯 丢帧平移事件时间戳（ch97）**：管线丢帧后事件时间戳整体前移（实测某 IN 事件 920→880ms）——布尔型告警（进/出/越线）不受影响，但**测速/精确计时/时长统计类应用是硬伤**；对策：事件一律用**采集端真实时间戳**（`ts_ms` 来自采集而非帧序号推算）。
- **Stage 接口在完整管线要吃 `TrackedFrame`（ch97）**：EVENTS_MODEL 的 `Stage::process(&Frame)` 是 ch83 教学起点；真管线里绊线/ROI/停留都按 `track_id` 判定，故检测与事件之间由跟踪器把 `Frame` 升级为带稳定 id 的 `TrackedFrame` 再喂事件层。去重窗口（`track_id`+时间窗）只压抑**同一目标抖动重复穿线**，防不了 **track_id 断裂**（id 换新→同物体被重计）——诚实边界，靠更稳的跟踪器/ReID 缓解。
- **🎯 级联域偏移的 OCR 变体：长宽比失真（ch99 实测，§20 家族再添一例）**：用「**居中 + 保长宽比**」的漂亮字形训练字符分类器，接上真实分割器后**整串精确率从 85% 崩到 20.5%、且不报错**——因为分割器抠出的字形是「**紧贴外接框 + 拉伸填满**」（长宽比失真），分布与居中字形不一致。修法同 ch101/ch102：**下游分类器必须用分割器的真实产出训练**（同一「分割 → 归一化」链），逐字 69.6%→93.6%、整串 20.5%→85.0%。凡「分割/检测 → 分类」两段式（缺陷、车牌、OCR、票据）都要对齐这条链的几何归一化方式（居中 vs 拉伸、padding vs crop），否则上游产出与下游训练分布不一致，静默掉点。

### 21. candle GAN / 扩散 生成模型 实测补充（ch70/71 确认，P15 收官；供附录 D 及生成类复用）

- **🎯 GAN 模式崩溃三对照（ch70，教学最有说服力）**：6 离散模式小数据上——① 朴素(G:D=1:1、同 lr、无标签平滑)训练不稳、`D_loss` 一度跌 0.069、只覆盖 **1/6**；② 加稳定技巧(G 走 2 步、D lr 减半、真标签平滑到 0.9)样本变清晰(精度距离 0.551≈真实 0.326)、`D_loss` 稳在 ~0.7，但**仍崩到 1/6**——干净证明"稳定技巧治标不治本"；③ **条件 GAN**(类别 one-hot 同时喂 G 和 D)→ **覆盖 6/6**、召回距离 7.59→0.67、`D_loss≈1.38=2ln2`(真假各半的理论均衡点)。多样性用"精度/召回距离"(生成→真实最近邻 & 真实→生成最近邻)量化。
- **🎯 谱偏置（spectral bias）坑（ch71，数据几何要配网络归纳偏置）**：小 MLP 学"环形多模态"(9 团同半径、**靠角度区分**)只能覆盖**一半**模式，**加高频 Fourier 特征也救不回**；把布局换成"3×3 网格"(**靠坐标区分**、低频)→ 立刻 **9/9 全覆盖**。教训：多模态生成任务，数据的可分性方向要落在网络容易表达的低频/坐标轴上。
- **极小 DDPM 玩具要点（ch71）**：预测噪声 ε + MSE 损失(0.99→0.28)；`T=100`、`β 1e-4→0.10`，**小 T 必须把 βT 调大让累积 `√ᾱ_T→0`**(实测 0.075，否则终点不够接近纯噪声)；反向采样用 **fixedlarge 方差 `√β_t`** 利于覆盖各模式；实测采样均值对齐 (0,0)、9/9 均衡、88% 落目标流形。
- **cGAN 条件失效坑**：类别 one-hot **维度低**(6)时会被高维图像信号淹没 → 判别器直接忽略条件、条件化失效；把 one-hot 的 on 值**放大(×5)**再拼接才生效。
- **candle 0.11 API 补充**：`Tensor::detach()` 返回 **`Tensor`(非 `Result`)**；两个独立网络各持自己的 `AdamW`(各含各自 `Var`)→ `backward_step` 只更新本优化器的 Var，**G/D 交替训练无需靠 detach 保正确性**，detach 只为省算力(冻结生成图那半边计算图)。数值稳定 BCE 里标量 `x*y` 用 `logits.affine(y, 0.0)`；`mean_keepdim(0)`/`broadcast_sub`/`broadcast_as`/`Tensor::cat` 均可用(复用 §10 自写稳定 BCE `relu(x)-x*y+log(1+exp(-|x|))`)。
- **诚实 Stable Diffusion 边界（ch71）**：真 SD = 潜空间扩散 + CLIP 文本条件 + 交叉注意力 + 巨型 U-Net + CFG；candle 官方 example 需下载数 GB 权重、CPU 极慢 → 正文**只讲机制、不跑 SD、不编数字**(全书"纯 R 不足处诚实说明"基调)。
