# 第 3 章 · Rust 图像处理上手：image 与 ndarray

> **本章导读**：上一章我们把图像拆到了最小零件——像素，还手写双重循环遍历过它们。但工程里没人天天手搓字节数组。本章请出 Rust 视觉工程的两块基石：`image`（读写图片、做基础操作，地位相当于 Python 的 Pillow）和 `ndarray`（多维数组，也就是"张量"，相当于 Rust 版 NumPy）。学完本章，你能把一张 JPG 读进来、随手改像素、裁剪缩放存盘；更关键的是——把它和 `ndarray` 张量自由互转，并亲手把 **HWC** 布局翻成模型爱吃的 **CHW** 布局。这一步，正是第 10 章"模型预处理"的地基。

**学习目标**：

- 会用 `image`：`open` 读图、`to_rgb8` 拿到 `RgbImage`、`get_pixel`/`put_pixel`、裁剪、缩放、`save`
- 理解 `into_raw()` 拿到的一维字节，就是上一章说的"HWC、packed 内存"
- 会用 `ndarray`：`Array3`、`shape`、双方括号索引、切片、`ArrayView`
- 吃透本章重点：image ↔ ndarray 互转，以及 HWC ↔ CHW 的轴变换
- 能给图片画一个矩形框，为后面画检测框热身

## 3.1 起步：新建项目与两块基石

先把摊子支起来。新建一个 cargo 项目，加上两个核心依赖：

```bash
cargo new ch03_demo && cd ch03_demo
cargo add image ndarray
cargo add anyhow          # 示例里用它做简洁的错误处理（可选）
```

两个 crate 的分工，一句话记住：

- **`image`**：管"图片"这个概念——从文件读进来、按像素操作、存回磁盘。它懂 JPEG/PNG/WebP 等格式，懂"这是一张 RGB 图"。
- **`ndarray`**：管"一堆数字"这个概念——多维数组的创建、索引、切片、变形、批量运算。它不懂图片，只懂张量。

预处理的本质，就是把前者变成后者：一张图 → 一个符合模型输入形状的张量。本章就是打通这条路。再把上一章 2.5 节的类型梳理接上，画一张贯穿全章的"全景图"：

```text
image::open(path)  ──►  DynamicImage       "万金油"枚举（还不知道是 RGB 还是别的）
                            │  to_rgb8()
                            ▼
                        RgbImage  =  ImageBuffer<Rgb<u8>, Vec<u8>>
                            │  into_raw()
                            ▼
                        Vec<u8>             一维裸字节（HWC、packed）
                            │  Array3::from_shape_vec((h, w, 3), ..)
                            ▼
                        Array3<u8>  ──(permuted_axes)──►  CHW  ──(mapv/insert_axis)──►  Array4<f32> [1,3,H,W]
```

上面这条竖线，从磁盘一路走到"模型能吃的张量"。本章 3.2~3.5 讲上半段（image），3.6 讲 ndarray，3.7 打通中间那道最重要的转换。

## 3.2 image 入门：读图、尺寸、像素、保存

先把最常用的四件事跑通。准备一张 `test.jpg` 放在项目根目录（沿用第 1 章约定的那张测试图）：

```rust,ignore
use image::{open, Rgb};

fn main() -> anyhow::Result<()> {
    // 读图：得到"万金油"类型 DynamicImage
    let dyn_img = open("test.jpg")?;

    // 转成确定的 RGB8：RgbImage 就是 ImageBuffer<Rgb<u8>, Vec<u8>>
    let mut img = dyn_img.to_rgb8();

    // 尺寸：dimensions() 返回 (宽, 高)，注意是"宽在前"
    let (w, h) = img.dimensions();
    println!("宽 = {}, 高 = {}", w, h);

    // 读单个像素：get_pixel(x, y) 返回 &Rgb<u8>，可以像数组一样 p[0]/p[1]/p[2]
    let p = img.get_pixel(10, 20);
    println!("(10,20) = R{} G{} B{}", p[0], p[1], p[2]);

    // 写单个像素：put_pixel(x, y, 颜色)
    img.put_pixel(10, 20, Rgb([255, 0, 0]));   // 把这个点涂成纯红

    // 保存：扩展名决定编码器（.png 走 PNG，.jpg 走 JPEG）
    img.save("out.png")?;
    Ok(())
}
```

几个要点：

- **为什么先 `to_rgb8()`？** `open()` 读文件得到的是 `DynamicImage`——一个"可能是 RGB、可能是 RGBA、可能是灰度"的枚举（第 2 章 2.5 节讲过）。我们不想每次都判断类型，就在流程入口一次性转成确定的 `RgbImage`，之后所有代码都只面对"三通道 u8"这一种情况，简单又高效。
- **坐标约定**：`get_pixel(x, y)` 里 **x 是列、y 是行、原点在左上角**（第 2 章 2.2 节），别和数学课本的坐标搞混。
- **`dimensions()` 返回 `(宽, 高)`**，先宽后高。记牢这个顺序，3.7 节会有个专门坑在这上面。

## 3.3 遍历像素：转灰度与通道分离

逐像素遍历，最省心的是 `enumerate_pixels()` 迭代器，它一次给你 `(x, y, 像素)`，不用自己写双重循环下标。两个常见操作：

```rust,ignore
use image::{open, GrayImage, Luma};

fn main() -> anyhow::Result<()> {
    let img = open("test.jpg")?.to_rgb8();
    let (w, h) = img.dimensions();

    // (A) 转灰度：加权公式 Gray = 0.299R + 0.587G + 0.114B（原理见第 2 章 2.3 节与附录 A.5）
    //     其实 img.to_luma8() 一行就能得到灰度图，这里手写是为了看清过程
    let mut gray = GrayImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        let g = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        gray.put_pixel(x, y, Luma([g as u8]));   // 灰度图每像素只有一个通道
    }
    gray.save("gray.png")?;

    // (B) 通道分离：把 R 通道单独抠成一张灰度图（G、B 同理）
    let mut red = GrayImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        red.put_pixel(x, y, Luma([p[0]]));       // 只取 R 分量
    }
    red.save("red_channel.png")?;
    Ok(())
}
```

通道分离在调试时很有用：把 R/G/B 分别存出来看，能一眼发现"是不是通道顺序反了"（第 4 章会专门讲 BGR 这个经典坑）。

一句提醒：像素运算要当心 `u8` 溢出。`200u8 + 100` 在 debug 模式下会直接 panic。要给整图提亮，别 `*c = *c + 50`，用 `c.saturating_add(50)`（到 255 就封顶）。这类"会不会超 255、会不会低于 0"的判断，是图像代码最常见的错误来源。

## 3.4 裁剪与缩放

裁剪取 ROI（感兴趣区域，第 2 章 2.7 节），缩放改尺寸——两个高频几何操作：

```rust,ignore
use image::{open, imageops, imageops::FilterType};

fn main() -> anyhow::Result<()> {
    let dyn_img = open("test.jpg")?;

    // 缩放：这里只演示最基础的调用。resize 会"等比"缩到不超过 320×320
    //   —— 四种插值算法怎么选，留到第 9 章几何变换；
    //      接到模型输入的完整 letterbox 流程，留到第 10 章预处理。
    let small = dyn_img.resize(320, 320, FilterType::Triangle);
    small.save("small.png")?;

    // 裁剪：先转 RgbImage，再从 (x=50, y=30) 抠出一块 200×150 的 ROI
    //   crop_imm 不改原图，返回一个"子视图"，用 to_image() 落成新图再保存
    let img = dyn_img.to_rgb8();
    let roi = imageops::crop_imm(&img, 50, 30, 200, 150).to_image();
    roi.save("roi.png")?;
    Ok(())
}
```

- `crop_imm` 的 `imm` 是 immutable（不可变）——它不动原图，只给你一个"看子区域"的视图。所以要保存得先 `.to_image()` 把视图变成一张真正独立的图。
- `resize` 这里点到为止就好。真正喂模型时用的既不是等比 `resize` 也不是拉变形的 `resize_exact`，而是 **letterbox**（等比缩放后用灰边填充到目标尺寸），第 10 章专门拆解。

## 3.5 into_raw：把"图像就是一串字节"落到实处

上一章反复强调：一张 RGB 图在内存里就是一个 `Vec<u8>`，一行一行、每像素三字节地排着。现在用 `into_raw()` 亲眼验证：

```rust,ignore
use image::open;

fn main() -> anyhow::Result<()> {
    let img = open("test.jpg")?.to_rgb8();
    let (w, h) = img.dimensions();

    // into_raw()：拿到最底层的一维字节数组（会"消费"掉 img）
    let bytes: Vec<u8> = img.into_raw();

    println!("字节总数 = {}", bytes.len());   // 应等于 w * h * 3
    println!("w*h*3   = {}", w * h * 3);

    // 前 3 个字节 = 左上角像素 (0,0) 的 R、G、B
    println!("像素(0,0) = R{} G{} B{}", bytes[0], bytes[1], bytes[2]);
    // 索引公式（第 2 章 2.2 节）：像素 (x, y) 的第 c 个通道 = bytes[(y*w + x)*3 + c]
    Ok(())
}
```

这就是所谓的 **HWC + packed 布局**：

- **HWC**：字节按"先行 (Height)、再列 (Width)、最后通道 (Channel)"的顺序排。
- **packed（打包/交错）**：同一个像素的 R、G、B 三个字节是**挨在一起**的（`RGB RGB RGB…`），而不是"所有 R 放一堆、所有 G 放一堆"。

记住"HWC + packed"这个词，它是相机、显示器、`image` 库共同的天然布局。而 3.7 节你会看到，深度学习模型偏偏想要另一种排法。

## 3.6 ndarray 入门：张量的操作台

第二块基石登场。你会反复听到一个词——**张量（tensor）**，定义很朴素：

> **张量 = 多维数组。** 0 维是一个数，1 维是一条数组，2 维是表格，3 维是"一叠表格"（比如一张 `高×宽×通道` 的图），4 维是"一摞纸"（比如 `批×通道×高×宽` 的一批图）。

`ndarray` 就是操作这种多维数组的库。核心 API 一次演示清楚：

```rust,ignore
use ndarray::{s, Array3, ArrayView3, Axis};

fn main() {
    // 创建：Array3::<f32> 表示"3 维、元素是 f32"；括号里是形状 (高, 宽, 通道)
    let mut a = Array3::<f32>::zeros((4, 6, 3));    // 4 高 × 6 宽 × 3 通道，全填 0
    println!("shape = {:?}", a.shape());            // [4, 6, 3]
    println!("维数  = {}", a.ndim());                // 3

    // 索引：双方括号 a[[y, x, c]]（注意不是 a[y][x][c]）
    a[[2, 3, 0]] = 1.0;                              // 第 2 行、第 3 列、R 通道 设为 1.0
    println!("a[[2,3,0]] = {}", a[[2, 3, 0]]);

    // 切片：s! 宏取一块子区域，返回的是"视图"(ArrayView)，零拷贝
    let roi = a.slice(s![0..2, 0..3, ..]);           // 左上角 2 行 3 列的全部通道
    println!("roi shape = {:?}", roi.shape());       // [2, 3, 3]

    // 沿某个轴汇总：Axis(2) 就是最后那个"通道"维，这里按通道求和
    let sum_c = a.sum_axis(Axis(2));
    println!("按通道求和后 shape = {:?}", sum_c.shape());  // [4, 6]

    // 视图可以传给函数，不复制数据（预处理里经常这么干）
    println!("平均值 = {:.3}", mean_of(a.view()));
}

// 只读借用一个 [H, W, 3] 张量：参数用 ArrayView3 而不是 Array3，避免整块数据被复制
fn mean_of(a: ArrayView3<f32>) -> f32 {
    a.mean().unwrap_or(0.0)
}
```

三个必须养成的习惯：

1. **先打印 shape。** 拿到任何张量，第一句调试代码永远是 `println!("{:?}", x.shape())`。80% 的张量 bug，是"形状和你以为的不一样"。
2. **`shape` 不等于内存排布。** `[3, 640, 640]` 和 `[640, 640, 3]` 元素个数一样，但语义天差地别——前者通道在前 (CHW)，后者通道在后 (HWC)。下一节就吃这个。
3. **索引用双方括号** `a[[y, x, c]]`，因为一个元素的"坐标"本身是多维的。`ArrayView`/`ArrayViewMut` 是"借来的视图"，传参优先用它，省掉一次拷贝。

## 3.7 image ↔ ndarray 互转（本章重点）

预处理的核心动作，就是把 `image` 的图变成 `ndarray` 的张量，并调整布局。为什么要调布局？因为二者的默认排法不同：

```text
同样一张 2×2 的小图，两种"看法"：

HWC（像素在一起）  一个像素的 RGB 挨着放，方便"取第 (x,y) 个像素"
   [ R G B | R G B | R G B | R G B ]      ← image / 相机 / 显示器的天然布局

CHW（通道在一起）  同一通道的所有像素挨着放，方便"整层一起做卷积"
   [ R R R R | G G G G | B B B B ]         ← 绝大多数深度学习模型要的布局
```

用一组小数字走一遍最清楚。假设这张 2×2 图的四个像素是：

```text
(0,0)=红(255,0,0)   (1,0)=绿(0,255,0)
(0,1)=蓝(0,0,255)   (1,1)=白(255,255,255)

HWC 字节流（into_raw 拿到的就是它）：
   255,0,0 , 0,255,0 , 0,0,255 , 255,255,255

CHW 字节流（把三个通道各自拢到一起）：
   R: 255,0,0,255   G: 0,255,0,255   B: 0,0,255,255
```

同样 12 个字节，只是换了顺序。`ndarray` 的 `permuted_axes` 就是干这个换轴的活儿。下面写一个"图 → 模型输入张量"的标准函数——把 `RgbImage` 变成 `[1, 3, H, W]` 的 f32 张量（batch=1、通道在前、归一化到 0~1），这正是 YOLOv8 模型要的输入（第 1 章设定的 `1×3×640×640`）：

```rust,ignore
use image::RgbImage;
use ndarray::{Array3, Array4, Axis};

/// RgbImage → [1, 3, H, W] 的 f32 张量（像素值归一化到 0~1）
fn img_to_tensor(img: &RgbImage) -> Array4<f32> {
    let (w, h) = img.dimensions();          // 注意：宽在前、高在后

    // 第 1 步：拿到 HWC、packed 的字节（as_raw 只借用，clone 出一份自己的 Vec）
    let raw: Vec<u8> = img.as_raw().clone();

    // 第 2 步：一维字节 → 三维数组 [H, W, 3]（形状元组是 高、宽、通道）
    let hwc = Array3::from_shape_vec((h as usize, w as usize, 3), raw)
        .expect("字节数应当恰好等于 H*W*3");

    // 第 3 步：HWC → CHW，把"通道轴"从第 2 位挪到第 0 位
    let chw = hwc.permuted_axes([2, 0, 1]);  // 逻辑形状变成 [3, H, W]

    // 第 4 步：u8 → f32 并归一化；mapv 会顺手把数据重排成连续内存
    let chw_f32 = chw.mapv(|v| v as f32 / 255.0);

    // 第 5 步：最前面插一个长度为 1 的 batch 维 → [1, 3, H, W]
    chw_f32.insert_axis(Axis(0))
}
```

`permuted_axes([2, 0, 1])` 怎么读？它说"新数组的第 0 轴 = 旧的第 2 轴、第 1 轴 = 旧的第 0 轴、第 2 轴 = 旧的第 1 轴"，于是 `[H, W, C]` 变成 `[C, H, W]`。这条转换链——`into_raw → from_shape_vec → permuted_axes → mapv → insert_axis`——会伴随你整个职业生涯，务必吃透。第 10 章的 letterbox、归一化、组 batch，全都建在它之上。

## 3.8 在图上画一个矩形框

有了检测结果，最后一步总是"在原图上把框画出来"。这里先热个身：手动描点画一个空心矩形。顺便引入全书统一的**边界框**类型 `BBox`（左上-右下像素坐标），第 16 章画真正的检测框时会直接复用它：

```rust,ignore
use image::{Rgb, RgbImage};

/// 边界框：左上-右下像素坐标（全书统一类型，见第 3.8 节）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// 在图上画一个空心矩形框（逐边描点，正好复习像素写入）
fn draw_bbox(img: &mut RgbImage, b: &BBox, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    // 把坐标夹到画布范围内，防止 put_pixel 越界 panic
    let x1 = (b.x1.max(0.0) as u32).min(w - 1);
    let y1 = (b.y1.max(0.0) as u32).min(h - 1);
    let x2 = (b.x2.max(0.0) as u32).min(w - 1);
    let y2 = (b.y2.max(0.0) as u32).min(h - 1);

    for x in x1..=x2 {           // 上、下两条横边
        img.put_pixel(x, y1, color);
        img.put_pixel(x, y2, color);
    }
    for y in y1..=y2 {           // 左、右两条竖边
        img.put_pixel(x1, y, color);
        img.put_pixel(x2, y, color);
    }
}

fn main() -> anyhow::Result<()> {
    let mut img = image::open("test.jpg")?.to_rgb8();
    let b = BBox { x1: 60.0, y1: 40.0, x2: 260.0, y2: 190.0 };
    draw_bbox(&mut img, &b, Rgb([255, 0, 0]));   // 画一个红框
    img.save("boxed.png")?;
    Ok(())
}
```

真实项目里不必自己描点——`imageproc` 这个 crate 的 `draw_hollow_rect_mut` 一行就能画框、`draw_text_mut` 还能写类别文字。但自己写一遍，你会更清楚"画框"本质上就是往特定坐标写像素。到了第 16 章，我们画的将是一批 `Detection`（内含 `BBox` + 置信度 + 类别号），把模型输出可视化成人能看懂的框。

## 3.9 串起来：一个完整可跑的例子

把本章的主线走一遍：**读图 → 打印尺寸 → 转 ndarray → 改成 CHW → 再存回图**。最后存出来的图应当和原图一模一样，这就证明我们的布局变换没搞错：

```rust,ignore
use image::{open, RgbImage};
use ndarray::Array3;

fn main() -> anyhow::Result<()> {
    // 1. 读图 + 打印尺寸
    let img: RgbImage = open("test.jpg")?.to_rgb8();
    let (w, h) = img.dimensions();
    println!("读入尺寸 (w, h) = ({}, {})", w, h);

    // 2. 图 → ndarray：先拿字节，再 reshape 成 [H, W, 3]（HWC）
    let raw: Vec<u8> = img.into_raw();
    let hwc: Array3<u8> = Array3::from_shape_vec((h as usize, w as usize, 3), raw)?;
    println!("HWC 张量 shape = {:?}", hwc.shape());   // [H, W, 3]

    // 3. 改成 CHW：把通道轴挪到最前（permuted_axes 只换"看法"，不搬数据）
    let chw = hwc.permuted_axes([2, 0, 1]);
    println!("CHW 张量 shape = {:?}", chw.shape());   // [3, H, W]

    // 4. 存图：CHW → HWC → 按逻辑顺序收字节 → RgbImage → 保存
    let hwc_back = chw.permuted_axes([1, 2, 0]);           // 转回 [H, W, 3]
    let bytes: Vec<u8> = hwc_back.iter().copied().collect(); // iter() 按逻辑顺序取，安全
    let out = RgbImage::from_raw(w, h, bytes).expect("字节数与宽高匹配，不会失败");
    out.save("roundtrip.png")?;
    println!("已存回 roundtrip.png（应与原图一致）");
    Ok(())
}
```

这里藏着本章**最容易翻车的一个坑**，务必看第 4 步的注释：`permuted_axes` 之后，数据在内存里并没有真正搬家，只是"步长 (stride)"变了。如果你图省事直接调 `into_raw_vec()`，拿到的是**原始内存顺序**（还是 CHW），而不是你以为的 HWC——存出来就是一张彻底错乱的图。用 `.iter()` 遍历则永远按**逻辑顺序**取值，所以 `.iter().copied().collect()` 是安全的。这个 bug 不报错、只出乱图，极难排查，记它一辈子。

跑一下，输出大概是：

```text
读入尺寸 (w, h) = (1920, 1080)
HWC 张量 shape = [1080, 1920, 3]
CHW 张量 shape = [3, 1080, 1920]
已存回 roundtrip.png（应与原图一致）
```

顺便印证 3.2 节强调的顺序陷阱：`dimensions()` 给的是 `(1920, 1080)`（宽, 高），而 `shape()` 给的是 `[1080, 1920, 3]`（高, 宽, 通道）——一个宽在前、一个高在前。`from_shape_vec` 和 `from_raw` 的参数顺序要是写反了，你会得到一张旋转、错位的图。

## 3.10 性能提示与常见坑

教学代码怎么好懂怎么写，但上线代码要讲究。几条最常见的性能坑，一表打尽：

| 坑 | 现象 | 正确做法 |
| --- | --- | --- |
| 用 `get_pixel` 双循环扫全图 | 每次调用都做越界检查 + 下标乘法，扫一张 4K 图很慢 | 用 `pixels_mut()` 迭代器 / `as_raw()` 切片 / `ndarray` 的 `mapv` 批量算 |
| 反复调用 `to_rgb8()` | `DynamicImage` 是枚举，每转一次就重新分配、拷贝一份内存 | 进算法前**只转一次**成 `RgbImage`，之后全程只碰这份 buffer |
| 直接对 `DynamicImage` 逐像素访问 | 每次访问都要 `match` 一遍"到底是哪种像素类型"的枚举分支 | 同上，先在边界处落成具体的 `ImageBuffer` |
| `permuted_axes` 后调 `into_raw_vec` | 拿到的是**原始内存顺序**，不是你以为的 HWC/CHW → 出乱图 | 用 `.iter()` 按逻辑顺序收集，或先 `.as_standard_layout()` 整理成连续内存 |
| `u8` 相加溢出 | debug 下 `200 + 100` 直接 panic | 用 `saturating_add`，或先转 `f32`/`i32` 算完再 `clamp` 回 `u8` |

核心心法就一句：**在流水线入口把 `DynamicImage` 落成一份确定布局的 buffer，中间尽量批量运算、避免逐像素函数调用，需要连续内存时记得 `.as_standard_layout()`。** 工程上还会用 `rayon` 并行遍历像素、用 `fast_image_resize` 替代通用缩放（第 10 章会用到），这些优化到具体场景再展开，别一上来就过度设计。

## 3.11 小结

- `image` 管"图片"、`ndarray` 管"张量"；预处理的本质，就是把前者变成后者。
- image 主线 API：`open` → `to_rgb8`（得到 `RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>`）→ `dimensions`/`get_pixel`/`put_pixel`/`enumerate_pixels` → `crop_imm`/`resize` → `save`。
- `into_raw()` 拿到的一维 `Vec<u8>` 就是"HWC + packed"内存，坐标 `(x,y,c)` 对应 `bytes[(y*w+x)*3+c]`。
- ndarray 记四样：`shape`、双方括号索引 `a[[y,x,c]]`、`s!` 切片（得 `ArrayView`）、`Axis`；调试第一句永远是打印 `shape`。
- 本章重点转换链：`into_raw → from_shape_vec((h,w,3)) → permuted_axes([2,0,1]) → mapv → insert_axis`，把 HWC 图变成模型要的 `[1,3,H,W]` CHW 张量。
- 两个夺命坑：`dimensions()` 是"宽高"而 `shape()` 是"高宽"；`permuted_axes` 后别用 `into_raw_vec`，要用 `.iter()` 或先 `as_standard_layout`。
- 下一章（第 4 章）我们深入色彩空间与内存布局，把 RGB/BGR/YUV/NV12 和这里的字节排布彻底讲透。

## 3.12 练习

1. 改写 3.9 节的程序：在转成 CHW 之后、存回之前，把 **R 通道整体清零**（提示：`chw.slice_mut(s![0, .., ..]).fill(0)`），存出来看看画面变成什么样，并解释为什么。
2. 写一个函数 `tensor_to_img`，把 `[3, H, W]` 的 `f32` 张量（0~1）转回 `RgbImage`（提示：`mapv(|v| (v*255.0).round() as u8)` → `permuted_axes([1,2,0])` → `.iter().copied().collect()` → `from_raw`）。注意别踩 `into_raw_vec` 的坑。
3. 用 3.8 节的 `BBox` 和 `draw_bbox`，在同一张图上画三个不同颜色的框，坐标自定。再想一想：如果框的 `x2` 超出了图像宽度会发生什么？我们的夹取代码保护到了吗？
4. 分别用 `get_pixel` 双循环和 `pixels_mut()` 迭代器实现"整图亮度 +30"，各跑一张大图，用 `std::time::Instant` 计时对比，直观感受两种写法的差距。
