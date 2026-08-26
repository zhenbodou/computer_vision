# 第 31 章 把图像当数据：图像与张量互转、批处理

> **本章导读**
> - 学完你能：把一张 `RgbImage` 变成神经网络能"吃"的规范张量，再变回来，全程纯 Rust。
> - 讲清两种通道布局 **HWC** 与 **CHW** 的区别，并动手互转（这是最容易翻车的地方）。
> - 掌握推理前的三步数值预处理：**类型转换 → 归一化 → 按通道标准化**，以及为什么要加一个 **batch 维**。
> - 这是"图像处理"通往"深度学习"的桥。第八部分（分类）、第九部分（检测）做推理时，输入预处理就是本章这套。

## 场景：照片喂进模型前，差了哪几步

假设你手上有一个训练好的分类模型（比如后面第八部分会用的 ResNet），你信心满满地用 [第 7 章](../p02-image-basics/ch07-load-save.md) 学的 `image::open` 读进一张猫的照片，拿到一个 `RgbImage`，直接塞给模型——结果要么类型对不上编译不过，要么勉强跑起来输出一堆乱码般的概率。

问题在于：**模型和图像库说的不是同一种"数据方言"。** `image` 库给你的是一堆 `u8`（0~255 的整数）、按"一个像素的 R/G/B 挨在一起"的方式排列；而神经网络要的是**规整的浮点张量**——数值是 `f32`、被缩放到某个固定范围、通道的排列顺序还不一样，最前面往往还得多一个维度。这中间的"翻译"活儿，就是本章要教的。

> 📖 **术语**：**张量（tensor）**。可以先朴素地理解成"多维数组"——一维是向量，二维是矩阵，三维、四维就是张量。深度学习里的图像通常是三维（通道、高、宽）或四维（多一个批量维）张量。本章用 [第 29 章](ch29-ndarray.md) 的 `ndarray` 来表示张量：`Array3<f32>` 就是一个三维 `f32` 张量。

> 💡 **提示**：在 Python 里，这几步预处理被 `torchvision.transforms` 一行 `ToTensor()` + `Normalize()` 包好了。**纯 Rust 生态目前没有这种一行到位的封装**，得自己把几步写出来。别怕——逻辑非常简单，本章写完你会发现，"自己动手"反而让你彻底看清预处理到底在干什么，这正是"会调包"和"懂原理"的分水岭。

## 两种通道布局：HWC 与 CHW

一张彩色图有三个维度：高 H、宽 W、通道 C（RGB 三通道）。三个维度在内存里"谁先谁后"，有两套约定：

- **HWC**（Height × Width × Channel）：`image` 库和绝大多数图像工具的默认。内存里**一个像素的 R、G、B 挨在一起**，一个像素接一个像素铺开。
- **CHW**（Channel × Height × Width）：PyTorch 及大多数深度学习模型的默认。内存里**同一个通道的所有像素连续**——先把整张图的 R 铺完，再铺 G，再铺 B。

```text
HWC（image 库默认）：像素优先，R/G/B 交错
  内存: [R G B][R G B][R G B] ...     取值 t[[y, x, c]]

CHW（模型默认）：通道优先，整片 R、整片 G、整片 B
  内存: [R R R ...][G G G ...][B B B ...]   取值 t[[c, y, x]]
```

同样是"第 y 行、第 x 列、第 c 通道"这个数，两种布局在内存里的落点完全不同，索引写法也不同（`[[y, x, c]]` vs `[[c, y, x]]`）。**把布局搞反，是新手最常见的 bug**：模型不会报错，只会给你莫名其妙的结果，因为它把"宽"当成了"通道"去理解。

好在 `ndarray` 转布局非常轻巧：`permuted_axes` 只是重新解释各维度的步长（stride），**不搬动任何数据**（零拷贝）。HWC 的三个轴编号是 `0=H, 1=W, 2=C`，要变成 CHW，就把它们重排成 `[2, 0, 1]`（新的第 0 轴取原来的第 2 轴 C，新的第 1 轴取原来的 H，新的第 2 轴取原来的 W）。

## 数值预处理：从 u8 到规范的 f32

布局理顺了，还要处理"数值本身"。分两步：

**第一步，归一化（normalize）到 [0,1]。** 把 `u8` 的 `0~255` 转成 `f32` 并除以 255：

\\[ x_{\text{norm}} = v / 255 \\]

为什么？神经网络喜欢**小而集中的输入**（梯度更稳、训练更顺），`0~255` 这种大整数直接喂进去数值容易失衡。

**第二步，按通道标准化（standardize）。** 很多预训练模型（尤其在 ImageNet 上训的）还要求把每个通道**减均值、除标准差**：

\\[ x_{\text{std}} = (x - \mu_c) / \sigma_c \\]

注意下标 `c`——**均值 \\( \mu_c \\) 和标准差 \\( \sigma_c \\) 是按通道分别给的**，R、G、B 各一套。ImageNet 上最常用的一组是：

- 均值 mean = `[0.485, 0.456, 0.406]`（对应 R、G、B）
- 标准差 std = `[0.229, 0.224, 0.225]`

这组数字是在 ImageNet 上百万张图上统计出来的。**用哪个模型，就配它训练时用的那组 mean/std**，配错了精度会明显下降。标准化之后，数据大致以 0 为中心，取值范围也不再是 `[0,1]` 而是大约 `[-2.5, 2.7]`——待会儿代码会实测印证。

> ⚠️ **注意**：这几步是**有顺序**的。真实流程是：先 `resize` 到模型要求的输入尺寸（见 [第 10 章](../p02-image-basics/ch10-geometry.md)）→ 归一化到 `[0,1]` → 再标准化。标准化一定是在 `[0,1]` 的基础上做，别拿 `0~255` 的原始值去减 `0.485`。

## 批量维度 NCHW：为什么要多一维

模型实际要的往往不是 `[C, H, W]` 三维，而是 `[N, C, H, W]` **四维**——最前面多一个 **N（batch，批量）**。哪怕你只推理一张图，也得把它包成"1 张的批"：`[1, 3, H, W]`。

为什么？因为 GPU 一次并行处理**一批**图片效率最高，模型的接口便统一设计成"吃一批"。一张图就是 batch 大小为 1 的特例。加这一维用 `insert_axis(Axis(0))` 即可，`Array3` 就升成了 `Array4`。

## 动手写代码：往返转换 + 标准化

我们写四个函数：`image_to_chw_tensor`（图像→CHW 张量）、`chw_tensor_to_image`（CHW 张量→图像）、`standardize_imagenet`（按通道标准化），再对一张合成图做**往返转换并验证像素与原图完全一致**。输入图用代码合成，自包含。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
ndarray = "0.17"
```

完整代码（`examples/ch31_image_as_tensor.rs`）：

```rust
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::{Array3, Array4, Axis};

/// 合成一张彩色小图：红随 x、绿随 y、蓝随对角线渐变。
/// 每个像素 RGB 组合几乎都不同，往返或通道搞反都能立刻看出来。
fn make_image(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let r = (255 * x / (w - 1)) as u8;
        let g = (255 * y / (h - 1)) as u8;
        let b = (255 * (x + y) / (w + h - 2)) as u8;
        Rgb([r, g, b])
    })
}

/// 图像 → CHW 张量：u8[0,255] 归一化到 f32[0,1]，并从 HWC 转成 CHW。
fn image_to_chw_tensor(img: &RgbImage) -> Array3<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    // 先按图像天然的 HWC 布局(高, 宽, 通道)塞进数组，顺手除以 255 归一化
    let hwc = Array3::from_shape_fn((h, w, 3), |(y, x, c)| {
        img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    });
    // HWC → CHW：permuted_axes 只改 strides、不搬数据（零拷贝）
    // [2,0,1]：原轴 2(C)→新轴 0，原轴 0(H)→新轴 1，原轴 1(W)→新轴 2
    let chw = hwc.permuted_axes([2, 0, 1]);
    // 此刻内存仍是 HWC 顺序，只是"看起来"像 CHW。
    // 交给模型或做 flatten 前，用 as_standard_layout 拷成真正的 CHW 连续内存
    chw.as_standard_layout().to_owned()
}

/// CHW 张量 → 图像：反归一化 f32[0,1]→u8[0,255]（round+clamp），CHW 转回 HWC。
fn chw_tensor_to_image(t: &Array3<f32>) -> RgbImage {
    let (c, h, w) = (t.shape()[0], t.shape()[1], t.shape()[2]);
    assert_eq!(c, 3, "期望 3 通道 CHW 张量");
    // 反归一化：*255 后必须 round + clamp，否则 f32→u8 的直接截断会累积误差
    let to_u8 = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            // 索引顺序：CHW 张量用 [通道, 行 y, 列 x]；图像 put_pixel 用 (列 x, 行 y)
            let px = Rgb([
                to_u8(t[[0, y, x]]), // 通道 0 = R
                to_u8(t[[1, y, x]]), // 通道 1 = G
                to_u8(t[[2, y, x]]), // 通道 2 = B
            ]);
            img.put_pixel(x as u32, y as u32, px);
        }
    }
    img
}

/// 按通道做 ImageNet 标准化：(x - mean) / std，三个通道各不相同。
fn standardize_imagenet(t: &Array3<f32>) -> Array3<f32> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut out = t.clone();
    for c in 0..3 {
        // index_axis_mut(Axis(0), c) 取出第 c 个通道的二维可变视图，就地标准化
        out.index_axis_mut(Axis(0), c)
            .mapv_inplace(|v| (v - mean[c]) / std[c]);
    }
    out
}

fn main() {
    let img = make_image(64, 48);
    println!("原图尺寸: {}x{} (宽x高)", img.width(), img.height());

    // 1) 图像 → CHW 张量
    let chw = image_to_chw_tensor(&img);
    println!("CHW 张量形状: {:?}  (C, H, W)", chw.shape());
    println!("转换后是否为标准(连续)布局: {}", chw.is_standard_layout());

    // 抽查一个像素，确认 CHW 索引 [c, y, x] 的取值 == 原像素 / 255
    let (tx, ty) = (10usize, 5usize);
    let p = img.get_pixel(tx as u32, ty as u32).0;
    println!("像素(x={tx}, y={ty}) 原始 RGB = {p:?}");
    println!(
        "  CHW 取值   : R={:.4} G={:.4} B={:.4}",
        chw[[0, ty, tx]], chw[[1, ty, tx]], chw[[2, ty, tx]]
    );

    // 2) 加 batch 维：CHW → NCHW（一张图也要凑成"1 张的批"）
    let nchw: Array4<f32> = chw.clone().insert_axis(Axis(0));
    println!("NCHW 张量形状: {:?}  (N, C, H, W)", nchw.shape());

    // 3) 往返验证：CHW → 图像，与原图逐像素比最大误差
    let back = chw_tensor_to_image(&chw);
    let max_err = img.as_raw().iter().zip(back.as_raw().iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max().unwrap_or(0);
    println!("往返(图像→张量→图像)最大像素误差 = {max_err}");

    // 4) 按通道 ImageNet 标准化，打印每通道标准化后的均值与范围
    let norm = standardize_imagenet(&chw);
    for c in 0..3 {
        let ch = norm.index_axis(Axis(0), c);
        let lo = ch.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = ch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("标准化后通道{} 均值={:.3} 范围=[{:.3}, {:.3}]",
            c, ch.mean().unwrap(), lo, hi);
    }
}
```

运行 `cargo run --example ch31_image_as_tensor`，终端打印（数字真实可复现）：

```text
原图尺寸: 64x48 (宽x高)
CHW 张量形状: [3, 48, 64]  (C, H, W)
转换后是否为标准(连续)布局: true
像素(x=10, y=5) 原始 RGB = [40, 27, 34]
  CHW 取值   : R=0.1569 G=0.1059 B=0.1333
NCHW 张量形状: [1, 3, 48, 64]  (N, C, H, W)
往返(图像→张量→图像)最大像素误差 = 0
标准化后通道0 均值=0.057 范围=[-2.118, 2.249]
标准化后通道1 均值=0.188 范围=[-2.036, 2.429]
标准化后通道2 均值=0.409 范围=[-1.804, 2.640]
```

几个结论一目了然：

1. **形状对了**：`64x48` 的图变成 `[3, 48, 64]` 的 CHW 张量（注意 H=48 在前、W=64 在后），加 batch 维后是 `[1, 3, 48, 64]`。
2. **像素值对了**：`x=10, y=5` 处原始 `[40, 27, 34]`，CHW 里取 `40/255=0.1569`……逐个吻合，说明布局和归一化都没搞错。
3. **往返最大误差 = 0**：图像→张量→图像后**每个像素与原图丝毫不差**。因为 `v/255` 再 `*255` 后 `round()`，能精确还原 `0~255` 的每一个整数——这正是反归一化必须用 `round` 的原因。
4. **标准化后数据以 0 附近为中心**，范围落到约 `[-2.1, 2.6]`，不再是 `[0,1]`——和前面说的一致。

## 常见坑与提示

> 🕳️ **坑**：**HWC / CHW 搞反，模型不报错但结果全乱。** 最典型的是把 CHW 当 HWC 直接 `reshape`，数据没重排，通道就和空间位置搅在一起，颜色、形状全错。**转布局要用 `permuted_axes` 真正重排维度，绝不能用 `reshape`/`to_shape` 硬掰形状**——后者只改"形状标签"、不动数据顺序，是两码事。

> 🕳️ **坑**：**`into_shape` 在 ndarray 0.16/0.17 已被废弃。** 直接用会编译告警：`use of deprecated method ...: Use .into_shape_with_order() or .to_shape()`。原因是老 `into_shape` 的行为依赖内存顺序、容易出歧义。**改形状请用 `to_shape(...)`（借用、按需拷贝，返回 `CowArray`）或 `into_shape_with_order(...)`（消费、要求内存连续）。** 本章转布局根本没碰 reshape，用的是 `permuted_axes`。

> ⚠️ **注意**：**`permuted_axes` 之后不是连续内存，`as_slice()` 会返回 `None`。** 因为它只改了 strides。想把张量拉平喂给模型（很多推理接口要连续的 `&[f32]`），要先 `as_standard_layout()`（必要时拷贝一份）或 `to_owned()`，让内存真正变成 CHW 连续，代码里 `is_standard_layout()` 打印 `true` 就是这个意思。

> 🕳️ **坑**：**忘了除以 255、或标准化 mean/std 没按通道。** 忘除 255，输入就大了 255 倍，模型直接懵；mean/std 若用同一个标量套三个通道，等于没按通道校正，精度会掉。记住：**先 `/255` 到 `[0,1]`，再按通道 `(x - mean_c)/std_c`**。

> 💡 **提示**：**反归一化 `f32 → u8` 一定要 `round` + `clamp`。** 直接 `as u8` 是**向零截断**（`0.99` 变 `0`），误差会累积；而且标准化后可能出现负数或大于 1 的值，不 `clamp` 到 `[0,255]` 会溢出得到乱七八糟的数。本章 `to_u8` 闭包就是 `round().clamp(0.0, 255.0) as u8`。

> ⚠️ **注意**：**索引里 `(x, y)` 和 `(y, x)` 别搞混。** `image` 的 `get_pixel(x, y)`/`put_pixel(x, y)` 是"先列后行"，而张量索引 `t[[c, y, x]]` 是"先行后列"。这俩顺序相反，是另一个高频翻车点，写的时候多留个心眼。

## 本章小结

- 神经网络吃的是**规整浮点张量**，不是 `RgbImage`。推理前要把图片翻译成模型要求的**布局、数值范围和维度**。
- 两种通道布局：**HWC**（`image` 默认，像素交错）与 **CHW**（模型默认，通道连续）。用 `permuted_axes([2,0,1])` 零拷贝互转，别用 reshape 硬掰。
- 数值预处理三步：`u8 → f32`、**归一化** `/255` 到 `[0,1]`、**按通道标准化** `(x - mu_c)/sigma_c`（ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`）。
- 单张图也要加 **batch 维**凑成 **NCHW** `[1, C, H, W]`，用 `insert_axis(Axis(0))`。
- 往返转换实测**最大误差 0**，关键在反归一化用 `round + clamp`；`ndarray` 0.17 里 `into_shape` 已废弃，改用 `to_shape`/`into_shape_with_order`。
- **第八部分（分类）、第九部分（检测）做模型推理时，输入预处理就是本章这套**——先记牢，后面反复用。

## 动手练习

1. **改布局验证**：把 `image_to_chw_tensor` 里的 `permuted_axes([2, 0, 1])` 故意删掉（直接返回 HWC），再跑往返，观察 `chw_tensor_to_image` 里 `[[c, y, x]]` 的取值和形状打印会怎样错乱，亲手体会"布局搞反"的后果。
2. **写反标准化**：新增 `unstandardize_imagenet`，把标准化后的张量用 `x * sigma_c + mu_c` 还原，接着 `chw_tensor_to_image`，验证与原图误差依然为 0。
3. **换一组 mean/std**：查一个非 ImageNet 模型（如某些人脸模型用 `mean=0.5, std=0.5`），改成它的参数，观察标准化后取值范围的变化。
4. **进阶**：写一个 `batch_to_nchw(imgs: &[RgbImage]) -> Array4<f32>`，把多张**同尺寸**图拼成一个真正的 batch（提示：先各自转 CHW，再用 `ndarray::stack` 沿 `Axis(0)` 堆叠），打印形状确认第 0 维等于图片张数。

到这里，图像和张量之间的"任督二脉"就打通了——你已经能把任意一张图变成模型认得的输入、再变回来。带着这套预处理，下一部分我们正式踏入"学习"的世界：**机器是怎么从数据里学到规律的？** 👉 [第 32 章 机器学习是怎么"学"的](../p06-ml/ch32-ml-intro.md)

## 第五部分阶段验收：布局、稳定性与零拷贝边界

为 HWC/CHW/NHWC/NCHW 建立显式类型和 shape 检查，覆盖非连续 view、奇数 stride、空维度、
不同 dtype 与溢出。比较逐像素循环、连续切片和向量化实现，报告复制次数、分配、缓存行为
和 P95。线性代数求解使用 QR/SVD/Cholesky 等分解而非显式求逆，并按条件数构造病态矩阵，
验证误差与失败返回。所谓零拷贝必须由指针/分配证据证明，不能只依据 API 名称。
