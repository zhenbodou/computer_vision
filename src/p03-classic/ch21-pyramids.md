# 第 21 章 图像金字塔与多尺度处理

> **本章导读**
> - 想清楚一件事：同一个物体，离镜头远近不同，在图里就有大有小——**检测和匹配必须能应对多种尺度**。
> - 亲手用 `image` 的降采样 + `imageproc` 的高斯模糊，**手工搭一座高斯金字塔**（imageproc 没有现成的）。
> - 再搭一座**拉普拉斯金字塔**，并从它**无损重建**原图（本章实测最大误差 = 0）。
> - 搞懂一个新手必翻的坑：**降采样之前为什么一定要先模糊**（抗混叠）。

## 场景：同一个 Logo，忽大忽小

[第 20 章](ch20-template-matching.md) 我们学了模板匹配——拿一张小图（模板）去大图里滑动比对。可它有个致命软肋：**模板一旦确定，大小就锁死了**。你用一个 60×60 的 Logo 模板，图里那个 Logo 要是 120×120（相机凑近了拍），或者缩成 30×30（拍远了），匹配分数就会一落千丈，甚至完全找不到。

现实世界里，"远近导致大小不同"是常态：自动驾驶要同时框出眼前的大车和远处的小车；人脸检测要兼顾贴脸自拍和集体合影里的小脸。**一个只认一种尺寸的算法，注定不好用。**

解决思路很朴素：**既然不知道目标多大，那我就把图像做成一系列由大到小的版本，在每个尺寸上都找一遍。** 这一摞从大到小、层层缩小的图像，就叫**图像金字塔（image pyramid）**——底层最大最清晰，越往上越小越模糊，堆起来正像一座金字塔。

> 📖 **术语**：**多尺度（multi-scale）**。指同一套算法在图像的多个缩放版本上分别运行，从而适应目标的各种大小。图像金字塔是实现多尺度最经典、最省事的数据结构，也是后面 [第 24 章](../p04-features/ch24-descriptors.md) SIFT"尺度空间"的雏形。

## 高斯金字塔：模糊 + 减半

最常用的金字塔叫**高斯金字塔（Gaussian pyramid）**。构建规则简单到只有两步，逐层循环：

1. 对当前层做一次**高斯模糊**；
2. 把它**降采样（downsample）**到宽高各一半。

记原图为 \\( G_0 \\)，则第 \\( k+1 \\) 层由第 \\( k \\) 层生成：

\\[ G_{k+1} = \text{downsample}\big(\text{blur}(G_k)\big) \\]

于是尺寸序列是 \\( W\times H,\ \tfrac{W}{2}\times\tfrac{H}{2},\ \tfrac{W}{4}\times\tfrac{H}{4},\dots \\)，像这样：

```text
 第0层  ┌───────────────┐  256×192  原图，最清晰
        │               │
        └───────────────┘
 第1层    ┌───────┐        128×96
          └───────┘
 第2层      ┌───┐          64×48
            └───┘
 第3层       ┌─┐           32×24   最小、最模糊
             └─┘
```

**为什么要先模糊？** 这不是为了好看，而是**信号采样的硬道理**，也是新手最常踩的坑，我们下面单独用一个实验戳穿它。

## 降采样前不模糊，会发生什么：混叠

先讲直觉。你一定见过这些现象：拍一件细条纹衬衫，照片上却浮现出诡异的彩色波纹；视频里高速旋转的车轮看着像在倒转。这些"凭空冒出来的假花纹"有个统一的名字——**混叠（aliasing）**。

原因在于**采样定理**：当你把采样点变稀（降采样就是这么干的），凡是比新采样间隔还"细"的高频细节，都无法被如实记录，反而会被**折叠**成一个**错误的低频花纹**（学名摩尔纹 moiré）。

怎么破？**在降采样之前，先用高斯模糊把这些"过细"的高频成分抹掉**，留下的都是新网格能如实表示的低频信息，混叠就没了。顺序很关键：**先模糊，再抽样**。

我们用一个极端例子把它量化出来：一幅"1 像素黑、1 像素白"交替的竖条纹（周期 2，是图像能表示的最高频信号），降采样到一半：

- **方法 A（不模糊，直接抽样）**：每隔一列取一个点，恰好全落在黑列上——整幅图塌缩成**纯黑**，条纹信息荡然无存。
- **方法 B（先模糊，再抽样）**：黑白先被模糊成灰，抽样后仍是**灰**，整体亮度保住了。

本章代码实测（原图均值 127.5）：方法 A 得到均值 **0.0**（彻底跑偏），方法 B 得到均值 **123.6**（基本正确）。一个数字，胜过千言万语。

## 拉普拉斯金字塔：把"细节"单独存起来

高斯金字塔每上一层都会**丢掉一些细节**（被模糊掉了）。那些丢掉的东西去哪了？**拉普拉斯金字塔（Laplacian pyramid）** 就负责把它们捡回来存好。

它的定义是**相邻两层高斯之差**：把上一层（较小、较模糊的 \\( G_{k+1} \\)）**放大（expand）**回本层尺寸，再用本层 \\( G_k \\) 去减它：

\\[ L_k = G_k - \text{expand}(G_{k+1}) \\]

\\( \text{expand}(G_{k+1}) \\) 是"丢了细节的粗糙版"，\\( G_k \\) 是"带细节的清晰版"，两者之差 \\( L_k \\) 恰好就是**这一层独有的细节**（数学上是一段特定的频率，所以拉普拉斯金字塔又叫**带通 band-pass 金字塔**）。金字塔最顶层没有"下一层"可减，就直接保留最小的高斯层 \\( G_n \\) 作为重建的种子。

妙就妙在这个过程**完全可逆**。把式子挪一下项就是重建公式，自顶向下一层层加回去：

\\[ G_k = L_k + \text{expand}(G_{k+1}) \\]

> 💡 **提示**：拉普拉斯金字塔是图像融合（image blending）的看家本领。经典的"苹果+橙子=橘子苹果"无缝拼接，就是把两张图各自的拉普拉斯金字塔**按层**用平滑掩膜混合，再重建——每个频率段单独过渡，接缝自然到看不出来。

## 动手写代码：搭金字塔、重建、验证抗混叠

一个例子做三件事：① 搭 4 层高斯金字塔并存每层；② 搭拉普拉斯金字塔并**重建原图**、报告误差；③ 做抗混叠对照实验。输入图用代码合成，自包含。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

运行：`cargo run --example ch21_pyramids`。完整代码：

```rust
use image::imageops::{self, FilterType};
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;

/// 合成灰度测试图：渐变背景 + 一个亮圆 + 一个暗圆 + 右上角细棋盘纹理
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        let mut v = (x * 180 / (w - 1)) as i32 + 20; // 左到右渐变
        let (cx, cy, r) = (w as i32 / 3, h as i32 / 3, 40); // 左上亮圆
        if (xi - cx).pow(2) + (yi - cy).pow(2) < r * r { v = 240; }
        let (cx2, cy2, r2) = (2 * w as i32 / 3, 2 * h as i32 / 3, 28); // 右下暗圆
        if (xi - cx2).pow(2) + (yi - cy2).pow(2) < r2 * r2 { v = 30; }
        // 右上细棋盘纹理：高频细节，越往金字塔上层会被抹得越平
        if xi > 2 * w as i32 / 3 && yi < h as i32 / 3 && ((xi / 3 + yi / 3) % 2 == 0) { v = 200; }
        Luma([v.clamp(0, 255) as u8])
    })
}

/// 放大到指定尺寸——构建与重建拉普拉斯金字塔时都用它，保证一致
fn expand(img: &GrayImage, w: u32, h: u32) -> GrayImage {
    imageops::resize(img, w, h, FilterType::Triangle)
}

/// 高斯金字塔：每层 = 上一层「先高斯模糊、再降采样到 1/2」
fn gaussian_pyramid(base: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyr = vec![base.clone()];
    for _ in 1..levels {
        let prev = pyr.last().unwrap();
        let blurred = gaussian_blur_f32(prev, 1.0);          // ① 先模糊，抗混叠的关键
        let (w, h) = ((prev.width() / 2).max(1), (prev.height() / 2).max(1));
        let down = imageops::resize(&blurred, w, h, FilterType::Triangle); // ② 再减半
        pyr.push(down);
    }
    pyr
}

/// 拉普拉斯金字塔：L_i = G_i - expand(G_{i+1})；顶层存最小的高斯层。
/// 用 f32 存：相邻层之差有正有负，且 u8 相减恰好是精确整数。
fn laplacian_pyramid(gauss: &[GrayImage]) -> Vec<Vec<f32>> {
    let n = gauss.len();
    let mut lap: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let (w, h) = (gauss[i].width(), gauss[i].height());
        let up = expand(&gauss[i + 1], w, h);
        let diff = gauss[i].as_raw().iter()
            .zip(up.as_raw().iter())
            .map(|(&a, &b)| a as f32 - b as f32) // 清晰版 - 粗糙版 = 本层独有的细节
            .collect();
        lap.push(diff);
    }
    lap.push(gauss[n - 1].as_raw().iter().map(|&p| p as f32).collect()); // 顶层种子
    lap
}

/// 从拉普拉斯金字塔重建：自顶向下 G_i = L_i + expand(G_{i+1})
fn reconstruct(lap: &[Vec<f32>], sizes: &[(u32, u32)]) -> GrayImage {
    let n = lap.len();
    let (tw, th) = sizes[n - 1];
    let seed = lap[n - 1].iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    let mut cur: GrayImage = ImageBuffer::from_raw(tw, th, seed).unwrap();
    for i in (0..n - 1).rev() {
        let (w, h) = sizes[i];
        let up = expand(&cur, w, h);
        let out = up.as_raw().iter().zip(lap[i].iter())
            .map(|(&u, &l)| (u as f32 + l).round().clamp(0.0, 255.0) as u8) // 补回细节
            .collect();
        cur = ImageBuffer::from_raw(w, h, out).unwrap();
    }
    cur
}

fn mean(img: &GrayImage) -> f32 {
    img.as_raw().iter().map(|&p| p as u64).sum::<u64>() as f32 / img.as_raw().len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 高斯金字塔
    let base = make_scene(256, 192);
    let gauss = gaussian_pyramid(&base, 4);
    for (i, g) in gauss.iter().enumerate() {
        println!("第 {} 层尺寸 = {}x{}", i, g.width(), g.height());
        g.save(format!("out_ch21_gauss_l{}.png", i))?;
    }

    // 2. 拉普拉斯金字塔 + 无损重建
    let sizes: Vec<(u32, u32)> = gauss.iter().map(|g| (g.width(), g.height())).collect();
    let lap = laplacian_pyramid(&gauss);
    let recon = reconstruct(&lap, &sizes);
    let (mut max_diff, mut sum) = (0i32, 0i64);
    for (&a, &b) in base.as_raw().iter().zip(recon.as_raw().iter()) {
        let d = (a as i32 - b as i32).abs();
        max_diff = max_diff.max(d);
        sum += d as i64;
    }
    println!("重建 vs 原图：最大误差 = {max_diff}，平均误差 = {:.4}",
             sum as f64 / base.as_raw().len() as f64);

    // 3. 抗混叠对照：周期 2 竖条纹（最高频）降采样
    let stripes: GrayImage =
        ImageBuffer::from_fn(256, 64, |x, _| Luma([if x % 2 == 0 { 255 } else { 0 }]));
    let alias = imageops::resize(&stripes, 128, 32, FilterType::Nearest); // 不模糊直接抽
    let clean = imageops::resize(&gaussian_blur_f32(&stripes, 1.0), 128, 32, FilterType::Nearest);
    println!("条纹原图均值={:.1}；无预模糊={:.1}；先模糊={:.1}",
             mean(&stripes), mean(&alias), mean(&clean));
    Ok(())
}
```

运行输出（真实可复现）：

```text
第 0 层尺寸 = 256x192
第 1 层尺寸 = 128x96
第 2 层尺寸 = 64x48
第 3 层尺寸 = 32x24
重建 vs 原图：最大误差 = 0，平均误差 = 0.0000
条纹原图均值=127.5；无预模糊=0.0；先模糊=123.6
```

三个结论：

1. **高斯金字塔**尺寸逐层减半：256×192 → 128×96 → 64×48 → 32×24。打开各层 PNG，会看到右上角的细棋盘纹理**越往上越模糊、最后彻底消融**——这正是"丢细节"的可视化。
2. **拉普拉斯金字塔可无损重建原图**：最大误差 = 0。为什么这么干净？因为我们把拉普拉斯层存成了**精确的浮点差值**（u8 相减本就是整数），且构建与重建用的是**同一个 `expand` 函数**，一加一减完全抵消。拉普拉斯金字塔是一种**可逆变换**。
3. **抗混叠对照**一锤定音：不做预模糊，条纹直接塌成纯黑（均值 0.0，信息全丢）；先模糊再抽样，整体亮度保住了（123.6）。

## 常见坑与提示

> 🕳️ **坑**：**降采样前必须先模糊，顺序不能反。** 先抽样再模糊是没用的——错误的低频花纹（混叠）在抽样那一刻就已经产生，之后再模糊也救不回来。记住口诀：**先模糊、后减半**。

> ⚠️ **注意**：**`resize` 用的滤波器本身可能已经在抗混叠。** 本章降采样用了 `FilterType::Triangle`（三角/双线性），它会平均相邻像素，自带一点抗混叠效果；`Lanczos3` 更强。所以抗混叠对照实验里我故意用 `FilterType::Nearest`（最近邻，纯粹抽样、不做任何平均）来**把混叠现象单独逼出来**。实际工程中，直接用 Triangle/Lanczos3 降采样通常就够了；而构建金字塔时额外做一次显式 `gaussian_blur_f32`，好处是**能自己掌控模糊强度 σ**，这跟 SIFT 构建尺度空间的做法一脉相承。

> 🕳️ **坑**：**拉普拉斯层有正有负，别用 `u8` 存。** \\( L_k = G_k - \text{expand}(G_{k+1}) \\) 的取值范围约在 \\([-255, 255]\\)。若直接塞进 `u8`，负数会溢出回绕成大正数，重建立刻乱套。要用 `f32`/`i16` 存储；**仅在可视化时**才 `+128` 偏移后 clamp 到 `[0,255]`（灰色 128 代表"零细节"，亮/暗代表正/负）。

> 💡 **提示**：**层数别贪多。** 每上一层尺寸减半，第 \\( k \\) 层只有原图的 \\( 1/4^k \\) 个像素。一般 4~6 层足矣，缩到几十像素以下再往上分就没意义了。若尺寸不是 2 的整数倍，除法会向下取整，重建时**务必按每层实际尺寸放大**（代码里的 `sizes` 就是干这个的），否则会差一两个像素对不齐。

## 本章小结

- **多尺度**是为了应对"同一物体远近大小不同"：把图像做成由大到小的**金字塔**，在每层都找一遍。
- **高斯金字塔**：逐层「先高斯模糊、再降采样到 1/2」。模糊是为了**抗混叠**——降采样前不模糊，过细的高频会折叠成错误的低频花纹（摩尔纹）。
- **拉普拉斯金字塔**：\\( L_k = G_k - \text{expand}(G_{k+1}) \\)，存的是每层丢掉的**细节**（带通）；顶层保留最小高斯层。它**可逆**——本章从它无损重建原图，最大误差 = 0。
- 应用：多尺度模板匹配/检测、图像融合（按层混合再重建）、SIFT 尺度空间的前身。
- 坑：先模糊后减半、拉普拉斯用 `f32` 存、层数适度、按实际尺寸放大对齐。

## 动手练习

1. **多尺度模板匹配**：在一张大图里放一个 100×100 的目标，但只给一个 50×50 的模板。对大图建高斯金字塔，在**每一层**用 [第 20 章](ch20-template-matching.md) 的 `match_template` 匹配，看看目标在哪一层的匹配分数最高（那层的尺度最接近模板）。
2. **可视化细节流失**：把高斯金字塔每层都**放大回原始尺寸**并存图，逐张对比，直观感受"越上层越模糊、细节越少"。
3. **改用最近邻建金字塔**：把 `gaussian_pyramid` 里的 `gaussian_blur_f32` 那行删掉、降采样换成 `FilterType::Nearest`，观察高频的棋盘纹理区域是否出现混叠杂点，体会模糊的必要性。
4. **进阶·图像融合**：合成左半亮、右半暗的两张图，各建拉普拉斯金字塔，按层用一个"从左到右平滑过渡"的掩膜混合，再重建。对比它和"直接左右各取一半硬拼"的接缝差异。

下一章，我们换一副眼镜看图像——不再盯着一个个像素，而是把它看成**各种频率波纹的叠加**，用**傅里叶变换**在频域里做模糊、锐化和去噪。👉 [第 22 章 频域处理：傅里叶变换与频域滤波](ch22-frequency.md)
