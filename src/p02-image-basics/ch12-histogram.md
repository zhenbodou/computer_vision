# 第 12 章 直方图与直方图均衡化

> **本章导读**
> - 看懂**直方图（histogram）**：它就是"每种灰度有多少个像素"的一张统计柱状图，是判断曝光、辅助选阈值的利器。
> - 用 `[u32; 256]` 亲手统计灰度直方图，并用 `imageproc` 把它画成条形图（外加一个纯文本柱状图，随手可看）。
> - 理解**直方图均衡化（histogram equalization）**的原理——用累积分布函数（CDF）把挤成一团的灰度"摊平"，自动拉开对比度。
> - 手写一遍均衡化，再对照 `imageproc::contrast::equalize_histogram`，对一张偏暗图看前后对比。📊

## 场景引入：这张照片曝光对不对？

摄影师按下快门后，第一件事往往是瞄一眼相机屏幕角落的一个"小山丘"图——那就是**直方图**。它不看照片内容，只回答一个问题：**从最暗到最亮，各种明暗的像素分别有多少？**

- 山丘全堆在**左边**：像素大多很暗 → 照片**欠曝**（太黑）；
- 全堆在**右边**：大多很亮 → **过曝**（太白）；
- 挤在**中间窄窄一条**：明暗差距小 → **对比度低**（灰蒙蒙）；
- 从左到右**铺得比较开**：层次丰富，曝光通常比较健康。

一张小小的统计图，就能客观告诉你"曝光对不对、对比够不够"，不用凭肉眼猜。这一章我们就来统计它、画出来，并用它**自动**改善一张偏暗照片。

## 原理：从"计数"到"摊平"

### 直方图就是计数

灰度图每个像素取值 0~255。**灰度直方图**做的事简单到不能再简单：**准备 256 个计数器，扫一遍全图，看到值为 v 的像素就给第 v 个计数器加一。** 数组 `[u32; 256]` 正好当这 256 个计数器。

```text
计数器:  hist[0] hist[1] ... hist[128] ... hist[255]
含义  :  值=0的   值=1的      值=128的       值=255的
         像素数   像素数      像素数         像素数
```

> 💡 **提示**：直方图**丢掉了位置信息**——它只关心"有多少个某灰度的像素"，不关心它们在图的哪儿。所以两张完全不同的图可能有一模一样的直方图。但正因为它是全局统计，才特别适合快速判断曝光、对比度，以及为下一步"选一个分割阈值"提供依据（见 [第 16 章 阈值分割](../p03-classic/ch16-threshold.md)，Otsu 法就是在直方图上找最佳分界点）。

### 直方图均衡化：把拥挤的灰度摊开

一张灰蒙蒙的低对比度图，直方图挤在中间一小段——大量像素的值很接近，肉眼就分不出层次。**直方图均衡化**的思路非常直观：**把这些挤在一起的灰度值重新分配，让它们尽量均匀地铺满整个 0~255**，差距自然就拉开了，对比度随之提升。

怎么"摊"才科学？答案是用**累积分布函数（CDF, Cumulative Distribution Function）**。CDF 在灰度 \\( v \\) 处的值，就是"值 ≤ \\( v \\) 的像素总数"。均衡化的映射公式是：

\\[ I_{out}(v) = \text{round}\left( \frac{C(v) - C_{\min}}{N - C_{\min}} \times 255 \right) \\]

其中 \\( C(v) \\) 是灰度 \\( v \\) 处的累积计数（CDF），\\( N \\) 是总像素数，\\( C_{\min} \\) 是最小的非零累积值（用于把最暗的像素对齐到 0）。

直觉解释：**CDF 越陡的地方（像素越密集的灰度段），映射后被拉得越开**；CDF 平缓的地方（像素稀少）则被压缩。于是原本拥挤的中间调被拉伸开，稀疏的两端被压紧——整体分布被"摊平"，对比度提升。

```text
均衡化前(挤在中段)          均衡化后(摊满全程)
  ▁▁▂▅█▇▃▁▁▁▁▁▁            ▂▃▂▃▄▃▄▃▂▃▄▂▃
  0        128      255     0        128      255
```

## 动手写代码

依赖：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

### 统计直方图，并画成条形图

```rust
use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;

/// 统计灰度直方图：256 个计数器，扫一遍全图即可
fn histogram(img: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for px in img.pixels() {
        hist[px[0] as usize] += 1; // 值为 px[0] 的计数器 +1
    }
    hist
}

/// 把直方图画成 256×120 的条形图（白底黑柱），高度按最大频次归一化
fn draw_histogram(hist: &[u32; 256]) -> RgbImage {
    let (w, h) = (256u32, 120u32);
    let mut canvas: RgbImage = ImageBuffer::from_pixel(w, h, Rgb([255, 255, 255]));
    let max = *hist.iter().max().unwrap_or(&1) as f32;
    for i in 0..256u32 {
        // 该灰度的柱子高度，按最高柱归一化到画布高度
        let bar = (hist[i as usize] as f32 / max * h as f32).round() as u32;
        if bar > 0 {
            let top = h - bar; // 从底部往上画
            draw_filled_rect_mut(
                &mut canvas,
                Rect::at(i as i32, top as i32).of_size(1, bar), // 每根柱宽 1 像素
                Rgb([30, 30, 30]),
            );
        }
    }
    canvas
}

/// 附赠：在终端打印一个粗粒度的文本柱状图（把 256 档并成 32 档）
fn print_text_histogram(hist: &[u32; 256]) {
    let mut bins = [0u32; 32];
    for i in 0..256 {
        bins[i / 8] += hist[i]; // 每 8 个灰度合并成一档
    }
    let max = *bins.iter().max().unwrap_or(&1) as f32;
    for (k, &c) in bins.iter().enumerate() {
        let len = (c as f32 / max * 40.0).round() as usize;
        println!("{:>3}..{:<3} | {}", k * 8, k * 8 + 7, "█".repeat(len));
    }
}
```

### 合成一张偏暗图，做均衡化

先合成一张**又暗又平**的图（灰度全压在 20~90 的窄区间），正好给均衡化"表演"的机会。再分别用**手写版**和 `imageproc` 官方版做均衡化。

```rust
/// 合成一张偏暗、低对比的图：中心稍亮的圆盘，但整体压在暗部 [20,90]
fn make_dark_scene(w: u32, h: u32) -> GrayImage {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let r = w as f32 / 2.0;
    ImageBuffer::from_fn(w, h, |x, y| {
        let d = (((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / r).min(1.0);
        let v = 20.0 + 70.0 * (1.0 - d); // 中心≈90，边缘≈20，全落在暗部
        Luma([v.clamp(0.0, 255.0) as u8])
    })
}

/// 手写直方图均衡化：先求 CDF，再按公式生成 LUT，最后查表映射
fn equalize_manual(img: &GrayImage) -> GrayImage {
    let hist = histogram(img);
    let total = (img.width() * img.height()) as f32;

    // 1) 求累积分布 CDF：cdf[i] = hist[0] + ... + hist[i]
    let mut cdf = [0u32; 256];
    let mut acc = 0u32;
    for i in 0..256 {
        acc += hist[i];
        cdf[i] = acc;
    }
    // 2) 最小非零累积值，用于把最暗像素对齐到 0
    let cdf_min = *cdf.iter().find(|&&c| c > 0).unwrap_or(&0);

    // 3) 按公式生成映射查找表（复用上一章的 LUT 思想）
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let num = cdf[i].saturating_sub(cdf_min) as f32;
        let den = (total - cdf_min as f32).max(1.0);
        lut[i] = (num / den * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    // 4) 查表输出
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        Luma([lut[img.get_pixel(x, y)[0] as usize]])
    })
}
```

### 主程序：前后对比

```rust
use imageproc::contrast::equalize_histogram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dark = make_dark_scene(256, 256);
    dark.save("dark.png")?;

    // 均衡化前的直方图
    let h_before = histogram(&dark);
    draw_histogram(&h_before).save("hist_before.png")?;
    println!("== 均衡化前的直方图（灰度集中在暗部）==");
    print_text_histogram(&h_before);

    // 手写均衡化 vs imageproc 官方实现
    let eq_manual = equalize_manual(&dark);
    eq_manual.save("eq_manual.png")?;
    let eq_lib = equalize_histogram(&dark); // imageproc 一行搞定
    eq_lib.save("eq_lib.png")?;

    // 均衡化后的直方图（应明显铺开）
    let h_after = histogram(&eq_manual);
    draw_histogram(&h_after).save("hist_after.png")?;
    println!("\n== 均衡化后的直方图（明显铺开到全程）==");
    print_text_histogram(&h_after);

    Ok(())
}
```

运行后对比 `dark.png` 与 `eq_manual.png`：原本灰暗一片、几乎看不清的圆盘，均衡化后层次分明、对比强烈。再看 `hist_before.png` 和 `hist_after.png`：柱子从挤在左侧一小簇，变成铺满整条灰度轴。手写版 `eq_manual.png` 与官方 `eq_lib.png` 效果应当基本一致——你已经亲手复现了库函数的核心逻辑。

> 📖 **术语**：`imageproc::contrast::equalize_histogram(&GrayImage) -> GrayImage` 只处理**单通道灰度图**。想给彩色图做均衡化，切忌对 R/G/B 三通道分别均衡（会严重偏色）；正确做法是转到亮度/明度通道（如 YUV 的 Y、HSV 的 V），只对该通道均衡后再转回来——这正是 [第 9 章 颜色空间](ch09-color-spaces.md) 埋下的用处。

## 常见坑与提示

> ⚠️ **注意**：直方图计数器要用 `u32`（甚至更大），别用 `u8`。一张 4K 图有八百多万像素，某个灰度的计数轻松破万，`u8` 直接溢出回绕，统计全错。

> 🕳️ **坑**：全局直方图均衡化是"一刀切"的——它按整幅图的分布来拉伸，可能把本来平滑的天空、墙面**放大出噪点和斑块**，也可能让局部过曝。工程上更稳的是**限制对比度的自适应均衡化（CLAHE）**：分块做、并给对比度设上限。`imageproc` 现成的是全局版；CLAHE 可在理解本章原理后自行按块实现。

> 💡 **提示**：均衡化的映射本质仍是一张 `[u8;256]` 的 **LUT**（回顾 [第 11 章](ch11-point-ops.md)）——先用直方图/CDF 把表算出来，再逐像素查表。所有"看似复杂"的全局灰度变换，最后几乎都能落到这张 256 项的小表上，既统一又高效。

## 本章小结

- **直方图**用 `[u32;256]` 统计每种灰度的像素个数，是判断曝光（欠曝/过曝）、对比度高低、以及辅助选阈值的基础工具；它只统计数量、丢弃位置。
- **直方图均衡化**通过**累积分布函数 CDF** 把拥挤的灰度重新摊开到 0~255，自动提升对比度；映射公式的核心是用归一化后的 CDF 当作新的灰度值。
- 手写实现四步：**统计直方图 → 求 CDF → 按公式生成 LUT → 逐像素查表**；`imageproc::contrast::equalize_histogram` 一行即可完成同样的事，效果与手写版一致。
- 彩色图应在亮度通道上均衡；全局均衡可能放大噪声，进阶方案是分块限幅的 CLAHE。

## 动手练习

1. **读懂直方图**：分别对一张偏暗、一张偏亮、一张正常曝光的图（可用合成图或自己的照片）统计直方图，用 `print_text_histogram` 打印，观察"山丘"位置与曝光的对应关系。
2. **均衡化前后对比**：在同一张画布上，把均衡化前后的两个条形图上下拼在一起保存，直观展示"挤 → 摊平"。
3. **验证一致性**：写个小函数逐像素比较 `equalize_manual` 与 `equalize_histogram` 的输出，统计有多少像素不同、最大差值多少，思考差异来源（取整策略）。
4. **进阶：分块均衡（简化版 CLAHE）**：把图切成 8×8 个小块，各自独立做一次均衡化后拼回，和全局均衡化对比效果差异（先不做插值和限幅，感受"分块"这一步的作用）。

至此，第二部分"图像处理入门"就全部走完了——你已经能读写、改像素、转颜色空间、做几何变换、调亮度对比度、玩转直方图。下一部分，我们进入**经典图像处理算法**，从"卷积"这个贯穿始终的核心操作讲起。👉 [第 13 章 卷积原理与线性滤波](../p03-classic/ch13-convolution.md)

## 第二部分阶段验收：图像契约与数值一致性

建立一组黄金图覆盖灰度/RGB/RGBA、8/16 位、EXIF 方向、奇数尺寸和透明度。对每条处理链
记录颜色空间、transfer function、alpha 语义、坐标约定、插值与舍入；做“编码→解码”和
“几何变换→逆变换”的误差统计。基准测试必须区分解码、分配、像素计算与编码，并验证
峰值内存。若结果无法说明是线性 RGB 还是 gamma 编码 RGB，本部分不算通过。
