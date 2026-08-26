# 第 20 章 模板匹配：在大图里找小图

> **本章导读**
> - 学完你能：用**模板匹配**在一张大图里定位一个已知的小图案，并把它框出来。
> - 理解衡量"像不像"的几种度量：**误差平方和（SSD）**、**互相关（Cross Correlation）**，以及它们的**归一化**版本为什么能抗光照变化。
> - 亲手验证一个反差极大的现象：整体变亮时，非归一化方法被"骗"到错误位置，归一化方法稳如泰山。
> - 看清模板匹配的**死穴**——不抗旋转、不抗缩放——为第四部分的特征匹配埋下伏笔。

## 场景：在大图里找一个已知的小图案

想象几个真实需求：在一整版电路板照片里找出某个特定元件的位置；在工业产线上定位产品角上的**基准标记（fiducial mark）**好做对位；在一张屏幕截图里找出某个按钮图标在哪儿。它们都是同一个问题：**我手里有一张"要找的小图"（模板），想知道它出现在大图的哪个位置。**

这就是**模板匹配（template matching）**。它是最朴素、最直观的定位方法，不需要训练、不需要特征，思路简单到一句话能讲完——但正因为简单，它的适用边界也很清晰，摸清它的脾气非常有教育意义。

> 📖 **术语**：**模板匹配（template matching）**。给定一张小图（模板 template）和一张大图（image），在大图上逐位置计算模板与该处局部区域的**相似度**，相似度最高（或差异最小）的位置，就认为是模板出现的地方。

## 原理：滑动窗口 + 相似度打分

### 直觉：拿透明胶片去对齐

把模板想象成一张印着图案的**透明胶片**。你把它放在大图左上角，看它和底下盖住的那块"对不对得上"；然后向右挪一个像素，再看一次；一直挪到右下角。每个位置都打一个"像不像"的分，**分最高（最像）的那个位置，就是匹配结果**。

```text
   大图 image                模板 template 在每个位置滑动打分
 ┌───────────────┐          ┌───┐
 │ · · · · · · · │          │ ▣ │ ← 在 (x,y) 处，比较模板
 │ · ┌───┐ · · · │  滑动──►  └───┘   与它盖住的那块区域
 │ · │▣▣▣│ · · · │          得到一张“得分图”：
 │ · └───┘ · · · │          每个像素 = 该位置的相似度
 └───────────────┘
```

`imageproc` 的 `match_template` 返回的正是这样一张**得分图**（score map），尺寸比原图小一圈：宽 `image_w - template_w + 1`、高 `image_h - template_h + 1`（因为模板不能滑出边界）。得分图上每个像素，就是模板左上角放在该处时的相似度分数。

### 三种"像不像"的算法

怎么给"像不像"打分？记模板为 \\(T\\)、大图在 \\((x,y)\\) 处盖住的区域为 \\(I\\)，逐像素比较：

**① 误差平方和（SSD, Sum of Squared Errors）**：把对应像素的差**平方**再求和。

\\[ \text{SSD}(x,y) = \sum_{x',y'} \big( T(x',y') - I(x{+}x',\,y{+}y') \big)^2 \\]

**越小越像**，完全一致时等于 0。最符合直觉，但对整体亮度变化敏感。

**② 互相关（Cross Correlation）**：把对应像素**相乘**再求和。

\\[ \text{CC}(x,y) = \sum_{x',y'} T(x',y') \cdot I(x{+}x',\,y{+}y') \\]

**越大越像**。但它有个大毛病：**偏爱"亮"的区域**——因为大数乘大数结果就大。一块又白又亮但根本不是目标的区域，可能仅仅因为像素值高，就骗到比真实目标更高的分。

**③ 归一化版本**：把上面两者除以一个"能量归一化项"（模板与区域各自的模长）：

\\[ \text{NCC}(x,y) = \frac{\sum T \cdot I}{\sqrt{\sum T^2 \cdot \sum I^2}} \\]

归一化的效果是**只看"形状/图案像不像"，不看整体明暗**。整幅图变亮变暗（相当于像素值乘一个系数），分子分母同步缩放、约掉了，得分几乎不变。这就是**抗光照**的来历。`imageproc` 提供四种方法：`SumOfSquaredErrors`、`SumOfSquaredErrorsNormalized`、`CrossCorrelation`、`CrossCorrelationNormalized`。

## 动手写代码：找回模板并对比归一化

我们合成一张大图：灰底、左上角放一个"靶心"图案（**待找的目标**）、右下角故意放一块**很亮的方块**（专门用来骗非归一化方法）。然后从大图里裁出靶心当模板，用三种方法找它，再把整幅图**变亮 1.4 倍**后重来一遍，对比归一化与非归一化的表现。

`Cargo.toml` 依赖：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch20_template_matching.rs`）：

```rust
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};

/// 合成大图：灰底 + 左上“靶心”(目标) + 右下一块很亮的方块(干扰) + 若干填充物
fn make_big(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([90]));
    draw_filled_circle_mut(&mut img, (70, 60), 18, Luma([150])); // 靶心：外圈亮
    draw_filled_circle_mut(&mut img, (70, 60), 8, Luma([30]));   // 靶心：内芯暗
    // 干扰：一块接近白的方块，值很高，专门骗“非归一化互相关”
    draw_filled_rect_mut(&mut img, Rect::at(200, 120).of_size(46, 46), Luma([250]));
    draw_filled_circle_mut(&mut img, (150, 150), 16, Luma([120])); // 其它填充物
    draw_filled_rect_mut(&mut img, Rect::at(30, 150).of_size(40, 30), Luma([60]));
    img
}

/// 整体“变亮”：像素乘系数后饱和截断到 [0,255]（模拟增益/曝光变化）
fn brighten(img: &GrayImage, factor: f32) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        Luma([(img.get_pixel(x, y)[0] as f32 * factor).min(255.0) as u8])
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let big = make_big(300, 200);
    let (tx, ty, tw, th) = (50u32, 40, 40, 40); // 目标真实位置(50,40)与模板尺寸 40x40

    // 从大图里裁一小块当模板
    let template = image::imageops::crop_imm(&big, tx, ty, tw, th).to_image();

    // 方法一：SSD 误差平方和，越小越匹配（完全一致时为 0）
    let sse = match_template(&big, &template, MatchTemplateMethod::SumOfSquaredErrors);
    let e = find_extremes(&sse);
    println!("[SSE] 最佳(最小)位置 = {:?}, 最小误差 = {}", e.min_value_location, e.min_value);

    // 方法二：非归一化互相关，越大越匹配 —— 会被“亮”骗
    let cc = match_template(&big, &template, MatchTemplateMethod::CrossCorrelation);
    let e_cc = find_extremes(&cc);
    println!("[CC 非归一化] 位置 = {:?}, 最大值 = {:.0}", e_cc.max_value_location, e_cc.max_value);

    // 方法三：归一化互相关，越大越匹配 —— 只看形状
    let ncc = match_template(&big, &template, MatchTemplateMethod::CrossCorrelationNormalized);
    let e_ncc = find_extremes(&ncc);
    println!("[NCC 归一化] 位置 = {:?}, 最大值 = {:.4}", e_ncc.max_value_location, e_ncc.max_value);

    // 整体变亮 1.4 倍后再比一次
    let bright = brighten(&big, 1.4);
    let cc_b = match_template(&bright, &template, MatchTemplateMethod::CrossCorrelation);
    let ncc_b = match_template(&bright, &template, MatchTemplateMethod::CrossCorrelationNormalized);
    let e_cc_b = find_extremes(&cc_b);
    let e_ncc_b = find_extremes(&ncc_b);
    println!("[变亮后 CC] 位置 = {:?}, 最大值 = {:.0}", e_cc_b.max_value_location, e_cc_b.max_value);
    println!("[变亮后 NCC] 位置 = {:?}, 最大值 = {:.4}", e_ncc_b.max_value_location, e_ncc_b.max_value);

    // 画框可视化：绿=NCC 找对的位置，红=非归一化被骗到的位置
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(big.clone()).to_rgb8();
    let (gx, gy) = e_ncc.max_value_location;
    draw_hollow_rect_mut(&mut vis, Rect::at(gx as i32, gy as i32).of_size(tw, th), Rgb([0, 255, 0]));
    let (rx, ry) = e_cc.max_value_location;
    draw_hollow_rect_mut(&mut vis, Rect::at(rx as i32, ry as i32).of_size(tw, th), Rgb([255, 0, 0]));
    vis.save("out_ch20_matched.png")?;
    Ok(())
}
```

运行 `cargo run --example ch20_template_matching`，终端打印（数字真实可复现）：

```text
[SSE] 最佳(最小)位置 = (50, 40), 最小误差 = 0
[CC 非归一化] 位置 = (200, 120), 最大值 = 45465000
[NCC 归一化] 位置 = (50, 40), 最大值 = 1.0000
[变亮后 CC] 位置 = (200, 120), 最大值 = 46374288
[变亮后 NCC] 位置 = (50, 40), 最大值 = 1.0000
```

这五行把模板匹配的性格暴露得淋漓尽致：

1. **SSD 完美命中**：最小误差位置 `(50, 40)` 正是目标真实位置，误差 `0`（因为模板就是从这儿裁的，逐像素一模一样）。
2. **非归一化互相关被"亮"骗了**：它选了 `(200, 120)`——那块又白又亮、跟靶心毫不相干的干扰方块！只因为它像素值高，乘出来的和就大。
3. **归一化互相关一击即中**：位置 `(50, 40)` 正确，得分 `1.0000`（完美匹配的理论上限）。
4. **变亮后见真章**：整幅图乘 1.4 变亮，非归一化互相关的**得分变了**（45465000 → 46374288）且**依旧被骗**在 `(200, 120)`；而归一化互相关**位置纹丝不动、得分仍是 1.0000**——对整体亮度变化完全免疫。

打开 `out_ch20_matched.png`：**绿框**稳稳套住左上角的靶心（归一化的功劳），**红框**却套在了右下那块亮方块上（非归一化的翻车现场）。

## 常见坑与提示

> 🕳️ **坑**：**非归一化方法对光照极度敏感，工程上几乎总用归一化版本。** 上面的数字就是铁证：只要图里有个更亮的区域，或者整体明暗一变，`CrossCorrelation` / `SumOfSquaredErrors` 就可能给出离谱结果。除非你能保证光照绝对恒定，否则请默认选 `CrossCorrelationNormalized`（或 `SumOfSquaredErrorsNormalized`）。

> ⚠️ **注意**：**模板不能比大图大，否则 `match_template` 直接 panic。** 它要求模板的宽和高都**严格小于**大图对应的尺寸。裁模板、缩放图像时留个心眼，别让模板"撑破"了图。

> 💡 **提示**：**要找多个目标，一次 `find_extremes` 不够。** `find_extremes` 只返回**一个**最值位置。若大图里有多个相同图案，得自己在得分图上**设阈值**挑出所有高分点，再做**非极大抑制**（见 [第 15 章](ch15-edges.md) 里 Canny 的思路，或对得分图找局部极大值），否则同一个目标周围会冒出一簇几乎一样高的点。

> 🕳️ **坑**：**模板匹配不抗旋转、不抗缩放，这是它的死穴。** 模板一旦转个角度或大小变了，逐像素比对立刻失配、得分暴跌。它只适合**目标姿态、大小基本固定**的场景（工业对位、UI 图标定位等）。想在这上面做多尺度，可以配合下一章的图像金字塔在几个尺度上各匹配一遍；但要真正做到**旋转/缩放不变**，就得请出第四部分的**局部特征**了。

## 本章小结

- 模板匹配 = **滑动窗口 + 相似度打分**：模板在大图上逐位置滑动，`match_template` 产出一张比原图小一圈的**得分图**，`find_extremes` 从中取极值位置。
- 相似度度量：**SSD** 越小越像（完全一致为 0）；**互相关**越大越像但**偏爱亮区**；**归一化**版本除以能量项，**抗整体明暗变化**——实测非归一化被亮方块骗到 `(200,120)`，归一化稳定命中 `(50,40)` 且得分恒为 1.0。
- 工程默认用**归一化**方法；注意**模板不能比图大**（否则 panic）；**多目标**需阈值 + 非极大抑制。
- 模板匹配**不抗旋转、不抗缩放**，只适合姿态固定的场景——这正是特征匹配要解决的问题。

## 动手练习

1. **换方法对比**：把主匹配从 `SumOfSquaredErrors` 换成 `SumOfSquaredErrorsNormalized`，再把 `brighten` 的系数调成 `0.6`（变暗），观察归一化与非归一化谁还能找对位置。
2. **多目标检测**：在 `make_big` 里再画一个一模一样的靶心，然后遍历归一化得分图，把所有得分 `> 0.9` 的位置都框出来（记得对相邻高分点做简单去重）。
3. **画得分热力图**：把归一化得分图（`Image<Luma<f32>>`）线性拉伸到 `0..255` 存成灰度图，直观看看"哪里亮=哪里像"，找找靶心位置是不是最亮的那个点。
4. **感受死穴**：把模板用 [第 10 章](../p02-image-basics/ch10-geometry.md) 的旋转变换转 15° 再去匹配，看看得分掉得多惨、位置是否还对——亲身体会"为什么需要特征"。

模板匹配靠的是**逐像素**死板比对，一转一缩就崩。下一章我们先补上**多尺度**这块拼图——用图像金字塔在不同分辨率上处理图像；而彻底摆脱旋转缩放束缚的**局部特征**，我们留到第四部分再隆重登场。👉 [第 21 章 图像金字塔与多尺度处理](ch21-pyramids.md)
