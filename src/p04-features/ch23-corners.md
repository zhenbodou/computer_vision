# 第 23 章 角点检测（Harris、FAST、Shi-Tomasi）

> **本章导读**
> - 想清楚一件事：为什么图像拼接、匹配、跟踪都离不开"特征点"这种可重复检测的**锚点**。
> - 建立直觉：什么样的点才是"好点"——为什么**角点**比边缘、比平坦区更值得当锚点。
> - 摸懂两类经典角点检测器：**Harris**（自相关/结构张量的思路，简述）与 **FAST**（看圆环、极快）。
> - 动手用 `imageproc` 的 `corners_fast9` 检测角点，并用**非极大值抑制（NMS）**把扎堆的点收拾干净。

## 场景：把两张照片拼成一张全景图

你举着手机拍全景，手机在背后干的事是：连拍好几张有**重叠**的照片，再把它们无缝拼接。可拼接的前提是——程序得知道"第一张照片右边那栋楼的屋角，就是第二张照片左边那栋楼的屋角"，也就是**在两张图里找到同一个物理点**。

怎么找？总不能拿整张图去比对。聪明的办法是：先在每张图里挑出一批**又稳又好认的点**，再拿这些点跨图配对。这批点就叫**特征点（feature point）**，也叫**关键点（keypoint）**或**兴趣点（interest point）**。图像拼接、目标跟踪、三维重建、SLAM……几乎所有"要在多张图之间建立联系"的任务，第一步都是它。

> 📖 **术语**：**特征点 / 关键点（keypoint）**。图像里一个**位置明确、易于重复检测、且对光照和视角变化相对稳定**的小地方。好的关键点应满足两个要求：一是**可重复（repeatable）**——换个角度、换个光照再拍，还能在同一个物理位置检测到它；二是**可区分（distinctive）**——它周围长得有辨识度，不容易和别处搞混。本章先解决"在哪"（检测位置），下一章再解决"长啥样"（[第 24 章 描述子](ch24-descriptors.md)）。

## 什么样的点才是"好点"

想象你拿一个小窗口在图像上滑动，观察窗口里的灰度怎么变。按变化情况，图像上的点分三类：

```text
   平坦区              边缘               角点
 ┌───────┐         ┌───────┐         ┌───────┐
 │       │         │■■■■■■■│         │■■■│   │
 │       │         │■■■■■■■│         │■■■│   │
 │       │         │       │         │───┘   │
 └───────┘         └───────┘         └───────┘
 往任意方向挪     沿边挪不变，       往任意方向挪
 灰度都不变       垂直边挪才变       灰度都剧烈变
```

- **平坦区**（比如一面白墙）：窗口往哪个方向挪，里面几乎没变化。这种点**没法定位**——你说不清它到底在墙的哪个位置，换张图根本找不回来。
- **边缘**（比如墙和天空的交界线）：窗口**沿着**边缘方向挪，灰度不变；只有**垂直**边缘挪才变。这叫**孔径问题（aperture problem）**——你能确定它在"哪条线上"，却说不清在线上的**哪一点**，滑来滑去都一样。
- **角点**（比如窗户的直角、棋盘格的交叉点）：窗口**往任意方向**挪，灰度都会明显变化。于是这个点被"卡死"在唯一的位置上——这正是我们想要的**好锚点**。

一句话：**角点 = 沿两个方向灰度都剧烈变化的点**，它比边缘、比平坦区都更容易被精确、可重复地定位。这就是"角点检测"成为特征工程第一课的原因。

## Harris 角点：用"结构张量"量化变化

怎么把上面的直觉变成能算的数？1988 年的 Harris 检测器给了一个经典答案。它考察：窗口从 \\( (x,y) \\) 挪动一个小位移 \\( (u,v) \\) 时，窗口内灰度的**加权平方差**有多大。经过泰勒展开，这个"变化能量"可以近似写成一个 2×2 矩阵 \\( M \\)（称为**结构张量**或**二阶矩矩阵**）主导的二次型，其中 \\( I_x, I_y \\) 是图像在该点的水平/垂直梯度（见 [第 15 章 边缘检测](../p03-classic/ch15-edges.md)）：

\\[ M = \sum_{(x,y)\in W} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\\\ I_x I_y & I_y^2 \end{bmatrix} \\]

这个 \\( M \\) 的两个**特征值** \\( \lambda_1, \lambda_2 \\) 就编码了"窗口在两个主方向上灰度变化的剧烈程度"：

- 两个都小 → 平坦区；
- 一大一小 → 边缘；
- **两个都大 → 角点**。

直接求特征值有点费劲，Harris 用一个巧妙的**响应函数**绕开了它（利用"行列式=特征值之积、迹=特征值之和"）：

\\[ R = \det(M) - k\,(\operatorname{trace} M)^2 = \lambda_1 \lambda_2 - k\,(\lambda_1 + \lambda_2)^2 \\]

其中 \\( k \\) 是个经验系数（常取 `0.04~0.06`）。\\( R \\) 很大说明是角点，\\( R \\) 为负说明是边缘，\\( R \\) 很小说明是平坦区。

> 💡 **提示**：**Shi-Tomasi** 是 Harris 的近亲。它认为与其用 \\( R \\) 那个组合式，不如直接看**较小的那个特征值** \\( \min(\lambda_1, \lambda_2) \\)：只要它够大，就说明两个方向都变化剧烈，就是好角点。这个判据更稳，就是经典的"Good Features to Track"，也是光流跟踪（[第 59 章](../p13-tracking/ch59-optical-flow.md)）里常用的选点方法。原理和 Harris 一脉相承，本章不再单独实现。

Harris/Shi-Tomasi 又准又稳，但要算梯度、算矩阵、求特征值，**偏慢**。当你要在每秒几十帧的视频上实时找几千个点时，就轮到又快又糙的 FAST 上场了。

## FAST：只看一圈，快到飞起

FAST（Features from Accelerated Segment Test）的思路简单粗暴到可爱：**判断一个点是不是角点，只需看它周围一圈像素。**

具体做法：以待测点 P 为中心，取半径 3 的一圈共 **16** 个像素（Bresenham 圆）。设 P 的灰度是 \\( I \\)、给定一个阈值 \\( t \\)。如果这一圈里存在**连续 N 个**像素**都比 \\( I + t \\) 亮**，或者**都比 \\( I - t \\) 暗**，那 P 就是角点。

```text
        16 3  4              连续一段(比如底部这 9 个)
     15    ●    5            都明显比中心暗 → P 是角点
   14        ●    6
   13        P    7          中心 P 的灰度 = I
   12        ●    8          阈值 = t
     11    ●    9
        10 ... 
```

`N=9` 的版本叫 **FAST-9**（`imageproc` 的 `corners_fast9`），`N=12` 叫 FAST-12。为什么这样就能判角点？回想角点的定义"往任意方向灰度都大变"——如果圆环上有连续一大段都比中心暗（或亮），说明中心处在明暗交界的"拐角"上，正是角点特征。而它快就快在：实现里可以**先只查上下左右 4 个点**，一旦这 4 个里不满足条件就立刻排除，绝大多数非角点像素一两次比较就被踢掉，根本不用看全 16 个。

FAST 还会给每个角点打一个**分数（score）**：让该点仍能被判为角点的**最大阈值 \\( t \\)**。分数越高，角点越"结实"、对比越强——这个分数待会做非极大值抑制时正好派上用场。

## 动手写代码：FAST 检测 + 非极大值抑制

我们合成一张自包含的灰度图：白底上放三个**对比度不同**的深色方块，再加一小片**棋盘格**（棋盘格的交叉点是教科书级的角点）。然后：① 试不同阈值看角点数量怎么变；② 做一次 NMS 把扎堆的角点收拢。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch23_corners.rs`）：

```rust
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::corners::{corners_fast9, Corner};
use imageproc::drawing::{draw_cross_mut, draw_filled_rect_mut};
use imageproc::rect::Rect;
use imageproc::suppress::suppress_non_maximum;

/// 合成灰度“照片”：白底 + 三个不同对比度的方块 + 一小片棋盘格。
/// 方块颜色越浅、与背景对比越弱，角点“分数”越低，越容易被高阈值滤掉。
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([235])); // 近白背景
    draw_filled_rect_mut(&mut img, Rect::at(40, 40).of_size(70, 70), Luma([30]));   // 强对比
    draw_filled_rect_mut(&mut img, Rect::at(150, 55).of_size(55, 55), Luma([130])); // 中对比
    draw_filled_rect_mut(&mut img, Rect::at(60, 165).of_size(55, 55), Luma([185])); // 弱对比
    let (ox, oy, cell, n) = (250i32, 150i32, 18i32, 4);
    for gy in 0..n {
        for gx in 0..n {
            if (gx + gy) % 2 == 0 {
                draw_filled_rect_mut(
                    &mut img,
                    Rect::at(ox + gx * cell, oy + gy * cell).of_size(cell as u32, cell as u32),
                    Luma([20]),
                );
            }
        }
    }
    img
}

/// 把角点画成十字标记，输出到彩色图上便于观察
fn draw_corners(gray: &GrayImage, corners: &[Corner], color: Rgb<u8>) -> RgbImage {
    let mut canvas = RgbImage::new(gray.width(), gray.height());
    for (x, y, p) in gray.enumerate_pixels() {
        let v = p[0];
        canvas.put_pixel(x, y, Rgb([v, v, v])); // 灰度铺成底图
    }
    for c in corners {
        // draw_cross_mut 的坐标是 i32，Corner 的 x/y 是 u32，要转一下
        draw_cross_mut(&mut canvas, color, c.x as i32, c.y as i32);
    }
    canvas
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gray = make_scene(360, 260);

    // ① 阈值高低对角点数量的影响
    for t in [20u8, 70, 120, 170] {
        let corners = corners_fast9(&gray, t);
        println!("threshold={:>3} -> FAST 角点数 = {}", t, corners.len());
    }

    // ② 取一个中等阈值，观察角点“扎堆”
    let threshold = 40u8;
    let corners = corners_fast9(&gray, threshold);
    println!("\nthreshold={} 时原始角点数 = {}", threshold, corners.len());

    // ③ 非极大值抑制（NMS）：把每个角点的分数写进一张响应图，
    //    再用 suppress_non_maximum 只保留局部窗口里分数最高的那个。
    let mut response = GrayImage::new(gray.width(), gray.height());
    for c in &corners {
        // corners_fast9 的 score 本质是 u8 阈值，可无损塞进 Luma<u8>
        response.put_pixel(c.x, c.y, Luma([c.score as u8]));
    }
    let radius = 6u32;
    let suppressed = suppress_non_maximum(&response, radius);
    let kept: Vec<Corner> = suppressed
        .enumerate_pixels()
        .filter(|(_, _, p)| p[0] > 0)
        .map(|(x, y, p)| Corner::new(x, y, p[0] as f32))
        .collect();
    println!("NMS(radius={}) 后保留角点数 = {}", radius, kept.len());

    // ④ 可视化：原始角点(红) vs NMS 后角点(绿)
    draw_corners(&gray, &corners, Rgb([255, 0, 0])).save("out_ch23_fast_raw.png")?;
    draw_corners(&gray, &kept, Rgb([0, 200, 0])).save("out_ch23_fast_nms.png")?;
    Ok(())
}
```

运行 `cargo run --example ch23_corners`，终端输出（数字真实可复现）：

```text
threshold= 20 -> FAST 角点数 = 156
threshold= 70 -> FAST 角点数 = 132
threshold=120 -> FAST 角点数 = 108
threshold=170 -> FAST 角点数 = 108

threshold=40 时原始角点数 = 156
NMS(radius=6) 后保留角点数 = 26
```

三个结论：

1. **阈值越高，角点越少**：`20 → 156`、`70 → 132`、`120 → 108`。阈值就是"对比度门槛"——它先滤掉了最弱对比的浅灰方块（185 vs 235），再滤掉中对比方块（130 vs 235），最后 `120` 和 `170` 都是 `108`，因为只剩下强对比方块和棋盘格，它们对比极强、分数很高，再抬阈值也纹丝不动。
2. **原始角点严重扎堆**：`156` 个点里，一个真实的角落往往被检出好几次（相邻像素都满足条件），挤成一小簇。
3. **NMS 把 156 个收拢到 26 个**：每个真实角落只保留"分数最高"的那一个代表。打开 `out_ch23_fast_nms.png`，你会看到绿色十字干净地落在每个方块的 4 个直角和棋盘格的交叉点上，不再是红图里那种一坨一坨的红斑。

`suppress_non_maximum` 的工作方式值得说一句：它不直接吃角点列表，而是吃一张**响应图**（每个像素放它的角点分数），输出一张"只在 \\( (2r+1)\times(2r+1) \\) 邻域内取到最大值的位置才保留、其余清零"的图。所以我们先把角点分数"画"回一张 `GrayImage`，抑制完再把非零像素捡回来当作最终角点。

## 常见坑与提示

> 🕳️ **坑**：**FAST 角点天生扎堆，不做 NMS 几乎没法用。** 真实角落附近的一圈像素往往都满足 FAST 条件，于是一个角被报成十几个点。如果直接拿去匹配，既浪费算力又制造混乱。**非极大值抑制是标配的下一步**，别省。抑制半径 `radius` 要按图像分辨率和你想要的角点密度来调：太小抑制不干净，太大会把靠得近的两个真角点误并成一个。

> ⚠️ **注意**：**阈值是一门平衡的艺术。** 阈值调低，角点多、召回高，但会混进大量弱角点和噪声点；调高则只留最"结实"的角点，稳但可能太少、在弱纹理区一个都找不到。工程上常见做法是：设一个偏低的阈值多检一些，再用 NMS + "按分数取前 N 个"来控制数量和质量。

> 🕳️ **坑**：**`draw_cross_mut` 的坐标是 `i32`，而 `Corner` 的 `x/y` 是 `u32`。** 直接传会类型不匹配，记得 `c.x as i32`。这类"库 A 用 u32、库 B 要 i32"的小转换，在 Rust 里编译器会拦住你，别嫌烦——它替你挡掉了越界和溢出的坑。

> 💡 **提示**：**FAST 只告诉你"角点在哪"，不告诉你"它多大、朝哪转"。** 它没有**尺度（scale）**和**方向（orientation）**信息：把图放大一倍，原来的角点可能就检不到了；把图旋转，检测到的还是同一批点，但你无从描述它们各自的朝向。`imageproc` 提供了 `corners::oriented_fast`，它在 FAST 基础上用"灰度质心"估计每个角点的主方向（这正是 ORB 的做法），为下一章的"带方向的描述子"铺路。

## 本章小结

- 图像拼接、匹配、跟踪都需要在多张图之间找**同一个物理点**，于是需要可重复、可区分的**特征点/关键点**。
- 按窗口滑动时的灰度变化，图像点分为平坦区、边缘、角点；**角点沿两个方向灰度都剧烈变化**，最适合当锚点（平坦区无法定位，边缘有孔径问题）。
- **Harris** 用结构张量 \\( M \\) 的两个特征值刻画变化，靠响应函数 \\( R = \det(M) - k\,(\operatorname{trace} M)^2 \\) 判角点；**Shi-Tomasi** 改看 \\( \min(\lambda_1,\lambda_2) \\)，更稳。二者准但偏慢。
- **FAST** 只看半径 3 的 16 像素圆环上有没有"连续 N 个都显著更亮或更暗"，极快，适合实时；`imageproc::corners::corners_fast9` 返回 `Vec<Corner{x,y,score}>`。
- FAST 角点会扎堆，必须配**非极大值抑制（`suppress_non_maximum`）**；阈值控制角点的数量与质量的平衡。
- FAST **不带尺度和方向**，这正是下一章描述子与 ORB 要补上的短板。

## 动手练习

1. **调阈值与半径**：把 NMS 的 `radius` 从 `6` 改成 `3` 和 `12`，观察保留角点数怎么变；再把 FAST 阈值设成 `10` 和 `100`，看棋盘格区域的角点密度变化。
2. **按分数取前 N 个**：不用 NMS，改成把 `corners` 按 `score` 从大到小排序，只取前 20 个画出来。对比它和 NMS 的结果有什么不同（提示：排序法可能仍然扎堆）。
3. **换 FAST-12**：把 `corners_fast9` 换成 `corners_fast12`，同阈值下角点数量是多变少？想想为什么"要求连续 12 个"比"连续 9 个"更严格。
4. **进阶——加噪声**：用 [第 14 章](../p03-classic/ch14-nonlinear-filter.md) 或 `imageproc::noise::gaussian_noise` 给图加点高斯噪声，看看低阈值下会不会冒出许多"假角点"，体会阈值和去噪对鲁棒性的作用。

检测到角点只是万里长征第一步——我们还只知道这些点**在哪**，却不知道它们**长什么样**，也就没法跨图判断"这两个点是不是同一个"。下一章，我们就给每个关键点写一张独一无二的"指纹"。👉 [第 24 章 局部特征描述子（SIFT、ORB、BRIEF）](ch24-descriptors.md)
