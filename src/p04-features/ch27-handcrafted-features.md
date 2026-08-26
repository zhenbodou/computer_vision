# 第 27 章 经典手工特征：HOG、LBP、Haar

> **本章导读**
> - 前几章的角点、SIFT/ORB/BRIEF 都是**局部描述子**——描述"某个点周围长什么样"。这章换个视角：怎么用**一个向量概括整块图像或区域**。
> - 学完你能：动手算出 **HOG（方向梯度直方图）** 特征——擅长描述**形状/轮廓**（经典行人检测）；算出 **LBP（局部二值模式）** 直方图——擅长描述**纹理**（人脸、材质）。
> - 认识 **Haar 特征**的思路（人脸检测的老功臣），知道它留到第十二部分细讲。
> - 想通一件事：**这些特征向量，正是下一步喂给传统机器学习分类器的"输入"**（第六部分）。

## 场景：怎么让机器判断"这块区域里有没有行人"

假设你要做一个行人检测器：滑动一个 `64×128` 的小窗口扫过整张图，每挪到一个位置，就要回答"这个窗口里是不是一个人"。

到这里你已经会找角点、会算 ORB 描述子了（见 [第 24 章](ch24-descriptors.md)）。但那些是**局部**的——描述的是"这个角点周围的花纹"。而"是不是人"是一个关于**整块区域**的判断：它关乎轮廓的整体形状（头、肩、腿的排布），不是某一个点。

我们需要一种能把**一整块图像浓缩成一个定长向量**的方法，让这个向量既能抓住关键信息（形状、纹理），又能直接丢给分类器去学。这一章的 HOG 和 LBP，正是深度学习普及之前最能打的两员大将。

> 📖 **术语**：**手工特征（hand-crafted feature）**。指由人根据经验设计好计算规则的特征提取器（HOG、LBP、Haar 都是），区别于深度学习里**自动学出来**的特征。手工特征快、稳、不需要训练数据，至今在工业界（尤其算力受限的场合）仍在用。

## HOG：用"边缘方向的分布"描述形状

**直觉**：一个物体的形状，本质上是由它的**边缘走向**勾勒出来的。人的轮廓里，肩膀是近乎水平的边、身体两侧是竖直的边。HOG（Histogram of Oriented Gradients）的核心思想就是：**不关心边缘具体在哪，只统计各个方向的边缘各有多少**。

具体分四步：

1. **算梯度**：对整幅灰度图求梯度（本质是 [第 15 章](../p03-classic/ch15-edges.md) 的 Sobel）。每个像素得到梯度的**幅值** \\( m \\)（边缘有多强）和**方向** \\( \theta \\)（边缘朝哪）：

\\[ m = \sqrt{g_x^2 + g_y^2}, \qquad \theta = \arctan2(g_y, g_x) \\]

2. **分格统计（cell）**：把图切成一个个 `cell_side × cell_side` 像素的小方格（cell）。每个 cell 内，把像素按方向投进若干个 bin（比如 9 个方向），**用幅值加权**累加，得到这个 cell 的方向直方图。

3. **分块归一化（block）**：把相邻的 `block_side × block_side` 个 cell 拼成一个 block，对 block 内的直方图做一次 L2 归一化（\\( v \leftarrow v / \lVert v \rVert_2 \\)）。这一步是为了**抵消光照明暗的影响**——同样的形状，在亮处和暗处梯度幅值不同，归一化后就拉回同一尺度。block 之间可以重叠，隔 `block_stride` 个 cell 取一个。

4. **拼接**：把所有 block 的归一化直方图首尾接起来，就是最终的 HOG 特征向量。

方向可以是**无符号**的（`signed=false`，把 \\( \theta \\) 和 \\( \theta+180° \\) 当同一个，范围 `0~180°`，形状任务常用）或**有符号**的（`signed=true`，范围 `0~360°`，区分明暗过渡方向）。

### cell / block / stride 与维度

这三个参数的关系是新手最容易懵的地方，画个图就清楚了（下面是横向，纵向同理）：

```text
一行 8 个 cell（每个 8x8 像素）：
[c0][c1][c2][c3][c4][c5][c6][c7]
 └─block0─┘                        block_side=2, stride=1
     └─block1─┘                    → 每次挪 1 个 cell，相邻 block 重叠一半
         └─block2─┘
             ...            共 (8-2)/1 + 1 = 7 个 block
```

设每行 \\( C_x \\) 个 cell、每列 \\( C_y \\) 个、block 边长 \\( s \\) 个 cell、步长 \\( t \\) 个 cell、方向数 \\( o \\)，则：

\\[ B_x = \frac{C_x - s}{t} + 1, \qquad D = B_x \cdot B_y \cdot o \cdot s^2 \\]

拿经典的行人检测配置（Dalal & Triggs 2005）算一遍：图 `64×128`、cell `8×8`、block `2×2` cell、步长 1 cell、9 个方向。于是 \\( C_x=8, C_y=16 \\)，\\( B_x=7, B_y=15 \\)，每个 block 有 \\( 9\times2^2=36 \\) 维，总维度 \\( 7\times15\times36 = 3780 \\)。等下代码会打印出这个数字对上。

## LBP：用"和邻居比大小"描述纹理

HOG 抓形状，**LBP（Local Binary Pattern，局部二值模式）抓纹理**。纹理是什么？是像素明暗的**局部起伏规律**——砖墙的接缝、皮肤的毛孔、布料的织纹。

**直觉**：一个像素是不是处在某种纹理里，看它和周围邻居的**相对明暗**就知道。LBP 的做法极简单：拿中心像素 \\( p \\) 和它周围 8 个邻居逐个比大小，比中心暗的记 0、不暗的记 1（imageproc 的约定是"中心严格更亮"记 1），按固定顺序拼成一个 8 位二进制数：

```text
邻居编号（位号）      比大小(中心=10)        二进制码
 7  0  1            7  1  0                第0位=1(11<10?否→...)
 6  p  2    －－▶   6  p  2    －－▶   0b... （每位: 中心是否比该邻居亮）
 5  4  3            5  1  0
```

\\[ \text{LBP}(p) = \sum_{i=0}^{7} s(p - n_i)\cdot 2^i, \qquad s(z) = \begin{cases} 1 & z > 0 \\\\ 0 & z \le 0 \end{cases} \\]

于是每个像素得到一个 `0~255` 的 LBP 码。**把整幅图所有像素的 LBP 码统计成一个 256 维直方图**，就是这幅图的纹理特征向量——不同纹理，直方图形状明显不同。

两个实用变体（imageproc 直接提供）：

- **旋转不变**：图一旋转，同一块纹理的 LBP 码会"循环移位"。`min_shift(code)` 取该码所有循环移位里的**最小值**当代表，这样旋转前后归到同一个码——纹理再转也认得。
- **uniform 模式**：如果一个码里 0/1 的跳变次数 \\( \le 2 \\)（比如 `00011100`），叫 **uniform**。自然图像里绝大多数像素都是 uniform 的（对应边、角、平坦这些基本结构）。`count_transitions(code)` 数跳变次数；只保留 uniform 码能把 256 维压到 59 维，又快又不太掉精度。

## Haar 特征：了解一下

标题里还有个 **Haar 特征**，这里点到为止（它的主场在人脸检测，见 [第 56 章](../p12-face/ch56-face-detection.md)）。Haar 特征长得像一块块黑白相间的矩形模板，它的值 = **白色区域像素和 − 黑色区域像素和**。比如"眼睛比脸颊暗"就能用一个"上黑下白"的模板捕捉到。

它凭什么快？靠**积分图（integral image）**：预先算好"从左上角到每个点的矩形区域和"，之后任意矩形的和都能用 4 次加减 O(1) 算出。Viola-Jones 人脸检测器（2001）就是靠成千上万个 Haar 特征 + 级联分类器，第一次在普通 CPU 上做到了实时人脸检测。imageproc 里有 `haar` 与 `integral_image` 模块，我们留到人脸部分再动手。

## 动手写代码：算 HOG 维度 + 提取 LBP 纹理特征

一个例子做两件事：① 对合成的"行人剪影"算 HOG，打印维度并可视化；② 对两种合成纹理算 LBP 直方图，演示纹理区分、uniform 占比和旋转不变性。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch27_handcrafted_features.rs`）：

```rust
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::hog::{cell_histograms, hog, render_hist_grid, HogOptions, HogSpec};
use imageproc::local_binary_patterns::{count_transitions, local_binary_pattern, min_shift};

/// 合成一个 64x128 的"行人剪影"：浅灰背景 + 深色人形（头 + 躯干 + 两条腿）
fn make_person(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as i32, y as i32);
        let cx = w as i32 / 2;
        let head_cy = h as i32 / 6;
        let head_r = w as i32 / 6;
        let in_head = (xf - cx).pow(2) + (yf - head_cy).pow(2) < head_r * head_r;
        let in_body = xf > cx - w as i32 / 5 && xf < cx + w as i32 / 5
            && yf > h as i32 / 4 && yf < 3 * h as i32 / 4;
        let in_legs = yf >= 3 * h as i32 / 4
            && ((xf > cx - w as i32 / 5 && xf < cx - w as i32 / 20)
                || (xf > cx + w as i32 / 20 && xf < cx + w as i32 / 5));
        if in_head || in_body || in_legs { Luma([40u8]) } else { Luma([200u8]) }
    })
}

/// 竖条纹纹理
fn make_stripes(w: u32, h: u32, period: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, _y| {
        if (x / period) % 2 == 0 { Luma([220]) } else { Luma([40]) }
    })
}

/// 方格棋盘纹理
fn make_checker(w: u32, h: u32, period: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        if ((x / period) + (y / period)) % 2 == 0 { Luma([220]) } else { Luma([40]) }
    })
}

/// 整幅图的 256 维 LBP 直方图（跳过一圈边界像素，那里 LBP 无定义）
fn lbp_histogram(img: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let (w, h) = (img.width(), img.height());
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                hist[code as usize] += 1;
            }
        }
    }
    hist
}

/// 旋转不变版：用 min_shift 把每个码归一化到"最小循环移位"
fn lbp_histogram_ri(img: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let (w, h) = (img.width(), img.height());
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                hist[min_shift(code) as usize] += 1;
            }
        }
    }
    hist
}

/// 两直方图先各自归一化为频率，再算 L1 距离（越小越像）
fn l1_distance(a: &[u32; 256], b: &[u32; 256]) -> f32 {
    let sa = a.iter().sum::<u32>().max(1) as f32;
    let sb = b.iter().sum::<u32>().max(1) as f32;
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x as f32 / sa - y as f32 / sb).abs())
        .sum()
}

/// uniform 模式（0/1 跳变 <= 2 次）的像素占比
fn uniform_ratio(img: &GrayImage) -> f32 {
    let (w, h) = (img.width(), img.height());
    let (mut total, mut uni) = (0u32, 0u32);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                total += 1;
                if count_transitions(code) <= 2 { uni += 1; }
            }
        }
    }
    uni as f32 / total.max(1) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 一、HOG：描述形状/轮廓 ----------
    let person = make_person(64, 128);
    // Dalal-Triggs 经典设置：9 个无符号方向、8x8 的 cell、2x2 cell 的 block、步长 1 cell
    let opts = HogOptions::new(9, false, 8, 2, 1);
    let feat = hog(&person, opts)?;
    println!("[HOG] 图像 64x128, cell=8px, block=2cell, stride=1cell, orientations=9(unsigned)");
    println!("[HOG] 特征向量维度 = {}", feat.len());

    // 手工核对维度：cells 8x16 → blocks 7x15 → 每 block 9*2*2=36 → 7*15*36
    let spec = HogSpec::from_options(64, 128, opts).unwrap();
    println!("[HOG] 公式核对维度 = 7*15*36 = {}", spec.descriptor_length());

    // 可视化每个 cell 的方向直方图（"星形"图），存 PNG
    let mut grid = cell_histograms(&person, spec);
    let vis = render_hist_grid(16, &grid.view_mut(), opts.signed);
    vis.save("out_ch27_hog_vis.png")?;

    // 故意给一组不合法的参数，看它如何报错（cell 边长必须整除宽高）
    match hog(&person, HogOptions::new(9, false, 7, 2, 1)) {
        Ok(_) => println!("[HOG] （意外）竟然通过了"),
        Err(e) => println!("[HOG] 非法参数报错示例：{e}"),
    }

    // ---------- 二、LBP：描述纹理 ----------
    let stripes = make_stripes(96, 96, 6);
    let checker = make_checker(96, 96, 8);
    let h_stripes = lbp_histogram(&stripes);
    let h_checker = lbp_histogram(&checker);
    println!("\n[LBP] 竖条纹 vs 方格：直方图 L1 距离 = {:.3}（越大越不同 → 纹理可区分）",
        l1_distance(&h_stripes, &h_checker));
    println!("[LBP] uniform 模式占比：竖条纹 {:.1}% , 方格 {:.1}%",
        uniform_ratio(&stripes) * 100.0, uniform_ratio(&checker) * 100.0);

    // 旋转不变：竖条纹旋转 90° 变横条纹
    let rot = image::imageops::rotate90(&stripes);
    let d_plain = l1_distance(&lbp_histogram(&stripes), &lbp_histogram(&rot));
    let d_ri = l1_distance(&lbp_histogram_ri(&stripes), &lbp_histogram_ri(&rot));
    println!("[LBP] 竖条纹 vs 其旋转90°：普通 LBP 距离 = {:.3} , min_shift 旋转不变距离 = {:.3}",
        d_plain, d_ri);
    Ok(())
}
```

运行 `cargo run --example ch27_handcrafted_features`，终端打印（数字可复现）：

```text
[HOG] 图像 64x128, cell=8px, block=2cell, stride=1cell, orientations=9(unsigned)
[HOG] 特征向量维度 = 3780
[HOG] 公式核对维度 = 7*15*36 = 3780
[HOG] 非法参数报错示例：Invalid HoG options: cell side 7 does not evenly divide width 64, cell side 7 does not evenly divide height 128

[LBP] 竖条纹 vs 方格：直方图 L1 距离 = 0.234（越大越不同 → 纹理可区分）
[LBP] uniform 模式占比：竖条纹 100.0% , 方格 97.3%
[LBP] 竖条纹 vs 其旋转90°：普通 LBP 距离 = 0.319 , min_shift 旋转不变距离 = 0.000
```

几个结论：

1. **HOG 维度 `3780`**，和公式 \\( 7\times15\times36 \\) 严丝合缝——这正是无数论文里出现的那个"3780 维 HOG"。
2. **参数非法会直接报错**：`cell_side=7` 无法整除 `64`，`hog` 返回 `Err`。这是 HOG 一个绕不开的约束（下面细说）。
3. **LBP 能区分纹理**：竖条纹和方格的直方图 L1 距离 `0.234`，明显不为零。
4. **旋转不变性立竿见影**：竖条纹旋转 90° 后，普通 LBP 直方图距离 `0.319`（认不出是同一种纹理了），而用 `min_shift` 后距离降到 `0.000`——旋转前后被认成了完全相同的纹理。

## 常见坑与提示

> 🕳️ **坑**：**HOG 的图像尺寸不能乱来。** `cell_side` 必须**整除**图像的宽和高；而且 `(每行 cell 数 − block_side)` 必须能被 `block_stride` **整除**，否则 `hog` 直接返回 `Err`（就像上面 `cell_side=7` 的例子）。所以实际用 HOG 前，通常要先把图 `resize` 到一个"规整"的尺寸（如 `64×128`）。

> ⚠️ **注意**：**HOG 参数决定维度，维度要和分类器对齐。** 同一张图，换一组 `HogOptions` 得到的向量长度完全不同。训练分类器（第六部分的 SVM）时用的是什么参数，推理时就必须一模一样，否则维度对不上、结果全错。把参数连同模型一起存档是个好习惯。

> 🕳️ **坑**：**LBP 边界像素没有定义。** `local_binary_pattern` 对第一行/列、最后一行/列的像素返回 `None`（它们凑不齐 8 个邻居）。所以遍历时要从 `1` 到 `w-1`、`h-1`，并用 `if let Some(code)` 稳妥地跳过 `None`，别 `unwrap` 到崩溃。

> 💡 **提示**：**LBP 是逐像素、逐点比较，天生偏慢。** 上面对 `96×96` 的图就要做近万次 8 邻域比较。真要在大图或实时场景用，考虑：只在感兴趣区域算、降采样、或改用查表法（`min_shift`、uniform 都可以预先打成 256 项的查找表，imageproc 内部就备了 `MIN_SHIFT` 表）。

> 💡 **提示**：**HOG 抓形状、LBP 抓纹理，两者常常"拼起来用"。** 比如人脸识别里，把一张脸的 HOG 向量和 LBP 直方图**首尾接成一个长向量**，同时利用形状和纹理线索，往往比单用一种更稳。

## 本章小结

- 本章的 HOG、LBP、Haar 都是**手工全局特征**：把一整块图像/区域浓缩成一个定长向量，区别于前几章描述单个关键点的**局部描述子**。
- **HOG** 把图切成 cell，统计每个 cell 的**梯度方向直方图**，再按 block 归一化后拼接——擅长描述**形状/轮廓**，经典配置产出 `3780` 维向量。注意 `cell_side` 要整除宽高。
- **LBP** 让每个像素和 8 邻居比明暗得到一个二进制码，统计整图的码直方图——擅长描述**纹理**；`min_shift` 做旋转不变、`count_transitions` 判 uniform 模式（可把 256 维压到 59 维）。
- **Haar** 特征靠黑白矩形之差 + 积分图，是经典人脸检测（Viola-Jones）的基石，细节留到 [第 56 章](../p12-face/ch56-face-detection.md)。
- **承上启下**：这些特征向量本身不下结论，它们是**送给分类器的"输入"**。下一步就是把它们喂给传统机器学习模型（第六部分的 [HOG + SVM 滑动窗口检测](../p06-ml/ch34-sliding-window.md)），让机器学会"看到这样的向量就判为行人"。

## 动手练习

1. **换个 HOG 配置**：把 `orientations` 改成 18、`cell_side` 改成 16，重新跑，用维度公式先手算再和程序输出对照。故意把图设成 `64×130` 看它怎么报错。
2. **可视化 HOG**：打开代码生成的 `out_ch27_hog_vis.png`，观察"星形"图——人形轮廓处的星星是不是更亮、方向更集中？
3. **LBP 分类小实验**：再合成第三种纹理（如斜条纹），算它和竖条纹、方格的 LBP 直方图两两 L1 距离，验证"同类近、异类远"。
4. **进阶**：把一张图切成上下左右四个区域，分别算 LBP 直方图再拼接成一个长向量（空间分块 LBP）。想一想：这样比"整图一个直方图"多保留了什么信息？

下一章我们更进一步：当图不止几张、而是成千上万张时，怎么**以图搜图**？答案是把这些局部特征打包成"视觉词袋"。👉 [第 28 章 词袋模型与图像检索基础（BoVW）](ch28-bovw-retrieval.md)
