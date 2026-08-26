# 第 24 章 局部特征描述子（SIFT、ORB、BRIEF）

> **本章导读**
> - 想清楚：检测到关键点只知道它**在哪**，要跨图判断"是不是同一个点"，还得知道它**长什么样**——这就是**描述子**。
> - 建立直觉：描述子就是给关键点写的一串"数字/比特指纹"；比较两个指纹用**汉明距离（Hamming distance）**，快到只用一条 CPU 指令。
> - 动手用 `imageproc` 的 `brief` 给 FAST 角点算 **BRIEF** 二进制描述子，亲手算两个描述子的汉明距离。
> - 认清 BRIEF 的软肋（不抗旋转/缩放），并知道 **SIFT / ORB** 是怎么补上这块短板的。

## 场景：两张照片里，哪个点才是"同一个屋角"

接着上一章的全景拼接。我们已经用 FAST 在两张有重叠的照片里各检出了一批角点。现在的问题是：左图有 130 个角点，右图也有 130 个角点，**怎么知道左图第 7 个点和右图第 42 个点其实是同一栋楼的同一个屋角？**

光靠坐标不行——两张图是从不同位置拍的，同一个屋角在两张图里的像素坐标完全不同。光靠"它是不是角点"也不行——130 个点全是角点，没法区分。

我们真正需要的，是给每个关键点**周围的那一小片图像**做个总结：把它长什么样（明暗怎么分布、纹理什么走向）压缩成一串数字。这串数字就像人的**指纹**：同一个物理点在两张图里，尽管坐标不同，但周围长得几乎一样，指纹就应该几乎一致；不同的点，指纹则大不相同。于是"找对应点"就变成了"**比指纹**"。这串指纹，就叫**描述子（descriptor）**。

> 📖 **术语**：**描述子（descriptor）**。对关键点**邻域（周围一小块，通常几十像素见方的 patch）**的一段紧凑编码，用来刻画"这个点周围长什么样"。它可以是一串浮点数（如 SIFT 的 128 维向量），也可以是一串二进制比特（如 BRIEF）。好的描述子要**可区分**（不同点的描述子差得远）又**有不变性**（同一个点在不同光照、视角、尺度下描述子尽量稳定）。检测（上一章）负责"在哪"，描述（本章）负责"长啥样"，两步合起来才能做匹配（[第 25 章](ch25-matching-ransac.md)）。

## BRIEF：随机比大小，攒出一串比特

最直白的描述子长什么样？**BRIEF**（Binary Robust Independent Elementary Features）给了一个快到极致的方案，思路简单得有点狡猾：

> 在关键点周围的小 patch 里，**预先随机选好一批点对** \\( (p, q) \\)。对每一对，只问一个是非题：**\\( p \\) 处比 \\( q \\) 处暗吗？** 暗记 `1`，否则记 `0`。把 M 个是非题的答案排成一行，就是一个 M 位的二进制串——这就是 BRIEF 描述子。

```text
   关键点周围的 patch（比如 31x31）
   ┌───────────────┐
   │   p1●----○q1   │   对 1: I(p1) < I(q1)? → 1
   │      ○q2       │   对 2: I(p2) < I(q2)? → 0
   │   p2●   p3●    │   对 3: I(p3) < I(q3)? → 1
   │        ＼      │        ...
   │         ○q3    │   256 个问题 → 256 位: 1010 1100 ...
   └───────────────┘
```

关键点在于：这批**点对的位置是一次选定、全程复用**的。同一个物理点在两张图里，邻域几乎一样，那么这 256 个"谁更暗"的答案也几乎一样，得到的比特串自然高度相似。为什么用"比大小"而不用绝对亮度？因为**相对明暗对整体光照变化更稳**——你把整个 patch 调亮一点，每一对的大小关系基本不变，描述子纹丝不动。（实际实现里还会先对采样点周围的小区域求个平均再比较，用来抗噪声，`imageproc` 借助积分图把这一步做得飞快。）

## 汉明距离：两串指纹差多少，数一下不同的比特

有了两个比特串，怎么衡量它们像不像？答案朴素得可爱：**数一数有多少位不一样**。这个"不同比特的个数"就是**汉明距离（Hamming distance）**。

计算它有一个快到飞起的技巧：先把两串**异或（XOR，\\( \oplus \\)）**——相同的位得 `0`，不同的位得 `1`——再数结果里有多少个 `1`（叫 **popcount** 或 `count_ones`）：

\\[ d_H(a, b) = \operatorname{popcount}(a \oplus b) \\]

现代 CPU 有专门的 popcount 指令，256 位描述子的汉明距离也就几条指令的事。这正是二进制描述子的最大卖点：**匹配时要在成千上万个描述子里反复比距离，二进制 + 汉明距离比"浮点向量算欧氏距离"快一两个数量级**。汉明距离越小，两个点越像；等于 0 表示两串比特完全一致。

## 动手写代码：给 FAST 角点算 BRIEF 指纹

我们把上一章的 FAST 检测和本章的 BRIEF 串起来：合成图 → FAST 检测角点 → 转成 `Point<u32>` → 算 BRIEF → 打印比特长度、算汉明距离。最后再做个小实验，亲眼看看 BRIEF 对**平移**和**旋转**的不同反应。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch24_descriptors.rs`）：

```rust
use image::imageops;
use image::{GrayImage, Luma};
use imageproc::binary_descriptors::brief::{brief, TestPair};
use imageproc::binary_descriptors::BinaryDescriptor; // hamming_distance / get_size 在此 trait
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::point::Point;
use imageproc::rect::Rect;

/// 合成灰度“照片”：白底 + 两个深色方块 + 一小片棋盘格，制造角点。
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([235]));
    draw_filled_rect_mut(&mut img, Rect::at(60, 55).of_size(70, 70), Luma([30]));
    draw_filled_rect_mut(&mut img, Rect::at(170, 70).of_size(55, 55), Luma([45]));
    let (ox, oy, cell, n) = (250i32, 150i32, 16i32, 4);
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

/// 把整幅图平移 (dx, dy)，空出来处填 fill。内容原样搬家，用于验证平移不变性。
fn translate(src: &GrayImage, dx: i32, dy: i32, fill: u8) -> GrayImage {
    let (w, h) = (src.width(), src.height());
    let mut out = GrayImage::from_pixel(w, h, Luma([fill]));
    for (x, y, p) in src.enumerate_pixels() {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
            out.put_pixel(nx as u32, ny as u32, *p);
        }
    }
    out
}

/// 用一个确定性小 PRNG 生成固定的 256 组测试点对（都落在 31x31 patch 内）。
/// 真实项目里 brief(.., None) 会自动随机生成；这里固定下来是为了：
/// ① 输出可复现；② 跨图匹配本就必须复用同一组点对，否则比特无从对应。
fn fixed_test_pairs(length: usize) -> Vec<TestPair> {
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || -> u32 {
        state = state // 经典 LCG 常数
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % 31 // 落在 [0, 31)
    };
    let mut pairs = Vec::with_capacity(length);
    for _ in 0..length {
        let (x0, y0, x1, y1) = (next(), next(), next(), next());
        pairs.push(TestPair { p0: Point::new(x0, y0), p1: Point::new(x1, y1) });
    }
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (360u32, 260u32);
    let gray = make_scene(w, h);

    // ① FAST 检测角点
    let corners = corners_fast9(&gray, 40);
    println!("FAST 检出角点数 = {}", corners.len());

    // ② 转成 BRIEF 需要的 Point<u32>。BRIEF patch 是 31x31，关键点必须
    //    离四边 >= 16px，否则 brief() 会因“某点贴边”而整体返回 Err。先过滤掉。
    let margin = 16u32;
    let keypoints: Vec<Point<u32>> = corners
        .iter()
        .filter(|c| c.x >= margin && c.x + margin < w && c.y >= margin && c.y + margin < h)
        .map(|c| (*c).into()) // Corner -> Point<u32>（imageproc 已实现 From）
        .collect();
    println!("离边缘 >= {}px、可用于 BRIEF 的关键点数 = {}", margin, keypoints.len());

    // ③ 算 BRIEF 描述子（256 位）。用固定测试点对，保证可复现、跨图可比。
    let test_pairs = fixed_test_pairs(256);
    let (descriptors, _) = brief(&gray, &keypoints, 256, Some(&test_pairs))?;
    let d0 = &descriptors[0];
    println!("每个描述子 = {} 位 = {} 个 u128 打包", d0.get_size(), d0.bits.len());

    // ④ 汉明距离：同一个点对自己 = 0；两个不同角点通常差很多
    let da = &descriptors[0];
    let bi = descriptors.len() / 2;
    let db = &descriptors[bi];
    println!("Hamming(点0, 点0)         = {}", da.hamming_distance(da));
    println!("Hamming(点0, 点{})         = {} / {} 位", bi, da.hamming_distance(db), da.get_size());

    // ⑤ 挑一个关键点，验证“平移不变”与“旋转敏感”。必须用同一组 test_pairs 才可比。
    let kp = keypoints[0];
    println!("\n选定关键点 kp = ({}, {})", kp.x, kp.y);

    // 平移：整幅图搬 (12, 9)，同一个物理点搬到 (kp.x+12, kp.y+9)
    let (dx, dy) = (12i32, 9i32);
    let shifted = translate(&gray, dx, dy, 235);
    let kp_shift = Point::new((kp.x as i32 + dx) as u32, (kp.y as i32 + dy) as u32);
    let (desc_orig, _) = brief(&gray, &[kp], 256, Some(&test_pairs))?;
    let (desc_shift, _) = brief(&shifted, &[kp_shift], 256, Some(&test_pairs))?;
    println!("平移后 Hamming(原图 kp, 平移图 kp) = {}   <- 平移几乎不变",
        desc_orig[0].hamming_distance(&desc_shift[0]));

    // 旋转 180°：同一个物理点映射到 (w-1-x, h-1-y)，邻域被转了 180°
    let rotated = imageops::rotate180(&gray);
    let kp_rot = Point::new(w - 1 - kp.x, h - 1 - kp.y);
    let (desc_rot, _) = brief(&rotated, &[kp_rot], 256, Some(&test_pairs))?;
    println!("旋转后 Hamming(原图 kp, 旋转图 kp) = {}   <- 旋转后指纹大变",
        desc_orig[0].hamming_distance(&desc_rot[0]));

    Ok(())
}
```

运行 `cargo run --example ch24_descriptors`，终端输出（数字真实可复现）：

```text
FAST 检出角点数 = 132
离边缘 >= 16px、可用于 BRIEF 的关键点数 = 132
每个描述子 = 256 位 = 2 个 u128 打包
Hamming(点0, 点0)         = 0
Hamming(点0, 点66)         = 124 / 256 位

选定关键点 kp = (60, 55)
平移后 Hamming(原图 kp, 平移图 kp) = 0   <- 平移几乎不变
旋转后 Hamming(原图 kp, 旋转图 kp) = 123   <- 旋转后指纹大变
```

逐条读懂这份输出：

1. **描述子是 256 位、打包成 2 个 `u128`。** `BriefDescriptor` 的 `bits` 字段是 `Vec<u128>`，256 位正好塞进 2 个 `u128`。`get_size()` 返回位数（256）。
2. **一个点和自己的汉明距离是 0**（`Hamming(点0,点0)=0`）——完全一致，天经地义。
3. **两个不同角点差了 124/256 位**，接近一半。这符合直觉：两个毫不相干的邻域，256 个"谁更暗"的是非题答案基本是随机的，对不上的大约占一半。所以匹配时我们要找的是**汉明距离特别小**（远低于一半）的那一对。
4. **平移不变**：把整张图搬了 `(12, 9)` 像素，同一个物理点的邻域没变，描述子**一模一样**，汉明距离 = 0。这说明 BRIEF 天然扛得住平移（也扛得住一定的光照变化）。
5. **旋转大变**：把图转 180°，同一个物理点的邻域被跟着转了，可 BRIEF 的采样点对位置是**固定不转**的，于是 256 个是非题的答案大面积翻转，汉明距离飙到 123——**几乎和两个陌生点一样远**。这就暴露了 BRIEF 的致命短板。

## SIFT 与 ORB：把不变性补上

BRIEF 快、省、抗平移和光照，但上面第 5 条说明它**不抗旋转、也不抗缩放**：相机一歪、物体一远，描述子就对不上了。业界有两条经典的补救路线：

- **SIFT**（Scale-Invariant Feature Transform）：老牌劲旅。它在**多个尺度**上找关键点（所以抗缩放），给每个点估计一个**主方向**（所以抗旋转），再用邻域的**梯度方向直方图**编码成一个 **128 维浮点向量**。SIFT 极其稳健、匹配质量高，代价是**慢、占内存**，且用欧氏距离比对。（它曾长期受专利保护，如今专利已过期。）

- **ORB**（Oriented FAST and Rotated BRIEF）：为实时而生，名字就是它的配方——**oriented FAST + rotated BRIEF**。它先用上一章提到的 `oriented_fast` 给每个 FAST 角点算一个主方向（灰度质心法），再在计算 BRIEF 时**把那批采样点对按这个方向一起旋转**，于是"物体转了，采样点对也跟着转"，描述子就转不动它了——补上了旋转不变性。ORB 保留了二进制描述子 + 汉明距离的极致速度，是 ORB-SLAM 等实时系统的主力特征。

> 💡 **提示**：**纯 Rust 里怎么用这些更强的特征？** `imageproc` 目前提供了 FAST/oriented FAST 和 BRIEF 这套 ORB 的核心积木。想要开箱即用的**尺度 + 旋转不变**特征，可以看纯 Rust 的 [`akaze`](https://crates.io/crates/akaze) crate（AKAZE 特征，接口和版本请自行 `cargo add` 后验证）。需要 SIFT 时，纯 Rust 生态尚不完善，可考虑用 AKAZE/ORB 替代，或走 ONNX 模型推理的学习型特征（后面部署章节会讲）。

## 常见坑与提示

> 🕳️ **坑**：**关键点贴边，`brief` 会整批报错，而不是跳过那一个点。** BRIEF 的 patch 是 31×31，关键点离图像边缘不足 16 像素时，`brief()` 直接返回 `Err(...)`，**整次调用都失败**。所以务必像代码里那样，先按 `margin` 把贴边的关键点过滤掉再喂给 `brief`。这也是为什么示例里"检出 132 个、可用 132 个"——恰好都在安全区；真实图像里贴边的点常常要被剔掉一批。

> ⚠️ **注意**：**比较两个描述子，必须来自同一组测试点对（test pairs）。** 汉明距离比的是"第 i 个问题的答案是否一致"，如果两个描述子问的根本不是同一批问题（点对不同），比出来的距离毫无意义。所以跨图匹配时，要么用第一次 `brief(.., None)` 返回的 `test_pairs`、之后一律 `Some(&test_pairs)` 复用（本章用固定点对是同一个道理），要么保证两次调用用的是同一套。`hamming_distance` 还要求两个描述子**位数相等**，否则会 panic。

> 🕳️ **坑**：**`length` 必须是 128 的倍数。** BRIEF 把比特打包进 `Vec<u128>`，所以位数得是 128 的整数倍（128、256、512……）。传个 `200` 会直接报错。位数越多越有区分度但越占空间、越慢，`256` 是常见的甜点值。

> 💡 **提示**：**`Corner` 转 `Point<u32>` 一步到位。** 上一章 `corners_fast9` 返回的是 `Vec<Corner>`，而 `brief` 要 `&[Point<u32>]`。`imageproc` 已为 `Corner` 实现了 `From<Corner> for Point<u32>`，所以直接 `.map(|c| (*c).into())` 就行，`(x, y)` 会原样搬过去，不用手动 `Point::new(c.x, c.y)`。

## 本章小结

- 检测只解决"关键点在哪"，要跨图判断"是不是同一个点"，还需要**描述子**——对关键点邻域的紧凑编码，就像一串"指纹"。
- **BRIEF** 在 patch 内用一批**固定的随机点对**做"谁更暗"的是非题，攒成一串二进制比特；它抗平移、抗光照，快而省。
- 比较二进制描述子用**汉明距离**：\\( d_H(a,b)=\operatorname{popcount}(a \oplus b) \\)，异或后数 `1`，一条 CPU 指令搞定，比浮点欧氏距离快得多。
- 实测印证：同一点自比 = 0、平移后 = 0（不变），但**旋转 180° 后汉明距离飙到与陌生点相当**——BRIEF **不抗旋转、不抗缩放**。
- **SIFT** 用多尺度 + 主方向 + 梯度直方图换来强不变性（128 维浮点，稳但慢）；**ORB = oriented FAST + rotated BRIEF**，给采样点对加上旋转，兼顾不变性与二进制的极速。
- `imageproc` API 要点：`brief(&gray, &keypoints, 256, None)` 返回 `(Vec<BriefDescriptor>, Vec<TestPair>)`；`length` 须为 128 的倍数；关键点须离边 ≥16px；比较须用同一组 test pairs。

## 动手练习

1. **改位数**：把 `length` 从 `256` 改成 `128` 和 `512`，看每个描述子的 `bits.len()` 怎么变（应是 1 和 4），以及"两个不同角点"的汉明距离随位数怎么变化。
2. **验证光照不变**：把 `shifted` 那张图整体加亮（每个像素 `+20`，注意别溢出 255），再算汉明距离，验证 BRIEF 对整体亮度变化确实很稳。
3. **量化旋转的破坏**：把旋转从 180° 换成"平移一点点 + 轻微内容改动"，或用 [第 10 章](../p02-image-basics/ch10-geometry.md) 的旋转函数转 10°、30°、90°，记录汉明距离随角度的增长——直观感受"不抗旋转"到底有多严重。
4. **进阶——真正匹配一次**：用 `imageproc::binary_descriptors::match_binary_descriptors`，把原图和它的平移版各自的描述子丢进去，看它能不能把对应的角点配上对。这正是下一章的主题。

现在每个关键点都有了指纹，我们终于能做那件从第 23 章就惦记着的事了：**把两张图里的点配成对，并剔除配错的**。下一章见。👉 [第 25 章 特征匹配与几何验证（RANSAC）](ch25-matching-ransac.md)
