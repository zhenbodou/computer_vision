# 第 65 章 立体视觉与深度：几何、匹配和置信度

> **本章导读**
> - 理解双目为什么能测深度：**视差（disparity）**——同一点在左右两眼里的位置差，越近差得越多。
> - 掌握核心公式 \\( Z = f B / d \\)，并搞懂它的**误差传播**：为什么远处深度天然测不准。
> - 动手合成一对已校正的左右图，按"代价→聚合→选择→细化"的标准流水线跑 **SAD 块匹配**，用**左右一致性**生成置信度、剔除无效区，再转深度并逐区对比真值。
> - 认清"单目深度"是深度学习任务，划清 metric（绝对尺度）与 relative（相对深度）的边界。

## 场景引入：为什么闭上一只眼就抓不准筷子

做个小实验：闭上一只眼，伸手去捏桌上的笔尖。你会发现比睁着两只眼难得多，常抓空。这不是错觉——**你的深度感很大程度来自两眼看同一物体的细微差异**。

把一根手指竖在鼻子前，交替闭左右眼，手指会在背景上"跳来跳去"，且离得越近跳得越远。这个"跳动量"就是**视差**，大脑用它反推"这东西离我多远"。双目立体视觉（stereo vision）做的就是同一件事：两台并排相机拍同一场景，靠视差算出每个点的深度。

## 原理：从视差到深度，再到误差

### 核心公式

设两台相机**已极线校正**——内参相同、光轴平行、严格共水平线（校正后同一个三维点在左右图落在**同一行**，匹配只需沿这行找，不必满图搜）。设基线（两光心间距）\\( B \\)、焦距 \\( f \\)（像素），点在左图列坐标 \\( x_L \\)、右图 \\( x_R \\)，**视差** \\( d = x_L - x_R \\)。相似三角形给出深度：

\\[ Z = \frac{f B}{d} \\]

一句话记住：**视差与深度成反比**。\\( d \\) 大 → 近；\\( d \\) 小 → 远；\\( d \to 0 \\) 表示无穷远。

```text
     左相机 O_L          右相机 O_R
        │  ＼           ／  │
        │    ＼   P    ／   │      P 越近 -> x_L 与 x_R 差越大 -> 视差 d 越大
        │      ＼ · ／      │
   ─────┼────────╳──────────┼─────  像面（已校正，同一行）
        │   x_L↗    ↖x_R    │
        └──── 基线 B ───────┘        Z = f·B / d
```

### 误差传播：为什么远处必然测不准

视差是从图像里估出来的，必然有误差 \\( \sigma_d \\)（哪怕只有半个像素）。对 \\( Z = fB/d \\) 求导，深度误差为：

\\[ \sigma_Z \approx \frac{f B}{d^2}\ \sigma_d = \frac{Z^2}{f B}\ \sigma_d \\]

关键在 \\( Z^2 \\)：**深度误差随距离平方增长**。代入本例 \\( fB = 500 \times 0.1 = 50 \\)、设视差误差 \\( \sigma_d = 0.5 \\) 像素：近处 \\( Z = 2.5 \\) m 时 \\( \sigma_Z \approx 2.5^2 / 50 \times 0.5 \approx 0.06 \\) m（约 2.5%）；远处 \\( Z = 10 \\) m 时 \\( \sigma_Z \approx 10^2 / 50 \times 0.5 = 1.0 \\) m（约 10%）——**同样的视差误差，远处的深度误差大了十几倍**。

这条公式是双目选型的"账本"：想在某个目标深度 \\( Z \\) 达到精度 \\( \sigma_Z \\)，就得**反推**所需的基线 \\( B \\)、焦距 \\( f \\) 和视差精度 \\( \sigma_d \\)（增大 \\( B \\) 或 \\( f \\) 都能减小误差，但基线太大又会加剧遮挡）。工程上必须先算这本账，而不是拍脑袋摆两个相机。

### 匹配的四步流水线

要算视差，得先在右图找到左图某点的对应。单个像素没法比（太多像素灰度相同），所以取周围一个**窗口（块）**去右图同一行滑动找最像。学术上，几乎所有立体匹配都遵循同一条流水线：

```text
  ① 代价computation  ② 代价聚合aggregation  ③ 视差选择selection  ④ 细化refinement
   逐像素算相似度      窗口内求和/加权          取代价最小的 d       亚像素、左右一致性
   (如 |L-R|)         (抗噪、更可靠)          (Winner-Take-All)    (剔除遮挡/误匹配)
```

我们用最经典的 **SAD（Sum of Absolute Differences）**：把①②合起来，在窗口里累加左右块的绝对差，③再取让它最小的视差：

\\[ \text{SAD}(d) = \sum_{(i, j) \in \text{窗口}} \big| L(x + i,\  y + j) - R(x - d + i,\  y + j) \big| \\]

完美匹配时 SAD 为 0。第④步的**细化**尤为关键——尤其是**左右一致性检验（left-right consistency）**：以左图为参考算一次视差，再以右图为参考算一次，若两者对不上（差 >1），说明这个点在某只相机里被遮挡或匹配错了，判为**无效**。这一步生成了"置信度"，让系统知道哪些深度可信、哪些该丢弃。

## 动手写代码

为让示例**自包含且真值已知**，我们合成一对已校正的左右图：伪随机纹理当背景（视差 5，远），贴一个纹理不同的前景矩形（视差 20，近）；右图就是各区域按各自视差左移的结果。然后跑双向 SAD、做左右一致性检验、转深度。依赖只需 `image`：

```toml
[dependencies]
image = "0.25"
```

新建 `code/examples/ch65_stereo_depth.rs`。先是常量、纹理与场景定义：

```rust
use image::{GrayImage, Luma};

const W: u32 = 256;
const H: u32 = 192;
const D0: u32 = 5;    // 背景视差（远）
const D1: u32 = 20;   // 前景视差（近）
const DMAX: u32 = 24; // 视差搜索上限
const R: i32 = 3;     // 块半径（7×7 窗口）

// 确定性伪随机纹理；seed 不同 => 前景/背景两套纹理，避免"背景恰好也匹配上"的合成假象
fn tex(x: u32, y: u32, seed: u64) -> u8 {
    let h = (x as u64).wrapping_mul(73856093)
        .wrapping_add((y as u64).wrapping_mul(19349663))
        .wrapping_add(seed);
    let h = (h ^ (h >> 13)).wrapping_mul(0x2545F4914F6CDD1D);
    ((h >> 33) & 0xFF) as u8
}
fn tex_bg(x: u32, y: u32) -> u8 { tex(x, y, 0xB6) }
fn tex_fg(x: u32, y: u32) -> u8 { tex(x, y, 0x3F) }

fn is_foreground_left(x: u32, y: u32) -> bool {
    (110..170).contains(&x) && (70..140).contains(&y)
}
fn is_foreground_right(x: u32, y: u32) -> bool { // 右图 = 左图左移 D1
    (90..150).contains(&x) && (70..140).contains(&y)
}
fn true_disp(x: u32, y: u32) -> u32 {
    if is_foreground_left(x, y) { D1 } else { D0 }
}

// SAD 块匹配。dir=-1：参考=left、目标=right，匹配位置 x-d；dir=+1：反过来（用于一致性检验）
fn disparity(reference: &GrayImage, target: &GrayImage, dir: i32) -> Vec<u32> {
    let mut disp = vec![0u32; (W * H) as usize];
    let px = |img: &GrayImage, x: i32, y: i32| img.get_pixel(x as u32, y as u32)[0] as i64;
    for y in R..(H as i32 - R) {
        for x in R..(W as i32 - R) {
            let (mut best_d, mut best_cost) = (0u32, i64::MAX);
            for d in 0..=DMAX as i32 {
                let tx = x + dir * d;
                if tx - R < 0 || tx + R >= W as i32 { continue; } // 越界跳过
                let mut cost = 0i64;
                for dy in -R..=R {
                    for dx in -R..=R {
                        cost += (px(reference, x + dx, y + dy) - px(target, tx + dx, y + dy)).abs();
                    }
                }
                if cost < best_cost { best_cost = cost; best_d = d as u32; }
            }
            disp[(y as u32 * W + x as u32) as usize] = best_d;
        }
    }
    disp
}
```

主流程：合成图 → 双向匹配 → 一致性检验 → 分区对比 → 转深度 → 存图：

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ① 合成已校正的左右图
    let left = GrayImage::from_fn(W, H, |x, y| {
        if is_foreground_left(x, y) { Luma([tex_fg(x, y)]) } else { Luma([tex_bg(x, y)]) }
    });
    let right = GrayImage::from_fn(W, H, |x, y| {
        if is_foreground_right(x, y) { Luma([tex_fg(x + D1, y)]) } // 前景（近，视差大）
        else { Luma([tex_bg(x + D0, y)]) }                          // 背景（远，视差小）
    });

    // ② 双向 SAD 块匹配
    let disp_l = disparity(&left, &right, -1);
    let disp_r = disparity(&right, &left, 1);

    // ③ 左右一致性检验：左图视差 dL 处回查右图对应点视差 dR，差 >1 判无效
    let mut valid = vec![false; (W * H) as usize];
    let (x_lo, x_hi) = ((DMAX as i32 + R) as u32, (W as i32 - DMAX as i32 - R) as u32);
    let (y_lo, y_hi) = (R as u32, (H as i32 - R) as u32);
    let (mut total, mut consistent, mut correct_all, mut correct_valid) = (0u64, 0u64, 0u64, 0u64);
    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let dl = disp_l[(y * W + x) as usize];
            let xr = x as i32 - dl as i32;
            let dr = disp_r[(y * W + xr as u32) as usize];
            let ok = (dl as i32 - dr as i32).abs() <= 1;
            valid[(y * W + x) as usize] = ok;
            let truth = true_disp(x, y);
            total += 1;
            if dl == truth { correct_all += 1; }
            if ok { consistent += 1; if dl == truth { correct_valid += 1; } }
        }
    }
    println!("评估区像素 {} 个", total);
    println!("  原始视差准确率（全部像素）        = {:.2}%", 100.0 * correct_all as f64 / total as f64);
    println!("  左右一致性判为有效的像素          = {} 个（{:.2}%），无效 {:.2}%",
        consistent, 100.0 * consistent as f64 / total as f64,
        100.0 * (total - consistent) as f64 / total as f64);
    println!("  有效像素中的视差准确率            = {:.2}%（一致性检验成功滤掉了遮挡带的错点）",
        100.0 * correct_valid as f64 / consistent as f64);

    // ④ 分区平均视差 vs 真值
    let region_avg = |x0: u32, x1: u32, y0: u32, y1: u32| -> f64 {
        let (mut s, mut n) = (0.0, 0.0);
        for y in y0..y1 { for x in x0..x1 { s += disp_l[(y * W + x) as usize] as f64; n += 1.0; } }
        s / n
    };
    let bg = region_avg(40, 90, 30, 60);
    let fg = region_avg(120, 152, 95, 135); // 避开遮挡边界
    println!("\n背景区平均视差：估计 {:.3} vs 真值 {}", bg, D0);
    println!("前景区平均视差：估计 {:.3} vs 真值 {}", fg, D1);

    // ⑤ 视差转深度 Z = f·B/d
    let (f, b) = (500.0_f64, 0.10_f64);
    let depth = |d: f64| f * b / d;
    println!("\n相机：f={f} px，基线 B={b} m");
    println!("背景深度：估计 {:.3} m（真值 {:.3} m）", depth(bg), depth(D0 as f64));
    println!("前景深度：估计 {:.3} m（真值 {:.3} m）", depth(fg), depth(D1 as f64));
    println!("=> 前景更近、背景更远，符合「视差越大越近」直觉 ✓");

    // ⑥ 存图：0..DMAX 拉伸到 0..255（前景亮、背景暗）；无效点涂黑
    left.save("out_ch65_left.png")?;
    right.save("out_ch65_right.png")?;
    let vis = GrayImage::from_fn(W, H, |x, y| {
        let idx = (y * W + x) as usize;
        if x >= x_lo && x < x_hi && y >= y_lo && y < y_hi && !valid[idx] {
            Luma([0]) // 无效（遮挡/误匹配）
        } else {
            Luma([(disp_l[idx] * 255 / DMAX) as u8])
        }
    });
    vis.save("out_ch65_disparity.png")?;
    println!("\n已存 out_ch65_left.png / out_ch65_right.png / out_ch65_disparity.png");
    Ok(())
}
```

运行：

```bash
cargo run --example ch65_stereo_depth
```

输出：

```text
评估区像素 37572 个
  原始视差准确率（全部像素）        = 97.63%
  左右一致性判为有效的像素          = 36493 个（97.13%），无效 2.87%
  有效像素中的视差准确率            = 99.89%（一致性检验成功滤掉了遮挡带的错点）

背景区平均视差：估计 5.000 vs 真值 5
前景区平均视差：估计 20.000 vs 真值 20

相机：f=500 px，基线 B=0.1 m
背景深度：估计 10.000 m（真值 10.000 m）
前景深度：估计 2.500 m（真值 2.500 m）
=> 前景更近、背景更远，符合「视差越大越近」直觉 ✓
```

读一下这几个数字，会发现它们讲了一个完整的工程故事：**原始**视差准确率 97.63%，剩下那 2.37% 的错点几乎全在前景边缘（遮挡带）；**左右一致性检验**把 2.87% 的像素判为无效并涂黑，而在它保留下来的"有效"像素里，准确率飙到 **99.89%**——这正是"置信度/无效区"的价值：**与其输出一张处处都有值、却在边缘偷偷出错的深度图，不如诚实地标出'这里我没把握'**。纯背景/前景区的平均视差 5.000/20.000 与真值分毫不差，代入 \\( Z = fB/d \\) 得 10 m / 2.5 m，前景近、背景远被如实还原。打开 `out_ch65_disparity.png`：中间一块亮矩形（前景）浮在暗背景上，边缘勾着一圈黑边——那就是被检出的遮挡区。

## 单目深度：一只眼睛能测距吗

只有一张图能估深度吗？人闭一只眼也能判断远近，靠的是**先验经验**：近大远小、遮挡、纹理梯度……这些没有几何上的唯一解，**传统算法难以可靠求出**。

> ⚠️ **注意：单目深度是深度学习任务，且要分清 metric 与 relative。** 主流方案（**MiDaS**、**Depth Anything** 等）用海量"图像–深度"数据训练神经网络，输入一张图直接回归深度。**纯 Rust 可以做推理**：把模型导出 ONNX，用 `tract`（纯 Rust）或 `candle` 加载运行即可（见[第 19 部分](../p19-deploy/ch94-onnx.md)）。但务必分清：多数单目模型输出的是 **relative（相对）深度**——只知道"谁比谁近"，**尺度未知**；要变成 **metric（绝对米数）**必须借助已知尺寸、地面假设或另一路传感器做**尺度对齐**。**没做尺度对齐的单目深度绝不能直接用于安全测距**（如自动驾驶的避障距离），这是纯 Rust 也好、任何语言也好都逾越不了的原理性限制。

## 常见坑与提示

> 🕳️ **坑：弱纹理、重复纹理、反光是块匹配三大杀手。** 一整面白墙每个块都一样，SAD 在哪都小，视差无从判断（本例故意用丰富随机纹理才这么准）；重复纹理（栅栏、百叶窗）会匹配到"错误但相似"处，给出**自信的错误**；镜面反光则让同一点在左右图亮度不同，直接违背匹配假设。实务上常主动投射随机散斑"人造纹理"来救场。

> 🕳️ **坑：绝不能用平滑把无效深度伪装成测量值。** 遮挡区、弱纹理区本就没有可靠答案。有人为了让深度图"好看"，直接对空洞做插值/平滑填充——这会**制造出看似精确、实则凭空捏造的深度**，在测量和安全场景里是重大隐患。正确做法是像本例那样**保留无效标记**，让下游知道这里不可信。

> 💡 **提示：块大小是一场权衡。** 窗口大 → 抗噪、平滑，但糊掉边缘细节；窗口小 → 边缘锐利，但对噪声敏感、弱纹理处易错。常用 5×5 到 11×11，按场景调。除左右一致性外，**代价尖锐度**（最优与次优代价的比值）和 **speckle 检查**（剔除孤立的小视差斑块）也是常用的置信度手段。

> ⚠️ **注意：块匹配的前提是"已极线校正"。** 本例假设左右图已校正、对应点在同一行，才能只沿一行搜。真实双目必须先做**立体校正**（用双目标定参数把两图摆正），否则要沿倾斜极线搜，代价大得多。双目标定是[第 64 章](ch64-calibration.md)单目标定的自然延伸。

## 本章小结

- 双目靠**视差**测深度：\\( d = x_L - x_R \\)，近大远小；核心公式 \\( Z = f B / d \\)，视差与深度成反比。
- **深度误差随距离平方增长**（\\( \sigma_Z \approx Z^2 \sigma_d / (fB) \\)）——远处天然测不准；基线/焦距/视差精度要由目标精度**反推**，先算误差预算再选型。
- 立体匹配遵循**代价 → 聚合 → 视差选择 → 细化**四步；SAD 块匹配 + WTA 是最经典实现。
- **左右一致性检验**生成置信度、剔除遮挡/误匹配：本例把 2.87% 的像素判为无效后，有效区准确率从 97.63% 升到 99.89%。**宁可标"无效"，也不要用平滑伪装成测量值。**
- **单目深度**是深度学习任务，多为 relative 深度、尺度未知，需尺度对齐才能变 metric，且不可直接用于安全测距。

## 动手练习

1. **多距离验证误差模型**：把前景视差依次设为 8、12、16、20，记录每档由 \\( Z = fB/d \\) 得到的深度，并与 \\( \sigma_Z \approx Z^2 \sigma_d/(fB) \\) 的预测对照，画出"距离–不确定度"曲线，亲手验证平方增长。
2. **按切片报告 bad-pixel**：分别统计"背景 / 前景 / 遮挡边界"三类区域的错误率（视差偏差 >1 的比例），体会误差如何随纹理与遮挡分布。
3. **改基线/分辨率验证误差预算**：把 `f` 或（等效地）基线 `b` 翻倍，重算远处深度精度，验证"增大 \\( fB \\) 能降低 \\( \sigma_Z \\)"。
4. **加代价尖锐度置信（进阶）**：在 `disparity` 里同时记录最优与次优 SAD，输出二者比值作为置信度；把低置信像素也涂黑，看它是否和左右一致性检出的无效区重合。

我们已经能从两张**位姿已知**的图恢复深度。可如果只给两张图、连相机怎么摆的都不知道，能反推它们的相对位姿吗？下一章的对极几何给出答案。👉 [第 66 章 对极几何、PnP 与鲁棒位姿估计](ch66-epipolar-pnp.md)
