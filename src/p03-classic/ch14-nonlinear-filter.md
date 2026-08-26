# 第 14 章 非线性滤波：中值与双边滤波（去噪）

> **本章导读**
> - 学完你能：说清楚为什么光有线性滤波（卷积）不够，并用**中值滤波**一招治好椒盐噪声。
> - 理解**中值滤波（median filter）**为什么能"无视极值、保住边缘"，并实测它对椒盐噪声完胜均值模糊。
> - 理解**双边滤波（bilateral filter）**如何同时看"空间距离"和"像素值差异"，做到**保边平滑**（磨皮不糊轮廓）。
> - 顺带认识最大/最小滤波。全程合成图像 + 可复现噪声，复制即可运行。

## 场景：老照片上的雪花点，和"磨皮不糊脸"

翻出一张老照片扫描件，上面常有一些**孤立的黑点和白点**，像撒了盐和胡椒——这就是经典的**椒盐噪声（salt-and-pepper noise）**，多由传感器坏点或传输错误造成。你可能想："上一章不是学了模糊吗，糊一下不就没了？"

试试就知道：均值/高斯模糊确实能让白点"淡"一些，但代价是**整张图都变糊了**，而且那个白点并没消失，只是被"抹匀"成了一小片灰雾，还把周围 9 个好像素也一起带脏了。这暴露了线性滤波的软肋——**它对所有邻居一视同仁，一个 255 的坏点会实打实地把平均值往上拽。**

再看另一个需求：美颜磨皮，我们想磨掉皮肤上的细小噪点，**但绝不想把眼睛、鼻子的轮廓也糊掉**。普通高斯模糊做不到——它糊皮肤的同时也糊了五官。

这两个场景，线性滤波都搞不定。本章的两位主角——**中值滤波**和**双边滤波**——正是为此而生。它们的共同点是：**非线性**。

## 原理：跳出"加权求和"的框

### 为什么需要非线性滤波

回忆 [第 13 章](ch13-convolution.md)：线性滤波（卷积）的本质是"邻域加权求和"，无论核怎么设计，输出永远是邻居的**线性组合**。这带来一个根本局限：**只要邻域里混进一个极端值（如椒盐噪点），它就会按权重实实在在地污染结果**；而且求和是"糊"的操作，边缘两侧的像素被混在一起，边缘必然变模糊。

**非线性滤波**跳出了"加权求和"这个框——它可以**排序取中间**（中值），也可以**根据像素值差异动态调整权重**（双边）。正因为不再是简单线性组合，它们能做到线性滤波做不到的事：去掉极值、保住边缘。

### 中值滤波：排序取中间，天生克制椒盐

**中值滤波**的规则简单到发指：把邻域里的像素值**从小到大排序，取正中间那个（中位数）**作为输出。就这么一步，却对椒盐噪声有奇效。为什么？看这个 3×3 邻域，中间混进了一个 255 的"盐"噪点：

```text
   邻域(中间 255 是盐噪点)      排序后： 48 50 51 52 54 55 58 60 255
      52  55  255                                    ↑
      50  54  58      ──▶                     取中位数 = 54 (第 5 个)
      48  51  60

   均值：(52+55+255+50+54+58+48+51+60)/9 ≈ 80   ← 被 255 拽高，污染
   中值：排序后取正中间 = 54                      ← 极值被挤到末尾，无视它
```

关键在于：**椒盐噪点的值是极端的（0 或 255），排序后它们必然被挤到队列的两端，而中位数取的是正中间，天然避开了这些极值。** 而均值会被 255 硬生生拽高到 80。这就是中值滤波"专治椒盐"的原理。

更妙的是，中值滤波**天然保边**：在一条边缘附近，邻域里要么大多是暗的、要么大多是亮的，中位数会落在多数派那边，于是边缘依旧清晰，不像均值那样在边缘处糊成过渡带。代价是它**没有对应的卷积核**（排序不是线性运算），所以不能用可分离等技巧加速，半径一大就慢（后面会说）。

### 双边滤波：既看远近，也看像素像不像

中值擅长椒盐，但对付**高斯噪声**（那种全图轻微起伏的"沙沙"噪点）并保边，主角是**双边滤波**。

它的思路极其聪明。回忆高斯模糊：邻居的权重只由**空间距离**决定（离得近权重大）。双边滤波在此基础上**再乘一个权重**——由**像素值差异**决定（值越接近权重越大）：

\\[ O(p) = \frac{1}{W_p} \sum_{q \in \Omega} \underbrace{G_s\big(\lVert p-q \rVert\big)}_{\text{空间近就大}} \cdot \underbrace{G_r\big(\lvert I(p)-I(q) \rvert\big)}_{\text{颜色像就大}} \cdot I(q) \\]

其中 \\( G_s \\) 是**空间高斯**（管远近，和普通高斯一样），\\( G_r \\) 是**值域高斯**（range，管"像素值像不像"），\\( W_p \\) 是把两个权重乘积归一化的系数。

这个"值域权重"是点睛之笔。想象核正骑在一条边缘上，中心是暗的皮肤、旁边是亮的边缘：

- 对于**同侧、值也接近**的邻居：两个高斯都大 → 正常参与平均 → 皮肤被平滑。
- 对于**跨过边缘、值差很大**的邻居：值域高斯 \\( G_r \\) 趋近 0 → 它几乎不参与平均 → **边缘对面的像素被"拒之门外"，边缘因此得以保留**。

一句话：**双边滤波只和"既挨得近、又长得像"的邻居做平均**，所以能平滑平坦区、却不跨过边缘——这正是"磨皮不糊脸"的秘诀。它有两个关键参数：**空间 sigma**（`spatial_sigma`，管邻域多大、糊多远）和**值域 sigma**（color sigma，管"差多少算不像"，越小越保边）。

### 最大/最小滤波：顺带认识

还有两个简单的非线性滤波：**最大滤波**取邻域最大值（会让亮区膨胀、吞掉暗的小点），**最小滤波**取邻域最小值（反之）。它们其实就是灰度图上的**膨胀与腐蚀**，是形态学的基础——我们放到 [第 17 章 形态学操作](ch17-morphology.md) 专门展开，这里先混个脸熟。

## 动手写代码：中值治椒盐、双边保边

一个完整例子分两部分：① 给合成图加**椒盐噪声**，比较中值 vs 均值/高斯谁能救回原图；② 给一条**阶跃边缘**加**高斯噪声**，比较双边 vs 高斯谁能保住边缘。为了让结论有数字支撑，我们用"相对干净图的平均绝对误差 MAE"和"边缘落差"来量化。噪声用一个可复现的小 PRNG 生成，自包含。

`Cargo.toml` 依赖：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch14_nonlinear_filter.rs`）：

```rust
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::bilateral::GaussianEuclideanColorDistance;
use imageproc::filter::{bilateral_filter, box_filter, gaussian_blur_f32, median_filter};

/// 极简可复现伪随机数（xorshift32）
struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x; x
    }
    fn unit(&mut self) -> f32 { (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32 }
    /// Box-Muller 生成标准差为 sigma 的高斯噪声
    fn gauss(&mut self, sigma: f32) -> f32 {
        let u1 = self.unit().max(1e-6);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
    }
}

/// 干净场景：灰底(128) + 亮圆盘(210) + 一根暗竖条(40)
fn make_shapes(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        let (cx, cy, r) = (w as i32 / 2, h as i32 / 2, 45);
        if (xi - cx).pow(2) + (yi - cy).pow(2) < r * r { return Luma([210]); }
        if xi > 3 * w as i32 / 4 && xi < 3 * w as i32 / 4 + 12 { return Luma([40]); }
        Luma([128])
    })
}

/// 加椒盐噪声：每个像素以概率 prob 变成纯黑(0)或纯白(255)
fn add_salt_pepper(src: &GrayImage, rng: &mut Rng, prob: f32) -> GrayImage {
    let mut out = src.clone();
    for p in out.pixels_mut() {
        let r = rng.unit();
        if r < prob / 2.0 { p.0[0] = 0; }
        else if r < prob { p.0[0] = 255; }
    }
    out
}

/// 竖直阶跃边缘：左半暗(70) 右半亮(190)
fn make_step(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, _| if x < w / 2 { Luma([70]) } else { Luma([190]) })
}

/// 加高斯噪声
fn add_gaussian_noise(src: &GrayImage, rng: &mut Rng, sigma: f32) -> GrayImage {
    let mut out = src.clone();
    for p in out.pixels_mut() {
        let v = p.0[0] as f32 + rng.gauss(sigma);
        p.0[0] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 平均绝对误差（相对某张参考图，越小越接近）
fn mae(a: &GrayImage, b: &GrayImage) -> f32 {
    let s: f64 = a.as_raw().iter().zip(b.as_raw().iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs() as f64).sum();
    (s / a.as_raw().len() as f64) as f32
}

/// 统计取到极值(0 或 255)的像素个数——用来数"残留的椒盐点"
fn count_extremes(img: &GrayImage) -> u32 {
    img.as_raw().iter().filter(|&&v| v == 0 || v == 255).count() as u32
}

/// 某一列(x 固定)的平均灰度——用来量边缘
fn col_mean(img: &GrayImage, x: u32) -> f32 {
    let h = img.height();
    (0..h).map(|y| img.get_pixel(x, y).0[0] as u32).sum::<u32>() as f32 / h as f32
}

/// 左侧平坦区(最左 1/4)的灰度标准差——衡量残余噪声
fn flat_std(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    let mut vals = vec![];
    for y in 0..h { for x in 0..(w / 4) { vals.push(img.get_pixel(x, y).0[0] as f32); } }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    var.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng(0x1234_5678);

    // ===== 第一部分：椒盐噪声下，中值 vs 均值 =====
    let clean = make_shapes(240, 200);
    let noisy = add_salt_pepper(&clean, &mut rng, 0.08); // 8% 椒盐

    let med = median_filter(&noisy, 1, 1);   // 3×3 中值
    let avg = box_filter(&noisy, 1, 1);       // 3×3 均值
    let gau = gaussian_blur_f32(&noisy, 1.0); // 高斯

    println!("== 椒盐噪声：谁能救回原图（相对干净图的 MAE，越小越好）==");
    println!("加噪后    MAE = {:.2}", mae(&noisy, &clean));
    println!("均值模糊  MAE = {:.2}", mae(&avg, &clean));
    println!("高斯模糊  MAE = {:.2}", mae(&gau, &clean));
    println!("中值滤波  MAE = {:.2}", mae(&med, &clean));
    println!("残留椒盐点(值=0或255)：加噪 {} -> 中值 {} / 均值 {}",
             count_extremes(&noisy), count_extremes(&med), count_extremes(&avg));

    // ===== 第二部分：双边滤波保边去噪 =====
    let step = make_step(240, 200);
    let step_noisy = add_gaussian_noise(&step, &mut rng, 15.0);

    let gb = gaussian_blur_f32(&step_noisy, 2.0);
    let bf = bilateral_filter(&step_noisy, 3, 2.0, GaussianEuclideanColorDistance::new(30.0));

    // 边缘落差：紧贴分界线左右两列的灰度差（越大 = 边缘越陡越清晰）
    let edge = 240 / 2;
    let local_step = |img: &GrayImage| col_mean(img, edge) - col_mean(img, edge - 1);
    println!("\n== 高斯噪声：双边保边、高斯抹边 ==");
    println!("干净阶跃  边缘落差 = {:.1}", local_step(&step));
    println!("加噪后    边缘落差 = {:.1}", local_step(&step_noisy));
    println!("高斯模糊  边缘落差 = {:.1}（边缘被抹平）", local_step(&gb));
    println!("双边滤波  边缘落差 = {:.1}（边缘保住了）", local_step(&bf));
    println!("左侧平坦区噪声(标准差)：加噪 {:.1} -> 高斯 {:.1} / 双边 {:.1}",
             flat_std(&step_noisy), flat_std(&gb), flat_std(&bf));

    clean.save("out_ch14_clean.png")?;
    noisy.save("out_ch14_saltpepper.png")?;
    med.save("out_ch14_median.png")?;
    step_noisy.save("out_ch14_step_noisy.png")?;
    bf.save("out_ch14_step_bilateral.png")?;
    Ok(())
}
```

运行 `cargo run --example ch14_nonlinear_filter`，终端打印（数字真实可复现）：

```text
== 椒盐噪声：谁能救回原图（相对干净图的 MAE，越小越好）==
加噪后    MAE = 10.44
均值模糊  MAE = 8.96
高斯模糊  MAE = 8.21
中值滤波  MAE = 0.10
残留椒盐点(值=0或255)：加噪 3941 -> 中值 0 / 均值 0

== 高斯噪声：双边保边、高斯抹边 ==
干净阶跃  边缘落差 = 120.0
加噪后    边缘落差 = 119.0
高斯模糊  边缘落差 = 24.7（边缘被抹平）
双边滤波  边缘落差 = 119.5（边缘保住了）
左侧平坦区噪声(标准差)：加噪 15.0 -> 高斯 2.2 / 双边 4.2
```

这组数字把两条结论钉死了：

1. **中值滤波对椒盐噪声完胜**：加噪后 MAE 是 10.44，均值模糊只降到 8.96、高斯 8.21（改善甚微），而**中值直接干到 0.10**——几乎完美还原干净图！道理正如前面所讲：中值把 255/0 这些极值挤到排序两端、无视掉，而均值/高斯只是把噪点"摊匀"给周围（所以 `残留椒盐点` 里均值也是 0，纯 0/255 消失了——但看 MAE 就露馅：噪声只是被摊开、没被真正去掉）。
2. **双边滤波保边、高斯抹边**：干净阶跃的边缘落差是 120，加噪后仍是 119。高斯模糊把它**抹平到只剩 24.7**（边缘糊掉了），而**双边滤波保住了 119.5**（边缘几乎原样）。同时看平坦区噪声：从 15.0 降到双边的 4.2——**边缘保住了，噪声也压下去了**，这正是双边滤波的价值。（高斯把平坦区压得更低到 2.2，但那是以牺牲边缘为代价的。）

打开 `out_ch14_median.png` 和 `out_ch14_step_bilateral.png`，眼睛会告诉你和数字一样的结论。

## 常见坑与提示

> 🕳️ **坑**：**中值滤波半径越大越慢，别无脑放大。** 中值要对邻域排序，`median_filter(img, r, r)` 的窗口是 \\( (2r+1)^2 \\) 个像素，半径从 1 加到 3，每像素要排序的元素从 9 个涨到 49 个，耗时陡增。它又**不可分离**（排序不是线性运算，没法拆成两次一维）。所以椒盐噪声通常 `r=1` 或 `2` 就够；噪点特别密才考虑更大，或先小半径多做几遍。

> ⚠️ **注意**：**双边滤波的两个 sigma 各管一头，别调反。** `spatial_sigma`（空间）管"糊多远、邻域多大"，值大 → 更平滑、更慢；值域 sigma（代码里 `GaussianEuclideanColorDistance::new(30.0)` 的 `30`）管"像素值差多少还算相似"，**值越小越保边**（只和很像的邻居平均），值越大就越接近普通高斯（连边缘也一起糊）。想磨皮不糊脸，就把值域 sigma 调小些。

> 💡 **提示**：**先辨噪声类型，再选滤波器。** 孤立黑白点（椒盐）→ **中值**；全图轻微起伏（高斯噪声）且要保边 → **双边**；不在乎边缘、只求快速柔化 → 高斯足矣。用错工具事倍功半：拿高斯去治椒盐，你只会得到一张又糊、噪点又还在的图（上面 MAE 8.21 就是证据）。

> 🕳️ **坑**：**双边滤波比想象中慢，且通道数有限制。** 它对每个像素都要现算一圈值域权重，比高斯慢不少；`imageproc` 的 `bilateral_filter` 最多支持 4 通道图像。实时视频里用双边要留意性能，或改用更快的近似（如引导滤波，纯 Rust 生态目前需自行实现）。

## 本章小结

- **线性滤波（卷积）**是"加权求和"，一个极值就能污染结果，且求和必然糊边——这是它去椒盐、保边的天生短板。
- **中值滤波**取邻域中位数，**排序天然避开极值**，对**椒盐噪声**几乎完美（实测 MAE 从 10.44 降到 0.10），且**保边**；代价是不可分离、半径大则慢。
- **双边滤波**在空间高斯之外再乘一个**值域高斯**，只和"既近又像"的邻居平均，从而**保边平滑**（实测边缘落差从高斯的 24.7 保到 119.5），是"磨皮不糊脸"的关键；两个 sigma 分管空间范围与保边强度。
- **最大/最小滤波**即灰度膨胀/腐蚀，留到形态学章节展开。
- API：`imageproc::filter::median_filter(img, x_radius, y_radius)`、`bilateral_filter(img, radius, spatial_sigma, GaussianEuclideanColorDistance::new(color_sigma))`。

## 动手练习

1. **加大椒盐**：把 `add_salt_pepper` 的概率从 `0.08` 提到 `0.3`，观察 `r=1` 的中值是否还扛得住；再把中值半径改成 `2`，看 MAE 有没有改善。
2. **调双边 sigma**：把值域 sigma 从 `30.0` 改成 `5.0` 和 `100.0` 各跑一次，看"边缘落差"和"平坦区噪声"怎么变——体会值域 sigma 越大越接近普通高斯（边缘落差越小）。
3. **组合拳**：真实图像常同时有椒盐和高斯噪声。试着"先中值去椒盐、再双边保边去高斯"，把两章的滤波器串起来，比单用一个效果好在哪。
4. **进阶**：给中值滤波和高斯模糊分别计时（`std::time::Instant`），把半径/sigma 逐步加大，画出耗时曲线，直观感受"中值不可分离、越大越贵"。

下一章我们从"去噪、平滑"转向"找结构"——如何让计算机自动找出图像里物体的**轮廓边缘**？这就要用到梯度与经典的 Sobel、Canny 算子。👉 [第 15 章 梯度与边缘检测](ch15-edges.md)
