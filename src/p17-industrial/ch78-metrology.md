# 第 78 章 亚像素边缘、精密测量与不确定度

> **本章导读**
> - 学完你能：用**亚像素（subpixel）**技术把边缘定位到"比一个像素还准"，测量零件的**尺寸、角度、直径**，并诚实回答"这套系统到底能不能量这个公差"。
> - 掌握两把手术刀：灰度剖面**抛物线拟合**求亚像素边位；**最小二乘拟合直线/圆**（nalgebra）把边缘点变成规整几何。
> - 亲手造一张**已知边距/倾角/直径**的图，测完**和真值对账算误差**——但也认清：一次漂亮的结果不代表能力。
> - 建立工业测量的"专家心智"：**边缘定义会带来系统偏差**、像素→物理必须**标定**、真正的验收靠**不确定度、重复性/再现性、GR&R**。

## 场景：0.01 毫米的公差，你敢接吗？

一块手机中框在 CNC 产线上下来，安装孔直径公差 ±0.01 毫米。相机视野宽 20 毫米、横向 2000 像素，于是**一个像素 ≈ 0.01 毫米**。如果边缘只能定位到整数像素，两条边一减，误差就把公差吃满——这活儿没法干，必须**亚像素**。

但真正的工业场景里，老板会追问一句更狠的："你这套系统**到底能不能**量这个 ±0.01 的公差？"回答它，靠的不是给他看一次量得准的截图，而是一份**不确定度报告**：系统的重复性有多少、换个人换台设备结果会漂多少、够不够把"合格"和"不合格"分开。本章前半段教你把边量准，后半段教你回答老板这句话——后者才是工业测量与"跑个 demo"的分水岭。

## 原理一：一条边的亚像素定位

### 直觉——边是"渐变"的，峰在梯度里

真实成像里，一条边不是"啪"地从黑跳到白，而是跨几个像素**平滑过渡**（镜头、传感器都会把它抹开）。把一条扫描线上的灰度画出来是条 S 形曲线；对它求**梯度**（相邻像素之差），过渡最陡处梯度最大——**梯度峰的位置就是边的位置**。

```text
灰度剖面 I(x)              梯度 g(x)=½(I(x+1)−I(x−1))
255 ┤        ______          ┤      ∧
    │       /                │     ╱ ╲    ← 峰值落在两个像素之间
    │      /                 │    ╱   ╲
  0 ┤_____/                0 ┤___╱     ╲___
    └───────────────x        └──────────────x
       ↑真实边在这             整数最高点 = 20，真峰 = 20.35
```

梯度的最高点仍落在某个**整数**像素。要抠出小数部分，用**抛物线拟合（parabola fitting）**：拿离散峰及左右邻居三个点拟一条抛物线，其**顶点**即亚像素峰位。设峰在下标 \\( 0 \\)、三点梯度值为 \\( g_{-1}, g_0, g_{+1} \\)（\\( g_0 \\) 最大），顶点偏移量：

\\[ \delta = \frac{1}{2}\cdot\frac{g_{-1}-g_{+1}}{g_{-1}-2g_0+g_{+1}} \\]

亚像素边位 \\( x_{\text{sub}} = x_{\text{peak}} + \delta \\)，\\( \delta \in (-0.5, 0.5) \\)。只用三个数几次加减乘除，就把精度从 1 像素提到 0.05 像素量级。

> 📖 **术语**：**边缘极性（edge polarity）与卡尺（caliper）**。卡尺工具在一小块 ROI 内沿设定方向拉**多条平行扫描线**，每条按指定**极性**（暗→亮取正梯度峰、亮→暗取负梯度峰）找一个亚像素边点，再把点集**拟合**成直线或圆。多条线求平均能压噪声。

## 原理二：从边缘点到规整几何——最小二乘拟合

工业测量要的是**尺寸/角度/直径**。做法是沿多条扫描线收集边缘点，再用**最小二乘（least squares）**拟合成直线或圆——正是 [第 30 章 nalgebra](../p05-numeric/ch30-nalgebra.md) 里 SVD 解超定方程的拿手好戏。

- **拟合直线（测角度）**：直接套 \\( y=mx+c \\) 遇到近竖直边会因斜率发散而翻车。工业上用**正交回归（全最小二乘）**：算点集**协方差矩阵**，其最大特征向量即直线方向。本章用 `symmetric_eigen` 求 2×2 协方差的特征向量。
- **拟合圆（测直径）**：用 **Kåsa 代数拟合**。圆方程 \\( x^2+y^2+Dx+Ey+F=0 \\) 对 \\( D,E,F \\) 线性，每个点写成一行 \\( x_i D + y_i E + F = -(x_i^2+y_i^2) \\)，堆成超定方程用 SVD 解，再换算圆心 \\( (-D/2,\,-E/2) \\)、半径 \\( \sqrt{D^2/4+E^2/4-F} \\)。

## 原理三：像素不是物理量——边缘定义与标定

这里有两个专家级的坑，新手几乎必栽：

**其一，边缘定义会引入系统偏差。** "边到底在哪"取决于你的定义——是取梯度峰（本章做法），还是取灰度过半（阈值法）？在**不对称**的过渡（一侧照明强、一侧弱，或离焦不均）下，这两种定义会给出**系统性不同**的位置。测同一条边，换个定义，尺寸可能整体偏 0.1~0.5 像素。所以：**同一个测量项，边缘定义/极性/搜索方向必须固定并写进规程**，否则不同班次、不同工程师量出来对不上。

**其二，像素→物理不能"全图一个 mm/pixel"。** 镜头有畸变、被测面有高度、透视下不同位置的放大率不同。把中心标定出的 `mm/pixel` 套到视野边缘，会有系统误差。正确做法是用**标定模型**（去畸变 + 每像素物理尺度）或对被测平面做**局部单应映射**；有高度变化时甚至要上三维（[第 64 章 相机标定](../p14-3d/ch64-calibration.md)、[第 66 章 PnP](../p14-3d/ch66-epipolar-pnp.md)）。本章为聚焦算法，全程用**像素**做单位、拿像素真值对账。

## 动手写代码：造真值、测量、算误差

三个**自验证**实验，每个都"先造已知答案的图，测完对账"：**A** 一维亚像素（边位真值 `20.37`）；**B** 卡尺量宽度 + 直线拟合量倾角（倾斜 4°、水平间距 120.5 像素的亮条，21 条扫描线）；**C** 径向卡尺 + 拟合圆（圆心 `(160.4,120.7)`、直径 140.6 像素，72 条射线）。

`Cargo.toml` 依赖：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
nalgebra = "0.35"
```

完整代码（`examples/ch78_metrology.rs`）：

```rust
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_circle_mut};
use nalgebra::{DMatrix, DVector, Matrix2};

const W: u32 = 320;
const H: u32 = 240;

/// 软边阶跃：中心在 x0、过渡宽度 edge_w 的“上升沿”，返回 [0,1]。
/// 用它合成的边缘不落在整数像素上，才能考验亚像素精度。
fn soft_step(x: f64, x0: f64, edge_w: f64) -> f64 {
    0.5 * (1.0 + ((x - x0) / edge_w).tanh())
}

/// 抛物线三点拟合，求离散峰附近的亚像素偏移 δ ∈ (-0.5, 0.5)。
fn parabola_offset(ym1: f64, y0: f64, yp1: f64) -> f64 {
    let denom = ym1 - 2.0 * y0 + yp1;
    if denom.abs() < 1e-12 { 0.0 } else { 0.5 * (ym1 - yp1) / denom }
}

/// 双线性采样：在“非整数坐标”处读到平滑灰度（径向卡尺要用）。
fn sample_bilinear(img: &GrayImage, x: f64, y: f64) -> f64 {
    let (w, h) = img.dimensions();
    if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return 0.0;
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let p = |xx, yy| img.get_pixel(xx, yy)[0] as f64;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

// ---------- 实验 A：一维亚像素边缘 ----------
fn demo_1d(x0_true: f64, edge_w: f64) {
    let n = 40usize;
    let prof: Vec<f64> = (0..n).map(|x| 255.0 * soft_step(x as f64, x0_true, edge_w)).collect();
    let grad: Vec<f64> = (0..n)
        .map(|i| if i == 0 || i + 1 >= n { 0.0 } else { 0.5 * (prof[i + 1] - prof[i - 1]) })
        .collect();
    let mut peak = 1;
    for i in 1..n - 1 {
        if grad[i].abs() > grad[peak].abs() { peak = i; }
    }
    let delta = parabola_offset(grad[peak - 1].abs(), grad[peak].abs(), grad[peak + 1].abs());
    let x_sub = peak as f64 + delta;
    println!("[A 一维亚像素]");
    println!("  整数峰位  = {peak} px（只能精确到 1 像素）");
    println!("  亚像素边位 = {x_sub:.4} px，真值 = {x0_true}，误差 = {:.4} px", (x_sub - x0_true).abs());
    println!("  对比：整数峰误差 = {:.4} px（亚像素把精度提高了一个量级）\n", (peak as f64 - x0_true).abs());
}

// ---------- 实验 B：卡尺量宽度 + 直线拟合量倾角 ----------
/// 合成一根“软边亮条”：左边在 xl0+k*y、右边在 xr0+k*y（k=tan(倾角)）。
fn make_bar(xl0: f64, xr0: f64, k: f64, edge_w: f64) -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        let xf = x as f64;
        let l = xl0 + k * y as f64;
        let r = xr0 + k * y as f64;
        let v = soft_step(xf, l, edge_w) * (1.0 - soft_step(xf, r, edge_w));
        Luma([(255.0 * v).round().clamp(0.0, 255.0) as u8])
    })
}

/// 在第 y 行找左（上升沿）、右（下降沿）两条边的亚像素 x 坐标。
fn row_edges(img: &GrayImage, y: u32) -> (f64, f64) {
    let w = img.width() as usize;
    let prof: Vec<f64> = (0..w).map(|x| img.get_pixel(x as u32, y)[0] as f64).collect();
    let grad: Vec<f64> = (0..w)
        .map(|i| if i == 0 || i + 1 >= w { 0.0 } else { 0.5 * (prof[i + 1] - prof[i - 1]) })
        .collect();
    let mut lp = 1; // 左：最大“正”梯度（暗→亮）
    for i in 1..w - 1 { if grad[i] > grad[lp] { lp = i; } }
    let lx = lp as f64 + parabola_offset(grad[lp - 1], grad[lp], grad[lp + 1]);
    let mut rp = 1; // 右：最“负”梯度（亮→暗）；取负号变成峰再拟合
    for i in 1..w - 1 { if grad[i] < grad[rp] { rp = i; } }
    let rx = rp as f64 + parabola_offset(-grad[rp - 1], -grad[rp], -grad[rp + 1]);
    (lx, rx)
}

/// 全最小二乘（正交回归）拟合直线，返回与竖直方向的夹角°。
fn line_angle_from_vertical(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in pts {
        let (dx, dy) = (x - mx, y - my);
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    let scatter = Matrix2::new(sxx, sxy, sxy, syy);
    let se = scatter.symmetric_eigen(); // 较大特征值对应的特征向量 = 直线方向
    let big = if se.eigenvalues[0] >= se.eigenvalues[1] { 0 } else { 1 };
    let dir = se.eigenvectors.column(big);
    let (mut dx, mut dy) = (dir[0], dir[1]);
    if dy < 0.0 { dx = -dx; dy = -dy; }
    dx.atan2(dy).to_degrees()
}

fn demo_bar(xl0: f64, xr0: f64, tilt_deg: f64, edge_w: f64) {
    let k = tilt_deg.to_radians().tan();
    let bar = make_bar(xl0, xr0, k, edge_w);
    bar.save("out_ch78_bar.png").ok();
    let rows: Vec<u32> = (20..=220).step_by(10).collect();
    let mut widths = Vec::new();
    let mut left_pts = Vec::new();
    for &y in &rows {
        let (lx, rx) = row_edges(&bar, y);
        widths.push(rx - lx);
        left_pts.push((lx, y as f64));
    }
    let w_mean = widths.iter().sum::<f64>() / widths.len() as f64;
    let w_true = xr0 - xl0; // 水平卡尺测的是水平间距
    let angle = line_angle_from_vertical(&left_pts);
    println!("[B 卡尺测宽度 + 直线拟合测倾角]（{} 条扫描线）", rows.len());
    println!("  水平宽度 = {w_mean:.4} px，真值 = {w_true}，误差 = {:.4} px", (w_mean - w_true).abs());
    println!("  倾角 = {angle:.4}°，真值 = {tilt_deg}°，误差 = {:.4}°\n", (angle - tilt_deg).abs());
}

// ---------- 实验 C：径向卡尺 + 拟合圆 ----------
fn make_disk(cx: f64, cy: f64, rad: f64, edge_w: f64) -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
        let v = 1.0 - soft_step(d, rad, edge_w); // 内亮外暗
        Luma([(255.0 * v).round().clamp(0.0, 255.0) as u8])
    })
}

fn rough_centroid(img: &GrayImage, thresh: u8) -> (f64, f64) {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0u64);
    for (x, y, p) in img.enumerate_pixels() {
        if p[0] >= thresh { sx += x as f64; sy += y as f64; n += 1; }
    }
    (sx / n as f64, sy / n as f64)
}

/// 从粗中心沿角度 theta 打射线，找亚像素边界点（内亮外暗=下降沿）。
fn ray_edge(img: &GrayImage, cx0: f64, cy0: f64, theta: f64, r_max: f64) -> Option<(f64, f64)> {
    let n = r_max as usize;
    let (dx, dy) = (theta.cos(), theta.sin());
    let prof: Vec<f64> = (0..n).map(|i| sample_bilinear(img, cx0 + i as f64 * dx, cy0 + i as f64 * dy)).collect();
    let grad: Vec<f64> = (0..n)
        .map(|i| if i == 0 || i + 1 >= n { 0.0 } else { 0.5 * (prof[i + 1] - prof[i - 1]) })
        .collect();
    let mut rp = 1;
    for i in 1..n - 1 { if grad[i] < grad[rp] { rp = i; } }
    if rp < 1 || rp + 1 >= n { return None; }
    let r_sub = rp as f64 + parabola_offset(-grad[rp - 1], -grad[rp], -grad[rp + 1]);
    Some((cx0 + r_sub * dx, cy0 + r_sub * dy))
}

/// Kåsa 代数最小二乘拟合圆：解 A·[D,E,F]ᵀ = b，返回 (cx, cy, r)。
fn fit_circle(pts: &[(f64, f64)]) -> (f64, f64, f64) {
    let m = pts.len();
    let mut a = Vec::with_capacity(m * 3);
    let mut b = Vec::with_capacity(m);
    for &(x, y) in pts {
        a.push(x); a.push(y); a.push(1.0);
        b.push(-(x * x + y * y));
    }
    let am = DMatrix::from_row_slice(m, 3, &a);
    let bv = DVector::from_column_slice(&b);
    let sol = am.svd(true, true).solve(&bv, 1e-9).expect("SVD solve");
    let (cx, cy) = (-sol[0] / 2.0, -sol[1] / 2.0);
    let r = ((sol[0] * sol[0] + sol[1] * sol[1]) / 4.0 - sol[2]).sqrt();
    (cx, cy, r)
}

fn demo_circle(cx_t: f64, cy_t: f64, rad_t: f64, edge_w: f64) {
    let disk = make_disk(cx_t, cy_t, rad_t, edge_w);
    let (cx0, cy0) = rough_centroid(&disk, 128);
    let m = 72; // 均匀打 72 条射线
    let mut pts = Vec::new();
    for i in 0..m {
        let theta = i as f64 / m as f64 * std::f64::consts::TAU;
        if let Some(p) = ray_edge(&disk, cx0, cy0, theta, rad_t + 20.0) { pts.push(p); }
    }
    let (cx, cy, r) = fit_circle(&pts);
    println!("[C 径向卡尺 + 拟合圆]（{} 个边界点）", pts.len());
    println!("  粗中心（质心）= ({cx0:.2}, {cy0:.2})");
    println!("  拟合圆心 = ({cx:.4}, {cy:.4})，真值 = ({cx_t}, {cy_t})，误差 = {:.4} px",
        ((cx - cx_t).powi(2) + (cy - cy_t).powi(2)).sqrt());
    println!("  直径 = {:.4} px，真值 = {}，误差 = {:.4} px", 2.0 * r, 2.0 * rad_t, (2.0 * r - 2.0 * rad_t).abs());
    // 可视化：边界点(红十字) + 拟合圆(绿)
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(disk).to_rgb8();
    for &(x, y) in &pts {
        draw_cross_mut(&mut vis, Rgb([255, 0, 0]), x.round() as i32, y.round() as i32);
    }
    draw_hollow_circle_mut(&mut vis, (cx.round() as i32, cy.round() as i32), r.round() as i32, Rgb([0, 255, 0]));
    vis.save("out_ch78_circle.png").ok();
}

fn main() {
    demo_1d(20.37, 1.6);
    demo_bar(90.3, 210.8, 4.0, 1.6);
    demo_circle(160.4, 120.7, 70.3, 1.6);
}
```

运行 `cargo run --example ch78_metrology`，终端打印（数字真实可复现）：

```text
[A 一维亚像素]
  整数峰位  = 20 px（只能精确到 1 像素）
  亚像素边位 = 20.3450 px，真值 = 20.37，误差 = 0.0250 px
  对比：整数峰误差 = 0.3700 px（亚像素把精度提高了一个量级）

[B 卡尺测宽度 + 直线拟合测倾角]（21 条扫描线）
  水平宽度 = 120.4947 px，真值 = 120.50000000000001，误差 = 0.0053 px
  倾角 = 4.0002°，真值 = 4°，误差 = 0.0002°

[C 径向卡尺 + 拟合圆]（72 个边界点）
  粗中心（质心）= (160.41, 120.66)
  拟合圆心 = (160.3975, 120.7046)，真值 = (160.4, 120.7)，误差 = 0.0052 px
  直径 = 140.5570 px，真值 = 140.6，误差 = 0.0430 px
```

对账：**一维**整数峰误差 0.37 像素，亚像素拟合到 `20.3450`、误差仅 **0.0250 像素**；**宽度**21 条线平均 `120.4947`、误差 **0.0053 像素**（多线平均压住了单线抖动）；**倾角** `4.0002°`、误差 **0.0002°**；**圆**圆心误差 **0.0052 像素**、直径误差 **0.0430 像素**。打开 `out_ch78_circle.png`，红十字（亚像素边界点）整齐落在圆周、绿色拟合圆严丝合缝。

## 落地：不确定度与测量系统分析（专家必修）

上面的误差都 < 0.05 像素，够漂亮了吧？**但它证明不了系统能量 ±0.01 毫米的公差。** 因为那是**合成图、无噪声、真值已知**的理想情形。真实系统要过三关：

**第一关：分辨率 ≠ 精度。** "亚像素"只说明**数值分辨率**细到小数——它绝不等于**真实精度**。系统精度还被标定残差、噪声、振动、温漂死死拖住。报告里写"亚像素级"却不给不确定度，是外行话。

**第二关：误差预算（error budget）。** 把总不确定度拆成可量化的分量，逐项估计再合成（按方差相加、开方）：

| 误差来源 | 说明 |
|---|---|
| 标定残差 | 像素→物理换算与去畸变的残余误差 |
| 边缘重复性 | 同一条边多次测量的随机抖动（噪声、量化） |
| 镜头畸变 | 未完全校正的几何畸变，视野边缘更大 |
| 工件高度/姿态 | 被测面不在标定平面、有倾斜，放大率就变 |
| 景深/离焦 | 边缘变糊，峰变平，定位方差变大 |
| 温漂 | 温度变化导致镜头/机械/相机尺度漂移 |
| 量具参考不确定度 | 你的"真值"来自标准件，它本身也有不确定度 |

**第三关：测量系统分析（MSA）。** 用**同一件重复测**评估**重复性（repeatability）**；**换操作者/设备/时间**评估**再现性（reproducibility）**；用**偏倚（bias）**看系统性偏离、用 **GR&R** 综合评估量具能力，并给**置信区间**。关键判据是能不能把公差"看清"——常用**精度-公差比（P/T）**：测量波动占公差带的比例，一般要求 P/T < 10%~30%。

> 📖 **术语**：**GR&R（Gage Repeatability and Reproducibility，量具重复性与再现性）**。工业界评估"这把尺子/这套视觉能不能用"的标准方法：多操作者 × 多零件 × 多次重复，用方差分析拆出重复性与再现性，算出测量系统吃掉了多少公差带。**一次量得准，不代表 GR&R 过关。**

## 常见坑与提示

> 🕳️ **坑**：**扫描方向要垂直于边缘。** 亚像素靠"沿测量方向的灰度陡变"。扫描线与边近乎平行时剖面没有清晰台阶，峰又平又糊，位置乱跳。量竖直边就横扫，量圆就沿半径打射线（本章即如此）。

> ⚠️ **注意**：**边缘定义/极性必须固定并写进规程。** 不同定义（梯度峰 vs 灰度过半）在不对称过渡下会产生**系统偏差**。同一测量项，一旦换了定义或搜索方向，历史数据就对不上了。

> 🕳️ **坑**：**别用一个 `mm/pixel` 走天下。** 有畸变、有高度、有透视时，视野各处放大率不同。要用标定模型或局部单应（[第 64 章](../p14-3d/ch64-calibration.md)），否则边缘区域系统性偏差。

> 💡 **提示**：**噪声抖动结果，多点拟合来治。** 梯度对噪声敏感，单点边位可能抖零点几像素。对策：像卡尺那样多线/多点最小二乘（本章圆用 72 点），或先对剖面轻度高斯平滑再求梯度——别用会削峰的中值。

> ⚠️ **注意**：**验收看不确定度，不看一次结果。** 交付前务必做重复性/再现性/GR&R，用 P/T 比证明系统能区分公差；只贴一张"量得准"的截图，是要吃大亏的。

## 本章小结

- **亚像素边缘**：灰度剖面 → 梯度 → 峰上**抛物线三点拟合**，一个 \\( \delta \\) 公式把精度提高一个量级（实测 0.37 → 0.025 像素）。
- **卡尺 + 拟合几何**：多扫描线压噪声；正交回归（`symmetric_eigen`）拟合直线测角度、Kåsa（SVD）拟合圆测直径。实测宽度误差 0.005 像素、倾角 0.0002°、直径误差 0.043 像素。
- **像素≠物理**：边缘定义会带系统偏差；像素→物理必须标定，且不能全图一个 `mm/pixel`。
- **能力≠一次结果**：亚像素是**数值分辨率**不是**精度**；真正的验收靠**误差预算 + 重复性/再现性/GR&R + P/T 比**。

### 专家考核

1. **建误差地图**：用可溯源标准件（如量块/标准环规），在视野的**网格位置**和**多个高度**上各测多次，画出"位置/高度 → 系统偏差"的**误差地图**，据此决定是否需要局部标定或去畸变加强。
2. **多日 GR&R**：设计多操作者 × 多零件 × 多次重复的方案，**连续多日**采集，报告**偏倚、重复性、再现性、线性、稳定性**五项，并给出 P/T 比与置信区间——用数据回答"能不能量这个公差"。
3. **漂移与重标定门限**：故意改变**曝光、焦距、温度**，量化各自引起的测量漂移，定义触发**重新标定**的门限（如漂移超过公差带的某比例即报警重标）。
4. **不确定度合成**：把上面各分量按方差合成，写一页**不确定度报告**（含 k=2 扩展不确定度），这才是能交给客户/质量部门的东西。

下一章，我们从"量得准"进到"找得到、摆得正、引导机器人去抓去贴"。👉 [第 79 章 视觉定位、机器人引导与手眼标定](ch79-alignment-guidance.md)
