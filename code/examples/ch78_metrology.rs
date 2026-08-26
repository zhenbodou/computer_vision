//! 第 78 章 亚像素边缘与精密测量
//! 运行：cargo run --example ch78_metrology
//! 依赖：image 0.25.10, imageproc 0.27.0, nalgebra 0.35.0
//!
//! 三个"造真值 → 测量 → 对比误差"的自验证实验：
//!   A. 一维亚像素边缘（抛物线拟合梯度峰）—— 已知边位 x0
//!   B. 卡尺量亮条宽度 + 直线拟合量倾角 —— 已知宽度/倾角
//!   C. 径向卡尺 + 最小二乘拟合圆 —— 已知圆心/直径

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_circle_mut};
use nalgebra::{DMatrix, DVector, Matrix2};

const W: u32 = 320;
const H: u32 = 240;

/// 软边阶跃：中心在 x0、过渡宽度 edge_w 的"上升沿"，返回 [0,1]。
/// 用它合成的边缘不落在整数像素上，才能考验亚像素精度。
fn soft_step(x: f64, x0: f64, edge_w: f64) -> f64 {
    0.5 * (1.0 + ((x - x0) / edge_w).tanh())
}

/// 抛物线三点拟合，求离散峰附近的亚像素偏移 δ ∈ (-0.5, 0.5)。
/// 输入 y_{-1}, y_0, y_{+1}（y_0 为离散峰值），返回顶点相对 y_0 的偏移。
fn parabola_offset(ym1: f64, y0: f64, yp1: f64) -> f64 {
    let denom = ym1 - 2.0 * y0 + yp1;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        0.5 * (ym1 - yp1) / denom
    }
}

/// 双线性采样：让我们能在"非整数坐标"处读到平滑的灰度（径向卡尺要用）。
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
    // 采样一条软边的灰度剖面
    let prof: Vec<f64> = (0..n)
        .map(|x| 255.0 * soft_step(x as f64, x0_true, edge_w))
        .collect();
    // 中心差分求梯度
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 || i + 1 >= n {
                0.0
            } else {
                0.5 * (prof[i + 1] - prof[i - 1])
            }
        })
        .collect();
    // 找梯度幅值最大的整数像素
    let mut peak = 1;
    for i in 1..n - 1 {
        if grad[i].abs() > grad[peak].abs() {
            peak = i;
        }
    }
    let delta = parabola_offset(grad[peak - 1].abs(), grad[peak].abs(), grad[peak + 1].abs());
    let x_sub = peak as f64 + delta;
    println!("[A 一维亚像素]");
    println!("  整数峰位  = {peak} px（只能精确到 1 像素）");
    println!(
        "  亚像素边位 = {x_sub:.4} px，真值 = {x0_true}，误差 = {:.4} px",
        (x_sub - x0_true).abs()
    );
    println!(
        "  对比：整数峰误差 = {:.4} px（亚像素把精度提高了一个量级）\n",
        (peak as f64 - x0_true).abs()
    );
}

// ---------- 实验 B：卡尺量宽度 + 直线拟合量倾角 ----------
/// 合成一根"软边亮条"：左边在 xl0+k*y、右边在 xr0+k*y（k=tan(倾角)）。
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
        .map(|i| {
            if i == 0 || i + 1 >= w {
                0.0
            } else {
                0.5 * (prof[i + 1] - prof[i - 1])
            }
        })
        .collect();
    // 左边：最大"正"梯度（暗→亮）
    let mut lp = 1;
    for i in 1..w - 1 {
        if grad[i] > grad[lp] {
            lp = i;
        }
    }
    let lx = lp as f64 + parabola_offset(grad[lp - 1], grad[lp], grad[lp + 1]);
    // 右边：最"负"梯度（亮→暗）；取负号变成峰再拟合
    let mut rp = 1;
    for i in 1..w - 1 {
        if grad[i] < grad[rp] {
            rp = i;
        }
    }
    let rx = rp as f64 + parabola_offset(-grad[rp - 1], -grad[rp], -grad[rp + 1]);
    (lx, rx)
}

/// 全最小二乘（正交回归）拟合直线，返回 (与竖直方向的夹角°)。
fn line_angle_from_vertical(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in pts {
        let (dx, dy) = (x - mx, y - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let scatter = Matrix2::new(sxx, sxy, sxy, syy);
    let se = scatter.symmetric_eigen();
    // 取较大特征值对应的特征向量作为直线方向
    let big = if se.eigenvalues[0] >= se.eigenvalues[1] { 0 } else { 1 };
    let dir = se.eigenvectors.column(big);
    let (mut dx, mut dy) = (dir[0], dir[1]);
    if dy < 0.0 {
        dx = -dx;
        dy = -dy;
    }
    dx.atan2(dy).to_degrees() // 与竖直(y轴)夹角
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
    println!(
        "  水平宽度 = {w_mean:.4} px，真值 = {w_true}，误差 = {:.4} px",
        (w_mean - w_true).abs()
    );
    println!(
        "  倾角 = {angle:.4}°，真值 = {tilt_deg}°，误差 = {:.4}°\n",
        (angle - tilt_deg).abs()
    );
}

// ---------- 实验 C：径向卡尺 + 拟合圆 ----------
fn make_disk(cx: f64, cy: f64, rad: f64, edge_w: f64) -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
        let v = 1.0 - soft_step(d, rad, edge_w); // 内亮外暗
        Luma([(255.0 * v).round().clamp(0.0, 255.0) as u8])
    })
}

/// 阈值化后求质心，作为径向卡尺的"粗中心"。
fn rough_centroid(img: &GrayImage, thresh: u8) -> (f64, f64) {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0u64);
    for (x, y, p) in img.enumerate_pixels() {
        if p[0] >= thresh {
            sx += x as f64;
            sy += y as f64;
            n += 1;
        }
    }
    (sx / n as f64, sy / n as f64)
}

/// 从粗中心沿角度 theta 打一条射线，找亚像素边界点（内亮外暗=下降沿）。
fn ray_edge(img: &GrayImage, cx0: f64, cy0: f64, theta: f64, r_max: f64) -> Option<(f64, f64)> {
    let n = r_max as usize;
    let (dx, dy) = (theta.cos(), theta.sin());
    let prof: Vec<f64> = (0..n)
        .map(|i| sample_bilinear(img, cx0 + i as f64 * dx, cy0 + i as f64 * dy))
        .collect();
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 || i + 1 >= n {
                0.0
            } else {
                0.5 * (prof[i + 1] - prof[i - 1])
            }
        })
        .collect();
    let mut rp = 1;
    for i in 1..n - 1 {
        if grad[i] < grad[rp] {
            rp = i;
        }
    }
    if rp < 1 || rp + 1 >= n {
        return None;
    }
    let r_sub = rp as f64 + parabola_offset(-grad[rp - 1], -grad[rp], -grad[rp + 1]);
    Some((cx0 + r_sub * dx, cy0 + r_sub * dy))
}

/// Kåsa 代数最小二乘拟合圆：解 A·[D,E,F]ᵀ = b，返回 (cx, cy, r)。
fn fit_circle(pts: &[(f64, f64)]) -> (f64, f64, f64) {
    let m = pts.len();
    let mut a = Vec::with_capacity(m * 3);
    let mut b = Vec::with_capacity(m);
    for &(x, y) in pts {
        a.push(x);
        a.push(y);
        a.push(1.0);
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

    // 均匀打 72 条射线，收集亚像素边界点
    let m = 72;
    let mut pts = Vec::new();
    for i in 0..m {
        let theta = i as f64 / m as f64 * std::f64::consts::TAU;
        if let Some(p) = ray_edge(&disk, cx0, cy0, theta, rad_t + 20.0) {
            pts.push(p);
        }
    }
    let (cx, cy, r) = fit_circle(&pts);

    println!("[C 径向卡尺 + 拟合圆]（{} 个边界点）", pts.len());
    println!("  粗中心（质心）= ({cx0:.2}, {cy0:.2})");
    println!(
        "  拟合圆心 = ({cx:.4}, {cy:.4})，真值 = ({cx_t}, {cy_t})，误差 = {:.4} px",
        ((cx - cx_t).powi(2) + (cy - cy_t).powi(2)).sqrt()
    );
    println!(
        "  直径 = {:.4} px，真值 = {}，误差 = {:.4} px",
        2.0 * r,
        2.0 * rad_t,
        (2.0 * r - 2.0 * rad_t).abs()
    );

    // 可视化：边界点(红十字) + 拟合圆(绿)
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(disk).to_rgb8();
    for &(x, y) in &pts {
        draw_cross_mut(&mut vis, Rgb([255, 0, 0]), x.round() as i32, y.round() as i32);
    }
    draw_hollow_circle_mut(&mut vis, (cx.round() as i32, cy.round() as i32), r.round() as i32, Rgb([0, 255, 0]));
    vis.save("out_ch78_circle.png").ok();
    println!("  已保存 out_ch78_circle.png");
}

fn main() {
    demo_1d(20.37, 1.6);
    demo_bar(90.3, 210.8, 4.0, 1.6);
    demo_circle(160.4, 120.7, 70.3, 1.6);
}
