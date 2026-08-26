//! 第 79 章 视觉定位与引导
//! 运行：cargo run --example ch79_alignment
//! 依赖：image 0.25.10, imageproc 0.27.0, nalgebra 0.35.0
//!
//! A. 位姿估计：把工件模板旋转 θ 放到已知 (dx,dy) 处，用"旋转搜索 + 归一化
//!    互相关"定位，粗到细两级搜索，打印估计位姿 vs 真值误差。
//! B. 手眼标定的简化平面版：几对"像素↔机器人坐标"点，最小二乘解仿射，
//!    打印映射误差（引第 64 章标定、第 10 章几何变换）。

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::geometric_transformations::{rotate_about_center, Border, Interpolation};
use imageproc::rect::Rect;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};
use nalgebra::{DMatrix, DVector};

const BG: u8 = 40; // 统一背景灰度：模板、旋转填充、场景都用它，拼接无缝
const TS: u32 = 44; // 模板边长

/// 合成一个"有方向感"的工件模板：亮矩形 + 偏置的亮圆（打破旋转对称，θ 才唯一可辨）。
fn make_template() -> GrayImage {
    let mut t = GrayImage::from_pixel(TS, TS, Luma([BG]));
    draw_filled_rect_mut(&mut t, Rect::at(6, 15).of_size(32, 14), Luma([200]));
    draw_filled_circle_mut(&mut t, (13, 22), 5, Luma([255])); // 靠左端的"键"，破对称
    t
}

/// 把模板顺时针旋转 deg 度（其余部分用 BG 填充），保持 TS×TS 尺寸。
fn rotate_template(t: &GrayImage, deg: f64) -> GrayImage {
    rotate_about_center(
        t,
        (deg as f32).to_radians(),
        Interpolation::Bilinear,
        Border::Constant(Luma([BG])),
    )
}

/// 在 angles 里逐个旋转模板、做归一化互相关，返回得分最高的 (角度°, 左上角位置, 得分)。
fn search_pose(scene: &GrayImage, t: &GrayImage, angles: &[f64]) -> (f64, (u32, u32), f32) {
    let mut best = (0.0f64, (0u32, 0u32), f32::MIN);
    for &a in angles {
        let rot = rotate_template(t, a);
        let map = match_template(scene, &rot, MatchTemplateMethod::CrossCorrelationNormalized);
        let e = find_extremes(&map);
        if e.max_value > best.2 {
            best = (a, e.max_value_location, e.max_value);
        }
    }
    best
}

fn demo_pose() {
    let template = make_template();
    // —— 造真值：把模板转 θ_true 度，贴到场景 (px,py) 处 ——
    let (px, py, theta_true) = (80i64, 55i64, 12.7f64);
    let mut scene = GrayImage::from_pixel(200, 150, Luma([BG]));
    let stamped = rotate_template(&template, theta_true);
    image::imageops::overlay(&mut scene, &stamped, px, py);
    scene.save("out_ch79_scene.png").ok();

    // 工件中心真值（旋转是绕patch中心做的，故中心 = 左上角 + 半个模板）
    let (cx_true, cy_true) = (px as f64 + TS as f64 / 2.0, py as f64 + TS as f64 / 2.0);

    // —— 第 1 级：粗搜索（步长 3°）——
    let coarse: Vec<f64> = (0..=30).step_by(3).map(|d| d as f64).collect();
    let (a_c, _, s_c) = search_pose(&scene, &template, &coarse);

    // —— 第 2 级：在粗结果附近细搜索（步长 0.5°）——
    let mut fine = Vec::new();
    let mut a = a_c - 2.0;
    while a <= a_c + 2.0 + 1e-9 {
        fine.push(a);
        a += 0.5;
    }
    let (a_f, loc, s_f) = search_pose(&scene, &template, &fine);

    let (cx_est, cy_est) = (loc.0 as f64 + TS as f64 / 2.0, loc.1 as f64 + TS as f64 / 2.0);
    let pos_err = ((cx_est - cx_true).powi(2) + (cy_est - cy_true).powi(2)).sqrt();

    println!("[A 位姿估计]");
    println!("  粗搜索最佳角 = {a_c}°（得分 {s_c:.4}）");
    println!("  细搜索最佳角 = {a_f}°（得分 {s_f:.4}）");
    println!(
        "  估计位姿 (x,y,θ) = ({cx_est:.1}, {cy_est:.1}, {a_f:.2}°)"
    );
    println!("  真值位姿 (x,y,θ) = ({cx_true:.1}, {cy_true:.1}, {theta_true}°)");
    println!(
        "  位置误差 = {pos_err:.3} px，角度误差 = {:.3}°\n",
        (a_f - theta_true).abs()
    );

    // 可视化：在场景上框出定位结果、标中心
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(scene).to_rgb8();
    draw_hollow_rect_mut(&mut vis, Rect::at(loc.0 as i32, loc.1 as i32).of_size(TS, TS), Rgb([0, 255, 0]));
    draw_cross_mut(&mut vis, Rgb([255, 0, 0]), cx_est as i32, cy_est as i32);
    vis.save("out_ch79_located.png").ok();
}

/// 对一维目标 target = a*px + b*py + c 做最小二乘，返回 [a,b,c]。
fn solve_affine_row(pxs: &[f64], pys: &[f64], target: &[f64]) -> [f64; 3] {
    let n = pxs.len();
    let mut a = Vec::with_capacity(n * 3);
    for i in 0..n {
        a.push(pxs[i]);
        a.push(pys[i]);
        a.push(1.0);
    }
    let am = DMatrix::from_row_slice(n, 3, &a);
    let bv = DVector::from_column_slice(target);
    let sol = am.svd(true, true).solve(&bv, 1e-9).expect("SVD solve");
    [sol[0], sol[1], sol[2]]
}

fn demo_hand_eye() {
    // —— 真值仿射：机器人坐标(mm) = M · [px, py, 1] ——
    // 相机与机器人平面差一个 缩放 s、旋转 φ、平移 (tx,ty)
    let (s, phi) = (0.25f64, 8.0f64.to_radians()); // 0.25 mm/px，转 8°
    let (tx, ty) = (12.0f64, -30.0f64);
    let mx = [s * phi.cos(), -s * phi.sin(), tx];
    let my = [s * phi.sin(), s * phi.cos(), ty];
    let map_true = |px: f64, py: f64| {
        (mx[0] * px + mx[1] * py + mx[2], my[0] * px + my[1] * py + my[2])
    };

    // —— 标定点：6 个像素点 + 对应机器人读数（叠加固定小噪声模拟测量误差）——
    let pix = [(40.0, 30.0), (280.0, 35.0), (300.0, 200.0), (60.0, 210.0), (160.0, 120.0), (200.0, 80.0)];
    let noise = [(0.03, -0.04), (-0.02, 0.05), (0.04, 0.01), (-0.05, -0.02), (0.01, 0.03), (0.02, -0.03)];
    let (mut pxs, mut pys, mut rxs, mut rys) = (vec![], vec![], vec![], vec![]);
    for (i, &(px, py)) in pix.iter().enumerate() {
        let (rx, ry) = map_true(px, py);
        pxs.push(px);
        pys.push(py);
        rxs.push(rx + noise[i].0);
        rys.push(ry + noise[i].1);
    }

    // —— 最小二乘解出 M_est ——
    let ex = solve_affine_row(&pxs, &pys, &rxs);
    let ey = solve_affine_row(&pxs, &pys, &rys);
    let map_est = |px: f64, py: f64| {
        (ex[0] * px + ex[1] * py + ex[2], ey[0] * px + ey[1] * py + ey[2])
    };

    // RMS 残差（标定点上，单位 mm）
    let mut se = 0.0;
    for &(px, py) in &pix {
        let (tx_, ty_) = map_true(px, py);
        let (ex_, ey_) = map_est(px, py);
        se += (tx_ - ex_).powi(2) + (ty_ - ey_).powi(2);
    }
    let rms = (se / pix.len() as f64).sqrt();

    // 拿一个"没参与标定"的新像素点验证
    let (npx, npy) = (123.0, 156.0);
    let (gt_x, gt_y) = map_true(npx, npy);
    let (es_x, es_y) = map_est(npx, npy);
    let test_err = ((gt_x - es_x).powi(2) + (gt_y - es_y).powi(2)).sqrt();

    println!("[B 像素→机器人 仿射标定]（6 对点最小二乘）");
    println!("  真值 M 第1行 = [{:.5}, {:.5}, {:.3}]", mx[0], mx[1], mx[2]);
    println!("  估计 M 第1行 = [{:.5}, {:.5}, {:.3}]", ex[0], ex[1], ex[2]);
    println!("  标定点 RMS 残差 = {rms:.4} mm");
    println!("  新点像素({npx},{npy}) → 机器人估计=({es_x:.3},{es_y:.3}) mm，真值=({gt_x:.3},{gt_y:.3}) mm");
    println!("  新点映射误差 = {test_err:.4} mm");
}

fn main() {
    demo_pose();
    demo_hand_eye();
}
