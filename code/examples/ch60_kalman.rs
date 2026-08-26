//! 第 60 章 卡尔曼滤波与目标跟踪 —— 正文代码验证
//! 运行：cargo run --example ch60_kalman
//! 依赖：nalgebra 0.35.0（矩阵运算）、image 0.25.10 + imageproc 0.27.0（可视化存 PNG）
//!
//! 演示：匀速(CV)模型、状态 [x, y, vx, vy] 的完整卡尔曼滤波。
//! ① 合成一条 2D 匀速直线真值轨迹 + 高斯观测噪声，跑 KF，打印“原始观测 vs 滤波后”对真值的 RMSE；
//! ② 一条“转弯”的机动轨迹：小 Q 会滞后、大 Q 更跟手，演示 Q 的调参含义；
//! ③ 把真值/观测/滤波轨迹画到一张 PNG 上。

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

// ---------- 自包含 PRNG（xorshift64*）+ Box-Muller 正态，保证可复现 ----------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// 一遍匀速卡尔曼滤波。输入观测序列，返回每步滤波后的位置 (x,y)、卡尔曼增益 K[0,0]、平均 NIS。
/// dt: 帧间隔；sigma_a: 过程噪声（“目标加速度”的标准差，越大越信观测）；r_std: 观测噪声标准差。
/// NIS（归一化新息平方）= y^T S^-1 y，若 Q/R 设得一致，其均值应接近观测维度（这里是 2）。
fn run_kf(meas: &[(f64, f64)], dt: f64, sigma_a: f64, r_std: f64) -> (Vec<(f64, f64)>, Vec<f64>, f64) {
    // 状态转移 F：x' = x + vx·dt, y' = y + vy·dt, 速度不变（匀速假设）
    let f = Matrix4::new(
        1.0, 0.0, dt, 0.0, //
        0.0, 1.0, 0.0, dt, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    );
    // 观测 H：传感器只能测到位置 (x, y)，测不到速度
    let h = Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0,
    );
    // 过程噪声 Q：离散白噪声加速度模型，由 sigma_a² 统一决定
    let sa2 = sigma_a * sigma_a;
    let (t2, t3, t4) = (dt * dt, dt * dt * dt, dt * dt * dt * dt);
    let q = Matrix4::new(
        t4 / 4.0, 0.0, t3 / 2.0, 0.0, //
        0.0, t4 / 4.0, 0.0, t3 / 2.0, //
        t3 / 2.0, 0.0, t2, 0.0, //
        0.0, t3 / 2.0, 0.0, t2,
    ) * sa2;
    // 观测噪声 R：与我们注入的噪声匹配（对角，x/y 独立）
    let r = Matrix2::new(r_std * r_std, 0.0, 0.0, r_std * r_std);

    // 初值：位置用第一个观测，速度设 0；P 给大值表示“初始很不确定”
    let mut x = Vector4::new(meas[0].0, meas[0].1, 0.0, 0.0);
    let mut p = Matrix4::identity() * 500.0;
    let i4 = Matrix4::<f64>::identity();

    let mut out = vec![(x[0], x[1])];
    let mut gains = vec![1.0];
    let mut nis_sum = 0.0;
    let mut nis_cnt = 0usize;
    for z in meas.iter().skip(1) {
        // —— 预测（先验）——
        x = f * x;
        p = f * p * f.transpose() + q;
        // —— 更新（用观测校正）——
        let z = Vector2::new(z.0, z.1);
        let y = z - h * x; // 新息：观测 - 预测
        let s = h * p * h.transpose() + r; // 新息协方差
        let s_inv = s.try_inverse().unwrap();
        let k: Matrix4x2<f64> = p * h.transpose() * s_inv; // 卡尔曼增益
        x += k * y;
        // Joseph 形式更新协方差：数值上更稳，始终保持对称半正定
        let ikh = i4 - k * h;
        p = ikh * p * ikh.transpose() + k * r * k.transpose();
        out.push((x[0], x[1]));
        gains.push(k[(0, 0)]);
        // NIS = y^T S^-1 y（标量），做一致性自检
        nis_sum += (y.transpose() * s_inv * y)[(0, 0)];
        nis_cnt += 1;
    }
    (out, gains, nis_sum / nis_cnt as f64)
}

fn rmse(est: &[(f64, f64)], truth: &[(f64, f64)]) -> f64 {
    let s: f64 = est
        .iter()
        .zip(truth)
        .map(|(&(ax, ay), &(tx, ty))| (ax - tx).powi(2) + (ay - ty).powi(2))
        .sum();
    (s / est.len() as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dt = 1.0f64;
    let steps = 60usize;
    let r_std = 8.0f64; // 观测噪声标准差（像素）
    let mut rng = Rng::new(2024);

    // ============ ① 匀速直线：KF 大显身手 ============
    let (x0, y0, vx, vy) = (10.0, 20.0, 2.2, 1.3);
    let truth: Vec<(f64, f64)> = (0..steps)
        .map(|k| (x0 + vx * k as f64 * dt, y0 + vy * k as f64 * dt))
        .collect();
    let meas: Vec<(f64, f64)> = truth
        .iter()
        .map(|&(tx, ty)| (tx + r_std * rng.normal(), ty + r_std * rng.normal()))
        .collect();

    let (filt, gains, nis) = run_kf(&meas, dt, 0.05, r_std);

    println!("== ① 匀速直线轨迹（观测噪声 std={} px）==", r_std);
    println!(" 步  观测(x,y)         滤波(x,y)         真值(x,y)        增益K[0,0]");
    for &k in &[1usize, 2, 3, 10, 59] {
        println!(
            " {:>2}  ({:>6.1},{:>6.1})  ({:>6.1},{:>6.1})  ({:>6.1},{:>6.1})   {:.3}",
            k, meas[k].0, meas[k].1, filt[k].0, filt[k].1, truth[k].0, truth[k].1, gains[k]
        );
    }
    let rm = rmse(&meas, &truth);
    let rf = rmse(&filt, &truth);
    // 稳态 RMSE：跳过前 12 步“冷启动”（初速未知、协方差很大），看收敛后的真实水平
    let rm_ss = rmse(&meas[12..], &truth[12..]);
    let rf_ss = rmse(&filt[12..], &truth[12..]);
    println!("原始观测 RMSE = {:.3} px（全程）/ {:.3} px（稳态）", rm, rm_ss);
    println!("卡尔曼滤波 RMSE = {:.3} px（全程）/ {:.3} px（稳态）", rf, rf_ss);
    println!(
        "稳态误差降幅 = {:.1}%（增益 K 随时间收敛，滤波器越来越“信任”自己的预测）",
        (1.0 - rf_ss / rm_ss) * 100.0
    );
    println!("平均 NIS = {:.2}（观测维度=2，落在其附近说明 Q/R 与真实噪声一致）\n", nis);

    // ============ ② 机动（转弯）轨迹：Q 的调参含义 ============
    // 真值做匀速转弯：速度矢量每步旋转一个小角度 —— 违背“匀速直线”假设
    let mut mx = vec![(40.0f64, 150.0f64)];
    let (mut px, mut py) = (40.0f64, 150.0f64);
    let (mut mvx, mut mvy) = (3.0f64, 0.0f64);
    let turn = 0.06f64; // 每步转 0.06 弧度
    for _ in 1..steps {
        let (c, s) = (turn.cos(), turn.sin());
        let (nvx, nvy) = (mvx * c - mvy * s, mvx * s + mvy * c);
        mvx = nvx;
        mvy = nvy;
        px += mvx;
        py += mvy;
        mx.push((px, py));
    }
    let mmeas: Vec<(f64, f64)> = mx
        .iter()
        .map(|&(tx, ty)| (tx + r_std * rng.normal(), ty + r_std * rng.normal()))
        .collect();

    // 小 Q（过度相信匀速模型）vs 大 Q（更相信观测、更跟手）
    let (filt_small, _, _) = run_kf(&mmeas, dt, 0.05, r_std);
    let (filt_big, _, _) = run_kf(&mmeas, dt, 1.2, r_std);
    println!("== ② 转弯（机动）轨迹：Q 调参 ==");
    println!("原始观测         RMSE = {:.3} px", rmse(&mmeas, &mx));
    println!("小 Q(σa=0.05) 滤波 RMSE = {:.3} px  <- 太信匀速模型，转弯处滞后", rmse(&filt_small, &mx));
    println!("大 Q(σa=1.20) 滤波 RMSE = {:.3} px  <- 更信观测，跟得上但更抖\n", rmse(&filt_big, &mx));

    // ============ ③ 可视化：真值/观测/滤波 三条线 ============
    let (cw, ch) = (200u32, 130u32);
    let mut vis = RgbImage::from_pixel(cw, ch, Rgb([255, 255, 255]));
    // 观测点（红），真值线（绿），滤波线（蓝）—— 用轨迹①
    for m in &meas {
        draw_filled_circle_mut(&mut vis, (m.0 as i32, m.1 as i32), 1, Rgb([230, 120, 120]));
    }
    for wpair in truth.windows(2) {
        draw_line_segment_mut(&mut vis, (wpair[0].0 as f32, wpair[0].1 as f32), (wpair[1].0 as f32, wpair[1].1 as f32), Rgb([0, 170, 0]));
    }
    for wpair in filt.windows(2) {
        draw_line_segment_mut(&mut vis, (wpair[0].0 as f32, wpair[0].1 as f32), (wpair[1].0 as f32, wpair[1].1 as f32), Rgb([0, 90, 255]));
    }
    vis.save("out_ch60_kalman.png")?;
    println!("已存 out_ch60_kalman.png（红=观测散点，绿=真值，蓝=滤波）");
    Ok(())
}
