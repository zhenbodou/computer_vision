//! 第 61 章 粒子滤波（Particle Filter）—— 正文代码验证
//! 运行：cargo run --example ch61_particle_filter
//! 依赖：image 0.25.10 + imageproc 0.27.0（可视化）；核心仅用 std + 自带 PRNG
//!
//! 演示：2D 粒子滤波跟踪一个做“匀速+转弯”的目标。
//! 观测是【非线性】的：原点处传感器测“距离 range + 方位角 bearing”，且噪声较大——
//! 单看一帧只能把目标定在一片模糊的区域里，得靠多帧不断收紧后验分布。
//! 卡尔曼那套线性高斯假设不直接适用，粒子滤波却能自然表示这种任意形状的分布。
//! 工程要点：用【对数权重 + log-sum-exp】归一化避免数值下溢；N_eff 低于阈值才重采样；
//! 同时输出【加权均值】与【MAP（最大权重粒子）】两种估计。
//! 流程：predict(按运动模型扩散)->update(按观测对数似然累加)->systematic 重采样。

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};

// ---------- 自包含 PRNG（xorshift64*）+ Box-Muller 正态，固定种子保证可复现 ----------
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

#[derive(Clone, Copy)]
struct Particle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    lw: f64, // 对数权重（log-weight），比直接存权重更抗数值下溢
}

// 观测模型（非线性）：从原点(0,0)看目标 -> (距离, 方位角)
fn observe(x: f64, y: f64) -> (f64, f64) {
    ((x * x + y * y).sqrt(), y.atan2(x))
}

// 角度差归一化到 [-pi, pi]
fn wrap(mut a: f64) -> f64 {
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

// log-sum-exp：数值稳定地求 ln(Σ exp(lw))，用来归一化对数权重
fn log_sum_exp(lw: &[f64]) -> f64 {
    let m = lw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + lw.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

// systematic resampling：一根等间隔“梳子”扫过累积权重，权重大的粒子被复制多次
fn systematic_resample(ps: &[Particle], w: &[f64], rng: &mut Rng) -> Vec<Particle> {
    let n = ps.len();
    let mut cum = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        acc += w[i];
        cum[i] = acc;
    }
    let start = rng.uniform() / n as f64; // 梳子第一个齿落在 [0, 1/N)
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    for i in 0..n {
        let u = start + i as f64 / n as f64;
        while j < n - 1 && u > cum[j] {
            j += 1;
        }
        let mut np = ps[j];
        np.lw = 0.0; // 重采样后权重重置为均匀（对数权重清零即可，反正下一步会再归一化）
        out.push(np);
    }
    out
}

/// 跑一遍粒子滤波。meas 是每帧的 (距离, 方位角) 观测。
/// 返回 (加权均值估计, MAP估计, N_eff 序列)。resample=false 时【从不重采样】，演示退化。
fn run_pf(
    meas: &[(f64, f64)],
    n: usize,
    resample: bool,
    seed: u64,
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let (sig_r, sig_b) = (6.0f64, 0.12f64); // 观测噪声：距离 6px，方位 0.12rad(≈6.9°)
    let (q_pos, q_vel) = (0.6f64, 0.7f64); // 过程噪声：位置/速度每步扩散强度

    // 初始化：撒一个又宽又偏的先验，看它能不能被多帧观测收敛回真值
    let mut ps: Vec<Particle> = (0..n)
        .map(|_| Particle {
            x: 10.0 + 30.0 * rng.normal(),
            y: 65.0 + 30.0 * rng.normal(),
            vx: 3.0 * rng.normal(),
            vy: 3.0 * rng.normal(),
            lw: 0.0,
        })
        .collect();

    let mut est = Vec::with_capacity(meas.len());
    let mut est_map = Vec::with_capacity(meas.len());
    let mut neffs = Vec::with_capacity(meas.len());
    for &z in meas.iter() {
        // —— ① 预测：按匀速模型推进，并注入过程噪声（“扩散”粒子）——
        for p in ps.iter_mut() {
            p.x += p.vx + q_pos * rng.normal();
            p.y += p.vy + q_pos * rng.normal();
            p.vx += q_vel * rng.normal();
            p.vy += q_vel * rng.normal();
        }
        // —— ② 更新：把观测对数似然【累加】到对数权重上（不做 exp，避免下溢）——
        for p in ps.iter_mut() {
            let (r, b) = observe(p.x, p.y);
            let dr = r - z.0;
            let db = wrap(b - z.1);
            p.lw += -0.5 * (dr * dr / (sig_r * sig_r) + db * db / (sig_b * sig_b));
        }
        // —— 用 log-sum-exp 归一化得到线性权重 w ——
        let lws: Vec<f64> = ps.iter().map(|p| p.lw).collect();
        let lse = log_sum_exp(&lws);
        let w: Vec<f64> = lws.iter().map(|&v| (v - lse).exp()).collect();
        // —— 估计一：加权均值 ——
        let (mut ex, mut ey) = (0.0, 0.0);
        for (p, &wi) in ps.iter().zip(&w) {
            ex += wi * p.x;
            ey += wi * p.y;
        }
        est.push((ex, ey));
        // —— 估计二：MAP（权重最大的那个粒子）——多峰时比均值靠谱 ——
        let (mut best_i, mut best_w) = (0usize, f64::NEG_INFINITY);
        for (i, &wi) in w.iter().enumerate() {
            if wi > best_w {
                best_w = wi;
                best_i = i;
            }
        }
        est_map.push((ps[best_i].x, ps[best_i].y));
        // 有效粒子数 N_eff = 1 / Σw²：越小说明权重越集中在少数粒子上（退化）
        let neff = 1.0 / w.iter().map(|wi| wi * wi).sum::<f64>();
        neffs.push(neff);
        // —— ③ 重采样：N_eff 掉到一半以下才重采样（省算力又抗退化/样本贫化）——
        if resample && neff < n as f64 / 2.0 {
            ps = systematic_resample(&ps, &w, &mut rng);
        }
    }
    (est, est_map, neffs)
}

fn errors(est: &[(f64, f64)], truth: &[(f64, f64)]) -> Vec<f64> {
    est.iter()
        .zip(truth)
        .map(|(&(ex, ey), &(tx, ty))| ((ex - tx).powi(2) + (ey - ty).powi(2)).sqrt())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = 30usize;
    // ---- 合成真值：目标从 (55,25) 出发，匀速并缓慢转弯，画出一条弧线 ----
    let mut truth = Vec::with_capacity(steps);
    let (mut x, mut y) = (55.0f64, 25.0f64);
    let (mut vx, mut vy) = (0.5f64, 3.2f64);
    let turn = 0.04f64;
    for _ in 0..steps {
        truth.push((x, y));
        let (c, s) = (turn.cos(), turn.sin());
        let (nvx, nvy) = (vx * c - vy * s, vx * s + vy * c);
        vx = nvx;
        vy = nvy;
        x += vx;
        y += vy;
    }
    // ---- 合成非线性观测：距离 + 方位角，各加高斯噪声（固定种子）----
    let mut rng = Rng::new(777);
    let (sig_r, sig_b) = (6.0f64, 0.12f64);
    let meas: Vec<(f64, f64)> = truth
        .iter()
        .map(|&(tx, ty)| {
            let (r, b) = observe(tx, ty);
            (r + sig_r * rng.normal(), b + sig_b * rng.normal())
        })
        .collect();
    // 先验中心（10,65）离真值起点有多远——这就是“还没看数据时”的误差起点
    let prior_err = ((10.0 - truth[0].0).powi(2) + (65.0 - truth[0].1).powi(2)).sqrt();
    println!("初始先验中心距真值约 {:.0} px（撒得又宽又偏，还没用上任何观测）", prior_err);

    let n = 3000usize;
    let (est, est_map, neff) = run_pf(&meas, n, true, 12345);
    let err = errors(&est, &truth);
    let err_map = errors(&est_map, &truth);

    println!("== 2D 粒子滤波（{} 个粒子，非线性“距离+方位”观测）==", n);
    println!(" 步   估计(x,y)         真值(x,y)        位置误差   N_eff   重采样?");
    for &k in &[0usize, 2, 5, 10, 15, 20, 29] {
        let resampled = neff[k] < n as f64 / 2.0;
        println!(
            " {:>2}  ({:>6.1},{:>6.1})  ({:>6.1},{:>6.1})   {:>6.2}   {:>6.0}   {}",
            k, est[k].0, est[k].1, truth[k].0, truth[k].1, err[k], neff[k], if resampled { "是" } else { "否" }
        );
    }
    let last10 = err[steps - 10..].iter().sum::<f64>() / 10.0;
    let last10_map = err_map[steps - 10..].iter().sum::<f64>() / 10.0;
    println!(
        "\n收敛过程：先验中心 ~{:.0} px --首帧观测--> 第0步 {:.1} px --多帧--> 末10步平均 {:.1} px（稳稳跟住真值）",
        prior_err, err[0], last10
    );
    println!("同一次运行，MAP 估计末 10 步平均误差 = {:.1} px（单峰时与均值接近；多峰时才见真章）", last10_map);

    // ---- 对比：不重采样会“退化” ----
    let (est_nr, _, neff_nr) = run_pf(&meas, n, false, 12345);
    let err_nr = errors(&est_nr, &truth);
    let last10_nr = err_nr[steps - 10..].iter().sum::<f64>() / 10.0;
    println!(
        "对照【从不重采样】：末 N_eff = {:.1}（几乎只剩 1 个有效粒子，严重退化），末 10 步平均误差 = {:.1} px",
        neff_nr[steps - 1],
        last10_nr
    );

    // ---- 可视化：真值弧线(绿)、估计(蓝)、传感器原点(红)。坐标平移到画布内 ----
    let (cw, ch) = (220u32, 200u32);
    let (ox, oy) = (55.0f32, 20.0f32); // 把原点画到画布左上
    let tf = |p: (f64, f64)| (ox + p.0 as f32, oy + p.1 as f32);
    let mut vis = RgbImage::from_pixel(cw, ch, Rgb([255, 255, 255]));
    let o = tf((0.0, 0.0));
    draw_filled_circle_mut(&mut vis, (o.0 as i32, o.1 as i32), 3, Rgb([230, 0, 0])); // 传感器原点
    for wpair in truth.windows(2) {
        let (a, b) = (tf(wpair[0]), tf(wpair[1]));
        draw_line_segment_mut(&mut vis, a, b, Rgb([0, 170, 0]));
    }
    for wpair in est.windows(2) {
        let (a, b) = (tf(wpair[0]), tf(wpair[1]));
        draw_line_segment_mut(&mut vis, a, b, Rgb([0, 90, 255]));
    }
    vis.save("out_ch61_particle.png")?;
    println!("已存 out_ch61_particle.png（红=传感器原点，绿=真值弧线，蓝=粒子滤波估计）");
    Ok(())
}
