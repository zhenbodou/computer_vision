// 第 71 章：扩散模型（极小 DDPM）
// candle 0.11.0，纯 CPU，无需下载任何模型/数据。
// 运行：cargo run --release
//
// 用 2D 玩具数据（two moons，两个交错的半月）演示 DDPM 全过程：
//   前向：按噪声调度一步步把数据加噪 → 纯高斯噪声
//   训练：一个小 MLP 学「预测噪声 ε」（输入 = 带噪样本 x_t + 时间步 t 的嵌入），损失 = MSE
//   反向：从纯噪声出发，一步步去噪，采样回数据分布
// 证据：训练 loss 下降；采样分布与目标分布的均值/协方差对比、落在流形附近的样本比例；
//       以及「从噪声到数据」的采样轨迹图 PNG。
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const DIM: usize = 2; // 数据维度（2D 点）
const T: usize = 100; // 扩散总步数
const N_FREQ: usize = 6; // 时间嵌入频率数
const EMB: usize = 2 * N_FREQ; // 时间嵌入维度
// 对坐标 x 也做 Fourier 特征（sin/cos 多频率）：普通 MLP 有「谱偏置」——偏爱低频、学不好
// 环形多模态这种高角频率结构；给它一组高频基函数，才能把每个模式都学出来（不然会「隔一个丢一个」）。
const SP_FREQ: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
const SP_FEAT: usize = DIM + DIM * (2 * SP_FREQ.len()); // 原始 2 维 + 每维每频率 sin/cos
const IN: usize = SP_FEAT + EMB; // 网络输入维度
const HID: usize = 256; // MLP 隐层宽度
const GRID: usize = 3; // 3x3 网格
const K_MODES: usize = GRID * GRID; // 目标分布的模式数（9 个高斯团，网格排布）
const SPACING: f32 = 1.6; // 相邻模式中心间距
const BLOB_S: f32 = 0.14; // 每个高斯团的标准差

// ---------- 极简随机数发生器（固定种子、可复现；candle CPU 不能 set_seed） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self { Self(s) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 { (self.next() >> 11) as f32 / (1u64 << 53) as f32 }
    fn range(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn randn(&mut self) -> f32 {
        let u1 = self.unit().max(1e-9);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
fn det_init(vm: &VarMap, seed: u64) -> Result<()> {
    let mut rng = Lcg::new(seed);
    let data = vm.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    for name in &names {
        let var = &data[name];
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let dev = var.device();
        if name.ends_with("bias") {
            var.set(&Tensor::zeros(dims, DType::F32, dev)?)?;
        } else {
            let fan_in = (n / dims[0].max(1)).max(1);
            let lim = (1.0 / fan_in as f32).sqrt();
            let vals: Vec<f32> = (0..n).map(|_| (rng.unit() * 2.0 - 1.0) * lim).collect();
            var.set(&Tensor::from_vec(vals, dims, dev)?)?;
        }
    }
    Ok(())
}

// ---------- 目标分布：3x3 网格上的 9 个高斯团（经典多模态玩具），并标准化到零均值单位方差 ----------
// 用多模态分布正好凸显扩散模型的一大优点：不像 GAN 那样模式崩溃，它会把所有模式都覆盖到。
fn mode_center_raw(k: usize) -> [f32; DIM] {
    let (i, j) = (k % GRID, k / GRID);
    let off = (GRID as f32 - 1.0) / 2.0; // 让网格以原点为中心
    [(i as f32 - off) * SPACING, (j as f32 - off) * SPACING]
}
fn make_data(n: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut v = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let c = mode_center_raw(i % K_MODES); // 均衡地落在各个模式上
        v.push(c[0] + rng.randn() * BLOB_S);
        v.push(c[1] + rng.randn() * BLOB_S);
    }
    v
}
// 标准化后各模式中心的位置（用于「模式覆盖率」评估）
fn mode_centers(m: &[f32; DIM], s: &[f32; DIM]) -> Vec<[f32; DIM]> {
    (0..K_MODES).map(|k| {
        let c = mode_center_raw(k);
        [(c[0] - m[0]) / s[0], (c[1] - m[1]) / s[1]]
    }).collect()
}
// 求每维均值/标准差
fn mean_std(v: &[f32]) -> ([f32; DIM], [f32; DIM]) {
    let n = v.len() / DIM;
    let mut m = [0f32; DIM];
    for i in 0..n { for d in 0..DIM { m[d] += v[i * DIM + d]; } }
    for d in 0..DIM { m[d] /= n as f32; }
    let mut s = [0f32; DIM];
    for i in 0..n { for d in 0..DIM { let e = v[i * DIM + d] - m[d]; s[d] += e * e; } }
    for d in 0..DIM { s[d] = (s[d] / n as f32).sqrt().max(1e-6); }
    (m, s)
}
fn standardize(v: &mut [f32], m: &[f32; DIM], s: &[f32; DIM]) {
    let n = v.len() / DIM;
    for i in 0..n { for d in 0..DIM { v[i * DIM + d] = (v[i * DIM + d] - m[d]) / s[d]; } }
}

// ---------- 时间步嵌入（正弦编码，把整数步 t 变成一个连续向量喂给网络） ----------
fn time_emb(t_norm: f32) -> [f32; EMB] {
    let mut e = [0f32; EMB];
    for k in 0..N_FREQ {
        let freq = std::f32::consts::PI * (1u32 << k) as f32; // π, 2π, 4π, 8π, ...
        e[2 * k] = (freq * t_norm).sin();
        e[2 * k + 1] = (freq * t_norm).cos();
    }
    e
}
// 把「带噪样本 x_t + 时间步 t」编码成网络输入：坐标的 Fourier 特征 + 时间嵌入
fn encode(x: &[f32], t_norm: f32) -> Vec<f32> {
    let mut f = Vec::with_capacity(IN);
    for d in 0..DIM { f.push(x[d]); } // 原始坐标
    for d in 0..DIM {
        for &fr in &SP_FREQ { f.push((fr * x[d]).sin()); f.push((fr * x[d]).cos()); } // 坐标的 Fourier 特征
    }
    f.extend_from_slice(&time_emb(t_norm)); // 时间嵌入
    f
}

// ---------- 噪声预测网络：MLP，输入 [x_t(2) + t 嵌入(8)] → 预测 ε(2) ----------
struct EpsNet { fc1: Linear, fc2: Linear, fc3: Linear, fc4: Linear }
impl EpsNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(IN, HID, vb.pp("fc1"))?,
            fc2: linear(HID, HID, vb.pp("fc2"))?,
            fc3: linear(HID, HID, vb.pp("fc3"))?,
            fc4: linear(HID, DIM, vb.pp("fc4"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.relu()?;
        let h = self.fc2.forward(&h)?.relu()?;
        let h = self.fc3.forward(&h)?.relu()?;
        self.fc4.forward(&h) // 直接回归 ε，不加激活
    }
}

// ---------- 噪声调度 ----------
struct Schedule { beta: Vec<f32>, alpha: Vec<f32>, abar: Vec<f32> }
fn make_schedule(beta_t: f32) -> Schedule {
    let mut beta = vec![0f32; T];
    for t in 0..T {
        beta[t] = 1e-4 + (beta_t - 1e-4) * t as f32 / (T - 1) as f32; // 线性 β schedule
    }
    let alpha: Vec<f32> = beta.iter().map(|b| 1.0 - b).collect();
    let mut abar = vec![0f32; T];
    let mut acc = 1.0f32;
    for t in 0..T { acc *= alpha[t]; abar[t] = acc; } // ᾱ_t = ∏ α
    Schedule { beta, alpha, abar }
}

// ---------- 反向采样：从纯噪声一步步去噪，返回若干时间步的快照（用于画轨迹） ----------
fn sample(net: &EpsNet, sch: &Schedule, m: usize, rng: &mut Lcg, dev: &Device, snap_at: &[usize]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
    let mut x: Vec<f32> = (0..m * DIM).map(|_| rng.randn()).collect(); // x_T ~ N(0,I)
    let mut snaps: Vec<Vec<f32>> = Vec::new();
    if snap_at.contains(&T) { snaps.push(x.clone()); }
    for t in (0..T).rev() {
        // 组装网络输入 [m, IN]：每个点做 encode(坐标, 时间)
        let t_norm = (t as f32 + 0.5) / T as f32;
        let mut inp = Vec::with_capacity(m * IN);
        for i in 0..m {
            inp.extend_from_slice(&encode(&x[i * DIM..i * DIM + DIM], t_norm));
        }
        let it = Tensor::from_vec(inp, (m, IN), dev)?;
        let eps = net.forward(&it)?.to_vec2::<f32>()?; // [m,2] 预测噪声
        // DDPM 反向一步：mean = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε̂)；t>0 再加 √β_t·z（注入噪声=探索各模式）
        let (a, ab, b) = (sch.alpha[t], sch.abar[t], sch.beta[t]);
        let inv_sqrt_a = 1.0 / a.sqrt();
        let coef = b / (1.0 - ab).sqrt();
        for i in 0..m {
            for d in 0..DIM {
                let mean = inv_sqrt_a * (x[i * DIM + d] - coef * eps[i][d]);
                x[i * DIM + d] = if t > 0 { mean + b.sqrt() * rng.randn() } else { mean };
            }
        }
        if snap_at.contains(&t) { snaps.push(x.clone()); }
    }
    Ok((x, snaps))
}

// ---------- 散点图渲染 ----------
const PANEL: u32 = 180;
const RANGE: f32 = 3.0; // 坐标可视范围 [-RANGE, RANGE]
fn to_px(v: f32) -> i64 {
    let pad = 10.0;
    let s = (PANEL as f32 - 2.0 * pad) / (2.0 * RANGE);
    (pad + (v + RANGE) * s) as i64
}
fn draw_points(canvas: &mut RgbImage, ox: i64, pts: &[f32], color: Rgb<u8>, r: i64) {
    let n = pts.len() / DIM;
    for i in 0..n {
        let cx = ox + to_px(pts[i * DIM]);
        let cy = PANEL as i64 - to_px(pts[i * DIM + 1]); // y 轴向上
        for dy in -r..=r { for dx in -r..=r {
            let (px, py) = (cx + dx, cy + dy);
            if px >= ox && px < ox + PANEL as i64 && py >= 0 && py < PANEL as i64 {
                canvas.put_pixel(px as u32, py as u32, color);
            }
        }}
    }
}
// 一排面板：每个面板先铺淡灰参考(真实分布)，再画该面板自己的点集
fn save_trajectory(path: &str, panels: &[&[f32]], reference: &[f32]) {
    let gap = 8u32;
    let cols = panels.len() as u32;
    let w = cols * PANEL + (cols - 1) * gap;
    let mut canvas = RgbImage::from_pixel(w, PANEL, Rgb([250, 250, 250]));
    for (idx, pts) in panels.iter().enumerate() {
        let ox = idx as i64 * (PANEL + gap) as i64;
        draw_points(&mut canvas, ox, reference, Rgb([210, 210, 210]), 1); // 淡灰：目标分布参考
        draw_points(&mut canvas, ox, pts, Rgb([30, 90, 200]), 1); // 蓝：当前点集
    }
    canvas.save(path).unwrap();
}

fn l2(a: &[f32], b: &[f32]) -> f32 { (0..DIM).map(|d| { let e = a[d] - b[d]; e * e }).sum::<f32>().sqrt() }
// 落在目标流形附近的比例：每个生成点到最近真实点的距离 < tau 的占比
fn manifold_fraction(gen: &[f32], real: &[f32], tau: f32) -> f32 {
    let (ng, nr) = (gen.len() / DIM, real.len() / DIM);
    let mut ok = 0usize;
    for i in 0..ng {
        let g = [gen[i * DIM], gen[i * DIM + 1]];
        let mut best = f32::INFINITY;
        for j in 0..nr { best = best.min(l2(&g, &[real[j * DIM], real[j * DIM + 1]])); }
        if best < tau { ok += 1; }
    }
    ok as f32 / ng as f32
}
// 模式覆盖：把每个生成点分给最近的模式中心；若在 thresh 内则算「命中」该模式。
// 返回 (覆盖的模式数[命中样本≥1%的模式], 每个模式的命中计数)。
fn ring_coverage(gen: &[f32], centers: &[[f32; DIM]], thresh: f32) -> (usize, Vec<usize>) {
    let n = gen.len() / DIM;
    let mut counts = vec![0usize; centers.len()];
    for i in 0..n {
        let g = [gen[i * DIM], gen[i * DIM + 1]];
        let (mut best, mut bk) = (f32::INFINITY, 0usize);
        for (k, c) in centers.iter().enumerate() {
            let d = l2(&g, c);
            if d < best { best = d; bk = k; }
        }
        if best < thresh { counts[bk] += 1; }
    }
    let cov = counts.iter().filter(|&&c| c as f32 >= 0.01 * n as f32).count();
    (cov, counts)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    println!("==== 极小 DDPM：在 9 个高斯团(2D 3x3 网格分布)上学「预测噪声」并采样 ====");

    // ---- 数据：生成 + 标准化 ----
    let mut drng = Lcg::new(1);
    let mut train = make_data(4000, &mut drng);
    let (m, s) = mean_std(&train);
    standardize(&mut train, &m, &s);
    let mut real_eval = make_data(1600, &mut drng);
    standardize(&mut real_eval, &m, &s); // 用同一套统计量标准化
    let centers = mode_centers(&m, &s); // 标准化坐标系下的 8 个模式中心
    let n_train = train.len() / DIM;

    // ---- 噪声调度 ----
    let sch = make_schedule(0.10);
    println!(
        "  噪声调度：T={T}，β 从 {:.4} 线性升到 {:.2}；√ᾱ_T = {:.4}（≈0 表示末步已几乎是纯噪声）\n",
        sch.beta[0], sch.beta[T - 1], sch.abar[T - 1].sqrt()
    );

    // ---- 训练：学预测 ε ----
    let vm = VarMap::new();
    let net = EpsNet::load(VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, beta1: 0.9, beta2: 0.999, weight_decay: 0.0, ..Default::default() })?;
    let batch = 256;
    let steps = 25000;
    let mut trng = Lcg::new(2024);
    println!("---- 训练（预测噪声 ε，损失 = MSE）----");
    for step in 1..=steps {
        // 学习率衰减：后半程线性从 1e-3 降到 2e-4，收敛更稳更细
        if step > steps / 2 {
            let frac = (step - steps / 2) as f64 / (steps / 2) as f64;
            opt.set_learning_rate(1e-3 + (2e-4 - 1e-3) * frac);
        }
        // 组装一个 batch：随机取 x0、随机时间步 t、随机噪声 ε，构造 x_t 与网络输入
        let mut inp = Vec::with_capacity(batch * IN);
        let mut tgt = Vec::with_capacity(batch * DIM);
        for _ in 0..batch {
            let i = trng.range(n_train);
            let t = trng.range(T); // 时间步 0..T-1
            let (sqrt_ab, sqrt_1mab) = (sch.abar[t].sqrt(), (1.0 - sch.abar[t]).sqrt());
            let mut xt = [0f32; DIM];
            for d in 0..DIM {
                let eps = trng.randn();
                xt[d] = sqrt_ab * train[i * DIM + d] + sqrt_1mab * eps; // 前向加噪：x_t = √ᾱ·x0 + √(1-ᾱ)·ε
                tgt.push(eps); // 目标就是这个噪声
            }
            inp.extend_from_slice(&encode(&xt, (t as f32 + 0.5) / T as f32));
        }
        let it = Tensor::from_vec(inp, (batch, IN), &dev)?;
        let tt = Tensor::from_vec(tgt, (batch, DIM), &dev)?;
        let pred = net.forward(&it)?;
        let loss = (&pred - &tt)?.sqr()?.mean_all()?; // MSE(预测噪声, 真实噪声)
        opt.backward_step(&loss)?;
        if step == 1 || step % 1000 == 0 {
            println!("  step {step:>5}  MSE(ε̂, ε) = {:.4}", loss.to_scalar::<f32>()?);
        }
    }

    // ---- 采样：从噪声一步步去噪，记录轨迹快照（结构在末段才「结晶」，故快照偏后段密集）----
    println!("\n---- 采样（从纯噪声反向去噪回数据分布）----");
    let snap_at = [T, 40, 20, 8, 0];
    let mut srng = Lcg::new(777);
    let (gen, snaps) = sample(&net, &sch, 1500, &mut srng, &dev, &snap_at)?;

    // ---- 分布匹配的真实数字 ----
    let (gm, _gs) = mean_std(&gen);
    let (rm, _rs) = mean_std(&real_eval);
    println!("  均值   目标=({:.2},{:.2})  生成=({:.2},{:.2})", rm[0], rm[1], gm[0], gm[1]);
    // 校准阈值：真实评估集中点到最近模式中心距离的最大值 ×1.2（作为「模式团半径」）
    let mut rmax = 0.0f32;
    for i in 0..real_eval.len() / DIM {
        let p = [real_eval[i * DIM], real_eval[i * DIM + 1]];
        let mut best = f32::INFINITY;
        for c in &centers { best = best.min(l2(&p, c)); }
        rmax = rmax.max(best);
    }
    let thresh = rmax * 1.2;
    let (rcov, rcnt) = ring_coverage(&real_eval, &centers, thresh);
    let (gcov, gcnt) = ring_coverage(&gen, &centers, thresh);
    println!("  模式覆盖：目标 {rcov}/{K_MODES}   生成 {gcov}/{K_MODES}   （扩散模型不像 GAN 那样模式崩溃）");
    println!("  各模式命中计数：目标 {rcnt:?}");
    println!("                  生成 {gcnt:?}");
    // 校准 tau：真实评估集内部最近邻距离的中位数 × 3
    let mut nn: Vec<f32> = Vec::new();
    for i in 0..(real_eval.len() / DIM).min(500) {
        let a = [real_eval[i * DIM], real_eval[i * DIM + 1]];
        let mut best = f32::INFINITY;
        for j in 0..real_eval.len() / DIM {
            if j == i { continue; }
            best = best.min(l2(&a, &[real_eval[j * DIM], real_eval[j * DIM + 1]]));
        }
        nn.push(best);
    }
    nn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = nn[nn.len() / 2] * 3.0;
    let frac = manifold_fraction(&gen, &real_eval, tau);
    println!("  落在目标流形附近(距最近真实点 < {tau:.3})的生成样本比例 = {:.1}%", frac * 100.0);

    // ---- 画轨迹图 ----
    let panels: Vec<&[f32]> = snaps.iter().map(|s| s.as_slice()).collect();
    save_trajectory("out_ch71_trajectory.png", &panels, &real_eval);
    println!("\n  已存轨迹图 out_ch71_trajectory.png（左→右快照 t = {snap_at:?}；淡灰=目标分布参考，蓝=当前样本）");
    // 目标分布 vs 最终生成 的并排对照
    save_trajectory("out_ch71_compare.png", &[&real_eval, &gen], &[]);
    println!("  已存对照图 out_ch71_compare.png（左=目标分布，右=DDPM 生成）");
    Ok(())
}
