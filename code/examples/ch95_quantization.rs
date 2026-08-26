// 第 95 章配套：手写 INT8 量化 demo（纯 std，无重依赖，结果可复现）
// 运行：cargo run --example ch95_quantization
//
// 目标：亲手把一段 f32 数据量化到 int8 再反量化，测量量化误差，
// 直观感受"精度换体积/速度"，并对比不同 scale 策略、per-tensor vs per-channel、
// 对称 vs 非对称量化。

/// 线性同余随机数（自带，避免引入 rand；同一 seed 每次运行结果一致）
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 均匀分布 [0, 1)
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
    }
    /// 近似标准正态（12 个均匀分布相加减 6，中心极限定理）
    fn next_gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.next_f32();
        }
        s - 6.0
    }
}

// ---------- 对称量化（权重常用）----------
// scale = amax / 127；q = clip(round(x/scale), -127, 127)；反量化 x' = q * scale
fn quantize_symmetric(x: &[f32], scale: f32) -> Vec<i8> {
    x.iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect()
}
fn dequantize_symmetric(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&qi| qi as f32 * scale).collect()
}

// ---------- 非对称（仿射）量化（激活常用）----------
// scale = (max-min)/255；zero_point = round(-min/scale)；q = clip(round(x/scale)+z, 0, 255)
// 反量化 x' = scale * (q - z)
fn quantize_affine(x: &[f32], scale: f32, zero_point: i32) -> Vec<u8> {
    x.iter()
        .map(|&v| ((v / scale).round() as i32 + zero_point).clamp(0, 255) as u8)
        .collect()
}
fn dequantize_affine(q: &[u8], scale: f32, zero_point: i32) -> Vec<f32> {
    q.iter().map(|&qi| scale * (qi as i32 - zero_point) as f32).collect()
}

// ---------- 误差度量 ----------
fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32
}
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn amax(x: &[f32]) -> f32 {
    x.iter().map(|v| v.abs()).fold(0.0, f32::max)
}

fn main() {
    println!("=== 第 95 章 · 手写 INT8 量化 ===\n");

    // ---- 1) 对称量化一段权重，看误差 ----
    let mut rng = Lcg::new(2026);
    // 造 4096 个近似正态的权重（均值 0），再塞几个离群值模拟长尾
    let mut w: Vec<f32> = (0..4096).map(|_| rng.next_gauss() * 0.05).collect();
    w[100] = 0.9; // 离群值
    w[2000] = -1.1; // 离群值

    let s_maxabs = amax(&w) / 127.0; // 教科书对称 scale
    let q = quantize_symmetric(&w, s_maxabs);
    let deq = dequantize_symmetric(&q, s_maxabs);
    println!("[1] 对称量化 4096 个权重（含 2 个离群值 ±~1.0）");
    println!("    amax = {:.4}, scale = amax/127 = {:.6}", amax(&w), s_maxabs);
    println!("    体积：f32 {} 字节 -> int8 {} 字节（1/4）", w.len() * 4, w.len());
    println!("    量化 MSE = {:.3e}, 最大绝对误差 = {:.5}", mse(&w, &deq), max_abs_err(&w, &deq));
    println!("    示例：w[0]={:.5} -> q={} -> 反量化={:.5}\n", w[0], q[0], deq[0]);

    // ---- 2) 不同 scale 策略：离群值如何"浪费"整数码 ----
    // 策略 A：max-abs（受离群值主导，scale 偏大，主体分辨率浪费）
    // 策略 B：截断到 3 倍标准差（clip 离群值，scale 变小，主体更精细但离群值饱和）
    // 策略 C：故意放大 2 倍 scale（更粗）
    let std_w = (w.iter().map(|v| v * v).sum::<f32>() / w.len() as f32).sqrt();
    let clip = 3.0 * std_w;
    let s_clip = clip / 127.0;
    let s_coarse = 2.0 * s_maxabs;

    let deq_a = dequantize_symmetric(&quantize_symmetric(&w, s_maxabs), s_maxabs);
    let deq_b = dequantize_symmetric(&quantize_symmetric(&w, s_clip), s_clip);
    let deq_c = dequantize_symmetric(&quantize_symmetric(&w, s_coarse), s_coarse);
    // 只统计"主体"（|w|<clip）的误差，看主体分辨率
    let body: Vec<usize> = (0..w.len()).filter(|&i| w[i].abs() < clip).collect();
    let body_mse = |d: &[f32]| -> f32 {
        body.iter().map(|&i| (w[i] - d[i]).powi(2)).sum::<f32>() / body.len() as f32
    };
    println!("[2] scale 策略对比（std={:.4}, 截断阈值 3σ={:.4}）", std_w, clip);
    println!("    A max-abs   : scale={:.6}  全体MSE={:.3e}  主体MSE={:.3e}  最大误差={:.4}", s_maxabs, mse(&w, &deq_a), body_mse(&deq_a), max_abs_err(&w, &deq_a));
    println!("    B 截断3σ    : scale={:.6}  全体MSE={:.3e}  主体MSE={:.3e}  最大误差={:.4}", s_clip, mse(&w, &deq_b), body_mse(&deq_b), max_abs_err(&w, &deq_b));
    println!("    C 放大2x    : scale={:.6}  全体MSE={:.3e}  主体MSE={:.3e}  最大误差={:.4}", s_coarse, mse(&w, &deq_c), body_mse(&deq_c), max_abs_err(&w, &deq_c));
    println!("    结论：截断离群值后主体 MSE 明显下降，代价是离群值处最大误差变大（长尾权衡）\n");

    // ---- 3) per-tensor vs per-channel ----
    // 3 个通道（行）尺度差异悬殊：0.1 / 1.0 / 10.0
    let mut rng = Lcg::new(7);
    let scales_true = [0.1f32, 1.0, 10.0];
    let n = 512;
    let mat: Vec<Vec<f32>> = scales_true
        .iter()
        .map(|&s| (0..n).map(|_| rng.next_gauss() * s).collect())
        .collect();
    let flat: Vec<f32> = mat.iter().flatten().copied().collect();
    // per-tensor：整块一个 scale
    let s_pt = amax(&flat) / 127.0;
    let deq_pt = dequantize_symmetric(&quantize_symmetric(&flat, s_pt), s_pt);
    let mse_pt = mse(&flat, &deq_pt);
    // per-channel：每行一个 scale
    let mut mse_pc_num = 0.0f32;
    let mut per_ch_report = Vec::new();
    for (ci, ch) in mat.iter().enumerate() {
        // 该通道在 per-tensor 全局 scale 下的 MSE
        let d_pt = dequantize_symmetric(&quantize_symmetric(ch, s_pt), s_pt);
        let ch_mse_pt = mse(ch, &d_pt);
        // 该通道用自己的 scale
        let s = amax(ch) / 127.0;
        let d = dequantize_symmetric(&quantize_symmetric(ch, s), s);
        let ch_mse_pc = mse(ch, &d);
        mse_pc_num += ch.iter().zip(&d).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
        per_ch_report.push((ci, ch_mse_pt, ch_mse_pc));
    }
    let mse_pc = mse_pc_num / flat.len() as f32;
    println!("[3] per-tensor vs per-channel（3 通道尺度 0.1/1.0/10.0）");
    println!("    per-tensor : 一个 scale={:.6}  MSE={:.3e}", s_pt, mse_pt);
    println!("    per-channel: 每通道各自 scale MSE={:.3e}", mse_pc);
    for (ci, m_pt, m_pc) in &per_ch_report {
        println!("      通道{}(尺度{:>4}): per-tensor MSE={:.3e}  per-channel MSE={:.3e}  改善 {:.0}x", ci, scales_true[*ci], m_pt, m_pc, m_pt / m_pc);
    }
    println!("    小尺度通道在 per-tensor 下被大尺度通道压垮，per-channel 各用各的 scale 才救得回来\n");

    // ---- 4) 对称 vs 非对称：量化一段带偏移的激活（有正有负、偏向正侧，最小值不为 0）----
    let mut rng = Lcg::new(99);
    // 均值 1.5、标准差 1.2 的分布：范围大致 [-2.x, 5.x]，不以 0 对称
    let act: Vec<f32> = (0..4096).map(|_| rng.next_gauss() * 1.2 + 1.5).collect();
    let (mn, mx) = (
        act.iter().cloned().fold(f32::MAX, f32::min),
        act.iter().cloned().fold(f32::MIN, f32::max),
    );
    // 对称：scale=amax/127，覆盖 [-amax, amax]，但数据偏向正侧 -> 负半轴大量码位浪费
    let s_sym = act.iter().map(|v| v.abs()).fold(0.0, f32::max) / 127.0;
    let deq_sym = dequantize_symmetric(&quantize_symmetric(&act, s_sym), s_sym);
    // 非对称：scale=(mx-mn)/255，zero_point 把真实最小值对齐到整数 0
    let s_aff = (mx - mn) / 255.0;
    let z = (-mn / s_aff).round() as i32;
    let deq_aff = dequantize_affine(&quantize_affine(&act, s_aff, z), s_aff, z);
    println!("[4] 对称 vs 非对称（带偏移激活，范围 [{:.3}, {:.3}]，不以 0 对称）", mn, mx);
    println!("    对称 int8 : scale={:.6}  MSE={:.3e}（覆盖 [-amax,amax]，负半轴码位大量浪费）", s_sym, mse(&act, &deq_sym));
    println!("    非对称 u8 : scale={:.6}, zero_point={}  MSE={:.3e}", s_aff, z, mse(&act, &deq_aff));
    println!("    非对称误差是对称的 1/{:.1}（zero_point 让 0~255 全部码位精准贴合真实范围）", mse(&act, &deq_sym) / mse(&act, &deq_aff));
    println!("    极端情形：若像 ReLU 输出那样全为正，对称会白白浪费一半码位，非对称优势更大");
}
