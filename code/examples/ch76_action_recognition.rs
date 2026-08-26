//! 第 76 章 动作识别与时序模型（纯 Rust 传统法 demo）
//! 运行：cargo run --example ch76_action_recognition
//! 依赖：image（存 MHI 运动历史图）+ std。
//!
//! 思路：单帧看不出「在做什么」，动作藏在帧与帧之间的变化里。
//! 我们合成三类动作的帧序列——静止 / 挥手 / 走过，用最朴素的**帧差运动能量**
//! 和**运动质心漂移**提「时序特征」，再用 1-NN（最近邻）做一个极简动作分类器，
//! 打印真实的逐帧能量曲线、混淆矩阵与准确率。全程确定性可复现（LCG 造噪声，不读系统时钟）。

use image::{GrayImage, Luma};

const W: u32 = 96;
const H: u32 = 64;
const N_FRAMES: usize = 16;
const SQ: i32 = 14; // 运动物体（方块）边长
const BG: u8 = 40; // 背景灰度
const FG: u8 = 210; // 前景（运动物体）灰度
const NOISE: f32 = 3.0; // 背景噪声幅度（远小于帧差阈值 → 静止时能量≈0）
const DIFF_TAU: i32 = 30; // 帧差阈值：噪声在此之下，运动边缘远在此之上

// ============ 确定性伪随机（LCG）：造可复现的噪声与个体差异 ============
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// 返回 [-1, 1) 的伪随机数
    fn signed(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

// ============ 三类动作 ============
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    Static, // 静止：物体不动，只有传感器噪声
    Wave,   // 挥手：物体原地左右往复摆动
    Walk,   // 走过：物体匀速横穿画面
}
impl Action {
    fn name(&self) -> &'static str {
        match self {
            Action::Static => "静止",
            Action::Wave => "挥手",
            Action::Walk => "走过",
        }
    }
    fn all() -> [Action; 3] {
        [Action::Static, Action::Wave, Action::Walk]
    }
    fn idx(&self) -> usize {
        match self {
            Action::Static => 0,
            Action::Wave => 1,
            Action::Walk => 2,
        }
    }
}

// ============ 合成帧序列 ============
// seed 变化 → 起点/速度/幅度/相位/噪声都不同，模拟「个体差异」，避免分类器只会背答案。
fn synth_clip(action: Action, seed: u64) -> Vec<GrayImage> {
    let mut rng = Lcg::new(seed);
    let cx0 = 48.0 + 6.0 * rng.signed(); // 中心 x 基准
    let cy = H as f32 * 0.5 + 6.0 * rng.signed(); // 垂直位置
    let amp = 17.0 + 5.0 * rng.signed().abs(); // 挥手幅度
    let speed = 3.4 + 0.8 * rng.signed(); // 走过速度（px/帧）
    let phase = rng.signed() * std::f32::consts::PI; // 挥手初相
    let freq = 0.22; // 挥手频率（周期/帧）

    let mut frames = Vec::with_capacity(N_FRAMES);
    for t in 0..N_FRAMES {
        let tf = t as f32;
        let cx = match action {
            Action::Static => cx0,
            Action::Wave => cx0 + amp * (2.0 * std::f32::consts::PI * freq * tf + phase).sin(),
            Action::Walk => 12.0 + speed * tf,
        };
        frames.push(render_frame(cx, cy, &mut rng));
    }
    frames
}

/// 画一帧：带噪声的背景 + 一个干净的亮方块。噪声先铺，方块后画（方块值恒定，静止时帧差≈0）。
fn render_frame(cx: f32, cy: f32, rng: &mut Lcg) -> GrayImage {
    let mut img = GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let n = (rng.signed() * NOISE) as i32;
            img.put_pixel(x, y, Luma([(BG as i32 + n).clamp(0, 255) as u8]));
        }
    }
    let x0 = (cx - SQ as f32 / 2.0).round() as i32;
    let y0 = (cy - SQ as f32 / 2.0).round() as i32;
    for yy in y0..y0 + SQ {
        for xx in x0..x0 + SQ {
            if xx >= 0 && yy >= 0 && (xx as u32) < W && (yy as u32) < H {
                img.put_pixel(xx as u32, yy as u32, Luma([FG]));
            }
        }
    }
    img
}

// ============ 时序特征提取 ============
struct ClipFeat {
    energy_curve: Vec<f32>, // 每帧运动像素占比（这就是「运动能量时间曲线」）
    mean_energy: f32,       // 平均运动能量：区分「静止 vs 有动作」
    straightness: f32,      // |净位移| / 路径总长 ∈ [0,1]：区分「走过 vs 挥手」
    reversals: u32,         // 运动质心水平速度符号翻转次数：挥手多、走过≈0
}

/// 帧差 → 运动像素占比（能量）+ 运动质心；再由质心序列算方向性特征。
fn extract(frames: &[GrayImage]) -> ClipFeat {
    let total = (W * H) as f32;
    let mut energy = Vec::with_capacity(frames.len() - 1);
    let mut centroids: Vec<(f32, f32)> = Vec::new();
    for t in 1..frames.len() {
        let (a, b) = (&frames[t - 1], &frames[t]);
        let (mut n, mut sx, mut sy) = (0u32, 0f32, 0f32);
        for y in 0..H {
            for x in 0..W {
                let d = (a.get_pixel(x, y).0[0] as i32 - b.get_pixel(x, y).0[0] as i32).abs();
                if d > DIFF_TAU {
                    n += 1;
                    sx += x as f32;
                    sy += y as f32;
                }
            }
        }
        energy.push(n as f32 / total);
        if n > 4 {
            centroids.push((sx / n as f32, sy / n as f32)); // 运动质心（帧差像素的重心）
        }
    }
    let mean_energy = energy.iter().sum::<f32>() / energy.len() as f32;

    // 由运动质心序列算：路径总长、净位移、方向翻转次数
    let (mut path, mut reversals) = (0f32, 0u32);
    let mut last_dir = 0f32;
    for w in centroids.windows(2) {
        let dx = w[1].0 - w[0].0;
        let dy = w[1].1 - w[0].1;
        path += (dx * dx + dy * dy).sqrt();
        if dx.abs() > 0.6 {
            let dir = dx.signum();
            if last_dir != 0.0 && dir != last_dir {
                reversals += 1;
            }
            last_dir = dir;
        }
    }
    let net = if centroids.len() >= 2 {
        let a = centroids[0];
        let b = *centroids.last().unwrap();
        ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
    } else {
        0.0
    };
    let straightness = if path > 1e-3 { (net / path).min(1.0) } else { 0.0 };

    ClipFeat {
        energy_curve: energy,
        mean_energy,
        straightness,
        reversals,
    }
}

// ============ 运动历史图 MHI：把一段时间的运动「拍扁」成一张图 ============
/// 有运动的像素点亮为最大值，其余每帧衰减 → 越近的运动越亮，形成「运动拖影」。
fn mhi(frames: &[GrayImage]) -> GrayImage {
    let mut hist = vec![0f32; (W * H) as usize];
    let decay = 255.0 / frames.len() as f32;
    for t in 1..frames.len() {
        let (a, b) = (&frames[t - 1], &frames[t]);
        for y in 0..H {
            for x in 0..W {
                let idx = (y * W + x) as usize;
                let d = (a.get_pixel(x, y).0[0] as i32 - b.get_pixel(x, y).0[0] as i32).abs();
                if d > DIFF_TAU {
                    hist[idx] = 255.0;
                } else {
                    hist[idx] = (hist[idx] - decay).max(0.0);
                }
            }
        }
    }
    GrayImage::from_fn(W, H, |x, y| Luma([hist[(y * W + x) as usize] as u8]))
}

// ============ 标准化 + 1-NN 分类 ============
type Feat = [f32; 2]; // [mean_energy, straightness]

/// 用训练集算每维的均值/标准差（供 z-score 标准化：两维量纲差异大，不标准化 NN 会被能量主导）。
fn norm_params(train: &[(Feat, Action)]) -> (Feat, Feat) {
    let n = train.len() as f32;
    let mut mean = [0f32; 2];
    for (f, _) in train {
        for k in 0..2 {
            mean[k] += f[k];
        }
    }
    for k in 0..2 {
        mean[k] /= n;
    }
    let mut var = [0f32; 2];
    for (f, _) in train {
        for k in 0..2 {
            var[k] += (f[k] - mean[k]).powi(2);
        }
    }
    let mut std = [0f32; 2];
    for k in 0..2 {
        std[k] = (var[k] / n).sqrt().max(1e-6);
    }
    (mean, std)
}

fn standardize(f: &Feat, mean: &Feat, std: &Feat) -> Feat {
    [(f[0] - mean[0]) / std[0], (f[1] - mean[1]) / std[1]]
}

/// 1-NN：在标准化后的特征空间里找最近的训练样本，用它的标签作预测。
fn classify_1nn(x: &Feat, train: &[(Feat, Action)], mean: &Feat, std: &Feat) -> Action {
    let xz = standardize(x, mean, std);
    let mut best = f32::INFINITY;
    let mut label = Action::Static;
    for (f, a) in train {
        let fz = standardize(f, mean, std);
        let d = (xz[0] - fz[0]).powi(2) + (xz[1] - fz[1]).powi(2);
        if d < best {
            best = d;
            label = *a;
        }
    }
    label
}

fn main() {
    let classes = Action::all();

    // ---- 训练集：每类 4 段（不同 seed）----
    let mut train: Vec<(Feat, Action)> = Vec::new();
    for &c in &classes {
        for k in 0..4u64 {
            let clip = synth_clip(c, 1000 + c.idx() as u64 * 100 + k);
            let f = extract(&clip);
            train.push(([f.mean_energy, f.straightness], c));
        }
    }
    let (mean, std) = norm_params(&train);

    // ---- 打印每类一段代表性 clip 的「运动能量时间曲线」+ 特征 ----
    println!("== 三类动作的运动能量时间曲线（每帧运动像素占比 ×1000）==");
    println!("帧号:        {}", (1..N_FRAMES).map(|t| format!("{t:>3}")).collect::<Vec<_>>().join(""));
    for &c in &classes {
        let clip = synth_clip(c, 7000 + c.idx() as u64); // 演示用固定 seed
        let f = extract(&clip);
        let bars: String = f
            .energy_curve
            .iter()
            .map(|e| format!("{:>3}", (e * 1000.0).round() as i32))
            .collect();
        println!(
            "{:<6}曲线: {}   | 均能={:.4} 直线度={:.2} 翻转={}",
            c.name(),
            bars,
            f.mean_energy,
            f.straightness,
            f.reversals
        );
        // 存该类的 MHI 运动历史图，供直观对照
        let path = format!("out_ch76_mhi_{}.png", match c {
            Action::Static => "static",
            Action::Wave => "wave",
            Action::Walk => "walk",
        });
        let _ = mhi(&clip).save(&path);
    }
    println!("（静止≈全程贴地；挥手上下起伏且原地打转；走过持续有能量且质心单向漂移）");
    println!("已存 3 张运动历史图 MHI：out_ch76_mhi_static/wave/walk.png\n");

    // ---- 测试集：每类 5 段（全新 seed），1-NN 分类，统计准确率 + 混淆矩阵 ----
    let mut confusion = [[0u32; 3]; 3]; // confusion[真][预测]
    let (mut correct, mut total) = (0u32, 0u32);
    println!("== 测试集逐段分类（1-NN，特征 = [均运动能量, 直线度]）==");
    for &c in &classes {
        for k in 0..5u64 {
            let clip = synth_clip(c, 42000 + c.idx() as u64 * 100 + k);
            let f = extract(&clip);
            let x = [f.mean_energy, f.straightness];
            let pred = classify_1nn(&x, &train, &mean, &std);
            confusion[c.idx()][pred.idx()] += 1;
            total += 1;
            let ok = pred == c;
            if ok {
                correct += 1;
            }
            println!(
                "真值 {:<4} 特征=[{:.4}, {:.2}] -> 预测 {:<4} {}",
                c.name(),
                x[0],
                x[1],
                pred.name(),
                if ok { "✓" } else { "✗ 错" }
            );
        }
    }

    println!("\n== 混淆矩阵（行=真值，列=预测）==");
    print!("          ");
    for &c in &classes {
        print!("{:>6}", c.name());
    }
    println!();
    for &c in &classes {
        print!("真值 {:<4}", c.name());
        for &p in &classes {
            print!("{:>6}", confusion[c.idx()][p.idx()]);
        }
        println!();
    }
    println!(
        "\n准确率 = {}/{} = {:.1}%",
        correct,
        total,
        100.0 * correct as f32 / total as f32
    );
    println!("提示：合成数据三类分得很开，准确率虚高；真实视频里动作类间差异小、类内差异大，远没这么容易。");
}
