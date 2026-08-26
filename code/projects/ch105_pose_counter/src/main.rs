// 第 105 章 项目：健身姿态计数器（关键点应用）
// candle 0.11.0，纯 CPU，可复现。运行：cd code/projects/ch105_pose_counter && cargo run --release
//
// 集成管线：合成"做 N 次深蹲"的侧面火柴人序列 → 热图姿态估计（复用第 53 章）
//   → 由 髋/膝/踝 三点算膝关节角 → 滞回双阈值状态机数重复次数
//   → 报计数准确率（数到的 vs 实际）、角度曲线 PNG、示例骨架图。
// 强调：① 用"关节角"而非绝对坐标（对人体尺度/位置不变）；② 滞回双阈值防临界抖动误计；
//       ③ 感知 vs 逻辑瓶颈定位——"理想关键点 vs 加噪关键点"随噪声升高的计数准确率曲线。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_circle_mut, draw_line_segment_mut};
use std::f32::consts::PI;

const IMG: usize = 48; // 纯 CPU candle 的可训练"甜点"尺寸
const J: usize = 5; // 侧面深蹲：头 肩 髋 膝 踝
const SIGMA: f32 = 2.0;
const BONES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 4)]; // 头-肩 肩-髋 髋-膝 膝-踝
                                                                     // 膝关节角阈值（度）：下阈值=蹲到位，上阈值=站直；两者之间是滞回死区
const TH_LO: f32 = 110.0;
const TH_HI: f32 = 150.0;
const TH_SINGLE: f32 = 130.0; // 单阈值计数器用的中间阈值（作对照）

// ---------- 确定性 LCG：离散随机务必走高位（API_NOTES §18） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // 取高位转 [0,1)
    }
    fn rangef(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.unit() as f64) * n as f64) as usize // [0,n)，从高位派生，避免 %k 低位退化
    }
    // 近似高斯噪声（Box-Muller），用于模拟"关键点估计误差"
    fn gauss(&mut self, sigma: f32) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

type Joints = [[f32; 2]; J];

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

// 由"深蹲深度 depth∈[0,1]"构造侧面火柴人 5 关节（0=站直，1=蹲到底）
fn squat_pose(depth: f32, base_x: f32, base_y: f32, scale: f32) -> Joints {
    let a_shin = 0.10 + 0.35 * depth; // 小腿前倾角随下蹲增大
    let g_thigh = 0.10 + 1.45 * depth; // 大腿从近竖直转到近水平
    let lean = 0.12 + 0.55 * depth; // 躯干前倾（下蹲时为平衡）
    let (ls, lt, torso, neck) = (9.0 * scale, 9.0 * scale, 9.0 * scale, 5.0 * scale);
    let ankle = [base_x, base_y];
    let knee = [ankle[0] + ls * a_shin.sin(), ankle[1] - ls * a_shin.cos()];
    let hip = [knee[0] - lt * g_thigh.sin(), knee[1] - lt * g_thigh.cos()]; // 髋随下蹲后移并降低
    let shoulder = [hip[0] + torso * lean.sin(), hip[1] - torso * lean.cos()];
    let head = [
        shoulder[0] + neck * lean.sin(),
        shoulder[1] - neck * lean.cos(),
    ];
    let mut js: Joints = [head, shoulder, hip, knee, ankle];
    for p in js.iter_mut() {
        p[0] = clampf(p[0], 2.0, IMG as f32 - 2.0);
        p[1] = clampf(p[1], 2.0, IMG as f32 - 2.0);
    }
    js
}

// 膝关节角：在膝(3)处，髋(2)与踝(4)两向量的夹角（度）
fn knee_angle(js: &Joints) -> f32 {
    let (h, k, a) = (js[2], js[3], js[4]);
    let v1 = [h[0] - k[0], h[1] - k[1]];
    let v2 = [a[0] - k[0], a[1] - k[1]];
    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let n1 = (v1[0].powi(2) + v1[1].powi(2)).sqrt().max(1e-6);
    let n2 = (v2[0].powi(2) + v2[1].powi(2)).sqrt().max(1e-6);
    (dot / (n1 * n2)).clamp(-1.0, 1.0).acos().to_degrees()
}

// 渲染一帧：画骨架（粗线+关节点）+ 像素噪声
fn render(js: &Joints, rng: &mut Lcg) -> RgbImage {
    let mut img = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    let bright = rng.rangef(200.0, 245.0) as u8;
    let col = Rgb([bright, bright, bright]);
    for &(a, b) in BONES.iter() {
        draw_thick_line(&mut img, js[a], js[b], 1, col);
    }
    for p in js.iter() {
        stamp_disk(&mut img, p[0], p[1], 1, col);
    }
    for _ in 0..12 {
        let x = rng.below(IMG) as u32;
        let y = rng.below(IMG) as u32;
        let v = (rng.unit() * 255.0) as u8;
        img.put_pixel(x, y, Rgb([v, v, v]));
    }
    img
}
fn stamp_disk(img: &mut RgbImage, cx: f32, cy: f32, r: i32, color: Rgb<u8>) {
    let (xi, yi) = (cx.round() as i32, cy.round() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r {
                continue;
            }
            let (x, y) = (xi + dx, yi + dy);
            if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}
fn draw_thick_line(img: &mut RgbImage, a: [f32; 2], b: [f32; 2], r: i32, color: Rgb<u8>) {
    let steps = ((a[0] - b[0]).abs().max((a[1] - b[1]).abs()) as i32 * 2).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        stamp_disk(
            img,
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            r,
            color,
        );
    }
}
fn img_to_chw(img: &RgbImage) -> Vec<f32> {
    let mut data = vec![0f32; 3 * IMG * IMG];
    for y in 0..IMG {
        for x in 0..IMG {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * IMG * IMG + y * IMG + x] = p[c] as f32 / 255.0;
            }
        }
    }
    data
}

// 生成一段深蹲序列的"真值关节"：N 次，每次深度平滑 0→peak→0
fn squat_sequence(n_reps: usize, rng: &mut Lcg) -> Vec<Joints> {
    let base_x = rng.rangef(22.0, 26.0);
    let base_y = rng.rangef(39.0, 42.0);
    let scale = rng.rangef(0.95, 1.1);
    let mut seq = Vec::new();
    // 起始站立几帧
    for _ in 0..6 {
        seq.push(squat_pose(0.0, base_x, base_y, scale));
    }
    for _ in 0..n_reps {
        let peak = rng.rangef(0.85, 1.0); // 每次下蹲深浅略不同
        let fpr = 22 + rng.below(9); // 每次 22~30 帧
        for f in 0..fpr {
            let p = f as f32 / (fpr - 1) as f32; // 0..1
            let depth = peak * (1.0 - (2.0 * PI * p).cos()) / 2.0; // 平滑 0→peak→0
            let depth = clampf(depth + rng.gauss(0.01), 0.0, 1.0); // 轻微抖动
            seq.push(squat_pose(depth, base_x, base_y, scale));
        }
    }
    for _ in 0..6 {
        seq.push(squat_pose(0.0, base_x, base_y, scale));
    }
    seq
}

// ---------- 计数器：滞回双阈值 vs 单阈值（对照） ----------
// 滞回：站立态角度跌破 lo → 进入"下蹲态"；下蹲态角度回升越过 hi → 计 1 次并回"站立态"
fn count_hysteresis(angles: &[f32], lo: f32, hi: f32) -> usize {
    let mut down = false; // false=站立态, true=下蹲态
    let mut n = 0;
    for &a in angles {
        if !down && a < lo {
            down = true;
        } else if down && a > hi {
            down = false;
            n += 1;
        }
    }
    n
}
// 单阈值：每次角度"下穿"阈值算一次（临界抖动会反复触发 → 多计）
fn count_single(angles: &[f32], thr: f32) -> usize {
    let mut below = false;
    let mut n = 0;
    for &a in angles {
        let now = a < thr;
        if now && !below {
            n += 1;
        }
        below = now;
    }
    n
}
// 时间平滑：滑动平均（窗口 w），压掉高频抖动（呼应事件层第 85 章时间窗思想）
fn smooth(a: &[f32], w: usize) -> Vec<f32> {
    let r = (w / 2) as i32;
    (0..a.len() as i32)
        .map(|i| {
            let (mut s, mut c) = (0f32, 0f32);
            for d in -r..=r {
                let j = i + d;
                if j >= 0 && (j as usize) < a.len() {
                    s += a[j as usize];
                    c += 1.0;
                }
            }
            s / c
        })
        .collect()
}

// ---------- 姿态估计器：编码器-解码器热图网络（复用第 53 章） ----------
struct PoseNet {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    dec1: Conv2d,
    dec2: Conv2d,
    head: Conv2d,
}
impl PoseNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let c0 = Conv2dConfig {
            padding: 0,
            ..Default::default()
        };
        Ok(Self {
            enc1: conv2d(3, 16, 3, c, vb.pp("enc1"))?,
            enc2: conv2d(16, 32, 3, c, vb.pp("enc2"))?,
            bott: conv2d(32, 32, 3, c, vb.pp("bott"))?,
            dec1: conv2d(32, 16, 3, c, vb.pp("dec1"))?,
            dec2: conv2d(16, 12, 3, c, vb.pp("dec2"))?,
            head: conv2d(12, J, 1, c0, vb.pp("head"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.enc1.forward(x)?.relu()?.max_pool2d(2)?;
        let x = self.enc2.forward(&x)?.relu()?.max_pool2d(2)?;
        let x = self.bott.forward(&x)?.relu()?;
        let x = x.upsample_nearest2d(IMG / 2, IMG / 2)?;
        let x = self.dec1.forward(&x)?.relu()?;
        let x = x.upsample_nearest2d(IMG, IMG)?;
        let x = self.dec2.forward(&x)?.relu()?;
        self.head.forward(&x)
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

fn build_heatmaps(all: &[Joints], dev: &Device) -> Result<Tensor> {
    let n = all.len();
    let mut hm = vec![0f32; n * J * IMG * IMG];
    let two_s2 = 2.0 * SIGMA * SIGMA;
    let rad = (3.0 * SIGMA).ceil() as i32;
    for (ni, js) in all.iter().enumerate() {
        for (k, kp) in js.iter().enumerate() {
            let (cxi, cyi) = (kp[0].round() as i32, kp[1].round() as i32);
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (x, y) = (cxi + dx, cyi + dy);
                    if x < 0 || x >= IMG as i32 || y < 0 || y >= IMG as i32 {
                        continue;
                    }
                    let r2 = (x as f32 - kp[0]).powi(2) + (y as f32 - kp[1]).powi(2);
                    let idx = ((ni * J + k) * IMG + y as usize) * IMG + x as usize;
                    hm[idx] = hm[idx].max((-r2 / two_s2).exp());
                }
            }
        }
    }
    Tensor::from_vec(hm, (n, J, IMG, IMG), dev)
}

fn decode_peak(hm: &[Vec<f32>]) -> (f32, f32) {
    let (mut br, mut bc, mut bv) = (0usize, 0usize, f32::MIN);
    for r in 0..IMG {
        for c in 0..IMG {
            if hm[r][c] > bv {
                bv = hm[r][c];
                br = r;
                bc = c;
            }
        }
    }
    let (mut px, mut py) = (bc as f32, br as f32);
    if bc >= 1 && bc + 1 < IMG {
        let (l, m, r) = (hm[br][bc - 1], hm[br][bc], hm[br][bc + 1]);
        let d = l + r - 2.0 * m;
        if d.abs() > 1e-6 {
            px += ((l - r) / (2.0 * d)).clamp(-0.5, 0.5);
        }
    }
    if br >= 1 && br + 1 < IMG {
        let (u, m, dn) = (hm[br - 1][bc], hm[br][bc], hm[br + 1][bc]);
        let d = u + dn - 2.0 * m;
        if d.abs() > 1e-6 {
            py += ((u - dn) / (2.0 * d)).clamp(-0.5, 0.5);
        }
    }
    (px, py)
}
fn predict_joints(model: &PoseNet, img: &RgbImage, dev: &Device) -> Result<Joints> {
    let x = Tensor::from_vec(img_to_chw(img), (1, 3, IMG, IMG), dev)?;
    let maps = model.forward(&x)?.i(0)?.to_vec3::<f32>()?;
    let mut js = [[0f32; 2]; J];
    for k in 0..J {
        let (x, y) = decode_peak(&maps[k]);
        js[k] = [x, y];
    }
    Ok(js)
}

// 画角度曲线 PNG：GT(绿) vs 预测(黄)，画上下阈值线
fn save_angle_plot(gt: &[f32], pred: &[f32], path: &str) {
    let (w, h) = (640u32, 260u32);
    let mut c = RgbImage::from_pixel(w, h, Rgb([22, 22, 26]));
    let (amin, amax) = (55.0f32, 175.0f32);
    let xof = |i: usize| 20.0 + i as f32 / (gt.len().max(2) - 1) as f32 * (w as f32 - 40.0);
    let yof = |a: f32| 10.0 + (amax - a) / (amax - amin) * (h as f32 - 20.0);
    // 阈值线
    let draw_h = |c: &mut RgbImage, a: f32, col: Rgb<u8>| {
        let y = yof(a);
        for x in 20..(w - 20) {
            c.put_pixel(x, y as u32, col);
        }
    };
    draw_h(&mut c, TH_LO, Rgb([70, 120, 240])); // 下阈值 蓝
    draw_h(&mut c, TH_HI, Rgb([240, 80, 80])); // 上阈值 红
    let plot = |c: &mut RgbImage, s: &[f32], col: Rgb<u8>| {
        for i in 1..s.len() {
            draw_line_segment_mut(c, (xof(i - 1), yof(s[i - 1])), (xof(i), yof(s[i])), col);
        }
    };
    plot(&mut c, gt, Rgb([40, 230, 90])); // GT 绿
    plot(&mut c, pred, Rgb([250, 220, 40])); // 预测 黄
    c.save(path).unwrap();
}
fn save_skeleton(img: &RgbImage, gt: &Joints, pred: &Joints, path: &str) {
    const S: u32 = 6;
    let mut c = image::imageops::resize(
        img,
        IMG as u32 * S,
        IMG as u32 * S,
        image::imageops::FilterType::Nearest,
    );
    let sf = S as f32;
    let (green, yellow) = (Rgb([40u8, 230, 90]), Rgb([250u8, 220, 40]));
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(
            &mut c,
            (gt[a][0] * sf, gt[a][1] * sf),
            (gt[b][0] * sf, gt[b][1] * sf),
            green,
        );
    }
    for p in gt.iter() {
        draw_hollow_circle_mut(&mut c, ((p[0] * sf) as i32, (p[1] * sf) as i32), 3, green);
    }
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(
            &mut c,
            (pred[a][0] * sf, pred[a][1] * sf),
            (pred[b][0] * sf, pred[b][1] * sf),
            yellow,
        );
    }
    for p in pred.iter() {
        draw_cross_mut(&mut c, yellow, (p[0] * sf) as i32, (p[1] * sf) as i32);
    }
    c.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：训练姿态估计器（复用第 53 章热图回归；随机深度/位置的深蹲帧）=====
    let mut trng = Lcg::new(7);
    let n_train = 260;
    let mut train_poses: Vec<Joints> = Vec::new();
    let mut train_imgs: Vec<RgbImage> = Vec::new();
    for _ in 0..n_train {
        let depth = trng.unit(); // 覆盖 0..1 全深度
        let js = squat_pose(
            depth,
            trng.rangef(22.0, 26.0),
            trng.rangef(39.0, 42.0),
            trng.rangef(0.95, 1.1),
        );
        train_imgs.push(render(&js, &mut trng));
        train_poses.push(js);
    }
    let mut xs = Vec::with_capacity(n_train * 3 * IMG * IMG);
    for img in &train_imgs {
        xs.extend_from_slice(&img_to_chw(img));
    }
    let x = Tensor::from_vec(xs, (n_train, 3, IMG, IMG), &dev)?;
    let target = build_heatmaps(&train_poses, &dev)?;
    println!("训练姿态估计器：{n_train} 张随机深蹲帧（48x48，{J} 关节），热图回归（复用第 53 章）");

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = PoseNet::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 3e-3,
            ..Default::default()
        },
    )?;
    for epoch in 1..=150 {
        if epoch == 100 {
            opt.set_learning_rate(5e-4);
        }
        let pred = net.forward(&x)?;
        let loss = pred.broadcast_sub(&target)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 30 == 0 || epoch == 1 {
            println!("  epoch {epoch:>3}  MSE {:.6}", loss.to_scalar::<f32>()?);
        }
    }
    vm.save("pose.safetensors")?;
    let vb2 =
        unsafe { VarBuilder::from_mmaped_safetensors(&["pose.safetensors"], DType::F32, &dev)? };
    let model = PoseNet::load(vb2)?;

    // ===== 第 2 步：演示序列（8 次深蹲）——图像 → 姿态 → 膝角 → 计数 =====
    let mut srng = Lcg::new(2025);
    let true_reps = 8;
    let seq = squat_sequence(true_reps, &mut srng);
    let (mut gt_ang, mut pred_ang) = (Vec::new(), Vec::new());
    let mut kp_err_sum = 0f32;
    let mut all_frames: Vec<(RgbImage, Joints, Joints)> = Vec::new();
    for js in seq.iter() {
        let img = render(js, &mut srng);
        let pj = predict_joints(&model, &img, &dev)?;
        for k in 0..J {
            kp_err_sum += ((pj[k][0] - js[k][0]).powi(2) + (pj[k][1] - js[k][1]).powi(2)).sqrt();
        }
        gt_ang.push(knee_angle(js));
        pred_ang.push(knee_angle(&pj));
        all_frames.push((img, *js, pj));
    }
    let kp_err = kp_err_sum / (seq.len() * J) as f32;
    let amin = gt_ang.iter().cloned().fold(f32::MAX, f32::min);
    let amax = gt_ang.iter().cloned().fold(f32::MIN, f32::max);
    println!(
        "\n===== 演示序列：{} 帧、真实 {} 次深蹲 =====",
        seq.len(),
        true_reps
    );
    println!("  姿态估计器关键点定位误差 = {kp_err:.3} 像素（这就是本项目的'感知质量'）");
    println!("  膝关节角范围：{amin:.0}° (蹲到底) ~ {amax:.0}° (站直)；阈值 下={TH_LO:.0}° 上={TH_HI:.0}°");

    let c_gt_h = count_hysteresis(&gt_ang, TH_LO, TH_HI);
    let c_pr_h = count_hysteresis(&pred_ang, TH_LO, TH_HI);
    let c_pr_s = count_single(&pred_ang, TH_SINGLE);
    println!("  计数结果（真实 {true_reps} 次）：");
    println!("    理想关键点 + 滞回双阈值 = {c_gt_h} 次  （逻辑上限：规则本身对不对）");
    println!("    预测关键点 + 滞回双阈值 = {c_pr_h} 次  （真实端到端）");
    println!("    预测关键点 + 单阈值     = {c_pr_s} 次  （对照：临界抖动会多计）");

    save_angle_plot(&gt_ang, &pred_ang, "angle_curve.png");
    // 按膝角挑三帧存示例：站直(角度最大)、半蹲(中间)、蹲到底(角度最小)
    let mut order: Vec<usize> = (0..gt_ang.len()).collect();
    order.sort_by(|&a, &b| gt_ang[a].partial_cmp(&gt_ang[b]).unwrap());
    let picks = [order[order.len() - 1], order[order.len() / 2], order[0]]; // 站直/半蹲/蹲到底
    for (i, &fi) in picks.iter().enumerate() {
        let (img, gt, pj) = &all_frames[fi];
        save_skeleton(img, gt, pj, &format!("pose_frame_{i}.png"));
    }
    println!("  已存 angle_curve.png（绿=理想角,黄=预测角,蓝/红=下/上阈值）与 pose_frame_0..2.png（站直/半蹲/蹲底）");

    // ===== 第 3 步：感知 vs 逻辑瓶颈——给理想关键点加不同强度高斯噪声，看计数准确率 =====
    // （用"加噪真值"作可控的估计器替身：噪声 σ 就是模拟的关键点误差，隔离出"感知要多准计数才靠谱"）
    println!("\n===== 感知瓶颈：关键点噪声 σ 对计数准确率的影响（40 段序列/档）=====");
    println!("  σ(px)   单阈值   滞回双阈值   滞回+平滑   （准确率=计数恰好正确的序列占比）");
    let n_seq = 40;
    for &sigma in &[0.0f32, 1.0, 2.0, 3.0, 5.0, 8.0] {
        let mut rng = Lcg::new(11);
        let mut nrng = Lcg::new(999);
        let (mut ok_s, mut ok_h, mut ok_hs) = (0usize, 0usize, 0usize);
        for _ in 0..n_seq {
            let n_reps = 5 + rng.below(5); // 5~9 次
            let seq = squat_sequence(n_reps, &mut rng);
            let angles: Vec<f32> = seq
                .iter()
                .map(|js| {
                    let mut noisy = *js;
                    for p in noisy.iter_mut() {
                        p[0] += nrng.gauss(sigma);
                        p[1] += nrng.gauss(sigma);
                    }
                    knee_angle(&noisy)
                })
                .collect();
            let sm = smooth(&angles, 5);
            if count_single(&angles, TH_SINGLE) == n_reps {
                ok_s += 1;
            }
            if count_hysteresis(&angles, TH_LO, TH_HI) == n_reps {
                ok_h += 1;
            }
            if count_hysteresis(&sm, TH_LO, TH_HI) == n_reps {
                ok_hs += 1;
            }
        }
        let pct = |k: usize| 100.0 * k as f32 / n_seq as f32;
        println!(
            "  {sigma:>4.0}    {:>5.1}%     {:>5.1}%      {:>5.1}%",
            pct(ok_s),
            pct(ok_h),
            pct(ok_hs)
        );
    }
    println!(
        "\n（对照上面的估计器误差 {kp_err:.2}px：只要感知误差落在计数器仍稳的区间，端到端就可靠；"
    );
    println!("  感知一旦太差，再好的逻辑也救不回。单阈值最脆、滞回更稳、滞回+平滑最强——");
    println!("  这就是'感知 vs 逻辑'的瓶颈定位：先看清是关键点不准，还是计数逻辑不够稳。）");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_counts_only_complete_repetitions() {
        let angles = [170.0, 140.0, 89.0, 100.0, 139.0, 151.0, 145.0, 85.0, 155.0];
        assert_eq!(count_hysteresis(&angles, 90.0, 150.0), 2);
        assert_eq!(count_hysteresis(&[170.0, 80.0, 100.0], 90.0, 150.0), 0);
    }

    #[test]
    fn smoothing_preserves_constant_signal_and_length() {
        let input = vec![42.0; 17];
        let output = smooth(&input, 5);
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|v| (*v - 42.0).abs() < 1e-6));
    }
}
