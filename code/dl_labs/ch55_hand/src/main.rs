// 第 55 章：合成手上的手部关键点检测（热图回归）+ 基于几何的手势识别
// candle 0.11.0，纯 CPU，可复现。运行：cd code/dl_labs/ch55_hand && cargo run --release
//
// 管线：合成手骨架（手腕 + 5 指各“指根/指尖”，随机姿态/伸屈）→ 造 J 张高斯热图
//       → 编码器-解码器网络热图回归（复用第 52/53 章）→ 解码 11 个关节
//       → 报定位误差 / PCK → 由关节几何判断每指伸/屈 → 规则识别手势 → 报准确率。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_circle_mut, draw_line_segment_mut};

const IMG: usize = 48;
const J: usize = 11; // 0 手腕；每指 base=1+2f, tip=2+2f（f=0..5：拇/食/中/无名/小）
const SIGMA: f32 = 2.0;
// 骨架：手腕→各指根（掌），各指根→指尖（指）
const BONES: [(usize, usize); 10] =
    [(0, 1), (0, 3), (0, 5), (0, 7), (0, 9), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10)];
// 5 种手势 = 5 指（拇/食/中/无名/小）的“伸直”模式
const GESTURES: [(&str, [bool; 5]); 5] = [
    ("握拳", [false, false, false, false, false]),
    ("张开", [true, true, true, true, true]),
    ("数字一", [false, true, false, false, false]),
    ("数字二", [false, true, true, false, false]),
    ("点赞", [true, false, false, false, false]),
];

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
    fn rangef(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
}

type Joints = [[f32; 2]; J];

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

// 手的规范布局（局部坐标，-y 朝上，手腕在下方）：每指的指根位置 + 伸直方向
const WRIST_L: [f32; 2] = [0.0, 10.0];
const BASE_L: [[f32; 2]; 5] = [[-8.0, 4.0], [-4.5, -3.5], [-0.5, -5.0], [3.5, -3.5], [7.0, -0.5]];
const DIR_L: [[f32; 2]; 5] =
    [[-0.80, -0.60], [-0.20, -0.98], [0.0, -1.0], [0.22, -0.97], [0.55, -0.84]];

// 合成一只手（给定手势）：随机相似位姿 + 每关节抖动 + 噪声。返回图 + 11 关节真值。
fn make_hand(rng: &mut Lcg, gesture: usize) -> (RgbImage, Joints) {
    let ext = GESTURES[gesture].1;
    let s = rng.rangef(0.95, 1.35);
    let theta = rng.rangef(-0.44, 0.44); // ±25°
    let cx = rng.rangef(22.0, 27.0);
    let cy = rng.rangef(24.0, 28.0);
    let (ct, st) = (theta.cos(), theta.sin());
    let jx = |r: &mut Lcg| r.rangef(-1.0, 1.0);
    let tf = |lx: f32, ly: f32| [cx + s * (ct * lx - st * ly), cy + s * (st * lx + ct * ly)];
    let mut js: Joints = [[0.0; 2]; J];
    js[0] = tf(WRIST_L[0], WRIST_L[1]); // 手腕
    let flen = 8.0f32; // 指长（局部）
    for f in 0..5 {
        let base = BASE_L[f];
        let tip_l = if ext[f] {
            // 伸直：沿外向伸长（指尖远离手腕）
            [base[0] + DIR_L[f][0] * flen, base[1] + DIR_L[f][1] * flen]
        } else {
            // 弯曲：指尖沿外向反方向折回半根手指（落在指根靠掌一侧，仍各自分开、不堆到手腕）
            [base[0] - DIR_L[f][0] * (0.5 * flen), base[1] - DIR_L[f][1] * (0.5 * flen)]
        };
        js[1 + 2 * f] = tf(base[0] + jx(rng), base[1] + jx(rng));
        js[2 + 2 * f] = tf(tip_l[0] + jx(rng), tip_l[1] + jx(rng));
    }
    for p in js.iter_mut() {
        p[0] = clampf(p[0], 2.0, IMG as f32 - 2.0);
        p[1] = clampf(p[1], 2.0, IMG as f32 - 2.0);
    }
    // 画手骨架：粗线 + 关节点
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
        let x = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let y = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let v = (rng.unit() * 255.0) as u8;
        img.put_pixel(x, y, Rgb([v, v, v]));
    }
    (img, js)
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
        stamp_disk(img, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, r, color);
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

struct Sample {
    chw: Vec<f32>,
    joints: Joints,
    gesture: usize,
}

// 造平衡数据集：每种手势 per 张
fn make_set(seed: u64, per: usize) -> Vec<Sample> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for _ in 0..per {
        for g in 0..GESTURES.len() {
            let (img, joints) = make_hand(&mut rng, g);
            out.push(Sample { chw: img_to_chw(&img), joints, gesture: g });
        }
    }
    out
}

fn stack(samples: &[Sample], dev: &Device) -> Result<(Tensor, Tensor)> {
    let n = samples.len();
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    for s in samples {
        xs.extend_from_slice(&s.chw);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    let mut hm = vec![0f32; n * J * IMG * IMG];
    let two_s2 = 2.0 * SIGMA * SIGMA;
    let rad = (3.0 * SIGMA).ceil() as i32;
    for (ni, s) in samples.iter().enumerate() {
        for (k, kp) in s.joints.iter().enumerate() {
            let (kx, ky) = (kp[0], kp[1]);
            let (cxi, cyi) = (kx.round() as i32, ky.round() as i32);
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (x, y) = (cxi + dx, cyi + dy);
                    if x < 0 || x >= IMG as i32 || y < 0 || y >= IMG as i32 {
                        continue;
                    }
                    let r2 = (x as f32 - kx).powi(2) + (y as f32 - ky).powi(2);
                    let idx = ((ni * J + k) * IMG + y as usize) * IMG + x as usize;
                    hm[idx] = hm[idx].max((-r2 / two_s2).exp());
                }
            }
        }
    }
    Ok((x, Tensor::from_vec(hm, (n, J, IMG, IMG), dev)?))
}

// ---------- 热图网络（与第 53 章同结构，头输出 J=11 张热图） ----------
struct HandNet {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    dec1: Conv2d,
    dec2: Conv2d,
    head: Conv2d,
}
impl HandNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        let c0 = Conv2dConfig { padding: 0, ..Default::default() };
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

fn predict_joints(model: &HandNet, chw: &[f32], dev: &Device) -> Result<Joints> {
    let x = Tensor::from_vec(chw.to_vec(), (1, 3, IMG, IMG), dev)?;
    let maps = model.forward(&x)?.i(0)?.to_vec3::<f32>()?;
    let mut js = [[0f32; 2]; J];
    for k in 0..J {
        let (x, y) = decode_peak(&maps[k]);
        js[k] = [x, y];
    }
    Ok(js)
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

// 几何规则：某指“伸直” ⟺ 指尖到手腕的距离 > 指根到手腕距离 × τ（弯曲时指尖折回，更近）
fn extended_pattern(js: &Joints) -> [bool; 5] {
    let wrist = js[0];
    let mut ext = [false; 5];
    for f in 0..5 {
        let base = js[1 + 2 * f];
        let tip = js[2 + 2 * f];
        ext[f] = dist(tip, wrist) > 1.15 * dist(base, wrist);
    }
    ext
}

// 由伸直模式匹配手势（精确匹配 5 种之一，否则 -1=未知）
fn classify(js: &Joints) -> i32 {
    let ext = extended_pattern(js);
    for (gi, (_, pat)) in GESTURES.iter().enumerate() {
        if *pat == ext {
            return gi as i32;
        }
    }
    -1
}

fn save_skeleton(chw: &[f32], gt: &Joints, pred: &Joints, gname: &str, path: &str) {
    const S: u32 = 6;
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    for y in 0..IMG {
        for x in 0..IMG {
            let v = (chw[y * IMG + x] * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
        }
    }
    let mut c = image::imageops::resize(&img, IMG as u32 * S, IMG as u32 * S, image::imageops::FilterType::Nearest);
    let sf = S as f32;
    let (green, yellow) = (Rgb([40u8, 230, 90]), Rgb([250u8, 220, 40]));
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(&mut c, (gt[a][0] * sf, gt[a][1] * sf), (gt[b][0] * sf, gt[b][1] * sf), green);
    }
    for p in gt.iter() {
        draw_hollow_circle_mut(&mut c, ((p[0] * sf) as i32, (p[1] * sf) as i32), 3, green);
    }
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(&mut c, (pred[a][0] * sf, pred[a][1] * sf), (pred[b][0] * sf, pred[b][1] * sf), yellow);
    }
    for p in pred.iter() {
        draw_cross_mut(&mut c, yellow, (p[0] * sf) as i32, (p[1] * sf) as i32);
    }
    let _ = gname;
    c.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：合成平衡数据（5 手势 × per 张）=====
    let train = make_set(7, 60); // 60×5 = 300 张
    let test = make_set(2025, 20); // 20×5 = 100 张
    println!("合成手数据：train {} 张，test {} 张（{} 种手势各均衡），每张 {J} 个关节", train.len(), test.len(), GESTURES.len());

    // ===== 第 2 步：热图回归训练 =====
    let (x, target) = stack(&train, &dev)?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = HandNet::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
    println!("\n开始训练手部热图回归网络（AdamW lr=3e-3→5e-4，MSE）...");
    println!("  {:>5}  {:>10}", "epoch", "MSE");
    let epochs = 250;
    for epoch in 1..=epochs {
        if epoch == 170 {
            opt.set_learning_rate(5e-4);
        }
        let pred = net.forward(&x)?;
        let loss = pred.broadcast_sub(&target)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {epoch:>5}  {:>10.6}", loss.to_scalar::<f32>()?);
        }
    }
    vm.save("hand.safetensors")?;
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["hand.safetensors"], DType::F32, &dev)? };
    let model = HandNet::load(vb2)?;

    // ===== 第 3 步：关键点定位误差 / PCK（参考=掌长=手腕到中指根）=====
    let (mut sum_e, mut n_e, mut ok2) = (0f32, 0f32, 0f32);
    let mut tip_e = 0f32; // 指尖平均误差
    let mut base_e = 0f32; // 指根平均误差
    for s in &test {
        let pred = predict_joints(&model, &s.chw, &dev)?;
        let palm = dist(s.joints[0], s.joints[5]).max(1.0);
        for k in 0..J {
            let e = dist(pred[k], s.joints[k]);
            sum_e += e;
            n_e += 1.0;
            if e <= 0.2 * palm {
                ok2 += 1.0;
            }
            if k >= 1 && k % 2 == 0 {
                tip_e += e; // 偶数(2,4,6,8,10)=指尖
            } else if k >= 1 {
                base_e += e; // 奇数(1,3,5,7,9)=指根
            }
        }
    }
    let ntest = test.len() as f32;
    println!("\n===== 关键点定位（{} 张测试手，共 {} 个关节）=====", test.len(), test.len() * J);
    println!("  平均定位误差 = {:.3} 像素，PCK@0.2 = {:.1}%", sum_e / n_e, 100.0 * ok2 / n_e);
    println!("  指根平均误差 = {:.3} 像素，指尖平均误差 = {:.3} 像素（指尖更难）", base_e / (5.0 * ntest), tip_e / (5.0 * ntest));

    // ===== 第 4 步：手势识别（关键点→几何→语义），比较“用真值点”与“用预测点” =====
    let (mut acc_gt, mut acc_pred) = (0usize, 0usize);
    let mut confuse = vec![vec![0usize; GESTURES.len() + 1]; GESTURES.len()]; // 最后一列=未知
    for s in &test {
        if classify(&s.joints) == s.gesture as i32 {
            acc_gt += 1;
        }
        let pred = predict_joints(&model, &s.chw, &dev)?;
        let pg = classify(&pred);
        if pg == s.gesture as i32 {
            acc_pred += 1;
        }
        let col = if pg < 0 { GESTURES.len() } else { pg as usize };
        confuse[s.gesture][col] += 1;
    }
    let n = test.len() as f32;
    println!("\n===== 手势识别（关键点→几何规则→手势）=====");
    println!("  用“真值关键点”识别准确率 = {:.1}%（几何规则上限）", 100.0 * acc_gt as f32 / n);
    println!("  用“预测关键点”识别准确率 = {:.1}%（端到端）", 100.0 * acc_pred as f32 / n);
    print!("  分手势（预测点）：");
    for (gi, (name, _)) in GESTURES.iter().enumerate() {
        let correct = confuse[gi][gi];
        let tot: usize = confuse[gi].iter().sum();
        print!("{}={}/{}  ", name, correct, tot);
    }
    println!();

    // ===== 第 5 步：存骨架 PNG（每种手势一张）=====
    for g in 0..GESTURES.len() {
        if let Some(s) = test.iter().find(|s| s.gesture == g) {
            let pred = predict_joints(&model, &s.chw, &dev)?;
            save_skeleton(&s.chw, &s.joints, &pred, GESTURES[g].0, &format!("hand_{g}.png"));
        }
    }
    println!("已存 hand_0..4.png（绿=真值骨架，黄=预测骨架；对应 5 种手势）");
    Ok(())
}
