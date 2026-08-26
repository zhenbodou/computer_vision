// 第 53 章：合成“火柴人”上的人体姿态估计（热图回归，candle 0.11.0，纯 CPU，可复现）
// 运行：cd code/dl_labs/ch53_pose && cargo run --release
//
// 管线：合成“火柴人”（J 个关节连成骨架，随机姿态）→ 造 J 张高斯目标热图
//       → 编码器-解码器网络输出 J 张热图 → MSE 训练 → 解码关节坐标（argmax+亚像素）
//       → 把关节按骨架连起来画出预测姿态 → 报 PCK 与平均关节误差。
// 另做一个“划分泄漏”对照实验（呼应第 47 章）：同一批“场景+近似重复帧”，
//       随机划分 vs 按场景分组划分，看关键点检测里泄漏怎么把 PCK 虚高。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_hollow_circle_mut, draw_line_segment_mut};

const IMG: usize = 48; // 输入 / 热图边长
const J: usize = 7; // 关节数
const SIGMA: f32 = 2.0; // 高斯目标标准差
// 关节：0 头 1 颈 2 左手 3 右手 4 髋 5 左脚 6 右脚
const JOINT_NAMES: [&str; J] = ["头", "颈", "左手", "右手", "髋", "左脚", "右脚"];
// 骨架：把关节连成肢体（头-颈、左右臂、躯干、左右腿）
const BONES: [(usize, usize); 6] = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)];

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed） ----------
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

// 一个“场景”= 一副固定姿态（关节的规范布局）。左肢体偏左、右肢体偏右，保证关节身份可学，
// 但四肢角度随机 → 姿态多样（抬手/垂手、叉腿/并腿）。
fn make_scene(rng: &mut Lcg) -> Joints {
    let scale = rng.rangef(0.85, 1.15);
    let bx = rng.rangef(21.0, 27.0); // 髋部中心 x
    let by = rng.rangef(26.0, 30.0); // 髋部 y
    let torso = 12.0 * scale;
    let arm = 9.0 * scale;
    let leg = 10.0 * scale;
    let pelvis = [bx, by];
    let neck = [bx + rng.rangef(-1.5, 1.5), by - torso];
    let head = [neck[0] + rng.rangef(-1.5, 1.5), neck[1] - 6.0 * scale];
    // 左臂偏左、右臂偏右；上下方向随机
    let lh = [neck[0] + rng.rangef(-arm, -0.3 * arm), neck[1] + rng.rangef(-0.7 * arm, 0.7 * arm)];
    let rh = [neck[0] + rng.rangef(0.3 * arm, arm), neck[1] + rng.rangef(-0.7 * arm, 0.7 * arm)];
    // 左腿偏左下、右腿偏右下
    let lf = [pelvis[0] + rng.rangef(-0.7 * leg, -0.1 * leg), pelvis[1] + rng.rangef(0.5 * leg, leg)];
    let rf = [pelvis[0] + rng.rangef(0.1 * leg, 0.7 * leg), pelvis[1] + rng.rangef(0.5 * leg, leg)];
    let mut js: Joints = [head, neck, lh, rh, pelvis, lf, rf];
    for p in js.iter_mut() {
        p[0] = clampf(p[0], 3.0, IMG as f32 - 3.0);
        p[1] = clampf(p[1], 3.0, IMG as f32 - 3.0);
    }
    js
}

// 沿线段每隔一步盖一个实心小圆，画出有粗细的“肢体”
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

// 把一个场景渲染成一帧：整体小平移 + 每关节小抖动 + 随机亮度 + 像素噪声（近似重复但非纯平移）
fn render_frame(scene: &Joints, rng: &mut Lcg) -> (RgbImage, Joints) {
    let (tx, ty) = (rng.rangef(-3.0, 3.0), rng.rangef(-3.0, 3.0));
    let mut js: Joints = *scene;
    for p in js.iter_mut() {
        p[0] = clampf(p[0] + tx + rng.rangef(-1.5, 1.5), 2.0, IMG as f32 - 2.0);
        p[1] = clampf(p[1] + ty + rng.rangef(-1.5, 1.5), 2.0, IMG as f32 - 2.0);
    }
    let mut img = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    let bright = rng.rangef(200.0, 245.0) as u8;
    let limb = Rgb([bright, bright, bright]);
    for &(a, b) in BONES.iter() {
        draw_thick_line(&mut img, js[a], js[b], 1, limb); // 肢体半径 1（直径约 3px）
    }
    for p in js.iter() {
        stamp_disk(&mut img, p[0], p[1], 1, limb); // 关节处点粗一点
    }
    // 像素噪声：撒一些随机灰点
    for _ in 0..14 {
        let x = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let y = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let v = (rng.unit() * 255.0) as u8;
        img.put_pixel(x, y, Rgb([v, v, v]));
    }
    (img, js)
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

// 一帧样本：像素向量 + 关节真值 + 它来自哪个场景（用于分组划分）
struct Sample {
    chw: Vec<f32>,
    joints: Joints,
    scene: usize,
}

fn stack(samples: &[&Sample], dev: &Device) -> Result<(Tensor, Tensor)> {
    let n = samples.len();
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    for s in samples {
        xs.extend_from_slice(&s.chw);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    // 目标热图 [N,J,IMG,IMG]
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

// ---------- 编码器-解码器热图网络（与第 52 章同结构，头输出 J 张热图） ----------
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
        let x = self.enc1.forward(x)?.relu()?.max_pool2d(2)?; // 48 -> 24
        let x = self.enc2.forward(&x)?.relu()?.max_pool2d(2)?; // 24 -> 12
        let x = self.bott.forward(&x)?.relu()?;
        let x = x.upsample_nearest2d(IMG / 2, IMG / 2)?; // 12 -> 24
        let x = self.dec1.forward(&x)?.relu()?;
        let x = x.upsample_nearest2d(IMG, IMG)?; // 24 -> 48
        let x = self.dec2.forward(&x)?.relu()?;
        self.head.forward(&x) // [N,J,48,48]
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

// 从一张热图解码峰值（argmax + 抛物线亚像素）
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

// 预测一帧的 J 个关节坐标
fn predict_joints(model: &PoseNet, chw: &[f32], dev: &Device) -> Result<Joints> {
    let x = Tensor::from_vec(chw.to_vec(), (1, 3, IMG, IMG), dev)?;
    let out = model.forward(&x)?;
    let maps = out.i(0)?.to_vec3::<f32>()?; // [J][IMG][IMG]
    let mut js = [[0f32; 2]; J];
    for k in 0..J {
        let (x, y) = decode_peak(&maps[k]);
        js[k] = [x, y];
    }
    Ok(js)
}

// 评估：返回 (平均关节误差像素, PCK@0.2, PCK@0.5)。参考长度=躯干长(颈-髋)。
fn eval(model: &PoseNet, val: &[&Sample], dev: &Device) -> Result<(f32, f32, f32)> {
    let (mut sum_e, mut n_e) = (0f32, 0f32);
    let (mut ok2, mut ok5) = (0f32, 0f32);
    for s in val {
        let pred = predict_joints(model, &s.chw, dev)?;
        let gt = &s.joints;
        let torso = ((gt[1][0] - gt[4][0]).powi(2) + (gt[1][1] - gt[4][1]).powi(2)).sqrt().max(1.0);
        for k in 0..J {
            let e = ((pred[k][0] - gt[k][0]).powi(2) + (pred[k][1] - gt[k][1]).powi(2)).sqrt();
            sum_e += e;
            n_e += 1.0;
            if e <= 0.2 * torso {
                ok2 += 1.0;
            }
            if e <= 0.5 * torso {
                ok5 += 1.0;
            }
        }
    }
    Ok((sum_e / n_e, 100.0 * ok2 / n_e, 100.0 * ok5 / n_e))
}

// 训练一个姿态网络（给定训练样本），返回可推理模型（经 safetensors 存取，模拟真实推理入口）
fn train(train: &[&Sample], epochs: usize, tag: &str, dev: &Device, verbose: bool) -> Result<PoseNet> {
    let (x, target) = stack(train, dev)?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = PoseNet::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
    if verbose {
        println!("  {:>5}  {:>10}", "epoch", "MSE");
    }
    for epoch in 1..=epochs {
        if epoch == (epochs * 2) / 3 {
            opt.set_learning_rate(5e-4);
        }
        let pred = net.forward(&x)?;
        let loss = pred.broadcast_sub(&target)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if verbose && (epoch % 20 == 0 || epoch == 1) {
            println!("  {epoch:>5}  {:>10.6}", loss.to_scalar::<f32>()?);
        }
    }
    let path = format!("pose_{tag}.safetensors");
    vm.save(&path)?;
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, dev)? };
    PoseNet::load(vb2)
}

// 画“预测骨架 vs 真值骨架”PNG：原图放大，绿=真值骨架，黄=预测骨架
fn save_skeleton(chw: &[f32], gt: &Joints, pred: &Joints, path: &str) {
    const S: u32 = 6;
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    for y in 0..IMG {
        for x in 0..IMG {
            let v = (chw[y * IMG + x] * 255.0) as u8; // 取 R 通道（灰度）
            img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
        }
    }
    let mut canvas = image::imageops::resize(&img, IMG as u32 * S, IMG as u32 * S, image::imageops::FilterType::Nearest);
    let green = Rgb([40u8, 230, 90]);
    let yellow = Rgb([250u8, 220, 40]);
    let sf = S as f32;
    // 真值骨架（绿）
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(&mut canvas, (gt[a][0] * sf, gt[a][1] * sf), (gt[b][0] * sf, gt[b][1] * sf), green);
    }
    for p in gt.iter() {
        draw_hollow_circle_mut(&mut canvas, ((p[0] * sf) as i32, (p[1] * sf) as i32), 4, green);
    }
    // 预测骨架（黄）
    for &(a, b) in BONES.iter() {
        draw_line_segment_mut(&mut canvas, (pred[a][0] * sf, pred[a][1] * sf), (pred[b][0] * sf, pred[b][1] * sf), yellow);
    }
    for p in pred.iter() {
        draw_cross_mut(&mut canvas, yellow, (p[0] * sf) as i32, (p[1] * sf) as i32);
    }
    canvas.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：造“场景 + 近似重复帧”数据（呼应第 47 章的划分泄漏设定）=====
    let n_scenes = 16;
    let frames_per_scene = 12;
    let mut scene_rng = Lcg::new(7);
    let scenes: Vec<Joints> = (0..n_scenes).map(|_| make_scene(&mut scene_rng)).collect();
    let mut frame_rng = Lcg::new(123);
    let mut all: Vec<Sample> = Vec::new();
    for (si, sc) in scenes.iter().enumerate() {
        for _ in 0..frames_per_scene {
            let (img, joints) = render_frame(sc, &mut frame_rng);
            all.push(Sample { chw: img_to_chw(&img), joints, scene: si });
        }
    }
    let total = all.len();
    println!("合成火柴人：{n_scenes} 个场景（姿态）× {frames_per_scene} 帧/场景 = {total} 帧，每帧 {J} 个关节");

    // 两种划分：分组（按场景，val 是全新姿态）vs 随机（打乱帧）
    let train_scenes = 12; // 前 12 场景做 train，后 4 场景做 val
    let grouped_train: Vec<&Sample> = all.iter().filter(|s| s.scene < train_scenes).collect();
    let grouped_val: Vec<&Sample> = all.iter().filter(|s| s.scene >= train_scenes).collect();
    // 随机划分：用 LCG 打乱下标，切成同样大小
    let mut idx: Vec<usize> = (0..total).collect();
    let mut shuf = Lcg::new(999);
    for i in (1..total).rev() {
        let j = (shuf.next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let n_train = grouped_train.len();
    let random_train: Vec<&Sample> = idx[..n_train].iter().map(|&i| &all[i]).collect();
    let random_val: Vec<&Sample> = idx[n_train..].iter().map(|&i| &all[i]).collect();
    // 诊断：随机 val 里有多少帧的场景也在随机 train 中（近似重复→泄漏）
    let rtrain_scenes: std::collections::HashSet<usize> = random_train.iter().map(|s| s.scene).collect();
    let leaked = random_val.iter().filter(|s| rtrain_scenes.contains(&s.scene)).count();
    println!(
        "分组划分：train {} 帧（{} 个场景），val {} 帧（{} 个全新场景，无泄漏）",
        grouped_train.len(), train_scenes, grouped_val.len(), n_scenes - train_scenes
    );
    println!(
        "随机划分：train {} 帧，val {} 帧；其中 {}/{} 个 val 帧的场景也出现在 train 中（近似重复→泄漏）",
        random_train.len(), random_val.len(), leaked, random_val.len()
    );

    // ===== 第 2 步：主实验——在“诚实的分组划分”上训练姿态网络 =====
    println!("\n===== 训练姿态网络（分组划分，AdamW lr=3e-3→5e-4，MSE 热图回归）=====");
    let epochs = 150;
    let model_g = train(&grouped_train, epochs, "grouped", &dev, true)?;
    let (err_g, pck2_g, pck5_g) = eval(&model_g, &grouped_val, &dev)?;
    println!("\n【分组 val（全新姿态）】平均关节误差 = {err_g:.3} 像素，PCK@0.2 = {pck2_g:.1}%，PCK@0.5 = {pck5_g:.1}%");
    // 分关节误差
    {
        let mut per = [0f32; J];
        for s in &grouped_val {
            let p = predict_joints(&model_g, &s.chw, &dev)?;
            for k in 0..J {
                per[k] += ((p[k][0] - s.joints[k][0]).powi(2) + (p[k][1] - s.joints[k][1]).powi(2)).sqrt();
            }
        }
        print!("分关节平均误差(像素)：");
        for k in 0..J {
            print!("{}={:.2}  ", JOINT_NAMES[k], per[k] / grouped_val.len() as f32);
        }
        println!();
    }

    // ===== 第 3 步：画预测骨架 PNG（分组 val 的全新姿态）=====
    for (n, s) in grouped_val.iter().take(3).enumerate() {
        let pred = predict_joints(&model_g, &s.chw, &dev)?;
        save_skeleton(&s.chw, &s.joints, &pred, &format!("pose_{n}.png"));
    }
    println!("已存 pose_0.png .. pose_2.png（绿=真值骨架，黄=预测骨架）");

    // ===== 第 4 步：对照实验——随机划分（泄漏）会把 PCK 虚高多少 =====
    println!("\n===== 对照：随机划分（泄漏）vs 分组划分（诚实），同网络同训练 =====");
    let model_r = train(&random_train, epochs, "random", &dev, false)?;
    let (err_r, pck2_r, pck5_r) = eval(&model_r, &random_val, &dev)?;
    println!("  划分方式        平均误差(px)   PCK@0.2    PCK@0.5");
    println!("  随机(泄漏)        {err_r:>6.3}      {pck2_r:>5.1}%    {pck5_r:>5.1}%");
    println!("  分组(诚实)        {err_g:>6.3}      {pck2_g:>5.1}%    {pck5_g:>5.1}%");
    println!("  → 随机划分把 PCK@0.2 虚高了 {:.1} 个百分点（关键点检测里的泄漏代价）", pck2_r - pck2_g);
    Ok(())
}
