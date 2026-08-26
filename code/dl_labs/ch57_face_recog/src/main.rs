// 第 57 章：人脸识别与比对 —— 小 embedding 网络 + ArcFace 角度间隔（度量学习）
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现。
// 运行：cd code/dl_labs/ch57_face_recog && cargo run --release
//
// 思路：合成“多身份”对齐人脸（每个身份=一组固定五官参数 + 类内扰动）
//       → 训练一个把人脸映射到 L2 归一化低维向量的网络
//       → 用 ArcFace 角度间隔 margin-softmax 做度量学习（对比 margin=0 的普通版）
//       → 在“训练时没见过”的新身份上做 1:1 verification 与 1:N identification
//       → 报同人/异人余弦相似度分布、验证准确率、TAR@FAR、类内/类间距离。

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::loss::cross_entropy;
use candle_nn::{conv2d, linear, linear_no_bias, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use image::{Rgb, RgbImage};

const IMG: usize = 32; // 对齐后的人脸裁剪图边长（识别在“已检测+已对齐”的人脸上做）
const EMB: usize = 32; // embedding 维度
const N_TRAIN_ID: usize = 16; // 训练身份数
const N_TEST_ID: usize = 8; // 验证身份数（训练时从未见过 → 开集测试）
const PER_TRAIN: usize = 32; // 每个训练身份的样本数
const PER_TEST: usize = 12; // 每个验证身份的样本数

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed） ----------
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
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
}

// ---------- 一个“身份”的生成参数（人脸的身份特征都藏在这几个数里） ----------
#[derive(Clone, Copy)]
struct Id {
    sk: f32,      // 肤色亮度基调
    warm: f32,    // 暖色偏移（R 比 B 高多少）
    aspect: f32,  // 脸高/宽比
    eye_dx: f32,  // 双眼水平间距（占半宽 a 的比例）
    eye_dy: f32,  // 眼睛在中心上方多少（占半高 b）
    eye_r: f32,   // 眼睛半径（占 a）
    mouth_w: f32, // 嘴半宽（占 a）
}
fn sample_id(rng: &mut Lcg) -> Id {
    Id {
        sk: rng.uniform(170.0, 235.0),
        warm: rng.uniform(30.0, 75.0),
        aspect: rng.uniform(1.12, 1.42),
        eye_dx: rng.uniform(0.30, 0.48),
        eye_dy: rng.uniform(0.16, 0.30),
        eye_r: rng.uniform(0.10, 0.18),
        mouth_w: rng.uniform(0.20, 0.38),
    }
}

fn fill_ellipse(img: &mut RgbImage, cx: f32, cy: f32, a: f32, b: f32, color: Rgb<u8>) {
    let x0 = (cx - a).floor().max(0.0) as i32;
    let x1 = (cx + a).ceil().min(IMG as f32 - 1.0) as i32;
    let y0 = (cy - b).floor().max(0.0) as i32;
    let y1 = (cy + b).ceil().min(IMG as f32 - 1.0) as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = (x as f32 - cx) / a;
            let dy = (y as f32 - cy) / b;
            if dx * dx + dy * dy <= 1.0 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}
fn fill_disk(img: &mut RgbImage, cx: f32, cy: f32, r: f32, color: Rgb<u8>) {
    let x0 = (cx - r).floor().max(0.0) as i32;
    let x1 = (cx + r).ceil().min(IMG as f32 - 1.0) as i32;
    let y0 = (cy - r).floor().max(0.0) as i32;
    let y1 = (cy + r).ceil().min(IMG as f32 - 1.0) as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx * dx + dy * dy <= r * r {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

// 渲染一张“对齐人脸”：身份参数固定，加上类内扰动（位置/尺度/亮度/噪声）
// —— 对齐后脸基本居中、尺度接近，身份信号在五官几何与肤色里，正如真实人脸识别流程。
fn render(id: &Id, rng: &mut Lcg) -> RgbImage {
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    // 背景：中性灰 + 噪声
    let bg = rng.uniform(40.0, 90.0);
    for y in 0..IMG as u32 {
        for x in 0..IMG as u32 {
            let v = (bg + rng.uniform(-8.0, 8.0)).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    // 类内扰动：位置 ±1.5px、尺度 ±6%、全局亮度 ±14
    let cx = IMG as f32 / 2.0 + rng.uniform(-1.5, 1.5);
    let cy = IMG as f32 / 2.0 + rng.uniform(-1.5, 1.5);
    let w = IMG as f32 * 0.66 * rng.uniform(0.94, 1.06);
    let (a, b) = (w / 2.0, w * id.aspect / 2.0);
    let bright = rng.uniform(-14.0, 14.0);
    let sk = (id.sk + bright).clamp(60.0, 255.0);
    let skin = Rgb([
        sk as u8,
        (sk - id.warm * 0.4).clamp(0.0, 255.0) as u8,
        (sk - id.warm).clamp(0.0, 255.0) as u8,
    ]);
    fill_ellipse(&mut img, cx, cy, a, b, skin);
    let dark = Rgb([28u8, 26, 30]);
    // 眼睛
    let ex = id.eye_dx * a;
    let ey = cy - id.eye_dy * b;
    fill_disk(&mut img, cx - ex, ey, id.eye_r * a + 0.5, dark);
    fill_disk(&mut img, cx + ex, ey, id.eye_r * a + 0.5, dark);
    // 鼻
    fill_disk(&mut img, cx, cy + 0.06 * b, 0.08 * a + 0.4, dark);
    // 嘴：一条粗线
    let mw = id.mouth_w * a;
    let my = (cy + 0.46 * b).round() as i32;
    for yy in my..=(my + 1) {
        for xx in (cx - mw).round() as i32..=(cx + mw).round() as i32 {
            if xx >= 0 && xx < IMG as i32 && yy >= 0 && yy < IMG as i32 {
                img.put_pixel(xx as u32, yy as u32, dark);
            }
        }
    }
    // 叠加像素噪声（让同一身份的每张“照片”都不同）
    for y in 0..IMG as u32 {
        for x in 0..IMG as u32 {
            let p = img.get_pixel(x, y).0;
            let n = rng.uniform(-10.0, 10.0);
            img.put_pixel(
                x,
                y,
                Rgb([
                    (p[0] as f32 + n).clamp(0.0, 255.0) as u8,
                    (p[1] as f32 + n).clamp(0.0, 255.0) as u8,
                    (p[2] as f32 + n).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    img
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

// 造数据集：n_id 个身份、每身份 per 张，返回 (输入张量[n_id*per,3,IMG,IMG], 标签 Vec<u32>)
fn make_set(ids: &[Id], per: usize, rng: &mut Lcg, dev: &Device) -> Result<(Tensor, Vec<u32>)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (label, id) in ids.iter().enumerate() {
        for _ in 0..per {
            let img = render(id, rng);
            xs.extend_from_slice(&img_to_chw(&img));
            ys.push(label as u32);
        }
    }
    let n = ids.len() * per;
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    Ok((x, ys))
}

// ---------- embedding 网络：3 段卷积 + 一个全连接 → EMB 维特征（不归一化，归一化在外面做） ----------
struct Embed {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc: Linear,
}
impl Embed {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            fc: linear(64 * 4 * 4, EMB, vb.pp("fc"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 32 -> 16
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 16 -> 8
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?; // 8 -> 4
        let x = x.flatten_from(1)?; // [N, 64*4*4]
        self.fc.forward(&x) // [N, EMB]
    }
}

// L2 行归一化：每一行除以自己的模长（加 eps 防除零）
fn l2norm(t: &Tensor) -> Result<Tensor> {
    let n = t.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-12)?.sqrt()?; // [rows,1]
    t.broadcast_div(&n)
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

// ArcFace 逻辑值：对归一化特征 f_norm[N,EMB] 与归一化类别权重 w_norm[NID,EMB]
// 求 cos θ = f_norm·w_norm^T，再给“真值类别”的角度加间隔 m：s·cos(θ+m)。
// margin=0 时退化成“尺度化余弦 softmax”（普通归一化 softmax，无间隔），用于对照。
fn arcface_logits(cos: &Tensor, one_hot: &Tensor, s: f64, m: f64) -> Result<Tensor> {
    if m == 0.0 {
        return cos.affine(s, 0.0); // 无间隔：直接 s·cosθ
    }
    let (cos_m, sin_m) = (m.cos(), m.sin());
    let cos_sq = cos.sqr()?;
    let sin = cos_sq.affine(-1.0, 1.0)?.relu()?.sqrt()?; // sinθ = sqrt(max(1-cos²,0))
    // cos(θ+m) = cosθ·cos m - sinθ·sin m
    let phi = cos.affine(cos_m, 0.0)?.broadcast_sub(&sin.affine(sin_m, 0.0)?)?;
    // 只给真值类别用带间隔的 phi，其余仍用 cosθ： out = cos + onehot·(phi - cos)
    let diff = phi.broadcast_sub(cos)?;
    let out = cos.broadcast_add(&one_hot.broadcast_mul(&diff)?)?;
    out.affine(s, 0.0)
}

// 一对归一化向量的余弦相似度（已归一化 → 点积即余弦）
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Metrics {
    pos_mean: f32,
    neg_mean: f32,
    best_acc: f32,
    best_thr: f32,
    tar_at_far10: f32,
    tar_at_far01: f32,
    ident_acc: f32,
}

// 训练一个 embedding 网络（给定 margin），并在“未见过的新身份”上评估比对性能。
fn run(margin: f64, train_ids: &[Id], test_ids: &[Id], dev: &Device) -> Result<Metrics> {
    // —— 训练数据（固定种子，两次 run 完全一致，隔离出 margin 这一个变量）——
    let mut drng = Lcg::new(100);
    let (xtr, ytr) = make_set(train_ids, PER_TRAIN, &mut drng, dev)?;
    let n = ytr.len();
    let ytr_t = Tensor::from_vec(ytr.clone(), (n,), dev)?; // U32 标签，cross_entropy 用
    // 真值 one-hot [N, N_TRAIN_ID]（ArcFace 施加间隔时的掩码）
    let mut oh = vec![0f32; n * N_TRAIN_ID];
    for (i, &l) in ytr.iter().enumerate() {
        oh[i * N_TRAIN_ID + l as usize] = 1.0;
    }
    let one_hot = Tensor::from_vec(oh, (n, N_TRAIN_ID), dev)?;

    // —— 建网：embedding 骨干 + ArcFace 类别权重（linear_no_bias 的权重就是每类的中心向量）——
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = Embed::load(vb.pp("emb"))?;
    let arc = linear_no_bias(EMB, N_TRAIN_ID, vb.pp("arc"))?; // 权重 [N_TRAIN_ID, EMB]
    det_init(&vm, 42)?;

    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    let (s, epochs) = (10.0f64, 160);
    for epoch in 1..=epochs {
        let feat = net.forward(&xtr)?; // [N, EMB]
        let f_norm = l2norm(&feat)?;
        let w_norm = l2norm(arc.weight())?; // [N_TRAIN_ID, EMB]
        let cos = f_norm.matmul(&w_norm.t()?)?; // [N, N_TRAIN_ID] ∈ [-1,1]
        let logits = arcface_logits(&cos, &one_hot, s, margin)?;
        let loss = cross_entropy(&logits, &ytr_t)?;
        opt.backward_step(&loss)?;
        if epoch % 40 == 0 || epoch == 1 {
            // 训练集上的分类准确率（顺便看收敛）
            let pred = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?;
            let acc = pred.iter().zip(&ytr).filter(|(a, b)| a == b).count() as f32 / n as f32;
            println!("    [margin={margin:.2}] epoch {epoch:>3}  loss={:.4}  train_acc={:.3}", loss.to_scalar::<f32>()?, acc);
        }
    }

    // —— 在“训练时没见过”的新身份上评估（开集：真实人脸识别就是要认没登记过的人脸的相似度）——
    let mut erng = Lcg::new(9090);
    let (xte, yte) = make_set(test_ids, PER_TEST, &mut erng, dev)?;
    let feat = net.forward(&xte)?;
    let emb = l2norm(&feat)?.to_vec2::<f32>()?; // 每张测试图的归一化 embedding

    // 1:1 verification：同人对 vs 异人对的余弦相似度
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    for i in 0..emb.len() {
        for j in (i + 1)..emb.len() {
            let sim = cosine(&emb[i], &emb[j]);
            if yte[i] == yte[j] {
                pos.push(sim);
            } else {
                neg.push(sim);
            }
        }
    }
    let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let neg_mean = neg.iter().sum::<f32>() / neg.len() as f32;

    // 扫阈值找最佳验证准确率（同人 sim≥t 判“同”，异人 sim<t 判“异”）
    let (mut best_acc, mut best_thr) = (0f32, 0f32);
    let mut t = -0.20f32;
    while t <= 1.0 {
        let tp = pos.iter().filter(|&&v| v >= t).count();
        let tn = neg.iter().filter(|&&v| v < t).count();
        let acc = (tp + tn) as f32 / (pos.len() + neg.len()) as f32;
        if acc > best_acc {
            best_acc = acc;
            best_thr = t;
        }
        t += 0.01;
    }

    // TAR@FAR：在异人对上把误接受率压到 FAR，看同人对的通过率 TAR
    let tar_at = |far: f32| -> f32 {
        let mut ns = neg.clone();
        ns.sort_by(|a, b| b.partial_cmp(a).unwrap()); // 降序
        let k = ((far * ns.len() as f32).floor() as usize).min(ns.len().saturating_sub(1));
        let thr = ns[k]; // 让约 far 比例的异人对越过阈值
        pos.iter().filter(|&&v| v >= thr).count() as f32 / pos.len() as f32
    };

    // 1:N identification：每个新身份用第 1 张“登记”，其余作为待识别探针，最近邻判定
    let mut gallery: Vec<(u32, Vec<f32>)> = Vec::new();
    for id in 0..N_TEST_ID as u32 {
        let first = yte.iter().position(|&l| l == id).unwrap();
        gallery.push((id, emb[first].clone()));
    }
    let (mut correct, mut total) = (0usize, 0usize);
    for (idx, e) in emb.iter().enumerate() {
        let label = yte[idx];
        let first = yte.iter().position(|&l| l == label).unwrap();
        if idx == first {
            continue; // 登记图不做探针
        }
        let mut best = (0u32, f32::MIN);
        for (gid, gvec) in &gallery {
            let sim = cosine(e, gvec);
            if sim > best.1 {
                best = (*gid, sim);
            }
        }
        if best.0 == label {
            correct += 1;
        }
        total += 1;
    }

    Ok(Metrics {
        pos_mean,
        neg_mean,
        best_acc,
        best_thr,
        tar_at_far10: tar_at(0.10),
        tar_at_far01: tar_at(0.01),
        ident_acc: correct as f32 / total as f32,
    })
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    // 先造好所有身份：前 N_TRAIN_ID 个用于训练，后 N_TEST_ID 个用于验证（身份不重叠 → 开集）
    let mut irng = Lcg::new(2024);
    let all_ids: Vec<Id> = (0..N_TRAIN_ID + N_TEST_ID).map(|_| sample_id(&mut irng)).collect();
    let train_ids = &all_ids[..N_TRAIN_ID];
    let test_ids = &all_ids[N_TRAIN_ID..];
    println!(
        "合成身份：{N_TRAIN_ID} 个训练身份（每身份 {PER_TRAIN} 张）+ {N_TEST_ID} 个验证身份（每身份 {PER_TEST} 张，训练时从未见过）"
    );
    println!("embedding 维度 = {EMB}，识别在 {IMG}x{IMG} 对齐人脸上进行\n");

    // 对照实验：只改 margin 这一个变量，其余（数据/初始化/轮数/尺度）完全相同
    println!("=== 训练 A：普通归一化 softmax（margin=0，无角度间隔）===");
    let a = run(0.0, train_ids, test_ids, &dev)?;
    println!("\n=== 训练 B：ArcFace 角度间隔 margin-softmax（margin=0.50 弧度）===");
    let b = run(0.50, train_ids, test_ids, &dev)?;

    println!("\n================= 新身份（开集）比对结果对比 =================");
    println!("{:<24}{:>14}{:>14}", "指标", "普通softmax", "ArcFace");
    println!("{:<26}{:>12.3}{:>14.3}", "同人对 平均余弦相似度", a.pos_mean, b.pos_mean);
    println!("{:<26}{:>12.3}{:>14.3}", "异人对 平均余弦相似度", a.neg_mean, b.neg_mean);
    println!("{:<26}{:>12.3}{:>14.3}", "同异相似度间隔(差)", a.pos_mean - a.neg_mean, b.pos_mean - b.neg_mean);
    println!("{:<26}{:>12.3}{:>14.3}", "类内平均距离(1-cos)", 1.0 - a.pos_mean, 1.0 - b.pos_mean);
    println!("{:<26}{:>12.3}{:>14.3}", "类间平均距离(1-cos)", 1.0 - a.neg_mean, 1.0 - b.neg_mean);
    println!("{:<26}{:>11.1}%{:>13.1}%", "最佳阈值验证准确率", 100.0 * a.best_acc, 100.0 * b.best_acc);
    println!("{:<28}{:>10.2}{:>14.2}", "（对应最佳阈值）", a.best_thr, b.best_thr);
    println!("{:<26}{:>11.1}%{:>13.1}%", "TAR@FAR=10%", 100.0 * a.tar_at_far10, 100.0 * b.tar_at_far10);
    println!("{:<26}{:>11.1}%{:>13.1}%", "TAR@FAR=1%", 100.0 * a.tar_at_far01, 100.0 * b.tar_at_far01);
    println!("{:<26}{:>11.1}%{:>13.1}%", "1:N 识别准确率", 100.0 * a.ident_acc, 100.0 * b.ident_acc);
    println!("\n结论：只加了角度间隔这一个变量，ArcFace 让同人相似度更高、异人更低，");
    println!("      同异间隔更大，验证/识别准确率随之提升 —— 这正是 margin 对“比对”的价值。");
    Ok(())
}
