// 第 44 章 · 多标签分类（multi-label）最小可运行 demo
// candle 0.11.0（candle-core + candle-nn），纯 CPU，结果可复现。
//
// 任务设定：每张 16x16 灰度图可能同时含有 3 个「相互独立」的属性——
//   标签 0：横条（一条水平亮带）
//   标签 1：竖条（一条竖直亮带）
//   标签 2：亮块（一个小方块，出现概率更低，用来演示标签不平衡）
// 一张图可以同时命中任意子集（既有横条又有亮块……），
// 这正是多标签任务和「类别互斥」的多分类的本质区别。
//
// 网络：conv(1->8) -> relu -> maxpool -> fc(512->32) -> relu -> fc(32->3)
// 输出层不做 softmax；每个标签是一个独立的 logit，用 sigmoid 压到 [0,1]，
// 损失用逐标签的二元交叉熵（BCE）。

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Conv2d, Conv2dConfig, Linear, Module, Optimizer};

const H: usize = 16; // 图像高
const W: usize = 16; // 图像宽
const N_LABELS: usize = 3; // 标签数
const N_TRAIN: usize = 512;
const N_TEST: usize = 256;
const EPOCHS: usize = 400;
const LABEL_NAMES: [&str; N_LABELS] = ["横条", "竖条", "亮块"];

// —— 自带的确定性伪随机数发生器（splitmix64），保证结果可复现，且不引入外部依赖 ——
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
    }
    fn bernoulli(&mut self, p: f32) -> bool {
        self.f32() < p
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

// 合成一张图 + 它的多标签真值。返回 (256 个像素[0,1], [横条,竖条,亮块])
fn make_sample(rng: &mut Rng) -> (Vec<f32>, [f32; N_LABELS]) {
    // 三个属性各自独立掷骰子（互不排斥，可同时成立）
    let has_hbar = rng.bernoulli(0.5);
    let has_vbar = rng.bernoulli(0.5);
    let has_block = rng.bernoulli(0.25); // 亮块更稀有：演示标签不平衡

    let mut img = vec![0.1f32; H * W]; // 背景偏暗
    let bright = 0.9f32;
    if has_hbar {
        let r = rng.range(2, H - 3);
        for y in r..r + 2 {
            for x in 0..W {
                img[y * W + x] = bright;
            }
        }
    }
    if has_vbar {
        let c = rng.range(2, W - 3);
        for y in 0..H {
            for x in c..c + 2 {
                img[y * W + x] = bright;
            }
        }
    }
    if has_block {
        let by = rng.range(0, H - 4);
        let bx = rng.range(0, W - 4);
        for y in by..by + 4 {
            for x in bx..bx + 4 {
                img[y * W + x] = bright;
            }
        }
    }
    for p in img.iter_mut() {
        *p = (*p + (rng.f32() - 0.5) * 0.1).clamp(0.0, 1.0); // 叠一点噪声
    }
    (
        img,
        [has_hbar as u8 as f32, has_vbar as u8 as f32, has_block as u8 as f32],
    )
}

// 生成一批数据，铺成 [N,1,H,W] 的像素向量与 [N,3] 的标签向量
fn make_dataset(rng: &mut Rng, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut xs = Vec::with_capacity(n * H * W);
    let mut ys = Vec::with_capacity(n * N_LABELS);
    for _ in 0..n {
        let (img, label) = make_sample(rng);
        xs.extend_from_slice(&img);
        ys.extend_from_slice(&label);
    }
    (xs, ys)
}

// 用自带 PRNG 生成一个 [-bound, bound] 均匀分布的张量（确定性初始化）
fn det_tensor(rng: &mut Rng, dims: &[usize], bound: f32, device: &Device) -> Result<Tensor> {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * bound).collect();
    Tensor::from_vec(data, dims, device)
}

// 造一个确定性初始化的全连接层，并把它的可训练参数塞进 vars
fn det_linear(rng: &mut Rng, in_dim: usize, out_dim: usize, device: &Device, vars: &mut Vec<Var>) -> Result<Linear> {
    let bound = (1.0 / in_dim as f32).sqrt(); // 类似 PyTorch 默认的 1/sqrt(fan_in)
    let w = Var::from_tensor(&det_tensor(rng, &[out_dim, in_dim], bound, device)?)?;
    let b = Var::from_tensor(&Tensor::zeros(out_dim, DType::F32, device)?)?;
    let lin = Linear::new(w.as_tensor().clone(), Some(b.as_tensor().clone()));
    vars.push(w);
    vars.push(b);
    Ok(lin)
}

// 小 CNN：一层卷积 + 池化 + 两层全连接，输出 3 个「独立」logit。
//
// 🕳️ 为什么自己造初始权重？candle 0.11.0 的 CPU 后端「无法设定随机种子」
//    （Device::set_seed 在 CPU 上直接报错，内部用 rand::rng() 取系统熵）。
//    于是每次跑，candle_nn::linear/conv2d 的随机初始化都不同，结果会小幅抖动。
//    为了让本章数字可复现，这里用自带的确定性 PRNG 亲手初始化每个参数，
//    再用 Var::from_tensor 包成可训练变量，交给优化器。
struct MultiLabelNet {
    conv: Conv2d,
    fc1: Linear,
    fc2: Linear,
    vars: Vec<Var>, // 所有可训练参数，交给优化器
}
impl MultiLabelNet {
    fn new(rng: &mut Rng, device: &Device) -> Result<Self> {
        let mut vars = Vec::new();
        // 卷积核 [out_c=8, in_c=1, 3, 3]
        let fan_in = 1 * 3 * 3;
        let cw = Var::from_tensor(&det_tensor(rng, &[8, 1, 3, 3], (1.0 / fan_in as f32).sqrt(), device)?)?;
        let cb = Var::from_tensor(&Tensor::zeros(8usize, DType::F32, device)?)?;
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = Conv2d::new(cw.as_tensor().clone(), Some(cb.as_tensor().clone()), cfg);
        vars.push(cw);
        vars.push(cb);
        // 池化到 8x8 后展平 = 8*8*8 = 512
        let fc1 = det_linear(rng, 8 * 8 * 8, 32, device, &mut vars)?;
        let fc2 = det_linear(rng, 32, N_LABELS, device, &mut vars)?;
        Ok(Self { conv, fc1, fc2, vars })
    }
    // 返回「原始 logit」，不做 sigmoid/softmax（BCE 的输入要 logit）
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?.relu()?; // [N,8,16,16]
        let x = x.max_pool2d(2)?; // [N,8,8,8]
        let x = x.flatten_from(1)?; // [N,512]
        let x = self.fc1.forward(&x)?.relu()?; // [N,32]
        let x = self.fc2.forward(&x)?; // [N,3] logits
        Ok(x)
    }
}

// 数值稳定版「带 logit 的二元交叉熵」。
// candle 0.11.0 自带 candle_nn::loss::binary_cross_entropy_with_logit，
// 但它内部是「教科书直译」：先 sigmoid 再取 log。一旦网络训得自信、logit 变大，
// sigmoid 会饱和到 0 或 1，log(0) = -inf，损失就变成 NaN（本机实测约 300 轮后必崩）。
// 这里改用等价但不溢出的形式：L = max(x,0) - x*y + log(1 + exp(-|x|))。
fn bce_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?; // max(x, 0)
    let x_y = (logits * targets)?; // x * y
    let abs_x = logits.abs()?;
    let softplus = (abs_x.neg()?.exp()? + 1.0)?.log()?; // log(1 + exp(-|x|))，恒不溢出
    let loss = ((relu_x - x_y)? + softplus)?;
    loss.mean_all()
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut rng = Rng::new(20260827);

    // —— 造数据 ——
    let (xtr, ytr) = make_dataset(&mut rng, N_TRAIN);
    let (xte, yte) = make_dataset(&mut rng, N_TEST);
    let x_train = Tensor::from_vec(xtr, (N_TRAIN, 1, H, W), &device)?;
    let y_train = Tensor::from_vec(ytr, (N_TRAIN, N_LABELS), &device)?;
    let x_test = Tensor::from_vec(xte, (N_TEST, 1, H, W), &device)?;
    let y_test = Tensor::from_vec(yte, (N_TEST, N_LABELS), &device)?;

    // 训练集里各标签的出现频率（看不平衡程度）
    let ytr_sum: Vec<f32> = y_train.sum(0)?.to_vec1()?;
    println!("训练集标签频率：");
    for i in 0..N_LABELS {
        println!(
            "  {}：{:.0}/{}  ({:.1}%)",
            LABEL_NAMES[i], ytr_sum[i], N_TRAIN, 100.0 * ytr_sum[i] / N_TRAIN as f32
        );
    }

    // —— 建网络 + 优化器（确定性初始化，见 MultiLabelNet::new 的说明）——
    let net = MultiLabelNet::new(&mut rng, &device)?;
    let mut opt = AdamW::new_lr(net.vars.clone(), 1e-3)?;

    // —— 训练：全批次，损失用逐标签 sigmoid + BCE ——
    println!("\n开始训练（BCE 损失）：");
    for epoch in 1..=EPOCHS {
        let logits = net.forward(&x_train)?;
        let loss = bce_with_logits(&logits, &y_train)?;
        opt.backward_step(&loss)?;
        if epoch % 50 == 0 || epoch == 1 {
            println!("  epoch {:>3}  loss = {:.4}", epoch, loss.to_scalar::<f32>()?);
        }
    }

    // —— 评估：sigmoid 得到每个标签的独立概率，再逐标签阈值 0.5 判定 ——
    let logits = net.forward(&x_test)?;
    let probs: Vec<Vec<f32>> = candle_nn::ops::sigmoid(&logits)?.to_vec2()?; // [N,3]，各 ∈ [0,1] 且彼此独立
    let truth: Vec<Vec<f32>> = y_test.to_vec2()?;

    let thr = 0.5f32;
    let (mut tp, mut fp, mut fn_, mut tn) = ([0usize; N_LABELS], [0usize; N_LABELS], [0usize; N_LABELS], [0usize; N_LABELS]);
    let mut correct_cells = 0usize; // 汉明准确率用
    let mut exact = 0usize; // 完全匹配（子集准确率）
    let mut pos_cnt = [0usize; N_LABELS]; // 真实正样本数（support）
    for i in 0..N_TEST {
        let mut all_ok = true;
        for l in 0..N_LABELS {
            let pred = probs[i][l] >= thr;
            let gt = truth[i][l] >= 0.5;
            if gt {
                pos_cnt[l] += 1;
            }
            if pred == gt {
                correct_cells += 1;
            } else {
                all_ok = false;
            }
            match (pred, gt) {
                (true, true) => tp[l] += 1,
                (true, false) => fp[l] += 1,
                (false, true) => fn_[l] += 1,
                (false, false) => tn[l] += 1,
            }
        }
        if all_ok {
            exact += 1;
        }
    }

    println!("\n逐标签指标（阈值 = {:.1}）：", thr);
    println!("  {:<6}{:>8}{:>8}{:>8}{:>10}{:>10}", "标签", "准确率", "精确率", "召回率", "F1", "正样本数");
    let mut macro_f1 = 0.0f32;
    for l in 0..N_LABELS {
        let acc = (tp[l] + tn[l]) as f32 / N_TEST as f32;
        let prec = if tp[l] + fp[l] > 0 { tp[l] as f32 / (tp[l] + fp[l]) as f32 } else { 0.0 };
        let rec = if tp[l] + fn_[l] > 0 { tp[l] as f32 / (tp[l] + fn_[l]) as f32 } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        macro_f1 += f1;
        println!("  {:<6}{:>8.3}{:>8.3}{:>8.3}{:>10.3}{:>10}", LABEL_NAMES[l], acc, prec, rec, f1, pos_cnt[l]);
    }
    macro_f1 /= N_LABELS as f32;

    let hamming_acc = correct_cells as f32 / (N_TEST * N_LABELS) as f32;
    let exact_acc = exact as f32 / N_TEST as f32;
    println!("\n整体指标：");
    println!("  汉明准确率（逐格正确率）    = {:.3}", hamming_acc);
    println!("  完全匹配率（3 个标签全对）  = {:.3}", exact_acc);
    println!("  macro-F1（各标签 F1 平均）  = {:.3}", macro_f1);

    // —— 为什么单看 accuracy 会误导：拿最稀有的「亮块」和「无脑全预测为负」的傻基线比 ——
    let l = 2usize; // 亮块
    let baseline_acc = (N_TEST - pos_cnt[l]) as f32 / N_TEST as f32;
    println!("\n标签「{}」：模型准确率看着不低，但真正有用的是 F1/召回。", LABEL_NAMES[l]);
    println!("  「无脑全预测为没有」这个傻基线的准确率 = {:.3}（因为它稀有），", baseline_acc);
    println!("  但它的召回率 = 0.000、F1 = 0.000 —— 单看 accuracy 完全被骗过去了。");

    // —— 抽样查看：每个标签的预测概率 + 阈值后的多标签预测 vs 真值 ——
    println!("\n抽样查看（概率 / 阈值{:.1}后预测 / 真值）：", thr);
    for i in 0..6 {
        let (p, g) = (&probs[i], &truth[i]);
        let pred: Vec<&str> = (0..N_LABELS).filter(|&l| p[l] >= thr).map(|l| LABEL_NAMES[l]).collect();
        let real: Vec<&str> = (0..N_LABELS).filter(|&l| g[l] >= 0.5).map(|l| LABEL_NAMES[l]).collect();
        println!("  #{i}: p=[横{:.2} 竖{:.2} 块{:.2}]  预测={:?}  真值={:?}", p[0], p[1], p[2], pred, real);
    }

    // —— 阈值不是只能取 0.5：对「亮块」扫一遍阈值，看哪个 F1 最高 ——
    println!("\n对「{}」扫描阈值，看 0.5 未必最优：", LABEL_NAMES[l]);
    let (mut best_thr, mut best_f1) = (0.5f32, -1.0f32);
    for k in 1..10 {
        let t = k as f32 / 10.0;
        let (mut ltp, mut lfp, mut lfn) = (0usize, 0usize, 0usize);
        for i in 0..N_TEST {
            match (probs[i][l] >= t, truth[i][l] >= 0.5) {
                (true, true) => ltp += 1,
                (true, false) => lfp += 1,
                (false, true) => lfn += 1,
                _ => {}
            }
        }
        let prec = if ltp + lfp > 0 { ltp as f32 / (ltp + lfp) as f32 } else { 0.0 };
        let rec = if ltp + lfn > 0 { ltp as f32 / (ltp + lfn) as f32 } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
        if f1 > best_f1 {
            best_f1 = f1;
            best_thr = t;
        }
        println!("  阈值 {:.1}: 精确率={:.3} 召回率={:.3} F1={:.3}", t, prec, rec, f1);
    }
    println!("  → 最佳阈值 ≈ {:.1}（F1 = {:.3}）", best_thr, best_f1);

    Ok(())
}
