# 第 44 章 细粒度、多标签与层级分类

> **本章导读**
> - 认清两种"长得像分类、规则却变了"的任务：**细粒度分类**（区分极像的子类）与**多标签分类**（一张图同时命中多个标签）。
> - 记牢多标签与多分类最核心的一处不同：输出层从 **softmax** 换成 **每个标签一个 sigmoid**，损失从交叉熵换成 **BCE（二元交叉熵）**。
> - 用 candle 亲手训练一个多标签小网络，**真的跑起来**，打印逐标签概率、阈值判定、per-label F1 与汉明准确率，并看清"为什么单看 accuracy 会骗人"。
> - 顺带认识**层级分类**，收好一串真实的坑（softmax 用错、阈值未必是 0.5、标签不平衡、BCE 数值溢出）。

前两章我们把"标准图像分类"这条线走通了：[第 42 章](ch42-candle-classify.md) 用 candle 加载模型、预处理、`forward` 再 `softmax` 拿类别；[第 43 章](ch43-finetune.md) 教你把它微调到自己的数据上。它们都默认了一件事——**一张图属于、且只属于一个类别**。可现实里，这个假设经常不成立，于是就有了本章这两道"变形题"。

## 场景：两道分类题，规则其实变了

**题一（细粒度）**：相册里两张鸟的照片，一张是麻雀、一张是文须雀。你我都得眯起眼，盯着喙的形状、眼周的纹路才分得清。它们同属"鸟"，**类间差异极小、还都藏在局部细节里**——这就是**细粒度分类（fine-grained classification）**：分的不是猫和狗，而是几百种长得差不多的子类。

**题二（多标签）**：一张街景照里，"有人""有车""有红绿灯"可以**同时成立**。这时候你要的不是"选一个最像的类别"，而是**给这张图挂上一串标签**。多分类那套"softmax 选一个"在这里直接崩掉——这就是**多标签分类（multi-label classification）**。

## 细粒度分类：难在"细节藏在局部"

分猫和狗，靠的是全局轮廓；分 200 种鸟，靠的是喙形、翅斑、眼纹这些**局部的微小差异**。普通分类器为什么在这儿吃力？因为一路卷积、池化下来，全局平均会把这些小差异"抹平"，而它们又对姿态、光照、背景特别敏感。

应对思路有三条（这里讲直觉，不必都完整实现）：

1. **高分辨率输入**：细节需要像素撑着。把 224×224 提到 448×448，喙尖上的纹路才有几个像素可看。
2. **局部/部件特征**：先定位"喙""翅膀"这些判别性部件（要用到关键点/部件定位，见 [第 52 章](../p11-keypoints/ch52-keypoint-basics.md)），把局部裁出来放大，再送去分类。
3. **注意力（attention）**：让网络自己学会"该盯哪一块"，把算力集中到最能区分子类的区域，而不是平均地看全图。

一个"放大局部再分类"的直觉示意：

```text
整图分类（看全局，容易"都是鸟"分不动）
┌───────────────┐
│   🐦  全景     │──►[分类器]──► "某种鸟"？子类难分
└───────────────┘
        │ 定位判别性部件（喙 / 翅 / 眼纹）
        ▼
放大局部再分类（盯着细节看）
┌────┐┌────┐┌────┐
│ 喙 ││ 翅 ││眼纹│──►[局部特征 + 注意力]──► 精确子类
└────┘└────┘└────┘
```

> ⚠️ **注意**：细粒度的强模型仍然要**更强的骨干网络 + 更多、更干净的数据**。小网络在这里只能把机制讲清楚，别指望它逼近 SOTA。此外，"归一化 embedding + ArcFace/Triplet"这类**度量学习**也是细粒度常用武器——人脸识别就是最典型的细粒度任务，我们留到 [第 57 章](../p12-face/ch57-face-recognition.md) 专门讲。

## 多标签分类：把"选一个"换成"逐个判断"

多标签和多分类，差别就一句话：**类别是互斥还是独立**。一张表看透：

| 维度 | 多分类（multi-class） | 多标签（multi-label） |
|---|---|---|
| 类别关系 | 互斥，有且仅有一个 | 独立，可同时成立多个 |
| 输出层 | 一个 **softmax** | **每个标签一个 sigmoid** |
| 概率含义 | 各类概率和为 1，此消彼长 | 每个 ∈ [0,1]，彼此无关 |
| 取结果 | `argmax` 取最大那一个 | **逐标签**与阈值比较 |
| 损失 | 交叉熵（cross-entropy） | **BCE**（逐标签的二元交叉熵） |
| 例子 | 猫 / 狗 / 车 三选一 | "有人""有车""有树"可全中 |

**为什么不能用 softmax？** softmax 会强制"各类概率之和为 1"，也就是各类**此消彼长**：你"有人"的概率涨一点，"有车"就得跌一点。这和"两者可以同时为真"直接矛盾。多标签要的是**对每个标签独立地问一句"有没有"**，互不牵制。

于是输出层换成 sigmoid，把每个 logit \\( z_c \\) 单独压到 [0,1]：

\\[ \sigma(z_c) = \frac{1}{1 + e^{-z_c}} \\]

（对比一下多分类的 softmax：\\( p_c = e^{z_c} / \sum_{j} e^{z_j} \\)，分母把所有类耦合在了一起。）

损失也随之改变。每个标签本质上是一个独立的二分类问题，用**二元交叉熵**，再把 \\( C \\) 个标签加起来求平均（\\( y_c \in \{0,1\} \\) 是真值，\\( \hat p_c = \sigma(z_c) \\)）：

\\[ L = -\frac{1}{C}\sum_{c=1}^{C}\big[\, y_c \log \hat p_c + (1 - y_c)\log(1 - \hat p_c) \,\big] \\]

最后**逐标签**拿概率和阈值比：\\( \hat p_c \ge 0.5 \\) 就判"命中"。注意这个 0.5 只是默认值，后面我们会实测看到它**未必最优**。

> 📖 **术语**：**层级分类（hierarchical classification）**。标签之间存在父子关系（"金毛" ⇒ "狗" ⇒ "动物"）。它可以看成"带约束的多标签"：预测必须满足 `子类为真 ⇒ 父类为真`。常见做法是叶子类与父类**联合损失**，或把叶子概率**沿树聚合**；落地时一定要单独报告"层级一致率"，任何"是金毛却不是狗"的输出都算违规。本章聚焦前两类任务，层级作为延伸了解即可。

## 动手：用 candle 训练一个多标签小网络

我们复用前两章那套 **candle 小 CNN**（卷积 → 池化 → 全连接）的思路，**唯一的改动就是输出层不接 softmax，而是 3 个独立 logit + sigmoid + BCE**——这就是多标签的全部秘密。预处理细节不再赘述（见 [第 42 章](ch42-candle-classify.md)）。

数据完全合成、自包含、可复现：16×16 灰度图，含 3 个**相互独立**的属性——横条、竖条、亮块，可任意共存（一张图能同时有横条和亮块）。其中"亮块"故意设得更稀有（出现概率 0.25），用来演示**标签不平衡**。

独立工程见 `code/dl_labs/ch44_multilabel/`（不参与主 `cv_examples` 构建）。`Cargo.toml`：

```toml
[package]
name = "ch44_multilabel"
version = "0.1.0"
edition = "2021"

# 声明为独立 workspace，使其与主 code/ 工程完全隔离
[workspace]

[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
```

完整 `src/main.rs`：

```rust
// candle 0.11.0（candle-core + candle-nn），纯 CPU，结果可复现。
// 每张 16x16 灰度图可能同时含 3 个「相互独立」的属性：横条 / 竖条 / 亮块。
// 输出层不做 softmax；每个标签一个独立 logit，用 sigmoid 压到 [0,1]，损失用逐标签 BCE。
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Conv2d, Conv2dConfig, Linear, Module, Optimizer};

const H: usize = 16;
const W: usize = 16;
const N_LABELS: usize = 3;
const N_TRAIN: usize = 512;
const N_TEST: usize = 256;
const EPOCHS: usize = 400;
const LABEL_NAMES: [&str; N_LABELS] = ["横条", "竖条", "亮块"];

// 自带确定性 PRNG（splitmix64），保证可复现，且不引入外部依赖
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

// 合成一张图 + 它的多标签真值
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

// 生成一批数据，铺成 [N,1,H,W] 像素向量与 [N,3] 标签向量
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

// 用自带 PRNG 生成 [-bound, bound] 均匀分布的张量（确定性初始化）
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
        let fan_in = 1 * 3 * 3; // 卷积核 [out_c=8, in_c=1, 3, 3]
        let cw = Var::from_tensor(&det_tensor(rng, &[8, 1, 3, 3], (1.0 / fan_in as f32).sqrt(), device)?)?;
        let cb = Var::from_tensor(&Tensor::zeros(8usize, DType::F32, device)?)?;
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = Conv2d::new(cw.as_tensor().clone(), Some(cb.as_tensor().clone()), cfg);
        vars.push(cw);
        vars.push(cb);
        let fc1 = det_linear(rng, 8 * 8 * 8, 32, device, &mut vars)?; // 池化到 8x8 后展平 = 512
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

    let ytr_sum: Vec<f32> = y_train.sum(0)?.to_vec1()?; // 各标签出现频率
    println!("训练集标签频率：");
    for i in 0..N_LABELS {
        println!(
            "  {}：{:.0}/{}  ({:.1}%)",
            LABEL_NAMES[i], ytr_sum[i], N_TRAIN, 100.0 * ytr_sum[i] / N_TRAIN as f32
        );
    }

    // —— 建网络 + 优化器（确定性初始化）——
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

    // —— 评估：sigmoid 得到每标签独立概率，再逐标签阈值 0.5 判定 ——
    let logits = net.forward(&x_test)?;
    let probs: Vec<Vec<f32>> = candle_nn::ops::sigmoid(&logits)?.to_vec2()?; // [N,3]，各 ∈ [0,1] 独立
    let truth: Vec<Vec<f32>> = y_test.to_vec2()?;

    let thr = 0.5f32;
    let (mut tp, mut fp, mut fn_, mut tn) =
        ([0usize; N_LABELS], [0usize; N_LABELS], [0usize; N_LABELS], [0usize; N_LABELS]);
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

    // —— 抽样查看：每标签概率 + 阈值后的多标签预测 vs 真值 ——
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
```

在工程目录里运行：

```bash
cd code/dl_labs/ch44_multilabel
cargo run --release
```

本机实测输出（固定随机种子，可复现）：

```text
训练集标签频率：
  横条：264/512  (51.6%)
  竖条：257/512  (50.2%)
  亮块：127/512  (24.8%)

开始训练（BCE 损失）：
  epoch   1  loss = 0.6919
  epoch  50  loss = 0.4459
  epoch 100  loss = 0.1955
  epoch 200  loss = 0.0674
  epoch 300  loss = 0.0200
  epoch 400  loss = 0.0063

逐标签指标（阈值 = 0.5）：
  标签         准确率     精确率     召回率        F1      正样本数
  横条       1.000   1.000   1.000     1.000       140
  竖条       1.000   1.000   1.000     1.000       125
  亮块       0.934   1.000   0.764     0.866        72

整体指标：
  汉明准确率（逐格正确率）    = 0.978
  完全匹配率（3 个标签全对）  = 0.934
  macro-F1（各标签 F1 平均）  = 0.955

抽样查看（概率 / 阈值0.5后预测 / 真值）：
  #1: p=[横1.00 竖1.00 块0.01]  预测=["横条", "竖条"]  真值=["横条", "竖条"]
  #5: p=[横1.00 竖1.00 块0.49]  预测=["横条", "竖条"]  真值=["横条", "竖条", "亮块"]

对「亮块」扫描阈值，看 0.5 未必最优：
  阈值 0.3: 精确率=1.000 召回率=0.917 F1=0.957
  阈值 0.5: 精确率=1.000 召回率=0.764 F1=0.866
  → 最佳阈值 ≈ 0.3（F1 = 0.957）
```

几点值得品味：横条、竖条模式简单又均衡，几乎满分；**亮块更稀有、更小、更难，F1 只有 0.866**。第 5 号样本尤其有意思——它真值里有亮块，模型也给了 0.49 的概率，可惜差一点点没过 0.5，就被漏判了。这直接引出下一节的话题。

## 评估：别用单一 accuracy

多标签任务里，**"准确率"这个词非常危险**。看代码里那段对比：亮块只占约 25%，那么一个"无脑全预测为没有"的傻瓜模型，准确率就有 **0.719**——可它召回率、F1 全是 0，一个亮块都没找到。如果你只盯着 0.719，会以为模型还凑合，其实它啥也没干。**标签越不平衡，这种"高准确率幻觉"越严重。**

所以多标签该看这些指标：

- **per-label precision / recall / F1**：逐标签算，别一锅烩。漏检致命就重点看召回，误报扰民就重点看精确率。
- **macro-F1**：各标签 F1 的简单平均（本例 0.955）。它对每个标签一视同仁，**不会被样本多的多数类淹没**，稀有标签拉胯会立刻体现出来。
- **mAP**：给每个标签画一条 PR 曲线、求平均精度（AP），再对所有标签平均。它衡量的是**排序质量**，和阈值无关，思路与目标检测的 mAP 一致（见 [第 45 章](../p09-detection/ch45-detection-principles.md)）。
- **汉明准确率 vs 完全匹配率**：前者是"逐格正确率"（本例 0.978），后者要求一张图 3 个标签**全对**才算对（本例 0.934）。完全匹配更严格，所以更低——报告时要分清你说的是哪一个。

而阈值本身也是可调的参数。看那张扫描表：阈值取 0.5 时，亮块召回只有 0.764（漏掉近 1/4）；把阈值降到 **0.3**，召回升到 0.917，F1 从 0.866 涨到 **0.957**。可见 **0.5 不是天经地义的，它应当在验证集上按你的漏报/误报成本挑出来，而且每个标签可以有各自的阈值。**

## 常见坑与提示

> 🕳️ **坑**：**把 softmax 用到多标签任务上。** 这是最常见的错误。症状很典型：概率加起来永远是 1、模型永远只敢报一个标签。多标签必须**逐标签 sigmoid + BCE**，绝不能 softmax + 交叉熵。

> ⚠️ **注意**：**阈值不是 0.5。** 尤其在标签稀有、漏报/误报代价不对称时，固定 0.5 往往吃亏。请在验证集上逐标签搜索阈值，按业务成本定。

> 🕳️ **坑**：**标签不平衡让稀有标签学不动、召回垫底。** 对策：给正样本更大的 loss 权重、对稀有标签过采样、或改用 focal loss（不平衡的系统讨论见 [第 43 章](ch43-finetune.md)）；评估时**务必抛弃 accuracy**，改用 per-label F1 / mAP。

> 🕳️ **坑**：**BCE 数值溢出成 NaN。** candle 0.11.0 自带的 `candle_nn::loss::binary_cross_entropy_with_logit` 是"先 sigmoid 再取 log"的直译版，logit 一大 sigmoid 就饱和、`log(0)` 变 `-inf`——本机实测约 300 轮后损失必然崩成 NaN。改用数值稳定形式 `max(x,0) - x*y + log(1+exp(-|x|))`（即代码里的 `bce_with_logits`）就稳了。

> 💡 **提示**：**细粒度要"更强特征 + 更多数据"。** 小网络只能讲清机制；真要区分近似子类，得上高分辨率、部件/注意力、更强骨干和更干净的标注，别指望小模型出奇迹。

## 本章小结

- **细粒度分类**：区分极像的子类，难点在"细微的局部差异"。思路是高分辨率、局部/部件特征、注意力；强模型还要更强骨干与更多数据。
- **多标签分类**：一张图可同时命中多个标签。与多分类唯一的本质区别是——输出层 **softmax → 每标签一个 sigmoid**，损失 **交叉熵 → BCE**，判定 **argmax → 逐标签阈值**。
- 我们用 candle 小 CNN 真跑通了多标签训练：横条/竖条近乎满分，稀有的亮块 F1 为 0.866，把阈值从 0.5 调到 0.3 可把它救到 0.957。
- 评估**别用单一 accuracy**（稀有标签下傻基线都能到 0.719），要看 per-label F1、macro-F1、mAP，并区分汉明准确率与完全匹配率。
- **层级分类**可看成带"子类为真则父类为真"约束的多标签，需单独报告层级一致率。

## 动手练习

1. **故意用错**：把输出层改成 `softmax` + 交叉熵重新训练，观察"概率和恒为 1、无法同时报多个标签"的现象，体会为什么多标签不能这么干。
2. **逐标签调阈值**：给每个标签各自扫一遍最优阈值，比较"统一 0.5"和"逐标签阈值"两种策略下的 macro-F1。
3. **加剧不平衡**：把亮块出现概率从 0.25 降到 0.08，看它的召回/F1 怎么塌，再用"给正样本更大 loss 权重"把它救回来。
4. **（选做）层级约束**：设计一条父子规则（如"含亮块则必含横条"），给模型输出加一个最小的层级一致性检查，统计违规率。

下一章，我们离开"这张图是什么"，走进 **[第九部分 目标检测](../p09-detection/ch45-detection-principles.md)**——不仅要说"图里有什么"，还要框出"在哪、有几个"。👉 [第 45 章](../p09-detection/ch45-detection-principles.md)
