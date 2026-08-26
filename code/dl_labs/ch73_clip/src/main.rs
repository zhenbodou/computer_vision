// 第 73 章：视觉-语言多模态（CLIP 教学版）
// candle 0.11.0，纯 CPU，无需下载任何模型/数据，一键 `cargo run --release` 跑通。
//
// 核心：把「图像」和「文本」映射进同一个嵌入空间，用对比损失（InfoNCE）
// 把配对的(图,文)拉近、不配对的推开；训练后用「文本提示」做零样本分类。
//
//   图像编码器：小 CNN → D 维嵌入
//   文本编码器：词表 embedding 表 + 均值池化 + 线性投影 → D 维嵌入
//   两者都 L2 归一化，点积即余弦相似度。
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::loss::cross_entropy;
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

const IMG: usize = 16; // 图像边长
const D: usize = 32; // 共享嵌入维度
const E: usize = 16; // 词向量维度
const VOCAB: usize = 10; // 词表大小
const L: usize = 5; // 每条描述的 token 数（"a photo of <color> <shape>"）
const INV_TEMP: f64 = 10.0; // 温度倒数 1/τ（缩放余弦相似度，让 softmax 更"尖"）

// 词表（token id → 词）：
// 0 <pad>  1 a  2 photo  3 of  4 red  5 green  6 blue  7 vertical  8 horizontal  9 square
const COLOR_TOK: [u32; 3] = [4, 5, 6]; // red / green / blue
const SHAPE_TOK: [u32; 3] = [7, 8, 9]; // vertical / horizontal / square
const COLOR_NAME: [&str; 3] = ["红", "绿", "蓝"];
const SHAPE_NAME: [&str; 3] = ["竖条", "横条", "方块"];

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

// ---------- 合成数据：16x16 RGB，黑底上画一个带颜色的形状 ----------
// class = color*3 + shape，共 9 类；返回 CHW 顺序的 [3*IMG*IMG] 像素，值域 [0,1]。
fn draw(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let color = class / 3;
    let shape = class % 3;
    let mut g = vec![0f32; 3 * IMG * IMG];
    let ch = color * IMG * IMG; // 该颜色对应的通道起点（R/G/B 通道）
    let val = 0.85 + rng.unit() * 0.15; // 轻微亮度抖动
    let mut set = |x: usize, y: usize| { g[ch + y * IMG + x] = val; };
    match shape {
        0 => { // 竖条：宽 4，随机 x
            let w = 4; let x0 = rng.range(IMG - w);
            for y in 0..IMG { for x in x0..x0 + w { set(x, y); } }
        }
        1 => { // 横条：高 4，随机 y
            let h = 4; let y0 = rng.range(IMG - h);
            for x in 0..IMG { for y in y0..y0 + h { set(x, y); } }
        }
        _ => { // 方块：8x8，随机位置
            let s = 8; let x0 = rng.range(IMG - s); let y0 = rng.range(IMG - s);
            for y in y0..y0 + s { for x in x0..x0 + s { set(x, y); } }
        }
    }
    g
}
// class → 文本提示的 token 序列 "a photo of <color> <shape>"
fn prompt(class: usize) -> [u32; L] {
    let color = class / 3;
    let shape = class % 3;
    [1, 2, 3, COLOR_TOK[color], SHAPE_TOK[shape]]
}
fn class_name(class: usize) -> String {
    format!("{}{}", COLOR_NAME[class / 3], SHAPE_NAME[class % 3])
}

// 把一张 CHW 的 [0,1] 图转成 interleaved RGB 图
fn chw_to_rgb(g: &[f32]) -> RgbImage {
    let mut raw = vec![0u8; 3 * IMG * IMG];
    for y in 0..IMG {
        for x in 0..IMG {
            for c in 0..3 {
                raw[(y * IMG + x) * 3 + c] = (g[c * IMG * IMG + y * IMG + x].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    RgbImage::from_raw(IMG as u32, IMG as u32, raw).unwrap()
}
// 存一张 3×3 数据集缩略图（行=颜色，列=形状），让读者看清 (图,文) 长什么样
fn save_montage(path: &str) {
    let (disp, gap) = (64u32, 6u32);
    let w = 3 * disp + 2 * gap;
    let mut canvas = RgbImage::from_pixel(w, w, Rgb([245, 245, 245]));
    let mut rng = Lcg::new(2024);
    for color in 0..3 {
        for shape in 0..3 {
            let img = chw_to_rgb(&draw(color * 3 + shape, &mut rng));
            let big = image::imageops::resize(&img, disp, disp, FilterType::Nearest);
            let (px, py) = (shape as u32 * (disp + gap), color as u32 * (disp + gap));
            image::imageops::overlay(&mut canvas, &big, px as i64, py as i64);
        }
    }
    canvas.save(path).unwrap();
}

// ---------- 图像编码器：小 CNN → D 维 ----------
struct ImageEncoder { c1: Conv2d, c2: Conv2d, fc: Linear }
impl ImageEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: conv2d(3, 8, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(8, 16, 3, cfg, vb.pp("c2"))?,
            fc: linear(16 * 4 * 4, D, vb.pp("fc"))?, // 16x16 →池化→8x8 →池化→4x4
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?.avg_pool2d(2)?; // [N,8,8,8]
        let h = self.c2.forward(&h)?.relu()?.avg_pool2d(2)?; // [N,16,4,4]
        self.fc.forward(&h.flatten_from(1)?) // [N,D]
    }
}

// ---------- 文本编码器：词表 embedding + 均值池化 + 线性投影 → D 维 ----------
struct TextEncoder { emb: Tensor, fc: Linear }
impl TextEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            emb: vb.get((VOCAB, E), "emb")?, // 词向量表 [VOCAB, E]，可训练
            fc: linear(E, D, vb.pp("fc"))?,
        })
    }
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (n, l) = tokens.dims2()?;
        let flat = tokens.reshape((n * l,))?; // [N*L]
        let looked = self.emb.index_select(&flat, 0)?; // 查表 [N*L, E]
        let seq = looked.reshape((n, l, E))?; // [N, L, E]
        let pooled = seq.mean(1)?; // 均值池化（词袋）→ [N, E]
        self.fc.forward(&pooled) // [N, D]
    }
}

// L2 归一化（加小 eps 防除零）：归一化后点积 = 余弦相似度
fn l2n(x: &Tensor) -> Result<Tensor> {
    let denom = x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, 1e-12)?.sqrt()?;
    x.broadcast_div(&denom)
}

// 对称 InfoNCE 对比损失
fn clip_loss(img_emb: &Tensor, txt_emb: &Tensor, dev: &Device) -> Result<Tensor> {
    let n = img_emb.dim(0)?;
    let logits = img_emb.matmul(&txt_emb.t()?)?.affine(INV_TEMP, 0.0)?; // [N,N] 相似度/温度
    let target = Tensor::arange(0u32, n as u32, dev)?; // 对角线下标（U32）
    let li = cross_entropy(&logits, &target)?; // 图→文
    let lt = cross_entropy(&logits.t()?.contiguous()?, &target)?; // 文→图
    (&li + &lt)?.affine(0.5, 0.0) // 对称平均
}

// 一批：训练类别各取一个实例（保证 batch 内文本互不相同，符合 InfoNCE 假设）
fn make_batch(classes: &[usize], rng: &mut Lcg, dev: &Device) -> Result<(Tensor, Tensor)> {
    let b = classes.len();
    let mut imgs = Vec::with_capacity(b * 3 * IMG * IMG);
    let mut toks: Vec<u32> = Vec::with_capacity(b * L);
    for &c in classes {
        imgs.extend(draw(c, rng));
        toks.extend_from_slice(&prompt(c));
    }
    let x = Tensor::from_vec(imgs, (b, 3, IMG, IMG), dev)?;
    let t = Tensor::from_vec(toks, (b, L), dev)?;
    Ok((x, t))
}

// 训练一个 CLIP（返回训练好的两个编码器 + 其 VarMap）
fn train_clip(classes: &[usize], steps: usize, seed: u64, dev: &Device)
    -> Result<(ImageEncoder, TextEncoder)> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let img_enc = ImageEncoder::load(vb.pp("img"))?;
    let txt_enc = TextEncoder::load(vb.pp("txt"))?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
    let mut rng = Lcg::new(seed);
    for step in 1..=steps {
        let (x, t) = make_batch(classes, &mut rng, dev)?;
        let ie = l2n(&img_enc.forward(&x)?)?;
        let te = l2n(&txt_enc.forward(&t)?)?;
        let loss = clip_loss(&ie, &te, dev)?;
        opt.backward_step(&loss)?;
        if step % 100 == 0 || step == 1 {
            println!("  step {step:>4}  对比损失 = {:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok((img_enc, txt_enc))
}

// 对所有 9 类各生成 per_class 张测试图，返回 (图像张量[T,3,H,W], 每张的真实类别)
fn make_test(per_class: usize, seed: u64, dev: &Device) -> Result<(Tensor, Vec<usize>)> {
    let mut rng = Lcg::new(seed);
    let mut imgs = Vec::new();
    let mut labels = Vec::new();
    for c in 0..9 {
        for _ in 0..per_class {
            imgs.extend(draw(c, &mut rng));
            labels.push(c);
        }
    }
    let t = labels.len();
    let x = Tensor::from_vec(imgs, (t, 3, IMG, IMG), dev)?;
    Ok((x, labels))
}

// 9 类文本提示 → [9, L] token 张量
fn all_prompts(dev: &Device) -> Result<Tensor> {
    let mut toks: Vec<u32> = Vec::new();
    for c in 0..9 { toks.extend_from_slice(&prompt(c)); }
    Tensor::from_vec(toks, (9, L), dev)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // 存一张数据集缩略图，直观看到 9 类 (图, 文) 配对
    save_montage("out_ch73_dataset.png");
    println!("已存数据集缩略图 out_ch73_dataset.png（行=红/绿/蓝，列=竖条/横条/方块）");
    println!("每张图配一句文本提示，例如 [1,2,3,4,7] = \"a photo of red vertical\"\n");

    // ===== 实验 A：9 类全部参与对比训练，做零样本分类 + 双向检索 =====
    println!("==== 实验 A：CLIP 教学版（9 类全参与训练）→ 零样本分类 ====");
    let all: Vec<usize> = (0..9).collect();
    let (ie_a, te_a) = train_clip(&all, 800, 1, &dev)?;

    // 零样本分类：测试图编码 vs 9 类文本提示编码，取最近
    let (x_test, labels) = make_test(20, 999, &dev)?;
    let img_emb = l2n(&ie_a.forward(&x_test)?)?; // [T,D]
    let txt_emb = l2n(&te_a.forward(&all_prompts(&dev)?)?)?; // [9,D]
    let sims = img_emb.matmul(&txt_emb.t()?)?; // [T,9]
    let pred = sims.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let correct = pred.iter().zip(&labels).filter(|(p, l)| **p as usize == **l).count();
    println!("  零样本分类准确率 = {}/{} = {:.1}%（随机猜 = {:.1}%）",
        correct, labels.len(), 100.0 * correct as f32 / labels.len() as f32, 100.0 / 9.0);

    // 双向检索 recall@1：9 类各取 1 张查询图 + 9 条文本，算 9x9 相似度
    let (x_q, ql) = make_test(1, 555, &dev)?;
    let qi = l2n(&ie_a.forward(&x_q)?)?; // [9,D]
    let s = qi.matmul(&txt_emb.t()?)?; // [9,9] image→text
    let i2t = s.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let t2i = s.t()?.contiguous()?.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let r_i2t = i2t.iter().zip(&ql).filter(|(p, l)| **p as usize == **l).count();
    let r_t2i = t2i.iter().enumerate().filter(|(i, p)| **p as usize == *i).count();
    println!("  图→文检索 recall@1 = {}/9 ；文→图检索 recall@1 = {}/9", r_i2t, r_t2i);

    // ===== 实验 B：留出 3 个"颜色×形状"组合不训练，测组合零样本泛化 =====
    println!("\n==== 实验 B：组合零样本泛化（留出 3 个组合不参与对比训练）====");
    // 留出：蓝方块(8)、绿横条(4)、红竖条(0)；其颜色与形状都在别的训练组合里出现过
    let held: Vec<usize> = vec![0, 4, 8];
    let train: Vec<usize> = (0..9).filter(|c| !held.contains(c)).collect();
    print!("  训练组合：");
    for &c in &train { print!("{} ", class_name(c)); }
    print!("\n  留出组合：");
    for &c in &held { print!("{} ", class_name(c)); }
    println!();
    let (ie_b, te_b) = train_clip(&train, 800, 1, &dev)?;

    let (x_test, labels) = make_test(20, 999, &dev)?;
    let img_emb = l2n(&ie_b.forward(&x_test)?)?;
    let txt_emb = l2n(&te_b.forward(&all_prompts(&dev)?)?)?;
    let sims = img_emb.matmul(&txt_emb.t()?)?;
    let pred = sims.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let (mut seen_ok, mut seen_n) = (0, 0);
    let (mut un_ok, mut un_n, mut un_color, mut un_shape) = (0, 0, 0, 0);
    for (p, l) in pred.iter().zip(&labels) {
        let (p, l) = (*p as usize, *l);
        if held.contains(&l) {
            un_n += 1;
            if p == l { un_ok += 1; }
            if p / 3 == l / 3 { un_color += 1; } // 颜色认对？
            if p % 3 == l % 3 { un_shape += 1; } // 形状认对？
        } else {
            seen_n += 1;
            if p == l { seen_ok += 1; }
        }
    }
    let pc = |a: usize, b: usize| 100.0 * a as f32 / b as f32;
    println!("  已见组合零样本准确率 = {}/{} = {:.1}%", seen_ok, seen_n, pc(seen_ok, seen_n));
    println!("  留出组合零样本准确率 = {}/{} = {:.1}%（模型从未见过这些图文配对！）", un_ok, un_n, pc(un_ok, un_n));
    println!("    └─ 拆开看：颜色认对 {}/{} = {:.1}%，形状认对 {}/{} = {:.1}%",
        un_color, un_n, pc(un_color, un_n), un_shape, un_n, pc(un_shape, un_n));
    for &hc in &held {
        let mut dist = [0usize; 9];
        for (p, l) in pred.iter().zip(&labels) { if *l == hc { dist[*p as usize] += 1; } }
        print!("    [{}] 预测→ ", class_name(hc));
        for c in 0..9 { if dist[c] > 0 { print!("{}×{} ", class_name(c), dist[c]); } }
        println!();
    }
    Ok(())
}
