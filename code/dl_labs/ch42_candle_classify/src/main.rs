// 第 42 章：用 candle 加载（我们自己"预训练"的）小 CNN 做图像分类
// candle 0.11.0，纯 CPU，无需下载任何大模型。
// 运行：cargo run --release
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::softmax;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

const SZ: usize = 28; // 模型输入边长（正方形）
const CLASSES: [&str; 3] = ["竖条", "横条", "方块"]; // 类别索引 → 名称（就是 ImageNet 里 1000 类标签表的迷你版）

// ---------- 极简随机数发生器（固定种子、可复现） ----------
// 🕳️ candle 0.11 的 CPU 后端无法 set_seed（会直接报错），所以随机性我们自己掌控。
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

/// 确定性地初始化 VarMap 里的所有权重：bias 置 0，weight 取 [-lim, lim] 均匀分布，
/// lim = 1/sqrt(fan_in)（PyTorch 默认风格）。因为 candle CPU 不能播种，只能这样保证每次跑结果一致。
fn det_init(vm: &VarMap, seed: u64) -> Result<()> {
    let mut rng = Lcg::new(seed);
    let data = vm.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort(); // 排序保证遍历顺序固定 → 随机序列固定
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

// ---------- 合成训练数据：竖条 / 横条 / 方块（白色图形 + 噪声，三通道） ----------
fn to_rgb_noisy(g: Vec<f32>, rng: &mut Lcg) -> Vec<f32> {
    // 加均匀噪声并钳到 [0,1]，再把单通道复制成 3 通道（CHW 顺序：先整片 R，再 G，再 B）
    let noisy: Vec<f32> = g.iter().map(|&v| (v + (rng.unit() - 0.5) * 0.3).clamp(0.0, 1.0)).collect();
    let mut out = Vec::with_capacity(3 * SZ * SZ);
    for _ in 0..3 { out.extend_from_slice(&noisy); }
    out
}
fn draw(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; SZ * SZ];
    match class {
        0 => { let x0 = 4 + rng.range(18); for y in 0..SZ { for x in x0..x0 + 3 { g[y * SZ + x] = 1.0; } } } // 竖条
        1 => { let y0 = 4 + rng.range(18); for x in 0..SZ { for y in y0..y0 + 3 { g[y * SZ + x] = 1.0; } } } // 横条
        _ => { let s = 8; let x0 = 3 + rng.range(SZ - s - 6); let y0 = 3 + rng.range(SZ - s - 6);            // 方块
               for y in y0..y0 + s { for x in x0..x0 + s { g[y * SZ + x] = 1.0; } } }
    }
    to_rgb_noisy(g, rng)
}
fn dataset(per_class: usize, rng: &mut Lcg) -> (Vec<f32>, Vec<u32>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..per_class { for c in 0..CLASSES.len() { xs.extend(draw(c, rng)); ys.push(c as u32); } }
    (xs, ys)
}

// ---------- 预处理：按通道标准化（务必与训练时一致！） ----------
// 我们这个模型训练时用 mean=std=0.5（把 [0,1] 映射到 [-1,1]）。
// 真正的 ImageNet 模型用 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]（见第 31 章）——
// 关键规则只有一条：推理必须复刻训练时的预处理，配错了精度会莫名其妙地掉。
fn normalize(t: &Tensor) -> Result<Tensor> {
    let dev = t.device();
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    t.broadcast_sub(&mean)?.broadcast_div(&std) // 广播：[N,3,H,W] 减/除 [1,3,1,1]
}
fn batch_to_tensor(xs: &[f32], n: usize, dev: &Device) -> Result<Tensor> {
    normalize(&Tensor::from_vec(xs.to_vec(), (n, 3, SZ, SZ), dev)?)
}

// ---------- 小 CNN：conv1/conv2/fc1 是"骨干"，head 是"分类头"（第 43 章会冻结骨干、换 head） ----------
struct SmallCnn { conv1: Conv2d, conv2: Conv2d, fc1: Linear, head: Linear }
impl SmallCnn {
    fn load(num_classes: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() }; // padding=1 → 卷积不改变边长
        Ok(Self {
            conv1: conv2d(3, 8, 3, cfg, vb.pp("conv1"))?,   // 3→8 通道，3x3
            conv2: conv2d(8, 16, 3, cfg, vb.pp("conv2"))?,  // 8→16 通道，3x3
            fc1: linear(16 * 7 * 7, 64, vb.pp("fc1"))?,     // 展平 784 → 64 维特征
            head: linear(64, num_classes, vb.pp("head"))?,  // 64 → 类别数（logits）
        })
    }
    // 骨干：输出 64 维特征
    fn features(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?; // [N,3,28,28] → [N,8,28,28]
        let x = x.max_pool2d(2)?;               // → [N,8,14,14]
        let x = self.conv2.forward(&x)?.relu()?;// → [N,16,14,14]
        let x = x.max_pool2d(2)?;               // → [N,16,7,7]
        let x = x.flatten_from(1)?;             // 展平 → [N,784]
        self.fc1.forward(&x)?.relu()            // → [N,64]
    }
    // 分类头：特征 → logits
    fn forward(&self, x: &Tensor) -> Result<Tensor> { self.head.forward(&self.features(x)?) }
}

fn accuracy(net: &SmallCnn, x: &Tensor, y: &[u32]) -> Result<f32> {
    let pred = net.forward(x)?.argmax(D::Minus1)?.to_vec1::<u32>()?; // 每行最大 logit 的下标 = 预测类别
    Ok(pred.iter().zip(y).filter(|(a, b)| a == b).count() as f32 / y.len() as f32)
}

// 用 image 库画一张彩色图形图（白图形、深色背景），尺寸任意——用于演示"真实图像先 resize"
fn draw_rgb(class: usize, sz: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(sz, sz, Rgb([20, 20, 20]));
    let white = Rgb([240, 240, 240]);
    match class {
        0 => { let x0 = sz * 4 / 10; for y in 0..sz { for x in x0..x0 + sz / 9 { img.put_pixel(x, y, white); } } }
        1 => { let y0 = sz * 4 / 10; for x in 0..sz { for y in y0..y0 + sz / 9 { img.put_pixel(x, y, white); } } }
        _ => { let a = sz / 4; let b = sz * 3 / 4; for y in a..b { for x in a..b { img.put_pixel(x, y, white); } } }
    }
    img
}

// 完整推理预处理：一张 RgbImage → 模型输入张量 [1,3,SZ,SZ]
// 步骤：resize 到模型尺寸 → HWC→CHW + 除以 255 归一化 → 加 batch 维 → 按通道标准化（见第 31 章）
fn preprocess(img: &RgbImage, dev: &Device) -> Result<Tensor> {
    let small = image::imageops::resize(img, SZ as u32, SZ as u32, FilterType::Triangle); // ① resize
    let (w, h) = (small.width() as usize, small.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = small.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * h * w + y * w + x] = p[c] as f32 / 255.0; // ② HWC→CHW 同时 /255 归一化
            }
        }
    }
    let t = Tensor::from_vec(data, (1, 3, h, w), dev)?; // ③ 加 batch 维 → [1,3,H,W]
    normalize(&t) // ④ 按通道标准化
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：搭小 CNN 并快速训练，得到"真会认三种形状"的权重 =====
    // （训练循环的原理见第 40/41 章；本章重点是随后的"存 → 取 → 推理"这条链路）
    let mut rng = Lcg::new(1);
    let (trx, tro) = dataset(60, &mut rng); // 180 张训练
    let (tex, teo) = dataset(20, &mut rng); // 60 张测试
    let (ntr, nte) = (tro.len(), teo.len());
    let xtr = batch_to_tensor(&trx, ntr, &dev)?;
    let ytr = Tensor::from_vec(tro.clone(), (ntr,), &dev)?; // 标签是 U32
    let xte = batch_to_tensor(&tex, nte, &dev)?;

    let vm = VarMap::new(); // VarMap 持有所有可训练参数（Var）
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = SmallCnn::load(CLASSES.len(), vb)?;
    det_init(&vm, 42)?; // 确定性初始化，保证可复现

    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("开始训练小 CNN（合成三类形状，AdamW，lr=1e-3）...");
    for epoch in 1..=40 {
        let loss = cross_entropy(&net.forward(&xtr)?, &ytr)?; // 交叉熵内部含 softmax
        opt.backward_step(&loss)?; // 反向 + 更新，一行搞定
        if epoch % 8 == 0 { println!("  epoch {epoch:>2}  loss = {:.4}", loss.to_scalar::<f32>()?); }
    }
    println!("训练完成：训练准确率 = {:.1}%   测试准确率 = {:.1}%\n",
        accuracy(&net, &xtr, &tro)? * 100.0, accuracy(&net, &xte, &teo)? * 100.0);

    // ===== 第 2 步：把权重存成 .safetensors =====
    vm.save("cnn.safetensors")?; // VarMap::save 内部就是调用 candle_core::safetensors::save
    println!("已保存 cnn.safetensors");
    // 看看文件里到底有哪些张量（这就是"模型权重"的真身）：
    let dict = candle_core::safetensors::load("cnn.safetensors", &dev)?; // 直接读成 name→Tensor 字典
    let mut keys: Vec<(String, Vec<usize>)> = dict.iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    keys.sort();
    println!("文件内张量：");
    for (k, shape) in &keys { println!("  {k:<13} 形状 {shape:?}"); }
    println!();

    // ===== 第 3 步：把权重加载回来（和加载 HuggingFace 上 ResNet 的 .safetensors 完全同一套 API） =====
    // ⚠️ from_mmaped_safetensors 是 unsafe（内存映射文件）
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["cnn.safetensors"], DType::F32, &dev)? };
    let model = SmallCnn::load(CLASSES.len(), vb2)?; // vb.pp("conv1") 会去文件里找 "conv1.weight"/"conv1.bias"
    println!("已从 safetensors 重新加载模型\n");

    // ===== 第 4 步：完整推理流水线，打印 top-1 与 softmax 概率 =====
    println!("推理（每类合成一张 64x64 测试图，先 resize 到 {SZ}x{SZ}）：");
    for true_c in 0..CLASSES.len() {
        let img = draw_rgb(true_c, 64);
        let input = preprocess(&img, &dev)?; // → [1,3,28,28]
        let logits = model.forward(&input)?; // → [1,3]
        let probs = softmax(&logits, D::Minus1)?.to_vec2::<f32>()?[0].clone(); // → 和为 1 的概率
        let top = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0] as usize;
        let dist: Vec<String> = probs.iter().enumerate().map(|(i, p)| format!("{}={:.3}", CLASSES[i], p)).collect();
        println!("  真实={:<4} → top-1={:<4} (prob={:.3})   分布 [{}]",
            CLASSES[true_c], CLASSES[top], probs[top], dist.join(", "));
    }
    Ok(())
}
