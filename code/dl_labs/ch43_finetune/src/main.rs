// 第 43 章：微调——冻结第 42 章的预训练骨干，只训练一个新的分类头
// candle 0.11.0，纯 CPU。运行：cargo run --release
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};

const SZ: usize = 28;
// 旧任务 A（骨干预训练用）：竖条 / 横条 / 方块
const TASK_A: [&str; 3] = ["竖条", "横条", "方块"];
// 新任务 B（我们要微调出的"专才"）：4 个全新形状
const TASK_B: [&str; 4] = ["左斜线", "右斜线", "十字", "圆盘"];

// ---------- 确定性随机数（candle CPU 不能 set_seed，随机性自己掌控） ----------
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

// ---------- 合成数据 ----------
fn to_rgb_noisy(g: Vec<f32>, rng: &mut Lcg) -> Vec<f32> {
    let noisy: Vec<f32> = g.iter().map(|&v| (v + (rng.unit() - 0.5) * 0.3).clamp(0.0, 1.0)).collect();
    let mut out = Vec::with_capacity(3 * SZ * SZ);
    for _ in 0..3 { out.extend_from_slice(&noisy); }
    out
}
fn draw_a(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; SZ * SZ];
    match class {
        0 => { let x0 = 4 + rng.range(18); for y in 0..SZ { for x in x0..x0 + 3 { g[y * SZ + x] = 1.0; } } }
        1 => { let y0 = 4 + rng.range(18); for x in 0..SZ { for y in y0..y0 + 3 { g[y * SZ + x] = 1.0; } } }
        _ => { let s = 8; let x0 = 3 + rng.range(SZ - s - 6); let y0 = 3 + rng.range(SZ - s - 6);
               for y in y0..y0 + s { for x in x0..x0 + s { g[y * SZ + x] = 1.0; } } }
    }
    to_rgb_noisy(g, rng)
}
fn draw_b(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; SZ * SZ];
    match class {
        0 => { for i in 0..SZ { let (x, y) = (i, i); for d in 0..2 { let xx = (x + d).min(SZ - 1); g[y * SZ + xx] = 1.0; } } } // 左斜线 '\'
        1 => { for i in 0..SZ { let (x, y) = (SZ - 1 - i, i); for d in 0..2 { let xx = x.saturating_sub(d); g[y * SZ + xx] = 1.0; } } } // 右斜线 '/'
        2 => { let c = SZ / 2; for y in 0..SZ { for x in c - 1..=c + 1 { g[y * SZ + x] = 1.0; } } for x in 0..SZ { for y in c - 1..=c + 1 { g[y * SZ + x] = 1.0; } } } // 十字 '+'
        _ => { let (cx, cy, r) = (SZ as i32 / 2, SZ as i32 / 2, 6i32); // 圆盘
               for y in 0..SZ as i32 { for x in 0..SZ as i32 { if (x - cx).pow(2) + (y - cy).pow(2) <= r * r { g[y as usize * SZ + x as usize] = 1.0; } } } }
    }
    to_rgb_noisy(g, rng)
}
fn dataset(per_class: usize, nc: usize, draw: fn(usize, &mut Lcg) -> Vec<f32>, rng: &mut Lcg) -> (Vec<f32>, Vec<u32>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..per_class { for c in 0..nc { xs.extend(draw(c, rng)); ys.push(c as u32); } }
    (xs, ys)
}
fn normalize(t: &Tensor) -> Result<Tensor> {
    let dev = t.device();
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    let std = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    t.broadcast_sub(&mean)?.broadcast_div(&std)
}
fn batch_to_tensor(xs: &[f32], n: usize, dev: &Device) -> Result<Tensor> {
    normalize(&Tensor::from_vec(xs.to_vec(), (n, 3, SZ, SZ), dev)?)
}

// ---------- 骨干（backbone）：conv1/conv2/fc1，输出 64 维特征。分类头单独放。 ----------
struct Backbone { conv1: Conv2d, conv2: Conv2d, fc1: Linear }
impl Backbone {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 8, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(8, 16, 3, cfg, vb.pp("conv2"))?,
            fc1: linear(16 * 7 * 7, 64, vb.pp("fc1"))?,
        })
    }
    fn features(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?;
        let x = x.max_pool2d(2)?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = x.max_pool2d(2)?;
        let x = x.flatten_from(1)?;
        self.fc1.forward(&x)?.relu()
    }
}
// 用"骨干+头"算准确率
fn acc(bb: &Backbone, head: &Linear, x: &Tensor, y: &[u32]) -> Result<f32> {
    let pred = head.forward(&bb.features(x)?)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
    Ok(pred.iter().zip(y).filter(|(a, b)| a == b).count() as f32 / y.len() as f32)
}

// ===== 预训练：在任务 A 上把骨干训好并存盘（等价于第 42 章，独立跑通用） =====
fn pretrain_and_save(dev: &Device) -> Result<()> {
    let mut rng = Lcg::new(1);
    let (trx, tro) = dataset(60, TASK_A.len(), draw_a, &mut rng);
    let n = tro.len();
    let x = batch_to_tensor(&trx, n, dev)?;
    let y = Tensor::from_vec(tro, (n,), dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let bb = Backbone::load(vb.clone())?;
    let head = linear(64, TASK_A.len(), vb.pp("head"))?; // 旧任务的头（3 类）
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    for _ in 1..=40 {
        let loss = cross_entropy(&head.forward(&bb.features(&x)?)?, &y)?;
        opt.backward_step(&loss)?;
    }
    vm.save("cnn.safetensors")?;
    println!("[预训练] 已在任务 A（{:?}）上训好骨干并存成 cnn.safetensors\n", TASK_A);
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // 第 0 步：准备好"预训练骨干"（第 42 章存下来的那个；这里为独立跑通重建一次）
    pretrain_and_save(&dev)?;

    // ===== 第 1 步：加载骨干，并"冻结"它 =====
    // from_mmaped_safetensors 读进来的是普通常量张量（不是 Var），天然不参与梯度更新 → 就是"冻结"。
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&["cnn.safetensors"], DType::F32, &dev)? };
    let backbone = Backbone::load(vb)?; // 只取 conv1/conv2/fc1，文件里旧的 head 直接忽略
    // 数一数被冻结的骨干参数量
    let dict = candle_core::safetensors::load("cnn.safetensors", &dev)?;
    let frozen: usize = ["conv1", "conv2", "fc1"].iter()
        .flat_map(|p| [format!("{p}.weight"), format!("{p}.bias")])
        .map(|k| dict[&k].elem_count()).sum();
    // 记录骨干某个权重的"指纹"（和值），训练后再比对，证明它没被动过
    let bb_w0 = backbone.conv1.weight().sum_all()?.to_scalar::<f32>()?;

    // ===== 第 2 步：换上新任务的分类头（4 类），这才是"可训练"的部分 =====
    let head_vm = VarMap::new(); // 单独的 VarMap，只装 head 的参数
    let head_vb = VarBuilder::from_varmap(&head_vm, DType::F32, &dev);
    let head = linear(64, TASK_B.len(), head_vb.pp("head"))?;
    det_init(&head_vm, 7)?;
    let trainable: usize = head_vm.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("冻结骨干参数 = {frozen} 个；新分类头可训练参数 = {trainable} 个（{} 个张量）",
        head_vm.all_vars().len());

    // ===== 第 3 步：准备新任务 B 的数据 =====
    let mut rng = Lcg::new(777);
    let (bx, bo) = dataset(60, TASK_B.len(), draw_b, &mut rng); // 240 张训练
    let (bex, beo) = dataset(20, TASK_B.len(), draw_b, &mut rng); // 80 张测试
    let (nb, nbe) = (bo.len(), beo.len());
    let xb = batch_to_tensor(&bx, nb, &dev)?;
    let yb = Tensor::from_vec(bo.clone(), (nb,), &dev)?;
    let xbe = batch_to_tensor(&bex, nbe, &dev)?;

    // ===== 第 4 步：微调前的准确率（新头是随机初始化的 → 约等于瞎猜 1/4） =====
    println!("\n微调前（随机初始化的新头）：训练 acc = {:.1}%   测试 acc = {:.1}%",
        acc(&backbone, &head, &xb, &bo)? * 100.0, acc(&backbone, &head, &xbe, &beo)? * 100.0);

    // ===== 第 5 步：只训练新头（骨干冻结）=====
    // 关键：优化器只拿到 head 的 Var（head_vm.all_vars()）→ 只有新头被更新，骨干纹丝不动。
    let mut opt = AdamW::new(head_vm.all_vars(), ParamsAdamW { lr: 1e-2, ..Default::default() })?;
    println!("\n开始微调（冻结骨干，只训新头，AdamW，lr=1e-2）...");
    for epoch in 1..=60 {
        let feat = backbone.features(&xb)?; // 前向经过冻结骨干
        let loss = cross_entropy(&head.forward(&feat)?, &yb)?;
        opt.backward_step(&loss)?; // 只更新 head
        if epoch % 15 == 0 { println!("  epoch {epoch:>2}  loss = {:.4}", loss.to_scalar::<f32>()?); }
    }

    // ===== 第 6 步：微调后的准确率 =====
    println!("\n微调后（训好的新头）：    训练 acc = {:.1}%   测试 acc = {:.1}%",
        acc(&backbone, &head, &xb, &bo)? * 100.0, acc(&backbone, &head, &xbe, &beo)? * 100.0);

    // 验证骨干确实没被动过：比对训练前后 backbone 内存里 conv1.weight 的和值
    let bb_w1 = backbone.conv1.weight().sum_all()?.to_scalar::<f32>()?;
    println!("\n[冻结验证] 骨干 conv1.weight 和值 微调前={bb_w0:.6} 微调后={bb_w1:.6}（完全一致，骨干未被更新）");
    Ok(())
}
