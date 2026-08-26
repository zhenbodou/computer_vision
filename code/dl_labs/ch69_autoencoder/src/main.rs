// 第 69 章：自编码器与表示学习
// candle 0.11.0，纯 CPU，无需下载任何模型/数据。
// 运行：cargo run --release
//
// 本程序依次做三件事：
//   1) 训练一个「欠完备自编码器」（16x16 灰度图 → 8 维瓶颈 → 重建），报重建 MSE/PSNR，存原图vs重建对照 PNG
//   2) 表示学习：冻结训练好的编码器，用它输出的 8 维 latent 当特征训一个线性分类器（对比直接用 256 原始像素）
//   3) 去噪自编码器（DAE）：输入加噪、目标是干净图，报去噪前后 PSNR，存 噪声图|DAE输出|干净图 对照 PNG
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::{GrayImage, Luma};

const SZ: usize = 16; // 图像边长（正方形灰度）
const LATENT: usize = 8; // 瓶颈（latent）维度：远小于 256 像素 → 强迫压缩
const CLASSES: [&str; 3] = ["竖条", "横条", "方块"];

// ---------- 极简随机数发生器（固定种子、可复现） ----------
// 🕳️ candle 0.11 的 CPU 后端无法 set_seed（会直接报错），随机性我们自己掌控。
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

/// 确定性初始化 VarMap 里所有权重：bias 置 0，weight 取 [-lim,lim] 均匀分布，lim=1/sqrt(fan_in)。
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

// ---------- 合成灰度图形：竖条 / 横条 / 方块（白图形、黑背景、位置随机） ----------
fn draw_shape(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; SZ * SZ];
    match class {
        0 => { let x0 = 3 + rng.range(SZ - 7); for y in 0..SZ { for x in x0..x0 + 3 { g[y * SZ + x] = 1.0; } } } // 竖条
        1 => { let y0 = 3 + rng.range(SZ - 7); for x in 0..SZ { for y in y0..y0 + 3 { g[y * SZ + x] = 1.0; } } } // 横条
        _ => { let s = 6; let x0 = 2 + rng.range(SZ - 9); let y0 = 2 + rng.range(SZ - 9);                        // 方块
               for y in y0..y0 + s { for x in x0..x0 + s { g[y * SZ + x] = 1.0; } } }
    }
    g
}
// 生成一批（无标签用得着 xs，有标签也返回 ys）。样本按 [c0,c1,c2,c0,c1,c2,...] 交替排列。
fn shape_dataset(per_class: usize, rng: &mut Lcg) -> (Vec<f32>, Vec<u32>) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..per_class {
        for c in 0..CLASSES.len() { xs.extend(draw_shape(c, rng)); ys.push(c as u32); }
    }
    (xs, ys)
}
fn to_tensor(xs: &[f32], n: usize, dev: &Device) -> Result<Tensor> {
    // 图像值本就在 [0,1]，直接喂（解码器末端是 sigmoid，输出也在 [0,1]，与目标同域，不需要标准化）
    Tensor::from_vec(xs.to_vec(), (n, 1, SZ, SZ), dev)
}

// ---------- 编码器：conv→pool→conv→pool→flatten→fc，输出 LATENT 维 latent ----------
struct Encoder { conv1: Conv2d, conv2: Conv2d, fc: Linear }
impl Encoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() }; // padding=1 → 卷积不改变边长
        Ok(Self {
            conv1: conv2d(1, 8, 3, cfg, vb.pp("conv1"))?,     // 1→8 通道
            conv2: conv2d(8, 16, 3, cfg, vb.pp("conv2"))?,    // 8→16 通道
            fc: linear(16 * 4 * 4, LATENT, vb.pp("fc"))?,     // 256 → 8（瓶颈）
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?; // [N,1,16,16]→[N,8,16,16]
        let x = x.max_pool2d(2)?;               // →[N,8,8,8]
        let x = self.conv2.forward(&x)?.relu()?;// →[N,16,8,8]
        let x = x.max_pool2d(2)?;               // →[N,16,4,4]
        let x = x.flatten_from(1)?;             // →[N,256]
        self.fc.forward(&x)                     // →[N,8]（latent，不加激活）
    }
}
// ---------- 解码器：fc→reshape→上采样→conv→上采样→conv→sigmoid，重建回 [N,1,16,16] ----------
struct Decoder { fc: Linear, conv1: Conv2d, conv2: Conv2d }
impl Decoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            fc: linear(LATENT, 16 * 4 * 4, vb.pp("fc"))?,     // 8 → 256
            conv1: conv2d(16, 8, 3, cfg, vb.pp("conv1"))?,    // 16→8 通道
            conv2: conv2d(8, 1, 3, cfg, vb.pp("conv2"))?,     // 8→1 通道（输出单通道图）
        })
    }
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let n = z.dim(0)?;
        let x = self.fc.forward(z)?.relu()?;    // [N,8]→[N,256]
        let x = x.reshape((n, 16, 4, 4))?;      // →[N,16,4,4]
        // candle 无转置卷积算子 → 用「最近邻上采样 + 卷积」实现上采样（讲清即可）
        let x = x.upsample_nearest2d(8, 8)?;    // →[N,16,8,8]
        let x = self.conv1.forward(&x)?.relu()?;// →[N,8,8,8]
        let x = x.upsample_nearest2d(SZ, SZ)?;  // →[N,8,16,16]
        let x = self.conv2.forward(&x)?;        // →[N,1,16,16]
        sigmoid(&x)                             // 压到 [0,1]，与目标图同域
    }
}
struct AutoEncoder { enc: Encoder, dec: Decoder }
impl AutoEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        // enc.* / dec.* 前缀 → safetensors 键名形如 enc.conv1.weight、dec.fc.weight
        Ok(Self { enc: Encoder::load(vb.pp("enc"))?, dec: Decoder::load(vb.pp("dec"))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> { self.dec.forward(&self.enc.forward(x)?) }
}

// ---------- PSNR（纯手写）：图像值域 [0,1] → MAX=1 → PSNR = -10 log10(MSE) ----------
fn mse_psnr(pred: &Tensor, target: &Tensor) -> Result<(f32, f32)> {
    let mse = (pred - target)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let psnr = if mse <= 1e-12 { 99.0 } else { -10.0 * mse.log10() };
    Ok((mse, psnr))
}

// ---------- 图像保存：把 [N,1,16,16] 里某几张拼成对照大图 ----------
fn tile_gray(pix: &[f32]) -> GrayImage {
    GrayImage::from_fn(SZ as u32, SZ as u32, |x, y| {
        Luma([(pix[y as usize * SZ + x as usize].clamp(0.0, 1.0) * 255.0) as u8])
    })
}
fn upscale(img: &GrayImage, factor: u32) -> GrayImage {
    image::imageops::resize(img, img.width() * factor, img.height() * factor, image::imageops::FilterType::Nearest)
}
// rows：每行若干张小图（Vec<f32>，长度 SZ*SZ），横向排开；行与行纵向堆叠。
fn save_grid(path: &str, rows: &[Vec<Vec<f32>>]) {
    let factor = 6u32;
    let cell = SZ as u32 * factor;
    let gap = 6u32;
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(1) as u32;
    let nrows = rows.len() as u32;
    let w = cols * cell + (cols - 1) * gap;
    let h = nrows * cell + (nrows - 1) * gap;
    let mut canvas = GrayImage::from_pixel(w, h, Luma([90])); // 灰底便于看清边界
    for (r, row) in rows.iter().enumerate() {
        let y = r as u32 * (cell + gap);
        for (c, pix) in row.iter().enumerate() {
            let x = c as u32 * (cell + gap);
            let tile = upscale(&tile_gray(pix), factor);
            image::imageops::overlay(&mut canvas, &tile, x as i64, y as i64);
        }
    }
    canvas.save(path).unwrap();
}
fn slice_of(flat: &[f32], i: usize) -> Vec<f32> { flat[i * SZ * SZ..(i + 1) * SZ * SZ].to_vec() }

// ========== 1) 训练欠完备自编码器 ==========
fn train_autoencoder(dev: &Device) -> Result<()> {
    println!("==== 1) 欠完备自编码器：16x16 灰度 → 瓶颈 {LATENT} 维 → 重建（MSE 损失） ====");
    let mut rng = Lcg::new(1);
    let (trx, _) = shape_dataset(60, &mut rng); // 180 张（无标签！自监督）
    let (tex, _) = shape_dataset(20, &mut rng); // 60 张测试
    let (ntr, nte) = (180, 60);
    let xtr = to_tensor(&trx, ntr, dev)?;
    let xte = to_tensor(&tex, nte, dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let ae = AutoEncoder::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    for epoch in 1..=400 {
        let recon = ae.forward(&xtr)?;                 // 前向：编码→瓶颈→解码
        let loss = (&recon - &xtr)?.sqr()?.mean_all()?; // MSE 重建损失：目标就是输入自己
        opt.backward_step(&loss)?;
        if epoch % 50 == 0 { println!("  epoch {epoch:>3}  重建 MSE = {:.5}", loss.to_scalar::<f32>()?); }
    }
    let recon_te = ae.forward(&xte)?;
    let (mse, psnr) = mse_psnr(&recon_te, &xte)?;
    println!("  测试集重建：MSE = {mse:.5}，PSNR = {psnr:.2} dB（把 256 个像素压进 {LATENT} 个数再还原）");
    vm.save("ae.safetensors")?;
    println!("  已存权重 ae.safetensors（供第 2 步冻结编码器复用）");

    // 每类挑第一张，存 原图 vs 重建 对照
    let flat = xte.flatten_all()?.to_vec1::<f32>()?;
    let rflat = recon_te.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?;
    let rows: Vec<Vec<Vec<f32>>> = (0..3).map(|c| vec![slice_of(&flat, c), slice_of(&rflat, c)]).collect();
    save_grid("out_ch69_ae_recon.png", &rows);
    println!("  已存对照图 out_ch69_ae_recon.png（每行：左=原图，右=重建）\n");
    Ok(())
}

// ========== 2) 表示学习：冻结编码器 + 线性探针 ==========
fn train_linear_probe(xtr: &Tensor, ytr: &Tensor, xte: &Tensor, yte: &[u32], in_dim: usize, dev: &Device, name: &str) -> Result<f32> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let clf = linear(in_dim, CLASSES.len(), vb.pp("probe"))?;
    det_init(&vm, 7)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-2, ..Default::default() })?;
    for _ in 1..=200 {
        let loss = cross_entropy(&clf.forward(xtr)?, ytr)?;
        opt.backward_step(&loss)?;
    }
    let pred = clf.forward(xte)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let acc = pred.iter().zip(yte).filter(|(a, b)| a == b).count() as f32 / yte.len() as f32;
    println!("  [{name:<12}] 线性分类器测试 acc = {:.1}%", acc * 100.0);
    Ok(acc)
}
fn representation_probe(dev: &Device) -> Result<()> {
    println!("==== 2) 表示学习：冻结编码器，用 8 维 latent 训练线性分类器（少样本，每类 8 张） ====");
    // from_mmaped_safetensors 读进来是常量张量（非 Var）→ 天然冻结；文件里 dec.* 键被静默忽略
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&["ae.safetensors"], DType::F32, dev)? };
    let enc = Encoder::load(vb.pp("enc"))?;
    let mut rng = Lcg::new(555);
    let (trx, tro) = shape_dataset(8, &mut rng);  // 24 张有标签（少样本）
    let (tex, teo) = shape_dataset(40, &mut rng); // 120 张测试
    let (ntr, nte) = (24, 120);
    let xtr = to_tensor(&trx, ntr, dev)?;
    let xte = to_tensor(&tex, nte, dev)?;
    let ytr = Tensor::from_vec(tro, (ntr,), dev)?;
    // 用冻结编码器把图片压成 8 维特征（只算一次；编码器是常量，不产生梯度）
    let ztr = enc.forward(&xtr)?; // [24,8]
    let zte = enc.forward(&xte)?; // [120,8]
    let acc_lat = train_linear_probe(&ztr, &ytr, &zte, &teo, LATENT, dev, "latent 8 维")?;
    // 基线：直接拿 256 个原始像素训同样的线性分类器
    let rtr = xtr.flatten_from(1)?; // [24,256]
    let rte = xte.flatten_from(1)?; // [120,256]
    let acc_raw = train_linear_probe(&rtr, &ytr, &rte, &teo, SZ * SZ, dev, "raw 256 像素")?;
    println!("  → 无监督学到的 8 维表示 acc={:.1}%  vs  256 原始像素 acc={:.1}%（少样本下低维表示更抗过拟合）\n",
        acc_lat * 100.0, acc_raw * 100.0);
    Ok(())
}

// ========== 3) 去噪自编码器（DAE） ==========
fn train_dae(dev: &Device) -> Result<()> {
    println!("==== 3) 去噪自编码器（DAE）：输入=加噪图，目标=干净图 ====");
    const SIGMA: f32 = 0.4; // 噪声强度（均匀噪声半幅）
    let mut rng = Lcg::new(2024);
    let per = 60;
    let (clean_tr, _) = shape_dataset(per, &mut rng);
    let n = per * 3;
    let noisy_tr: Vec<f32> = clean_tr.iter().map(|&v| (v + (rng.unit() - 0.5) * 2.0 * SIGMA).clamp(0.0, 1.0)).collect();
    let xin = to_tensor(&noisy_tr, n, dev)?;   // 加噪输入
    let xtar = to_tensor(&clean_tr, n, dev)?;  // 干净目标
    let (clean_te, _) = shape_dataset(20, &mut rng);
    let nte = 60;
    let noisy_te: Vec<f32> = clean_te.iter().map(|&v| (v + (rng.unit() - 0.5) * 2.0 * SIGMA).clamp(0.0, 1.0)).collect();
    let xin_te = to_tensor(&noisy_te, nte, dev)?;
    let xtar_te = to_tensor(&clean_te, nte, dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let ae = AutoEncoder::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    for epoch in 1..=400 {
        let recon = ae.forward(&xin)?;
        let loss = (&recon - &xtar)?.sqr()?.mean_all()?; // 关键：目标是干净图，不是带噪输入！
        opt.backward_step(&loss)?;
        if epoch % 100 == 0 { println!("  epoch {epoch:>3}  MSE(输出,干净) = {:.5}", loss.to_scalar::<f32>()?); }
    }
    let out_te = ae.forward(&xin_te)?.clamp(0.0, 1.0)?;
    let (_, psnr_before) = mse_psnr(&xin_te, &xtar_te)?;
    let (_, psnr_after) = mse_psnr(&out_te, &xtar_te)?;
    println!("  去噪前 PSNR(噪声图 vs 干净) = {psnr_before:.2} dB");
    println!("  去噪后 PSNR(DAE输出 vs 干净) = {psnr_after:.2} dB   （提升 {:.2} dB）", psnr_after - psnr_before);

    let nflat = xin_te.flatten_all()?.to_vec1::<f32>()?;
    let oflat = out_te.flatten_all()?.to_vec1::<f32>()?;
    let cflat = xtar_te.flatten_all()?.to_vec1::<f32>()?;
    let rows: Vec<Vec<Vec<f32>>> = (0..3)
        .map(|c| vec![slice_of(&nflat, c), slice_of(&oflat, c), slice_of(&cflat, c)])
        .collect();
    save_grid("out_ch69_dae.png", &rows);
    println!("  已存对照图 out_ch69_dae.png（每行：左=噪声输入，中=DAE去噪，右=干净目标）");
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    train_autoencoder(&dev)?;
    representation_probe(&dev)?;
    train_dae(&dev)?;
    Ok(())
}
