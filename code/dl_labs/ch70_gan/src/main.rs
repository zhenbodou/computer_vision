// 第 70 章：生成对抗网络 GAN
// candle 0.11.0，纯 CPU，无需下载任何模型/数据。
// 运行：cargo run --release
//
// 训练一个「小 GAN」（用全连接 MLP，够小够快、对抗机制看得最清）：
//   - 生成器 G：隐向量 z(16 维)[+类别标签] → MLP → 16x16 灰度小图（256 维，sigmoid 到 [0,1]）
//   - 判别器 D：16x16 小图[+类别标签] → MLP → 1 个真/假 logit
// 真实分布是 6 个「离散模式」（3 竖条 + 3 横条，位置固定 + 微噪），便于量化「模式覆盖率」。
// 判别器/生成器损失都用【数值稳定 BCE】；对抗训练交替更新 D、G。
//
// 三组对照实验，串起本章核心概念：
//   A) 朴素训练：1:1 更新、D/G 同学习率、无标签平滑  → 训练不稳
//   B) 稳定训练：G 多更新、D 学习率减半、标签平滑    → 训练稳、样本更清晰，但仍模式崩溃（只会画 1 个模式）
//   C) 条件 GAN：把类别标签喂给 G 和 D               → 每个标签「认领」一个模式，覆盖全部 6 个模式
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::{GrayImage, Luma};

const SZ: usize = 16; // 图像边长（正方形灰度）
const PX: usize = SZ * SZ; // 256
const LATENT: usize = 16; // 隐向量 z 的维度
const HID: usize = 128; // MLP 隐层宽度
const N_MODES: usize = 6; // 真实分布的模式数（3 竖条 + 3 横条）
const TAU_COVER: f32 = 4.0; // 判定「生成样本命中某模式」的距离阈值（远小于模式间最小距≈8.8）

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
    fn randn(&mut self) -> f32 { // 标准正态（Box-Muller），供隐向量 z 采样
        let u1 = self.unit().max(1e-9);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
/// 确定性初始化 VarMap 里所有权重：bias 置 0，weight 取 [-lim,lim] 均匀分布，lim=1/sqrt(fan_in)。
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

// ---------- 真实数据：6 个离散模式（干净模板 + 微噪） ----------
// 模式 0..3 竖条（列位置 {2,7,12}），模式 3..6 横条（行位置 {2,7,12}），条宽 3；模式间彼此不重叠。
fn clean_mode(m: usize) -> Vec<f32> {
    let pos = [2usize, 7, 12];
    let mut g = vec![0f32; PX];
    if m < 3 {
        let x0 = pos[m];
        for y in 0..SZ { for x in x0..(x0 + 3).min(SZ) { g[y * SZ + x] = 1.0; } }
    } else {
        let y0 = pos[m - 3];
        for x in 0..SZ { for y in y0..(y0 + 3).min(SZ) { g[y * SZ + x] = 1.0; } }
    }
    g
}
fn real_sample(m: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut g = clean_mode(m);
    for v in g.iter_mut() { *v = (*v + (rng.unit() - 0.5) * 0.10).clamp(0.0, 1.0); } // 微小均匀噪声
    g
}
fn build_real_pool(per_mode: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut pool = Vec::with_capacity(per_mode * N_MODES * PX);
    for _ in 0..per_mode {
        for m in 0..N_MODES { pool.extend(real_sample(m, rng)); }
    }
    pool // 布局：[m0,m1,...,m5, m0,m1,...] 循环 → 第 i 张的模式 = i % N_MODES
}

// 把一组模式下标做成 one-hot 标签张量 [n, N_MODES]（条件 GAN 用；非条件时用全零）
// on 值放大到 LBL_SCALE，让这 6 维标签信号不被 256 维图像/16 维 z 淹没，网络才「看得见」条件。
const LBL_SCALE: f32 = 5.0;
fn onehot(modes: &[usize], dev: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; modes.len() * N_MODES];
    for (i, &m) in modes.iter().enumerate() { v[i * N_MODES + m] = LBL_SCALE; }
    Tensor::from_vec(v, (modes.len(), N_MODES), dev)
}
fn zeros_lbl(n: usize, dev: &Device) -> Result<Tensor> { Tensor::zeros((n, N_MODES), DType::F32, dev) }

// ---------- 生成器 G：[z(16)+标签(6)] → 128 → 128 → 256，末层 sigmoid ----------
struct Generator { fc1: Linear, fc2: Linear, fc3: Linear }
impl Generator {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(LATENT + N_MODES, HID, vb.pp("fc1"))?,
            fc2: linear(HID, HID, vb.pp("fc2"))?,
            fc3: linear(HID, PX, vb.pp("fc3"))?,
        })
    }
    fn forward(&self, z: &Tensor, lbl: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[z, lbl], 1)?; // 拼接隐向量与标签（非条件实验里 lbl 全 0，等于没条件）
        let h = self.fc1.forward(&x)?.relu()?;
        let h = self.fc2.forward(&h)?.relu()?;
        candle_nn::ops::sigmoid(&self.fc3.forward(&h)?) // 压到 [0,1]
    }
}
// ---------- 判别器 D：[图(256)+标签(6)] → 128 → 1 个 logit ----------
struct Discriminator { fc1: Linear, fc2: Linear }
impl Discriminator {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(PX + N_MODES, HID, vb.pp("fc1"))?,
            fc2: linear(HID, 1, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor, lbl: &Tensor) -> Result<Tensor> {
        let x = Tensor::cat(&[x, lbl], 1)?;
        let h = self.fc1.forward(&x)?.relu()?;
        self.fc2.forward(&h) // [N,1] 原始 logit（不加 sigmoid）
    }
}

// ---------- 数值稳定 BCE-with-logits：L = mean( relu(x) - x*y + log(1+exp(-|x|)) ) ----------
// y 是标量目标（1.0=真、0.0=假，也可用标签平滑值如 0.9）。别用 candle 的 binary_cross_entropy_with_logit（会 NaN）。
fn bce(logits: &Tensor, y: f64) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.affine(y, 0.0)?; // x * y
    let e = logits.abs()?.neg()?.exp()?; // exp(-|x|)
    let sp = (&e + 1.0)?.log()?; // log(1+exp(-|x|))
    (&(&relu_x - &xy)? + &sp)?.mean_all()
}

fn sample_z(n: usize, rng: &mut Lcg, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = (0..n * LATENT).map(|_| rng.randn()).collect();
    Tensor::from_vec(v, (n, LATENT), dev)
}
// 取一个真实 mini-batch，同时返回它们的模式下标（条件 GAN 需要）
fn sample_real(pool: &[f32], n_pool: usize, batch: usize, rng: &mut Lcg, dev: &Device) -> Result<(Tensor, Vec<usize>)> {
    let mut buf = Vec::with_capacity(batch * PX);
    let mut labels = Vec::with_capacity(batch);
    for _ in 0..batch {
        let idx = rng.range(n_pool);
        buf.extend_from_slice(&pool[idx * PX..(idx + 1) * PX]);
        labels.push(idx % N_MODES);
    }
    Ok((Tensor::from_vec(buf, (batch, PX), dev)?, labels))
}

fn l2(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum::<f32>().sqrt() }

struct GenStats { avg_nn: f32, coverage: usize, hit_rate: f32 }
fn eval_samples(samples: &[f32], modes: &[Vec<f32>]) -> GenStats {
    let n = samples.len() / PX;
    let mut covered = [false; N_MODES];
    let (mut sum_nn, mut hit) = (0.0f32, 0usize);
    for i in 0..n {
        let s = &samples[i * PX..(i + 1) * PX];
        let (mut best, mut best_m) = (f32::INFINITY, 0usize);
        for (m, cm) in modes.iter().enumerate() {
            let d = l2(s, cm);
            if d < best { best = d; best_m = m; }
        }
        sum_nn += best;
        if best < TAU_COVER { hit += 1; covered[best_m] = true; }
    }
    GenStats { avg_nn: sum_nn / n as f32, coverage: covered.iter().filter(|&&b| b).count(), hit_rate: hit as f32 / n as f32 }
}
// Chamfer 式精度/召回：精度=生成→最近真实的平均距离；召回=真实→最近生成的平均距离（模式崩溃→召回距离飙高）
fn chamfer(gen: &[f32], real: &[f32]) -> (f32, f32) {
    let (ng, nr) = (gen.len() / PX, real.len() / PX);
    let mut prec = 0.0f32;
    for i in 0..ng {
        let g = &gen[i * PX..(i + 1) * PX];
        let mut best = f32::INFINITY;
        for j in 0..nr { best = best.min(l2(g, &real[j * PX..(j + 1) * PX])); }
        prec += best;
    }
    let mut rec = 0.0f32;
    for j in 0..nr {
        let r = &real[j * PX..(j + 1) * PX];
        let mut best = f32::INFINITY;
        for i in 0..ng { best = best.min(l2(r, &gen[i * PX..(i + 1) * PX])); }
        rec += best;
    }
    (prec / ng as f32, rec / nr as f32)
}

// 生成一个评估集：条件模式下按模式各生成若干张（应覆盖全部模式）；非条件模式下随机生成。
fn gen_eval(gen: &Generator, conditional: bool, per: usize, dev: &Device) -> Result<Vec<f32>> {
    let mut rng = Lcg::new(7777);
    if conditional {
        let mut out = Vec::new();
        for m in 0..N_MODES {
            let z = sample_z(per, &mut rng, dev)?;
            let lbl = onehot(&vec![m; per], dev)?;
            out.extend(gen.forward(&z, &lbl)?.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(out)
    } else {
        let n = per * N_MODES;
        let z = sample_z(n, &mut rng, dev)?;
        gen.forward(&z, &zeros_lbl(n, dev)?)?.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()
    }
}

// ---------- 图像保存 ----------
fn tile_gray(pix: &[f32]) -> GrayImage {
    GrayImage::from_fn(SZ as u32, SZ as u32, |x, y| {
        Luma([(pix[y as usize * SZ + x as usize].clamp(0.0, 1.0) * 255.0) as u8])
    })
}
fn upscale(img: &GrayImage, f: u32) -> GrayImage {
    image::imageops::resize(img, img.width() * f, img.height() * f, image::imageops::FilterType::Nearest)
}
fn save_grid(path: &str, rows: &[Vec<Vec<f32>>]) {
    let (factor, gap) = (5u32, 5u32);
    let cell = SZ as u32 * factor;
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(1) as u32;
    let nrows = rows.len() as u32;
    let w = cols * cell + (cols - 1) * gap;
    let h = nrows * cell + (nrows.max(1) - 1) * gap;
    let mut canvas = GrayImage::from_pixel(w, h, Luma([90]));
    for (r, row) in rows.iter().enumerate() {
        for (c, pix) in row.iter().enumerate() {
            let tile = upscale(&tile_gray(pix), factor);
            image::imageops::overlay(&mut canvas, &tile, (c as u32 * (cell + gap)) as i64, (r as u32 * (cell + gap)) as i64);
        }
    }
    canvas.save(path).unwrap();
}
fn slice_of(flat: &[f32], i: usize) -> Vec<f32> { flat[i * PX..(i + 1) * PX].to_vec() }

// 训练配置：把「稳定技巧」与「是否条件」做成旋钮
struct Cfg { g_steps: usize, lr_d: f64, lr_g: f64, real_label: f64, conditional: bool, tag: &'static str }

fn run_gan(dev: &Device, pool: &[f32], n_pool: usize, modes: &[Vec<f32>], real_eval: &[f32], cfg: &Cfg) -> Result<(usize, f32, f32, f32)> {
    let g_vm = VarMap::new();
    let gen = Generator::load(VarBuilder::from_varmap(&g_vm, DType::F32, dev))?;
    det_init(&g_vm, 42)?;
    let d_vm = VarMap::new();
    let disc = Discriminator::load(VarBuilder::from_varmap(&d_vm, DType::F32, dev))?;
    det_init(&d_vm, 123)?;
    let params = |lr: f64| ParamsAdamW { lr, beta1: 0.5, beta2: 0.999, weight_decay: 0.0, ..Default::default() };
    let mut g_opt = AdamW::new(g_vm.all_vars(), params(cfg.lr_g))?;
    let mut d_opt = AdamW::new(d_vm.all_vars(), params(cfg.lr_d))?;

    let (batch, steps) = (128, 10000);
    let mut trng = Lcg::new(2024);
    let z_vis = sample_z(N_MODES, &mut Lcg::new(999), dev)?; // 固定隐向量，观察演化
    let vis_lbl = if cfg.conditional { onehot(&(0..N_MODES).collect::<Vec<_>>(), dev)? } else { zeros_lbl(N_MODES, dev)? };
    let mut snapshots: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut last = (0usize, 0f32);

    for step in 1..=steps {
        // === 更新判别器 D：真判真、假判假 ===
        let (real, real_m) = sample_real(pool, n_pool, batch, &mut trng, dev)?;
        let real_lbl = if cfg.conditional { onehot(&real_m, dev)? } else { zeros_lbl(batch, dev)? };
        // 生成假样本：条件模式下随机指定要画哪个模式（并把该标签同时喂给 G 和 D）
        let fake_m: Vec<usize> = (0..batch).map(|_| trng.range(N_MODES)).collect();
        let fake_lbl = if cfg.conditional { onehot(&fake_m, dev)? } else { zeros_lbl(batch, dev)? };
        let z = sample_z(batch, &mut trng, dev)?;
        let fake = gen.forward(&z, &fake_lbl)?.detach();
        let d_loss = (&bce(&disc.forward(&real, &real_lbl)?, cfg.real_label)? + &bce(&disc.forward(&fake, &fake_lbl)?, 0.0)?)?;
        d_opt.backward_step(&d_loss)?;

        // === 更新生成器 G：让 D 把生成图判成「真」（每轮更新 g_steps 次）===
        let mut g_loss_val = 0.0f32;
        for _ in 0..cfg.g_steps {
            let fm: Vec<usize> = (0..batch).map(|_| trng.range(N_MODES)).collect();
            let fl = if cfg.conditional { onehot(&fm, dev)? } else { zeros_lbl(batch, dev)? };
            let z2 = sample_z(batch, &mut trng, dev)?;
            let g_loss = bce(&disc.forward(&gen.forward(&z2, &fl)?, &fl)?, 1.0)?;
            g_opt.backward_step(&g_loss)?;
            g_loss_val = g_loss.to_scalar::<f32>()?;
        }

        if step == 1 || step % 1000 == 0 {
            let ev = gen_eval(&gen, cfg.conditional, 50, dev)?;
            let st = eval_samples(&ev, modes);
            println!(
                "  [{}] step {step:>5}  D_loss={:.3}  G_loss={:.3}  |  平均最近距离={:.3}  模式覆盖={}/{}  命中率={:.1}%",
                cfg.tag, d_loss.to_scalar::<f32>()?, g_loss_val, st.avg_nn, st.coverage, N_MODES, st.hit_rate * 100.0
            );
            if [1usize, 1000, 3000, 10000].contains(&step) {
                let vis = gen.forward(&z_vis, &vis_lbl)?.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?;
                snapshots.push((0..N_MODES).map(|i| slice_of(&vis, i)).collect());
            }
            last = (st.coverage, st.hit_rate);
        }
    }

    let gen_final = gen_eval(&gen, cfg.conditional, 50, dev)?;
    let (prec, rec) = chamfer(&gen_final, real_eval);
    println!("  [{}] 最终 Chamfer：精度距离={prec:.3}（生成像不像真）  召回距离={rec:.3}（真实覆盖得全不全）", cfg.tag);

    save_grid(&format!("out_ch70_{}_progress.png", cfg.tag), &snapshots);
    let mut best_per_mode: Vec<Vec<f32>> = vec![vec![0f32; PX]; N_MODES];
    let mut best_d = vec![f32::INFINITY; N_MODES];
    let ng = gen_final.len() / PX;
    for i in 0..ng {
        let s = &gen_final[i * PX..(i + 1) * PX];
        for m in 0..N_MODES {
            let d = l2(s, &modes[m]);
            if d < best_d[m] { best_d[m] = d; best_per_mode[m] = s.to_vec(); }
        }
    }
    save_grid(&format!("out_ch70_{}_modes.png", cfg.tag), &vec![(0..N_MODES).map(clean_mode).collect::<Vec<_>>(), best_per_mode]);
    Ok((last.0, last.1, prec, rec))
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    println!("==== 小 GAN：6 个离散模式（3 竖条 + 3 横条）上的对抗训练 ====");
    let mut drng = Lcg::new(1);
    let pool = build_real_pool(64, &mut drng);
    let n_pool = 64 * N_MODES;
    let real_eval = build_real_pool(50, &mut drng);
    let modes: Vec<Vec<f32>> = (0..N_MODES).map(clean_mode).collect();
    let mut sum = 0.0f32;
    for i in 0..n_pool { sum += l2(&pool[i * PX..(i + 1) * PX], &modes[i % N_MODES]); }
    println!("  参照：真实样本到干净模式的平均距离 = {:.3}（生成样本平均最近距离越接近它越好）\n", sum / n_pool as f32);

    println!("---- 实验 A：朴素训练（1:1 更新、D/G 同学习率、无标签平滑）----");
    let a = run_gan(&dev, &pool, n_pool, &modes, &real_eval, &Cfg { g_steps: 1, lr_d: 2e-4, lr_g: 2e-4, real_label: 1.0, conditional: false, tag: "naive" })?;
    println!("\n---- 实验 B：稳定训练（G 每轮 2 步、D 学习率减半、标签平滑 0.9）----");
    let b = run_gan(&dev, &pool, n_pool, &modes, &real_eval, &Cfg { g_steps: 2, lr_d: 1e-4, lr_g: 2e-4, real_label: 0.9, conditional: false, tag: "stable" })?;
    println!("\n---- 实验 C：条件 GAN（把类别标签喂给 G 和 D）----");
    let c = run_gan(&dev, &pool, n_pool, &modes, &real_eval, &Cfg { g_steps: 2, lr_d: 1e-4, lr_g: 2e-4, real_label: 0.9, conditional: true, tag: "cgan" })?;

    println!("\n==== 对照小结（召回距离越小=真实分布覆盖得越全）====");
    println!("  A 朴素     ：模式覆盖 {}/{}  命中率 {:.1}%  精度距离 {:.3}  召回距离 {:.3}", a.0, N_MODES, a.1 * 100.0, a.2, a.3);
    println!("  B 稳定     ：模式覆盖 {}/{}  命中率 {:.1}%  精度距离 {:.3}  召回距离 {:.3}", b.0, N_MODES, b.1 * 100.0, b.2, b.3);
    println!("  C 条件 GAN ：模式覆盖 {}/{}  命中率 {:.1}%  精度距离 {:.3}  召回距离 {:.3}", c.0, N_MODES, c.1 * 100.0, c.2, c.3);
    Ok(())
}
