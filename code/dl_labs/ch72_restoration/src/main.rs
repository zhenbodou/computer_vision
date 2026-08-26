// 第 72 章：超分辨率、去噪与风格迁移
// candle 0.11.0 + 纯 Rust 基线，纯 CPU，无需下载任何模型/数据。
// 运行：cargo run --release
//
// 三个部分：
//   1) 超分辨率：手写双线性插值做基线；SRCNN 风格残差网络（candle）做 2x 放大，比 PSNR
//   2) 去噪：手写 3x3 高斯滤波做经典基线；DnCNN 风格残差网络（candle）做去噪，比 PSNR
//   3) 风格迁移：诚实说明真 NST 需下载 VGG；自包含 demo 用纯 Rust 颜色统计迁移演示「风格=统计量」
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{GrayImage, Luma, Rgb, RgbImage};

const HR: usize = 32; // 超分：高清边长
const LR: usize = 16; // 超分：低清边长（放大 2x）
const DN: usize = 24; // 去噪：图像边长
const ST: usize = 64; // 风格迁移：RGB 图边长

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

// ================= 纯 Rust 基础件：双线性插值 + PSNR + 高斯滤波 =================

/// 纯 Rust 双线性插值 resize（半像素中心对齐）。src 是行主序 [sh*sw] 的 [0,1] 灰度。
fn bilinear_resize(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    let rx = sw as f32 / dw as f32;
    let ry = sh as f32 / dh as f32;
    for dy in 0..dh {
        let fy = ((dy as f32 + 0.5) * ry - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for dx in 0..dw {
            let fx = ((dx as f32 + 0.5) * rx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            let top = src[y0 * sw + x0] * (1.0 - wx) + src[y0 * sw + x1] * wx;
            let bot = src[y1 * sw + x0] * (1.0 - wx) + src[y1 * sw + x1] * wx;
            out[dy * dw + dx] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

/// 纯手写 PSNR（图像值域 [0,1] → MAX=1 → PSNR = -10 log10(MSE)）。
fn psnr_vec(a: &[f32], b: &[f32]) -> f32 {
    let mse: f32 = a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum::<f32>() / a.len() as f32;
    if mse <= 1e-12 { 99.0 } else { -10.0 * mse.log10() }
}
/// 逐图 PSNR 求平均（每张 px 个像素）。
fn avg_psnr(pred: &[f32], target: &[f32], px: usize) -> f32 {
    let n = pred.len() / px;
    (0..n).map(|i| psnr_vec(&pred[i * px..(i + 1) * px], &target[i * px..(i + 1) * px])).sum::<f32>() / n as f32
}

/// 纯手写 3x3 高斯滤波（经典去噪基线之一），边界按复制处理。
fn gauss3(src: &[f32], side: usize) -> Vec<f32> {
    let k = [1., 2., 1., 2., 4., 2., 1., 2., 1.];
    let mut out = vec![0f32; side * side];
    let at = |x: i32, y: i32| -> usize {
        (y.clamp(0, side as i32 - 1) as usize) * side + x.clamp(0, side as i32 - 1) as usize
    };
    for y in 0..side as i32 {
        for x in 0..side as i32 {
            let mut s = 0.0;
            let mut ki = 0;
            for dy in -1..=1 {
                for dx in -1..=1 { s += src[at(x + dx, y + dy)] * k[ki]; ki += 1; }
            }
            out[y as usize * side + x as usize] = s / 16.0;
        }
    }
    out
}
/// 纯手写 3x3 中值滤波（经典去噪基线之一，边缘保持能力强），边界按复制处理。
fn median3(src: &[f32], side: usize) -> Vec<f32> {
    let mut out = vec![0f32; side * side];
    let at = |x: i32, y: i32| -> f32 {
        src[(y.clamp(0, side as i32 - 1) as usize) * side + x.clamp(0, side as i32 - 1) as usize]
    };
    for y in 0..side as i32 {
        for x in 0..side as i32 {
            let mut w = [0f32; 9];
            let mut k = 0;
            for dy in -1..=1 {
                for dx in -1..=1 { w[k] = at(x + dx, y + dy); k += 1; }
            }
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out[y as usize * side + x as usize] = w[4]; // 取 9 个值的中位数
        }
    }
    out
}

// ---------- 灰度图保存（把若干小图放大后拼成对照大图） ----------
fn tile_gray(pix: &[f32], side: usize) -> GrayImage {
    GrayImage::from_fn(side as u32, side as u32, |x, y| {
        Luma([(pix[y as usize * side + x as usize].clamp(0.0, 1.0) * 255.0) as u8])
    })
}
// rows：每格是 (像素, 该图边长)；统一放大到 disp×disp 显示。
fn save_grid(path: &str, rows: &[Vec<(Vec<f32>, usize)>], disp: u32) {
    let gap = 6u32;
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(1) as u32;
    let nrows = rows.len() as u32;
    let w = cols * disp + (cols - 1) * gap;
    let h = nrows * disp + (nrows - 1) * gap;
    let mut canvas = GrayImage::from_pixel(w, h, Luma([90]));
    for (r, row) in rows.iter().enumerate() {
        for (c, (pix, side)) in row.iter().enumerate() {
            let img = image::imageops::resize(&tile_gray(pix, *side), disp, disp, FilterType::Nearest);
            image::imageops::overlay(&mut canvas, &img, (c as u32 * (disp + gap)) as i64, (r as u32 * (disp + gap)) as i64);
        }
    }
    canvas.save(path).unwrap();
}

// ================= 1) 超分辨率：双线性基线 vs SRCNN =================

// 高清图：随机 2~3 条横/竖白条，制造大量边缘（低清→上采样后边缘会糊，正好让 SR 网络锐化）
fn draw_hr(rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; HR * HR];
    let bars = 2 + rng.range(2);
    for _ in 0..bars {
        let w = 2 + rng.range(3);
        if rng.range(2) == 0 {
            let x0 = rng.range(HR - w);
            for y in 0..HR { for x in x0..x0 + w { g[y * HR + x] = 1.0; } }
        } else {
            let y0 = rng.range(HR - w);
            for x in 0..HR { for y in y0..y0 + w { g[y * HR + x] = 1.0; } }
        }
    }
    g
}
// 造超分数据：返回（双线性上采样图 flat，HR 目标 flat）
fn make_sr_data(n: usize, rng: &mut Lcg) -> (Vec<f32>, Vec<f32>) {
    let (mut up, mut hr) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let h = draw_hr(rng);
        let lr = bilinear_resize(&h, HR, HR, LR, LR); // 退化：高清→低清
        let u = bilinear_resize(&lr, LR, LR, HR, HR); // 双线性放大回高清尺寸（这就是基线，也是 SRCNN 的输入）
        up.extend_from_slice(&u);
        hr.extend_from_slice(&h);
    }
    (up, hr)
}

// SRCNN 风格：3 层卷积，残差学习（在双线性图上做锐化修正）
struct SrCnn { c1: Conv2d, c2: Conv2d, c3: Conv2d }
impl SrCnn {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            c1: conv2d(1, 32, 5, Conv2dConfig { padding: 2, ..Default::default() }, vb.pp("c1"))?,
            c2: conv2d(32, 16, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c2"))?,
            c3: conv2d(16, 1, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("c3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?;
        let h = self.c2.forward(&h)?.relu()?;
        let r = self.c3.forward(&h)?; // 预测「双线性图 → 高清」的残差
        x + &r // 残差学习：初始残差≈0 → 起点≈双线性，训练只会把 PSNR 往上推
    }
}

fn superres(dev: &Device) -> Result<()> {
    println!("==== 1) 超分辨率：纯 Rust 双线性插值基线 vs SRCNN（2x 放大） ====");
    let mut rng = Lcg::new(1);
    let (up_tr, hr_tr) = make_sr_data(120, &mut rng);
    let (up_te, hr_te) = make_sr_data(40, &mut rng);
    let (ntr, nte) = (120, 40);
    let px = HR * HR;
    let xin = Tensor::from_vec(up_tr.clone(), (ntr, 1, HR, HR), dev)?;
    let xhr = Tensor::from_vec(hr_tr, (ntr, 1, HR, HR), dev)?;
    let xin_te = Tensor::from_vec(up_te.clone(), (nte, 1, HR, HR), dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = SrCnn::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    for epoch in 1..=300 {
        let sr = net.forward(&xin)?;
        let loss = (&sr - &xhr)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 60 == 0 { println!("  epoch {epoch:>3}  训练 MSE = {:.6}", loss.to_scalar::<f32>()?); }
    }
    // 测试集 PSNR：双线性基线 vs SRCNN
    let sr_te: Vec<f32> = net.forward(&xin_te)?.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?;
    let psnr_bilinear = avg_psnr(&up_te, &hr_te, px);
    let psnr_srcnn = avg_psnr(&sr_te, &hr_te, px);
    println!("  双线性插值基线 PSNR = {psnr_bilinear:.2} dB");
    println!("  SRCNN 超分       PSNR = {psnr_srcnn:.2} dB   （提升 {:.2} dB）", psnr_srcnn - psnr_bilinear);

    // 对照图：LR（放大显示）| 双线性 | SRCNN | HR
    let lr0 = bilinear_resize(&hr_te[0..px], HR, HR, LR, LR);
    let rows = vec![vec![
        (lr0, LR),
        (up_te[0..px].to_vec(), HR),
        (sr_te[0..px].to_vec(), HR),
        (hr_te[0..px].to_vec(), HR),
    ]];
    save_grid("out_ch72_sr.png", &rows, 128);
    println!("  已存对照图 out_ch72_sr.png（左→右：低清 | 双线性放大 | SRCNN | 高清原图）\n");
    Ok(())
}

// ================= 2) 去噪：高斯滤波基线 vs DnCNN =================

// 去噪用图：随机单个形状（竖条/横条/方块），24x24
fn draw_dn(rng: &mut Lcg) -> Vec<f32> {
    let mut g = vec![0f32; DN * DN];
    match rng.range(3) {
        0 => { let x0 = 4 + rng.range(DN - 9); for y in 0..DN { for x in x0..x0 + 4 { g[y * DN + x] = 1.0; } } }
        1 => { let y0 = 4 + rng.range(DN - 9); for x in 0..DN { for y in y0..y0 + 4 { g[y * DN + x] = 1.0; } } }
        _ => { let s = 9; let x0 = 3 + rng.range(DN - s - 4); let y0 = 3 + rng.range(DN - s - 4);
               for y in y0..y0 + s { for x in x0..x0 + s { g[y * DN + x] = 1.0; } } }
    }
    g
}
fn make_dn_data(n: usize, sigma: f32, rng: &mut Lcg) -> (Vec<f32>, Vec<f32>) {
    let (mut noisy, mut clean) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let c = draw_dn(rng);
        let no: Vec<f32> = c.iter().map(|&v| (v + (rng.unit() - 0.5) * 2.0 * sigma).clamp(0.0, 1.0)).collect();
        noisy.extend_from_slice(&no);
        clean.extend_from_slice(&c);
    }
    (noisy, clean)
}

// DnCNN 风格：残差网络，预测噪声，干净 = 噪声图 - 预测噪声
struct DnCnn { c1: Conv2d, c2: Conv2d, c3: Conv2d }
impl DnCnn {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: conv2d(1, 32, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(32, 32, 3, cfg, vb.pp("c2"))?,
            c3: conv2d(32, 1, 3, cfg, vb.pp("c3"))?,
        })
    }
    fn forward(&self, noisy: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(noisy)?.relu()?;
        let h = self.c2.forward(&h)?.relu()?;
        let resid = self.c3.forward(&h)?; // 预测噪声
        noisy - &resid // 干净 = 噪声图 - 噪声
    }
}

fn denoise(dev: &Device) -> Result<()> {
    println!("==== 2) 去噪：纯 Rust 经典滤波基线 vs DnCNN ====");
    const SIGMA: f32 = 0.35;
    let mut rng = Lcg::new(7);
    let (noisy_tr, clean_tr) = make_dn_data(150, SIGMA, &mut rng);
    let (noisy_te, clean_te) = make_dn_data(50, SIGMA, &mut rng);
    let (ntr, nte) = (150, 50);
    let px = DN * DN;
    let xin = Tensor::from_vec(noisy_tr, (ntr, 1, DN, DN), dev)?;
    let xtar = Tensor::from_vec(clean_tr, (ntr, 1, DN, DN), dev)?;
    let xin_te = Tensor::from_vec(noisy_te.clone(), (nte, 1, DN, DN), dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = DnCnn::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    for epoch in 1..=300 {
        let out = net.forward(&xin)?;
        let loss = (&out - &xtar)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 60 == 0 { println!("  epoch {epoch:>3}  训练 MSE = {:.6}", loss.to_scalar::<f32>()?); }
    }
    // 两个经典基线（逐图）：高斯 vs 中值
    let gauss_te: Vec<f32> = (0..nte).flat_map(|i| gauss3(&noisy_te[i * px..(i + 1) * px], DN)).collect();
    let median_te: Vec<f32> = (0..nte).flat_map(|i| median3(&noisy_te[i * px..(i + 1) * px], DN)).collect();
    let dncnn_te: Vec<f32> = net.forward(&xin_te)?.clamp(0.0, 1.0)?.flatten_all()?.to_vec1::<f32>()?;
    let psnr_noisy = avg_psnr(&noisy_te, &clean_te, px);
    let psnr_gauss = avg_psnr(&gauss_te, &clean_te, px);
    let psnr_median = avg_psnr(&median_te, &clean_te, px);
    let psnr_dncnn = avg_psnr(&dncnn_te, &clean_te, px);
    println!("  含噪图         PSNR = {psnr_noisy:.2} dB（去噪前）");
    println!("  高斯滤波基线   PSNR = {psnr_gauss:.2} dB（模糊了边缘，边缘密集时几乎不赚甚至倒亏）");
    println!("  中值滤波基线   PSNR = {psnr_median:.2} dB（保边更好，明显改善）");
    println!("  DnCNN 去噪     PSNR = {psnr_dncnn:.2} dB（学习式，大幅领先）");

    let rows = vec![vec![
        (noisy_te[0..px].to_vec(), DN),
        (median_te[0..px].to_vec(), DN),
        (dncnn_te[0..px].to_vec(), DN),
        (clean_te[0..px].to_vec(), DN),
    ]];
    save_grid("out_ch72_denoise.png", &rows, 128);
    println!("  已存对照图 out_ch72_denoise.png（左→右：含噪 | 中值滤波 | DnCNN | 干净原图）\n");
    Ok(())
}

// ================= 3) 风格迁移：纯 Rust 颜色统计迁移（诚实代理） =================

fn gen_content() -> Vec<u8> {
    // 冷色调：暗蓝渐变背景 + 两个亮块（结构）
    let mut v = vec![0u8; 3 * ST * ST];
    for y in 0..ST {
        for x in 0..ST {
            let i = (y * ST + x) * 3;
            let t = y as f32 / ST as f32;
            v[i] = 25; // R 低
            v[i + 1] = (40.0 + 70.0 * t) as u8; // G 中
            v[i + 2] = (120.0 + 100.0 * t) as u8; // B 高
        }
    }
    for (bx, by, bs) in [(10usize, 12usize, 14usize), (38, 34, 18)] {
        for y in by..by + bs { for x in bx..bx + bs { let i = (y * ST + x) * 3; v[i] = 235; v[i + 1] = 235; v[i + 2] = 240; } }
    }
    v
}
fn gen_reference() -> Vec<u8> {
    // 暖色调：橙红渐变（高红、中绿、低蓝）
    let mut v = vec![0u8; 3 * ST * ST];
    for y in 0..ST {
        for x in 0..ST {
            let i = (y * ST + x) * 3;
            let t = x as f32 / ST as f32;
            v[i] = (200.0 + 40.0 * t).min(255.0) as u8;
            v[i + 1] = (90.0 + 120.0 * t) as u8;
            v[i + 2] = (30.0 + 40.0 * (1.0 - t)) as u8;
        }
    }
    v
}
fn channel_stats(rgb: &[u8], n: usize) -> [(f32, f32); 3] {
    let mut mean = [0f32; 3];
    for i in 0..n { for c in 0..3 { mean[c] += rgb[i * 3 + c] as f32; } }
    for c in 0..3 { mean[c] /= n as f32; }
    let mut var = [0f32; 3];
    for i in 0..n { for c in 0..3 { let d = rgb[i * 3 + c] as f32 - mean[c]; var[c] += d * d; } }
    let mut out = [(0f32, 0f32); 3];
    for c in 0..3 { out[c] = (mean[c], (var[c] / n as f32).sqrt().max(1e-6)); }
    out
}
// 颜色迁移（Reinhard 式）：把 content 每个通道标准化后，套上 reference 的均值/标准差
fn color_transfer(content: &[u8], cs: &[(f32, f32); 3], rs: &[(f32, f32); 3], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            let z = (content[i * 3 + c] as f32 - cs[c].0) / cs[c].1;
            out[i * 3 + c] = (z * rs[c].1 + rs[c].0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}
fn save_rgb_row(path: &str, imgs: &[&[u8]]) {
    let gap = 8u32;
    let cols = imgs.len() as u32;
    let w = cols * ST as u32 + (cols - 1) * gap;
    let mut canvas = RgbImage::from_pixel(w, ST as u32, Rgb([245, 245, 245]));
    for (c, raw) in imgs.iter().enumerate() {
        let img = RgbImage::from_raw(ST as u32, ST as u32, raw.to_vec()).unwrap();
        image::imageops::overlay(&mut canvas, &img, (c as u32 * (ST as u32 + gap)) as i64, 0);
    }
    canvas.save(path).unwrap();
}
fn style_transfer() {
    println!("==== 3) 风格迁移：纯 Rust 颜色统计迁移（真 NST 需下载 VGG，见正文） ====");
    let n = ST * ST;
    let content = gen_content();
    let reference = gen_reference();
    let cs = channel_stats(&content, n);
    let rs = channel_stats(&reference, n);
    let out = color_transfer(&content, &cs, &rs, n);
    let os = channel_stats(&out, n);
    let names = ["R", "G", "B"];
    println!("  通道         内容图(均值,标准差)     参考图(均值,标准差)     输出图(均值,标准差)");
    for c in 0..3 {
        println!("  {}  {:>8.1},{:>6.1}       {:>8.1},{:>6.1}       {:>8.1},{:>6.1}",
            names[c], cs[c].0, cs[c].1, rs[c].0, rs[c].1, os[c].0, os[c].1);
    }
    println!("  → 输出图的每通道均值/标准差已对齐参考图 = 参考图的「色彩风格」被迁移到了内容图上");
    save_rgb_row("out_ch72_style.png", &[&content, &reference, &out]);
    println!("  已存对照图 out_ch72_style.png（左→右：内容图 | 参考风格图 | 迁移结果）");
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    superres(&dev)?;
    denoise(&dev)?;
    style_transfer();
    Ok(())
}
