// 第 58 章：活体检测（反欺骗）+ 人脸属性多任务分类
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现。
// 运行：cd code/dl_labs/ch58_liveness_attr && cargo run --release
//
// 两部分：
//  A) 活体检测：合成“真人脸”(带 3D 明暗/纹理) vs “假体攻击”(打印照片=扁平/低对比、
//     屏幕翻拍=摩尔纹高频)；先用纯手工特征(明暗变化 + 高频能量)做阈值判真假，
//     再用小 CNN 分类；两者都报真实准确率。
//  B) 属性分析：合成带已知属性(眼镜/微笑/帽子)的人脸，一个骨干 + 3 个分类头做
//     多任务分类，报每个属性的真实准确率。

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const IMG: usize = 32; // 对齐人脸裁剪图边长

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
    // 注意：只用高位（unit 取 >>11 的高位）派生离散随机；LCG 的低位质量极差
    // （bit0 会周期性地严格交替），千万别用 `next() % k` 或 `next() & 1`。
    fn boolean(&mut self) -> bool {
        self.unit() < 0.5
    }
    // 返回 0..n 的整数（用高位）
    fn below(&mut self, n: u32) -> u32 {
        ((self.unit() * n as f32) as u32).min(n - 1)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Live {
    Real,   // 真人：有 3D 明暗渐变 + 自然纹理
    Print,  // 打印照片：扁平、无立体明暗、对比被压低
    Screen, // 屏幕翻拍：叠加摩尔纹（高频周期纹理）
}

// 一张人脸的渲染选项：活体类型 + 属性
struct FaceOpts {
    live: Live,
    glasses: bool,
    smile: bool,
    hat: bool,
}

fn clampf(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

fn fill_ellipse_shaded(
    img: &mut RgbImage,
    cx: f32,
    cy: f32,
    a: f32,
    b: f32,
    sk: f32,
    warm: f32,
    ramp: bool,
    lx: f32,
    ly: f32,
    shade_k: f32,
) {
    let x0 = (cx - a).floor().max(0.0) as i32;
    let x1 = (cx + a).ceil().min(IMG as f32 - 1.0) as i32;
    let y0 = (cy - b).floor().max(0.0) as i32;
    let y1 = (cy + b).ceil().min(IMG as f32 - 1.0) as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = (x as f32 - cx) / a;
            let dy = (y as f32 - cy) / b;
            if dx * dx + dy * dy <= 1.0 {
                let mut base = sk;
                if ramp {
                    base += shade_k * (dx * lx + dy * ly); // 3D 明暗：随位置线性变化
                }
                img.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([clampf(base), clampf(base - warm * 0.4), clampf(base - warm)]),
                );
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

fn set_px(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
        img.put_pixel(x as u32, y as u32, color);
    }
}

// 渲染一张人脸（居中、带类内扰动）
fn render(o: &FaceOpts, rng: &mut Lcg) -> RgbImage {
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    // 背景
    let bg = rng.uniform(45.0, 95.0);
    for y in 0..IMG as u32 {
        for x in 0..IMG as u32 {
            let v = clampf(bg + rng.uniform(-8.0, 8.0));
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    // 脸几何
    let cx = IMG as f32 / 2.0 + rng.uniform(-1.5, 1.5);
    let cy = IMG as f32 / 2.0 + rng.uniform(-1.5, 1.5);
    let w = IMG as f32 * 0.70 * rng.uniform(0.94, 1.06);
    let (a, b) = (w / 2.0, w * 1.25 / 2.0);
    let sk = (rng.uniform(180.0, 228.0) + rng.uniform(-12.0, 12.0)).clamp(80.0, 255.0);
    let warm = rng.uniform(35.0, 70.0);
    let ramp = o.live != Live::Print; // 打印照片扁平无立体明暗
    let (lx, ly) = (rng.uniform(-1.0, 1.0), rng.uniform(-0.6, 1.0));
    let shade_k = rng.uniform(28.0, 52.0); // 明暗强度（随机 → 制造类间重叠，别太好分）
    fill_ellipse_shaded(&mut img, cx, cy, a, b, sk, warm, ramp, lx, ly, shade_k);

    let dark = Rgb([28u8, 26, 30]);
    // 眼睛
    let ex = 0.40 * a;
    let ey = cy - 0.22 * b;
    fill_disk(&mut img, cx - ex, ey, 0.13 * a + 0.6, dark);
    fill_disk(&mut img, cx + ex, ey, 0.13 * a + 0.6, dark);
    // 眼镜：横跨双眼的深色框条（属性信号）
    if o.glasses {
        for t in -1..=1 {
            let yy = ey.round() as i32 + t;
            for xx in (cx - 0.58 * a).round() as i32..=(cx + 0.58 * a).round() as i32 {
                set_px(&mut img, xx, yy, Rgb([20, 20, 24]));
            }
        }
    }
    // 鼻
    fill_disk(&mut img, cx, cy + 0.06 * b, 0.08 * a + 0.4, dark);
    // 嘴：微笑=上扬曲线，中性=直线
    let mw = 0.30 * a;
    let my = cy + 0.46 * b;
    let steps = (2.0 * mw).round() as i32;
    for s in 0..=steps {
        let xx = -mw + 2.0 * mw * s as f32 / steps.max(1) as f32;
        let dxn = xx / mw;
        // 微笑：嘴角上扬（图像 y 向下，故中间点 y 更大、两端更小 → ∪ 形）；中性=直线
        let yy = if o.smile { my + 2.4 * (1.0 - dxn * dxn) } else { my };
        set_px(&mut img, (cx + xx).round() as i32, yy.round() as i32, dark);
        set_px(&mut img, (cx + xx).round() as i32, yy.round() as i32 + 1, dark);
    }
    // 帽子：脸顶部一条彩色带（属性信号）
    if o.hat {
        let hat = Rgb([200u8, 60, 60]);
        let yt = (cy - b - 1.0).round() as i32;
        let yb = (cy - 0.55 * b).round() as i32;
        for yy in yt..=yb {
            for xx in (cx - a).round() as i32..=(cx + a).round() as i32 {
                let dx = (xx as f32 - cx) / a;
                let dy = (yy as f32 - cy) / b;
                if dx * dx + dy * dy <= 1.05 {
                    set_px(&mut img, xx, yy, hat);
                }
            }
        }
    }

    // 打印照片：全局对比压低（扁平、发灰）
    if o.live == Live::Print {
        for y in 0..IMG as u32 {
            for x in 0..IMG as u32 {
                let p = img.get_pixel(x, y).0;
                let f = |c: u8| clampf(128.0 + (c as f32 - 128.0) * 0.55);
                img.put_pixel(x, y, Rgb([f(p[0]), f(p[1]), f(p[2])]));
            }
        }
    }
    // 屏幕翻拍：叠加摩尔纹（高频周期）+ 轻微偏色
    if o.live == Live::Screen {
        let (fx, fy) = (rng.uniform(5.0, 10.0), rng.uniform(5.0, 10.0));
        let ph = rng.uniform(0.0, 6.28);
        let amp = rng.uniform(0.09, 0.17);
        for y in 0..IMG as u32 {
            for x in 0..IMG as u32 {
                let m = 1.0 + amp * (2.0 * std::f32::consts::PI * (fx * x as f32 + fy * y as f32) / IMG as f32 + ph).sin();
                let p = img.get_pixel(x, y).0;
                img.put_pixel(
                    x,
                    y,
                    Rgb([clampf(p[0] as f32 * m), clampf(p[1] as f32 * m), clampf(p[2] as f32 * m + 6.0)]),
                );
            }
        }
    }
    // 细纹理噪声（真人较多、打印很少、屏幕中等）
    let tex = match o.live {
        Live::Real => 7.0,
        Live::Print => 3.0,
        Live::Screen => 6.0,
    };
    for y in 0..IMG as u32 {
        for x in 0..IMG as u32 {
            let n = rng.uniform(-tex, tex);
            let p = img.get_pixel(x, y).0;
            img.put_pixel(x, y, Rgb([clampf(p[0] as f32 + n), clampf(p[1] as f32 + n), clampf(p[2] as f32 + n)]));
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

fn to_gray(img: &RgbImage) -> Vec<f32> {
    let mut g = vec![0f32; IMG * IMG];
    for y in 0..IMG {
        for x in 0..IMG {
            let p = img.get_pixel(x as u32, y as u32).0;
            g[y * IMG + x] = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        }
    }
    g
}

// 特征1：明暗变化能量 —— 中央 20x20 分成 5x5 个 4x4 块，取块均值的方差
// （真人 3D 明暗 → 大；打印扁平 → 小）
fn feat_shading(g: &[f32]) -> f32 {
    let (o, side, blk) = (6usize, 20usize, 4usize);
    let nb = side / blk; // 5
    let mut means = Vec::new();
    for by in 0..nb {
        for bx in 0..nb {
            let mut s = 0f32;
            for yy in 0..blk {
                for xx in 0..blk {
                    let y = o + by * blk + yy;
                    let x = o + bx * blk + xx;
                    s += g[y * IMG + x];
                }
            }
            means.push(s / (blk * blk) as f32);
        }
    }
    let m = means.iter().sum::<f32>() / means.len() as f32;
    means.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / means.len() as f32
}

// 特征2：高频能量 —— 在两块“脸颊”皮肤区（避开眼/嘴的强边缘，否则会淹没摩尔纹信号）
// 上算拉普拉斯响应平方均值。屏幕摩尔纹 → 大；打印平滑 → 小；真人皮肤纹理 → 中。
fn feat_hf(g: &[f32]) -> f32 {
    // 两块脸颊 patch：(行 14..20, 列 6..11) 与 (行 14..20, 列 21..26)，都在脸内的皮肤上
    let patches = [(14usize, 20usize, 6usize, 11usize), (14, 20, 21, 26)];
    let mut s = 0f32;
    let mut n = 0f32;
    for &(y0, y1, x0, x1) in &patches {
        for y in y0..y1 {
            for x in x0..x1 {
                let lap = 4.0 * g[y * IMG + x]
                    - g[(y - 1) * IMG + x]
                    - g[(y + 1) * IMG + x]
                    - g[y * IMG + x - 1]
                    - g[y * IMG + x + 1];
                s += lap * lap;
                n += 1.0;
            }
        }
    }
    s / n
}

// ---------- 共享骨干：3 段卷积 + 全连接 → 64 维特征 ----------
struct Backbone {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc: Linear,
}
impl Backbone {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            fc: linear(64 * 4 * 4, 64, vb.pp("fc"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 32->16
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 16->8
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?; // 8->4
        let x = x.flatten_from(1)?;
        self.fc.forward(&x)?.relu() // [N,64]
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

fn accuracy(logits: &Tensor, labels: &[u32]) -> Result<f32> {
    let pred = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    Ok(pred.iter().zip(labels).filter(|(a, b)| a == b).count() as f32 / labels.len() as f32)
}

// ============ A：活体检测 ============
fn liveness(dev: &Device) -> Result<()> {
    println!("========== A. 活体检测（反欺骗）==========");
    // 造数据：真人 = 标签1；打印/屏幕 = 假体 = 标签0（各占一半假体）
    let gen = |n: usize, rng: &mut Lcg| -> (Vec<RgbImage>, Vec<u32>, Vec<Live>) {
        let mut imgs = Vec::new();
        let mut ys = Vec::new();
        let mut kinds = Vec::new();
        for _ in 0..n {
            let r = rng.below(4); // 用高位派生：0,1=真人；2=打印；3=屏幕（各占 1/2、1/4、1/4）
            let live = if r < 2 { Live::Real } else if r == 2 { Live::Print } else { Live::Screen };
            let o = FaceOpts { live, glasses: rng.boolean(), smile: rng.boolean(), hat: rng.boolean() };
            imgs.push(render(&o, rng));
            ys.push(if live == Live::Real { 1 } else { 0 });
            kinds.push(live);
        }
        (imgs, ys, kinds)
    };
    let mut trng = Lcg::new(11);
    let (tr_imgs, tr_y, tr_kind) = gen(600, &mut trng);
    let mut erng = Lcg::new(77);
    let (te_imgs, te_y, te_kind) = gen(300, &mut erng);
    let n_real = tr_y.iter().filter(|&&v| v == 1).count();
    println!("训练集 {} 张（真人 {}，假体 {}）；测试集 {} 张", tr_y.len(), n_real, tr_y.len() - n_real, te_y.len());

    // ---- 手工特征：明暗能量 + 高频能量 ----
    let feats = |imgs: &[RgbImage]| -> Vec<(f32, f32)> {
        imgs.iter().map(|im| {
            let g = to_gray(im);
            (feat_shading(&g), feat_hf(&g))
        }).collect()
    };
    let tr_f = feats(&tr_imgs);
    let te_f = feats(&te_imgs);
    // 分类型统计特征均值（看可分性）
    let mean_of = |kinds: &[Live], f: &[(f32, f32)], k: Live, idx: usize| -> f32 {
        let vs: Vec<f32> = kinds.iter().zip(f).filter(|(kk, _)| **kk == k).map(|(_, ff)| if idx == 0 { ff.0 } else { ff.1 }).collect();
        vs.iter().sum::<f32>() / vs.len().max(1) as f32
    };
    let (r_sh, p_sh, s_sh) = (mean_of(&tr_kind, &tr_f, Live::Real, 0), mean_of(&tr_kind, &tr_f, Live::Print, 0), mean_of(&tr_kind, &tr_f, Live::Screen, 0));
    let (r_hf, p_hf, s_hf) = (mean_of(&tr_kind, &tr_f, Live::Real, 1), mean_of(&tr_kind, &tr_f, Live::Print, 1), mean_of(&tr_kind, &tr_f, Live::Screen, 1));
    println!("\n  手工特征各类型均值（训练集）：");
    println!("    明暗能量  真人={r_sh:7.1}  打印={p_sh:7.1}  屏幕={s_sh:7.1}   (真人/屏幕高、打印低)");
    println!("    高频能量  真人={r_hf:7.1}  打印={p_hf:7.1}  屏幕={s_hf:7.1}   (屏幕摩尔纹最高)");
    // 两阈值规则：真人 iff 明暗能量>=t1 且 高频能量<=t2
    let t1 = (r_sh + p_sh) / 2.0; // 用“真人 vs 打印”定明暗阈值（抓打印）
    let t2 = (r_hf + s_hf) / 2.0; // 用“真人 vs 屏幕”定高频阈值（抓屏幕摩尔纹）
    println!("    规则阈值：明暗>= {t1:.1} 且 高频<= {t2:.1} → 判真人");
    let rule_acc = te_f.iter().zip(&te_y).filter(|((sh, hf), &y)| {
        let pred = if *sh >= t1 && *hf <= t2 { 1 } else { 0 };
        pred == y
    }).count() as f32 / te_y.len() as f32;
    println!("    手工特征两阈值规则  测试准确率 = {:.1}%", 100.0 * rule_acc);

    // ---- 小 CNN 二分类 ----
    let to_tensor = |imgs: &[RgbImage]| -> Result<Tensor> {
        let mut xs = Vec::with_capacity(imgs.len() * 3 * IMG * IMG);
        for im in imgs {
            xs.extend_from_slice(&img_to_chw(im));
        }
        Tensor::from_vec(xs, (imgs.len(), 3, IMG, IMG), dev)
    };
    let xtr = to_tensor(&tr_imgs)?;
    let xte = to_tensor(&te_imgs)?;
    let ytr_t = Tensor::from_vec(tr_y.clone(), (tr_y.len(),), dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let bb = Backbone::load(vb.pp("bb"))?;
    let head = linear(64, 2, vb.pp("head"))?; // 真/假 两类
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("\n  训练小 CNN 活体分类器（AdamW lr=1e-3）...");
    for epoch in 1..=170 {
        let logits = head.forward(&bb.forward(&xtr)?)?;
        let loss = cross_entropy(&logits, &ytr_t)?;
        opt.backward_step(&loss)?;
        if epoch % 34 == 0 {
            println!("    epoch {epoch:>3}  loss={:.4}  train_acc={:.3}", loss.to_scalar::<f32>()?, accuracy(&logits, &tr_y)?);
        }
    }
    let te_logits = head.forward(&bb.forward(&xte)?)?;
    let cnn_acc = accuracy(&te_logits, &te_y)?;
    // 按攻击类型看 CNN 的召回（假体被识破的比例）
    let pred = te_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let recall_of = |k: Live| -> (usize, usize) {
        let mut caught = 0;
        let mut tot = 0;
        for (i, kk) in te_kind.iter().enumerate() {
            if *kk == k {
                tot += 1;
                if pred[i] == te_y[i] {
                    caught += 1;
                }
            }
        }
        (caught, tot)
    };
    let (rc, rt) = recall_of(Live::Real);
    let (pc, pt) = recall_of(Live::Print);
    let (sc, st) = recall_of(Live::Screen);
    println!("\n  ==== 活体检测结果（测试集）====");
    println!("    手工特征规则 准确率 = {:.1}%", 100.0 * rule_acc);
    println!("    小 CNN       准确率 = {:.1}%", 100.0 * cnn_acc);
    println!("    CNN 分类型正确率：真人 {rc}/{rt}、打印攻击 {pc}/{pt}、屏幕翻拍 {sc}/{st}");

    // 存几张示例图（放大）
    let save = |o: &FaceOpts, name: &str| {
        let mut r = Lcg::new(555);
        let im = render(o, &mut r);
        let big = image::imageops::resize(&im, IMG as u32 * 6, IMG as u32 * 6, image::imageops::FilterType::Nearest);
        big.save(name).unwrap();
    };
    save(&FaceOpts { live: Live::Real, glasses: false, smile: true, hat: false }, "live_real.png");
    save(&FaceOpts { live: Live::Print, glasses: false, smile: true, hat: false }, "live_print.png");
    save(&FaceOpts { live: Live::Screen, glasses: false, smile: true, hat: false }, "live_screen.png");
    println!("    示例图已存：live_real.png / live_print.png / live_screen.png");
    Ok(())
}

// ============ B：人脸属性多任务分类 ============
fn attributes(dev: &Device) -> Result<()> {
    println!("\n========== B. 人脸属性多任务分类 ==========");
    // 造数据：真人脸，随机赋 眼镜/微笑/帽子 三个独立二值属性
    let gen = |n: usize, rng: &mut Lcg| -> (Vec<RgbImage>, [Vec<u32>; 3]) {
        let mut imgs = Vec::new();
        let mut yg = Vec::new();
        let mut ys = Vec::new();
        let mut yh = Vec::new();
        for _ in 0..n {
            let (g, s, h) = (rng.boolean(), rng.boolean(), rng.boolean());
            let o = FaceOpts { live: Live::Real, glasses: g, smile: s, hat: h };
            imgs.push(render(&o, rng));
            yg.push(g as u32);
            ys.push(s as u32);
            yh.push(h as u32);
        }
        (imgs, [yg, ys, yh])
    };
    let mut trng = Lcg::new(303);
    let (tr_imgs, tr_y) = gen(640, &mut trng);
    let mut erng = Lcg::new(909);
    let (te_imgs, te_y) = gen(300, &mut erng);
    println!("训练集 {} 张、测试集 {} 张；每张带 眼镜/微笑/帽子 三个独立标签", tr_imgs.len(), te_imgs.len());

    let to_tensor = |imgs: &[RgbImage]| -> Result<Tensor> {
        let mut xs = Vec::with_capacity(imgs.len() * 3 * IMG * IMG);
        for im in imgs {
            xs.extend_from_slice(&img_to_chw(im));
        }
        Tensor::from_vec(xs, (imgs.len(), 3, IMG, IMG), dev)
    };
    let xtr = to_tensor(&tr_imgs)?;
    let xte = to_tensor(&te_imgs)?;
    let names = ["眼镜", "微笑", "帽子"];
    let ytr_t: Vec<Tensor> = tr_y.iter().map(|y| Tensor::from_vec(y.clone(), (y.len(),), dev).unwrap()).collect();

    // 一个骨干 + 3 个分类头（共享表示、多任务）
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let bb = Backbone::load(vb.pp("bb"))?;
    let heads: Vec<Linear> = (0..3).map(|i| linear(64, 2, vb.pp(format!("head{i}"))).unwrap()).collect();
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("\n  训练多任务网络（1 个骨干 + 3 个分类头，损失=3 项交叉熵之和）...");
    for epoch in 1..=130 {
        let feat = bb.forward(&xtr)?;
        let mut loss = cross_entropy(&heads[0].forward(&feat)?, &ytr_t[0])?;
        for i in 1..3 {
            loss = loss.broadcast_add(&cross_entropy(&heads[i].forward(&feat)?, &ytr_t[i])?)?;
        }
        opt.backward_step(&loss)?;
        if epoch % 26 == 0 {
            println!("    epoch {epoch:>3}  total_loss={:.4}", loss.to_scalar::<f32>()?);
        }
    }
    // 逐属性测试准确率
    let feat = bb.forward(&xte)?;
    println!("\n  ==== 属性分析结果（测试集，逐属性准确率）====");
    let mut sum = 0f32;
    for i in 0..3 {
        let logits = heads[i].forward(&feat)?;
        let acc = accuracy(&logits, &te_y[i])?;
        sum += acc;
        println!("    {:<4} 准确率 = {:.1}%", names[i], 100.0 * acc);
    }
    println!("    三属性平均准确率 = {:.1}%", 100.0 * sum / 3.0);
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    liveness(&dev)?;
    attributes(&dev)?;
    Ok(())
}
