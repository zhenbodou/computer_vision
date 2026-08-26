// 第 52 章：小型热图回归网络做关键点检测（candle 0.11.0，纯 CPU，可复现）
// 运行：cd code/dl_labs/ch52_heatmap && cargo run --release
//
// 管线：合成三角形图（3 个角点 = 3 个关键点）→ 造 K 张高斯目标热图
//       → 编码器-解码器网络输出 K 张热图 → MSE 训练热图逼近高斯核
//       → 解码：每张热图 argmax（+ 抛物线亚像素精化）得预测坐标
//       → 报平均定位误差（像素）→ 存“预测热图叠加原图”PNG。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const IMG: usize = 48; // 输入 / 热图边长（本章热图与原图同分辨率，便于理解）
const K: usize = 3; // 关键点个数：三角形的 3 个角
const SIGMA: f32 = 2.0; // 高斯目标的标准差（像素）
const KP_NAMES: [&str; K] = ["顶点", "左下角", "右下角"];

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed，随机性自己掌控） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
    // [lo, hi) 内的浮点数
    fn rangef(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
}

// 一张图的 K 个关键点真值坐标（像素）
type Kps = [[f32; 2]; K];

// 合成一张 48x48 图：一个白色实心三角形（顶点在上，两底角在下），返回 3 个角的坐标
fn make_triangle(rng: &mut Lcg) -> (RgbImage, Kps) {
    let mut img = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    // 随机中心、底半宽、高；再给每个顶点加小抖动，避免完全对称（更接近真实关键点）
    let cx = rng.rangef(16.0, 32.0);
    let cy = rng.rangef(18.0, 30.0);
    let hw = rng.rangef(9.0, 13.0); // 底边半宽
    let h = rng.rangef(16.0, 24.0); // 三角形高
    let jit = |r: &mut Lcg| r.rangef(-2.0, 2.0);
    let apex = [cx + jit(rng), cy - h / 2.0 + jit(rng)]; // 关键点 0：顶点
    let bl = [cx - hw + jit(rng), cy + h / 2.0 + jit(rng)]; // 关键点 1：左下角
    let br = [cx + hw + jit(rng), cy + h / 2.0 + jit(rng)]; // 关键点 2：右下角
    let kps: Kps = [apex, bl, br];
    // 用重心符号法把三角形填成实心白色
    let sign = |p: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    };
    for y in 0..IMG {
        for x in 0..IMG {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let d1 = sign(p, apex, bl);
            let d2 = sign(p, bl, br);
            let d3 = sign(p, br, apex);
            let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            if !(has_neg && has_pos) {
                img.put_pixel(x as u32, y as u32, Rgb([235, 235, 235]));
            }
        }
    }
    (img, kps)
}

// 一张 RgbImage → CHW 排布的 [3,IMG,IMG] 像素向量（/255 归一化）
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

// 造一批数据：返回 (输入张量 [N,3,IMG,IMG], 每张图的关键点真值)
fn make_dataset(rng: &mut Lcg, n: usize, dev: &Device) -> Result<(Tensor, Vec<Kps>)> {
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    let mut all = Vec::with_capacity(n);
    for _ in 0..n {
        let (img, kps) = make_triangle(rng);
        xs.extend_from_slice(&img_to_chw(&img));
        all.push(kps);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    Ok((x, all))
}

// 由关键点真值构造目标热图张量 [N,K,IMG,IMG]：每个关键点一张以其为中心的高斯核（峰值 1）
fn build_heatmaps(all: &[Kps], dev: &Device) -> Result<Tensor> {
    let n = all.len();
    let mut hm = vec![0f32; n * K * IMG * IMG];
    let two_s2 = 2.0 * SIGMA * SIGMA;
    let rad = (3.0 * SIGMA).ceil() as i32; // 只在 3σ 邻域内填，够用且省事
    for (ni, kps) in all.iter().enumerate() {
        for (k, kp) in kps.iter().enumerate() {
            let (kx, ky) = (kp[0], kp[1]);
            let (cxi, cyi) = (kx.round() as i32, ky.round() as i32);
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (x, y) = (cxi + dx, cyi + dy);
                    if x < 0 || x >= IMG as i32 || y < 0 || y >= IMG as i32 {
                        continue;
                    }
                    let r2 = (x as f32 - kx).powi(2) + (y as f32 - ky).powi(2);
                    let v = (-r2 / two_s2).exp();
                    let idx = ((ni * K + k) * IMG + y as usize) * IMG + x as usize;
                    if v > hm[idx] {
                        hm[idx] = v;
                    }
                }
            }
        }
    }
    Tensor::from_vec(hm, (n, K, IMG, IMG), dev)
}

// ---------- 编码器-解码器热图网络：先降采样抽特征，再上采样回原分辨率，1x1 头输出 K 张热图 ----------
struct HeatmapNet {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    dec1: Conv2d,
    dec2: Conv2d,
    head: Conv2d,
}
impl HeatmapNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() }; // 3x3 保持边长
        let c0 = Conv2dConfig { padding: 0, ..Default::default() }; // 1x1 头
        Ok(Self {
            enc1: conv2d(3, 12, 3, c, vb.pp("enc1"))?,
            enc2: conv2d(12, 24, 3, c, vb.pp("enc2"))?,
            bott: conv2d(24, 24, 3, c, vb.pp("bott"))?,
            dec1: conv2d(24, 12, 3, c, vb.pp("dec1"))?,
            dec2: conv2d(12, 8, 3, c, vb.pp("dec2"))?,
            head: conv2d(8, K, 1, c0, vb.pp("head"))?, // → [N,K,IMG,IMG]
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.enc1.forward(x)?.relu()?.max_pool2d(2)?; // 48 -> 24
        let x = self.enc2.forward(&x)?.relu()?.max_pool2d(2)?; // 24 -> 12
        let x = self.bott.forward(&x)?.relu()?; // 12（瓶颈）
        let x = x.upsample_nearest2d(IMG / 2, IMG / 2)?; // 12 -> 24（最近邻上采样）
        let x = self.dec1.forward(&x)?.relu()?; // 24
        let x = x.upsample_nearest2d(IMG, IMG)?; // 24 -> 48
        let x = self.dec2.forward(&x)?.relu()?; // 48
        self.head.forward(&x) // [N,K,48,48]，每个关键点一张热图
    }
}

// 确定性初始化：bias 置 0，weight 取 [-lim,lim] 均匀分布，lim=1/sqrt(fan_in)
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

// 从一张热图 [IMG][IMG] 解码峰值坐标：先 argmax，再抛物线亚像素精化
fn decode_peak(hm: &[Vec<f32>], subpixel: bool) -> (f32, f32) {
    let (mut br, mut bc, mut bv) = (0usize, 0usize, f32::MIN);
    for r in 0..IMG {
        for c in 0..IMG {
            if hm[r][c] > bv {
                bv = hm[r][c];
                br = r;
                bc = c;
            }
        }
    }
    let (mut px, mut py) = (bc as f32, br as f32);
    if subpixel {
        // 对峰值左右邻居做抛物线拟合，顶点相对偏移 ∈ [-0.5, 0.5]
        if bc >= 1 && bc + 1 < IMG {
            let (l, m, r) = (hm[br][bc - 1], hm[br][bc], hm[br][bc + 1]);
            let denom = l + r - 2.0 * m;
            if denom.abs() > 1e-6 {
                px += ((l - r) / (2.0 * denom)).clamp(-0.5, 0.5);
            }
        }
        if br >= 1 && br + 1 < IMG {
            let (u, m, d) = (hm[br - 1][bc], hm[br][bc], hm[br + 1][bc]);
            let denom = u + d - 2.0 * m;
            if denom.abs() > 1e-6 {
                py += ((u - d) / (2.0 * denom)).clamp(-0.5, 0.5);
            }
        }
    }
    (px, py)
}

// 在一批测试图上评估：返回 (argmax 平均误差, 亚像素平均误差, 每个关键点的亚像素平均误差)
fn eval(
    model: &HeatmapNet,
    x: &Tensor,
    all: &[Kps],
    dev: &Device,
) -> Result<(f32, f32, [f32; K])> {
    let _ = dev;
    let out = model.forward(x)?; // [N,K,IMG,IMG]
    let n = all.len();
    let (mut sum_arg, mut sum_sub) = (0f32, 0f32);
    let mut per_kp = [0f32; K];
    for ni in 0..n {
        let maps = out.i(ni)?.to_vec3::<f32>()?; // [K][IMG][IMG]
        for k in 0..K {
            let (ax, ay) = decode_peak(&maps[k], false);
            let (sx, sy) = decode_peak(&maps[k], true);
            let (tx, ty) = (all[ni][k][0], all[ni][k][1]);
            let e_arg = ((ax - tx).powi(2) + (ay - ty).powi(2)).sqrt();
            let e_sub = ((sx - tx).powi(2) + (sy - ty).powi(2)).sqrt();
            sum_arg += e_arg;
            sum_sub += e_sub;
            per_kp[k] += e_sub;
        }
    }
    let denom = (n * K) as f32;
    for e in per_kp.iter_mut() {
        *e /= n as f32;
    }
    Ok((sum_arg / denom, sum_sub / denom, per_kp))
}

// 把“预测热图叠加原图”存成 PNG：原图放大，热图值染红，真值=绿圈，预测(亚像素)=黄叉
fn save_overlay(img: &RgbImage, maps: &[Vec<Vec<f32>>], kps: &Kps, path: &str) {
    const S: u32 = 6;
    let base = image::imageops::resize(img, IMG as u32 * S, IMG as u32 * S, image::imageops::FilterType::Nearest);
    let mut canvas = base;
    // 叠加热图：每像素取 K 张热图的最大值，作为红色强度加上去
    for y in 0..IMG {
        for x in 0..IMG {
            let mut hv = 0f32;
            for k in 0..K {
                if maps[k][y][x] > hv {
                    hv = maps[k][y][x];
                }
            }
            let hv = hv.clamp(0.0, 1.0);
            if hv > 0.05 {
                for sy in 0..S {
                    for sx in 0..S {
                        let (cx, cy) = (x as u32 * S + sx, y as u32 * S + sy);
                        let p = canvas.get_pixel(cx, cy).0;
                        let r = (p[0] as f32 + hv * 220.0).min(255.0) as u8;
                        let g = (p[1] as f32 * (1.0 - 0.4 * hv)) as u8;
                        let b = (p[2] as f32 * (1.0 - 0.4 * hv)) as u8;
                        canvas.put_pixel(cx, cy, Rgb([r, g, b]));
                    }
                }
            }
        }
    }
    let green = Rgb([40u8, 230, 90]);
    let yellow = Rgb([250u8, 230, 40]);
    for k in 0..K {
        // 真值：绿色空心圆
        let (tx, ty) = ((kps[k][0] * S as f32) as i32, (kps[k][1] * S as f32) as i32);
        imageproc::drawing::draw_hollow_circle_mut(&mut canvas, (tx, ty), 5, green);
        // 预测（亚像素 argmax）：黄色十字
        let (px, py) = decode_peak(&maps[k], true);
        let (pxi, pyi) = ((px * S as f32) as i32, (py * S as f32) as i32);
        imageproc::drawing::draw_cross_mut(&mut canvas, yellow, pxi, pyi);
    }
    canvas.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：合成训练集 + 目标热图 =====
    let mut rng = Lcg::new(7);
    let n_train = 192;
    let (xtr, kps_tr) = make_dataset(&mut rng, n_train, &dev)?;
    let target = build_heatmaps(&kps_tr, &dev)?; // [N,K,IMG,IMG]
    println!("训练集：{n_train} 张 48x48 三角形图，每张 {K} 个关键点（顶点/左下/右下）");
    println!("目标热图：每个关键点一张 {IMG}x{IMG} 高斯核（σ={SIGMA}，峰值 1）\n");

    // ===== 第 2 步：搭网络 + 确定性初始化 + 训练（MSE 热图回归）=====
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = HeatmapNet::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;

    println!("开始训练热图回归网络（AdamW, lr=3e-3→5e-4, 损失=MSE）...");
    println!("  {:>5}  {:>10}", "epoch", "MSE");
    let epochs = 250;
    for epoch in 1..=epochs {
        if epoch == 160 {
            opt.set_learning_rate(5e-4); // 后段降学习率，把峰值“磨”得更准
        }
        let pred = net.forward(&xtr)?;
        let loss = pred.broadcast_sub(&target)?.sqr()?.mean_all()?; // MSE
        opt.backward_step(&loss)?;
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {epoch:>5}  {:>10.6}", loss.to_scalar::<f32>()?);
        }
    }

    // ===== 第 3 步：存 safetensors → 加载回来（真实推理入口）=====
    vm.save("heatmap.safetensors")?;
    let dict = candle_core::safetensors::load("heatmap.safetensors", &dev)?;
    let mut keys: Vec<(String, Vec<usize>)> =
        dict.iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    keys.sort();
    println!("\n已保存 heatmap.safetensors，文件内张量（注意 head.weight = [K,8,1,1]）：");
    for (k, shape) in &keys {
        println!("  {k:<12} 形状 {shape:?}");
    }
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["heatmap.safetensors"], DType::F32, &dev)? };
    let model = HeatmapNet::load(vb2)?;

    // ===== 第 4 步：在全新测试集上评估定位误差（argmax vs 亚像素）=====
    let mut test_rng = Lcg::new(2025);
    let n_test = 64;
    let (xte, kps_te) = make_dataset(&mut test_rng, n_test, &dev)?;
    let (e_arg, e_sub, per_kp) = eval(&model, &xte, &kps_te, &dev)?;
    println!("\n===== 定位误差（{n_test} 张全新测试图，共 {} 个关键点）=====", n_test * K);
    println!("  argmax 解码平均误差    = {e_arg:.3} 像素");
    println!("  亚像素精化平均误差     = {e_sub:.3} 像素");
    println!("  分关键点（亚像素）：{} = {:.3},  {} = {:.3},  {} = {:.3} 像素",
        KP_NAMES[0], per_kp[0], KP_NAMES[1], per_kp[1], KP_NAMES[2], per_kp[2]);

    // ===== 第 5 步：存 3 张“预测热图叠加原图”PNG =====
    for idx in 0..3 {
        let one = xte.i(idx)?.unsqueeze(0)?; // [1,3,IMG,IMG]
        let out = model.forward(&one)?;
        let maps = out.i(0)?.to_vec3::<f32>()?; // [K][IMG][IMG]
        // 还原这张测试图（从 CHW 张量取回像素）
        let chw = one.i(0)?.to_vec3::<f32>()?;
        let mut img = RgbImage::new(IMG as u32, IMG as u32);
        for y in 0..IMG {
            for x in 0..IMG {
                let v = (chw[0][y][x] * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
            }
        }
        save_overlay(&img, &maps, &kps_te[idx], &format!("kp_{idx}.png"));
    }
    println!("\n已存 kp_0.png .. kp_2.png（红=预测热图，绿圈=真值，黄叉=亚像素预测）");
    Ok(())
}
