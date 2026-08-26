// 第 49 章：小型 U-Net 语义分割（candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现）
// 运行：cd code/dl_labs/ch49_unet_seg && cargo run --release
//
// 管线：合成图 + 逐像素类别 GT → 搭小型 U-Net（编码器下采样 + 解码器上采样 + 跳跃连接）
//       → 逐像素交叉熵训练 → 报训练 loss / 逐像素准确率 / 各类 IoU / mIoU
//       → 存「输入 / 预测掩码 / GT 掩码」三联对照 PNG
//       → 跳跃连接开/关消融对比 + DeepLab 空洞卷积（dilation）最小演示。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const SZ: usize = 28; // 输入边长（正方形，需能被 4 整除以便两次 2x2 池化）
const NC: usize = 4; // 类别数：背景 / 竖条 / 横条 / 方块
const CLASSES: [&str; NC] = ["背景", "竖条", "横条", "方块"];
// 每个类别的可视化颜色（画掩码 PNG 用）
const PALETTE: [[u8; 3]; NC] = [[40, 40, 50], [220, 60, 60], [60, 200, 90], [70, 120, 230]];
// 每个类别在输入图里的“基色”（再叠噪声）：背景暗，三类各偏红/绿/蓝
const BASECOLOR: [[f32; 3]; NC] = [
    [32.0, 32.0, 42.0],
    [205.0, 70.0, 62.0],
    [70.0, 195.0, 92.0],
    [72.0, 116.0, 222.0],
];

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed，随机性我们自己掌控） ----------
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
    // [lo, hi) 内的整数
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    // 近似标准正态：12 个均匀分布相加减 6（中心极限定理）
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// 合成一张 32x32 RGB 图 + 它的逐像素类别掩码 [SZ*SZ]（0=背景）。
// 随机放 2~3 个形状：竖条(1)/横条(2)/方块(3)，各有基色 + 高斯噪声；后画的覆盖先画的。
fn make_sample(rng: &mut Lcg) -> (RgbImage, Vec<u32>) {
    let mut mask = vec![0u32; SZ * SZ]; // 先全是背景
    let n = rng.range(2, 4); // 2..=3 个形状
    // 记录每个形状的类别与像素矩形范围（含内部噪声靠画的时候采样）
    let mut shapes: Vec<(usize, i32, i32, i32, i32)> = Vec::new(); // (cls,x0,y0,w,h)
    let mut tries = 0;
    while (shapes.len() as i32) < n && tries < 60 {
        tries += 1;
        let cls = rng.range(1, NC as i32) as usize; // 1..=3
        let (w, h) = match cls {
            1 => (rng.range(3, 6), rng.range(12, 21)),   // 竖条：窄而高
            2 => (rng.range(12, 21), rng.range(3, 6)),   // 横条：宽而矮
            _ => {
                let s = rng.range(8, 13);
                (s, s)
            } // 方块
        };
        let x0 = rng.range(1, SZ as i32 - w - 1);
        let y0 = rng.range(1, SZ as i32 - h - 1);
        shapes.push((cls, x0, y0, w, h));
    }
    // 把类别写进掩码（后画的覆盖先画的 → 天然产生遮挡边界）
    for &(cls, x0, y0, w, h) in &shapes {
        for yy in 0..h {
            for xx in 0..w {
                let (x, y) = (x0 + xx, y0 + yy);
                if x >= 0 && x < SZ as i32 && y >= 0 && y < SZ as i32 {
                    mask[y as usize * SZ + x as usize] = cls as u32;
                }
            }
        }
    }
    // 依据掩码给每个像素上「基色 + 噪声」
    let mut img = RgbImage::new(SZ as u32, SZ as u32);
    for y in 0..SZ {
        for x in 0..SZ {
            let cls = mask[y * SZ + x] as usize;
            let base = BASECOLOR[cls];
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = base[c] + rng.gauss() * 22.0; // 叠加噪声，避免“颜色=类别”的平凡查表
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            img.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    (img, mask)
}

// 一张 RgbImage → CHW 排布 [3,SZ,SZ] 的像素向量（/255 归一化）
fn img_to_chw(img: &RgbImage) -> Vec<f32> {
    let mut data = vec![0f32; 3 * SZ * SZ];
    for y in 0..SZ {
        for x in 0..SZ {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * SZ * SZ + y * SZ + x] = p[c] as f32 / 255.0;
            }
        }
    }
    data
}

// 造一批数据：返回 (输入张量 [N,3,SZ,SZ], 目标掩码张量 U32 [N,SZ,SZ], 原图列表, 掩码列表)
fn make_dataset(
    rng: &mut Lcg,
    n: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor, Vec<RgbImage>, Vec<Vec<u32>>)> {
    let mut xs = Vec::with_capacity(n * 3 * SZ * SZ);
    let mut ys = Vec::with_capacity(n * SZ * SZ);
    let mut imgs = Vec::with_capacity(n);
    let mut masks = Vec::with_capacity(n);
    for _ in 0..n {
        let (img, mask) = make_sample(rng);
        xs.extend_from_slice(&img_to_chw(&img));
        ys.extend_from_slice(&mask);
        imgs.push(img);
        masks.push(mask);
    }
    let x = Tensor::from_vec(xs, (n, 3, SZ, SZ), dev)?;
    let y = Tensor::from_vec(ys, (n, SZ, SZ), dev)?; // U32 目标（逐像素类别）
    Ok((x, y, imgs, masks))
}

// ---------- 小型 U-Net：2 次下采样 + 2 次上采样 + 2 条跳跃连接 ----------
// 编码器： e1[16,SZ,SZ] --pool--> [16,SZ/2] --conv--> e2[32,SZ/2] --pool--> [32,SZ/4] --conv--> b[64,SZ/4]
// 解码器： 上采样 b → 减通道 → 与 e2 拼接(skip) → conv → 上采样 → 减通道 → 与 e1 拼接(skip) → conv → 1x1 头出 logits
struct UNet {
    enc1: Conv2d,     // 3 -> 16
    enc2: Conv2d,     // 16 -> 32
    bott: Conv2d,     // 32 -> 64（瓶颈）
    up2: Conv2d,      // 64 -> 32（上采样后减通道）
    dec2: Conv2d,     // 64 -> 32（拼接 e2 后：32+32=64）
    up1: Conv2d,      // 32 -> 16
    dec1: Conv2d,     // 32 -> 16（拼接 e1 后：16+16=32）
    head: Conv2d,     // 16 -> NC（1x1，逐像素分类头）
    use_skip: bool,   // 消融开关：false 时跳跃连接送零，模拟“没有 skip”
}
impl UNet {
    fn load(vb: VarBuilder, use_skip: bool) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() }; // 3x3 保持边长
        let c1 = Conv2dConfig { padding: 0, ..Default::default() }; // 1x1 头
        Ok(Self {
            enc1: conv2d(3, 16, 3, c, vb.pp("enc1"))?,
            enc2: conv2d(16, 32, 3, c, vb.pp("enc2"))?,
            bott: conv2d(32, 64, 3, c, vb.pp("bott"))?,
            up2: conv2d(64, 32, 3, c, vb.pp("up2"))?,
            dec2: conv2d(64, 32, 3, c, vb.pp("dec2"))?,
            up1: conv2d(32, 16, 3, c, vb.pp("up1"))?,
            dec1: conv2d(32, 16, 3, c, vb.pp("dec1"))?,
            head: conv2d(16, NC, 1, c1, vb.pp("head"))?,
            use_skip,
        })
    }
    // 前向：返回 logits [N,NC,SZ,SZ]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ---- 编码器（抽语义，逐步下采样）----
        let e1 = self.enc1.forward(x)?.relu()?; // [N,16,SZ,SZ]（细节层，供 skip）
        let p1 = e1.max_pool2d(2)?; // [N,16,SZ/2,SZ/2]
        let e2 = self.enc2.forward(&p1)?.relu()?; // [N,32,SZ/2,SZ/2]（中层，供 skip）
        let p2 = e2.max_pool2d(2)?; // [N,32,SZ/4,SZ/4]
        let b = self.bott.forward(&p2)?.relu()?; // [N,64,SZ/4,SZ/4]（瓶颈：感受野最大、分辨率最低）

        // ---- 解码器（恢复分辨率；candle 无转置卷积 → 用“最近邻上采样 + 卷积”）----
        let u2 = b.upsample_nearest2d(SZ / 2, SZ / 2)?; // → [N,64,SZ/2,SZ/2]
        let u2 = self.up2.forward(&u2)?.relu()?; // [N,32,SZ/2,SZ/2]
        let skip2 = if self.use_skip { e2.clone() } else { e2.zeros_like()? }; // 跳跃连接（或送零）
        let c2 = Tensor::cat(&[&u2, &skip2], 1)?; // 通道维拼接 → [N,64,SZ/2,SZ/2]
        let d2 = self.dec2.forward(&c2)?.relu()?; // [N,32,SZ/2,SZ/2]

        let u1 = d2.upsample_nearest2d(SZ, SZ)?; // → [N,32,SZ,SZ]
        let u1 = self.up1.forward(&u1)?.relu()?; // [N,16,SZ,SZ]
        let skip1 = if self.use_skip { e1.clone() } else { e1.zeros_like()? };
        let c1 = Tensor::cat(&[&u1, &skip1], 1)?; // → [N,32,SZ,SZ]
        let d1 = self.dec1.forward(&c1)?.relu()?; // [N,16,SZ,SZ]

        self.head.forward(&d1) // [N,NC,SZ,SZ]，每个像素一份类别 logits
    }
}

// 确定性初始化：bias 置 0，weight 取 [-lim,lim] 均匀分布，lim=1/sqrt(fan_in)（PyTorch 风格）
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

// 逐像素交叉熵：把 logits [N,NC,H,W] 摊平成 [N*H*W, NC]，target [N,H,W] 摊平成 [N*H*W]（U32）
fn pixel_ce(logits: &Tensor, target_u32: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = logits.dims4()?;
    let flat = logits
        .permute((0, 2, 3, 1))? // [N,H,W,C]
        .contiguous()? // permute 后内存非连续，reshape 前先连续化
        .reshape((n * h * w, c))?; // [N*H*W, C]
    let tgt = target_u32.reshape((n * h * w,))?; // [N*H*W]
    candle_nn::loss::cross_entropy(&flat, &tgt) // 内部 log_softmax + nll，target 必须 U32
}

// 在测试集上评估：逐像素准确率 + 各类 IoU + mIoU。
// 返回 (pixel_acc, per_class_iou[NC], mIoU)
fn evaluate(model: &UNet, x: &Tensor, masks: &[Vec<u32>]) -> Result<(f32, [f32; NC], f32)> {
    let logits = model.forward(x)?; // [N,NC,SZ,SZ]
    let nimg = masks.len();
    let mut correct = 0usize;
    let mut total = 0usize;
    // 每类的交、并计数
    let mut inter = [0usize; NC];
    let mut union = [0usize; NC];
    for k in 0..nimg {
        // 第 k 张图的预测：logits.i(k) → [NC,SZ,SZ]，对通道维取 argmax → [SZ,SZ]
        let pred = logits.i(k)?.argmax(0)?.to_vec2::<u32>()?; // [SZ][SZ]
        let gt = &masks[k];
        for y in 0..SZ {
            for x in 0..SZ {
                let p = pred[y][x] as usize;
                let g = gt[y * SZ + x] as usize;
                total += 1;
                if p == g {
                    correct += 1;
                }
                // 逐类累计混淆：命中该类的交与并
                for cls in 0..NC {
                    let in_p = p == cls;
                    let in_g = g == cls;
                    if in_p && in_g {
                        inter[cls] += 1;
                    }
                    if in_p || in_g {
                        union[cls] += 1;
                    }
                }
            }
        }
    }
    let pixel_acc = correct as f32 / total as f32;
    let mut ious = [0f32; NC];
    let mut sum = 0f32;
    for cls in 0..NC {
        ious[cls] = if union[cls] == 0 {
            1.0 // 该类在测试集里不存在也未误报 → IoU 记为 1（不惩罚）
        } else {
            inter[cls] as f32 / union[cls] as f32
        };
        sum += ious[cls];
    }
    Ok((pixel_acc, ious, sum / NC as f32))
}

// 训练一个 U-Net（use_skip 决定是否启用跳跃连接），返回训练好的模型和它的 VarMap
fn train_unet(
    use_skip: bool,
    xtr: &Tensor,
    ytr: &Tensor,
    epochs: usize,
    verbose: bool,
    dev: &Device,
) -> Result<(UNet, VarMap)> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = UNet::load(vb, use_skip)?;
    det_init(&vm, 1234)?; // 固定种子 → 两个消融模型从同一初始权重出发，公平对比
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    for epoch in 1..=epochs {
        if epoch == (epochs * 3 / 4) {
            opt.set_learning_rate(5e-4); // 后段降 lr，把边界磨细
        }
        let logits = net.forward(xtr)?;
        let loss = pixel_ce(&logits, ytr)?;
        opt.backward_step(&loss)?;
        if verbose && (epoch % 20 == 0 || epoch == 1) {
            println!("  epoch {epoch:>3}  逐像素交叉熵 loss = {:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok((net, vm))
}

// 把「输入图 | 预测掩码 | GT 掩码」拼成一张三联对照图并存 PNG（放大 SCALE 倍，掩码按调色板上色）
fn save_triptych(img: &RgbImage, pred: &[Vec<u32>], gt: &[u32], path: &str) {
    const SCALE: u32 = 6;
    const GAP: u32 = 4;
    let cell = SZ as u32 * SCALE;
    let canvas_w = cell * 3 + GAP * 2;
    let mut canvas = RgbImage::from_pixel(canvas_w, cell, Rgb([255, 255, 255]));
    // ① 输入图（放大）
    let big = image::imageops::resize(img, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &big, 0, 0);
    // ② 预测掩码上色
    let mut pm = RgbImage::new(SZ as u32, SZ as u32);
    for y in 0..SZ {
        for x in 0..SZ {
            pm.put_pixel(x as u32, y as u32, Rgb(PALETTE[pred[y][x] as usize]));
        }
    }
    let pmb = image::imageops::resize(&pm, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &pmb, (cell + GAP) as i64, 0);
    // ③ GT 掩码上色
    let mut gm = RgbImage::new(SZ as u32, SZ as u32);
    for y in 0..SZ {
        for x in 0..SZ {
            gm.put_pixel(x as u32, y as u32, Rgb(PALETTE[gt[y * SZ + x] as usize]));
        }
    }
    let gmb = image::imageops::resize(&gm, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &gmb, (cell * 2 + GAP * 2) as i64, 0);
    canvas.save(path).unwrap();
}

// DeepLab 空洞卷积（atrous / dilated convolution）最小演示：
// 同样的 3x3 卷积核，dilation=1 感受野 3x3；dilation=2 感受野扩到 5x5，但输出分辨率不变、参数量不变。
fn atrous_demo(dev: &Device) -> Result<()> {
    println!("\n===== DeepLab 空洞卷积（dilation）最小演示 =====");
    // 一块 8x8 的单通道特征（模拟瓶颈层的空间尺寸）
    let feat = Tensor::ones((1, 8, 8, 8), DType::F32, dev)?; // [N=1, C=8, 8, 8]
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    // 普通 3x3 卷积：padding=1 保持 8x8，感受野 3x3
    let normal = conv2d(8, 8, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("normal"))?;
    // 空洞 3x3 卷积：dilation=2 → 感受野 5x5；配 padding=2 仍保持 8x8。参数量与普通 3x3 完全相同。
    let atrous = conv2d(8, 8, 3, Conv2dConfig { padding: 2, dilation: 2, ..Default::default() }, vb.pp("atrous"))?;
    det_init(&vm, 7)?;
    let y_n = normal.forward(&feat)?;
    let y_a = atrous.forward(&feat)?;
    let rf = |d: usize| 1 + (3 - 1) * d; // 3x3 核在 dilation=d 下的一维感受野
    println!("  普通卷积  dilation=1：输出 {:?}，感受野 {}x{}，权重数 {}", y_n.dims(), rf(1), rf(1), normal.weight().elem_count());
    println!("  空洞卷积  dilation=2：输出 {:?}，感受野 {}x{}，权重数 {}", y_a.dims(), rf(2), rf(2), atrous.weight().elem_count());
    println!("  → 结论：空洞卷积用同样多的参数、不降低分辨率，就把感受野从 3x3 扩到 5x5（DeepLab 借此'看得更宽'）。");
    Ok(())
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：合成训练/测试数据 =====
    let mut rng = Lcg::new(20);
    let n_train = 96;
    let n_test = 48;
    let (xtr, ytr, _imgs_tr, _masks_tr) = make_dataset(&mut rng, n_train, &dev)?;
    let (xte, _yte, imgs_te, masks_te) = make_dataset(&mut rng, n_test, &dev)?;
    // 统计训练集各类像素占比（分割任务里“类别不平衡”很常见：背景像素远多于目标）
    let mut cls_px = [0usize; NC];
    for m in _masks_tr.iter() {
        for &c in m {
            cls_px[c as usize] += 1;
        }
    }
    let tot_px: usize = cls_px.iter().sum();
    println!("训练集 {n_train} 张 {SZ}x{SZ} 图，测试集 {n_test} 张。各类像素占比：");
    for c in 0..NC {
        println!("  {:<4} {:>5.1}%", CLASSES[c], 100.0 * cls_px[c] as f32 / tot_px as f32);
    }

    // ===== 第 2 步：训练带跳跃连接的小型 U-Net =====
    let epochs = 110;
    println!("\n开始训练小型 U-Net（带跳跃连接，AdamW, lr=2e-3→5e-4，逐像素交叉熵）...");
    let (net, vm) = train_unet(true, &xtr, &ytr, epochs, true, &dev)?;

    // 存 safetensors（与加载真实预训练分割模型是同一套 API）
    vm.save("unet.safetensors")?;
    let dict = candle_core::safetensors::load("unet.safetensors", &dev)?;
    let mut keys: Vec<(String, Vec<usize>)> = dict.iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    keys.sort();
    println!("\n已保存 unet.safetensors，文件内张量（部分）：");
    for (k, shape) in keys.iter().take(4) {
        println!("  {k:<12} 形状 {shape:?}");
    }

    // 从 safetensors 加载回来做推理（真实推理入口）
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["unet.safetensors"], DType::F32, &dev)? };
    let model = UNet::load(vb2, true)?;

    // ===== 第 3 步：在测试集上评估 逐像素准确率 / 各类 IoU / mIoU =====
    let (acc, ious, miou) = evaluate(&model, &xte, &masks_te)?;
    println!("\n===== 测试集评估（带跳跃连接）=====");
    println!("逐像素准确率 = {:.2}%", acc * 100.0);
    for c in 0..NC {
        println!("  IoU[{:<4}] = {:.3}", CLASSES[c], ious[c]);
    }
    println!("mIoU（各类 IoU 平均）= {:.3}", miou);

    // ===== 第 4 步：存几张「输入 / 预测 / GT」三联对照 PNG =====
    let logits = model.forward(&xte)?;
    for k in 0..5 {
        let pred = logits.i(k)?.argmax(0)?.to_vec2::<u32>()?;
        save_triptych(&imgs_te[k], &pred, &masks_te[k], &format!("seg_{k}.png"));
    }
    println!("\n已存三联对照图 seg_0.png .. seg_4.png（左：输入图，中：预测掩码，右：GT 掩码）");
    println!("掩码配色：背景=灰, 竖条=红, 横条=绿, 方块=蓝");

    // ===== 第 5 步：跳跃连接开/关 消融对比（同初始权重、同数据、同轮数）=====
    println!("\n===== 跳跃连接消融对比（skip on vs off，其余完全一致）=====");
    let (net_ns, _vm_ns) = train_unet(false, &xtr, &ytr, epochs, false, &dev)?; // 关闭 skip
    let (acc_s, _i_s, miou_s) = evaluate(&net, &xte, &masks_te)?; // 复用第 2 步训练好的 skip 模型
    let (acc_ns, _i_ns, miou_ns) = evaluate(&net_ns, &xte, &masks_te)?;
    println!("  有跳跃连接：逐像素准确率 {:.2}%   mIoU {:.3}", acc_s * 100.0, miou_s);
    println!("  无跳跃连接：逐像素准确率 {:.2}%   mIoU {:.3}", acc_ns * 100.0, miou_ns);
    println!("  → 跳跃连接把 mIoU 提升了 {:.3}（细节/边界靠它从编码器直送解码器恢复）", miou_s - miou_ns);

    // ===== 第 6 步：DeepLab 空洞卷积演示 =====
    atrous_demo(&dev)?;
    Ok(())
}
