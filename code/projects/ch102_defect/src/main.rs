// 第 102 章 项目：工业缺陷检测系统（candle 0.11.0，纯 CPU，无需下载模型；结果可复现）
// 运行：cd code/projects/ch102_defect && cargo run --release
//
// 两条路线，同一套后处理（分割掩码 → 阈值 → 连通域 → 缺陷框 + 严重度）：
//   路线 A（监督分割）：有缺陷标注时，用小 U-Net（复用第 49 章）逐像素分出缺陷区。
//   路线 B（异常检测，工业界更常用）：只用【正常样本】训自编码器（复用第 69 章），
//        重建误差高 = 缺陷。无需任何缺陷标注（真实产线缺陷稀少、形态未知）。
// 报真实数字：缺陷像素 IoU、缺陷级检出率/误报，并对比两条路线；存缺陷叠加 PNG。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use std::collections::VecDeque;

const SZ: usize = 32; // 图像边长（正方形灰度）
const LATENT: usize = 32; // 自编码器瓶颈维度（<< 1024 像素，强迫只学“正常纹理”）
const MIN_AREA: usize = 4; // 连通域最小面积（滤掉零星噪点）

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed；且只用高位，避免 LCG 低位规律） ----------
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
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // 取高位做 [0,1)
    }
    // [lo,hi) 内整数——用高位缩放，绝不用 self.next() % k（LCG 低位质量差）
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.unit() * (hi - lo) as f32) as i32
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// ---------- 一个缺陷：类型 + 逐像素掩码 + 外接框 ----------
#[derive(Clone)]
struct Defect {
    kind: usize, // 0 划痕 / 1 孔洞 / 2 污点
    mask: Vec<bool>,
    bbox: [i32; 4],
}
const KINDS: [&str; 3] = ["划痕", "孔洞", "污点"];

// 合成一张“工业表面”灰度图（值域 [0,1]）：水平纹理条纹 + 噪声（模拟拉丝/织纹）。
// add_defects 为真时叠 1~3 个缺陷（划痕/孔洞/污点），返回缺陷并集掩码与逐缺陷列表。
fn make_surface(rng: &mut Lcg, add_defects: bool) -> (Vec<f32>, Vec<bool>, Vec<Defect>) {
    let phase = rng.unit() * 6.28; // 每张图条纹相位不同 → 正常纹理有多样性，但仍低维（AE 可学）
    let bright = 0.48 + (rng.unit() - 0.5) * 0.06; // 整体亮度轻微波动
    let mut g = vec![0f32; SZ * SZ];
    for y in 0..SZ {
        let stripe = 0.11 * (y as f32 * 0.85 + phase).sin(); // 水平条纹
        for x in 0..SZ {
            let v = bright + stripe + rng.gauss() * 0.028; // 叠噪声
            g[y * SZ + x] = v.clamp(0.0, 1.0);
        }
    }
    let mut mask = vec![false; SZ * SZ];
    let mut defects = Vec::new();
    if add_defects {
        let n = rng.range(1, 4); // 1..=3 个缺陷
        for _ in 0..n {
            let kind = rng.range(0, 3) as usize;
            let mut dmask = vec![false; SZ * SZ];
            let (mut x0, mut y0, mut x1, mut y1) = (SZ as i32, SZ as i32, 0, 0);
            let mut paint = |x: i32, y: i32, val: f32, m: &mut Vec<bool>, bb: &mut (i32, i32, i32, i32)| {
                if x >= 0 && y >= 0 && x < SZ as i32 && y < SZ as i32 {
                    g[y as usize * SZ + x as usize] = val;
                    m[y as usize * SZ + x as usize] = true;
                    bb.0 = bb.0.min(x);
                    bb.1 = bb.1.min(y);
                    bb.2 = bb.2.max(x);
                    bb.3 = bb.3.max(y);
                }
            };
            let mut bb = (x0, y0, x1, y1);
            match kind {
                0 => {
                    // 划痕：一条亮线（宽 2），近竖直/斜，长 10~18 —— 打断水平条纹，最“异常”
                    let len = rng.range(10, 19);
                    let sx = rng.range(3, SZ as i32 - 3);
                    let sy = rng.range(2, SZ as i32 - len - 2).max(2);
                    let slope = (rng.unit() - 0.5) * 1.2; // 轻微倾斜
                    for t in 0..len {
                        let cx = sx + (slope * t as f32) as i32;
                        for w in 0..2 {
                            paint(cx + w, sy + t, 0.92, &mut dmask, &mut bb);
                        }
                    }
                }
                1 => {
                    // 孔洞：暗色圆盘，半径 2~4
                    let r = rng.range(2, 5);
                    let cx = rng.range(r + 1, SZ as i32 - r - 1);
                    let cy = rng.range(r + 1, SZ as i32 - r - 1);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            if dx * dx + dy * dy <= r * r {
                                paint(cx + dx, cy + dy, 0.08, &mut dmask, &mut bb);
                            }
                        }
                    }
                }
                _ => {
                    // 污点：灰色不规则团块，半径 3~5（用带扰动的圆）
                    let r = rng.range(3, 6);
                    let cx = rng.range(r + 1, SZ as i32 - r - 1);
                    let cy = rng.range(r + 1, SZ as i32 - r - 1);
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let rr = r as f32 - 0.8 + rng.unit() * 1.6; // 边界扰动 → 不规则
                            if (dx * dx + dy * dy) as f32 <= rr * rr {
                                paint(cx + dx, cy + dy, 0.32, &mut dmask, &mut bb);
                            }
                        }
                    }
                }
            }
            for (i, &d) in dmask.iter().enumerate() {
                if d {
                    mask[i] = true;
                }
            }
            (x0, y0, x1, y1) = bb;
            defects.push(Defect { kind, mask: dmask, bbox: [x0, y0, x1, y1] });
        }
    }
    (g, mask, defects)
}

fn gray_to_tensor(batch: &[Vec<f32>], dev: &Device) -> Result<Tensor> {
    let n = batch.len();
    let mut xs = Vec::with_capacity(n * SZ * SZ);
    for g in batch {
        xs.extend_from_slice(g);
    }
    Tensor::from_vec(xs, (n, 1, SZ, SZ), dev)
}

// ============================ 路线 A：监督分割（小 U-Net，复用第 49 章） ============================
// 1 通道输入 → 2 类输出（正常 / 缺陷）。编码 32→16→8，解码 8→16→32（全 2× 整数上采样，反向安全）。
struct UNet {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    up2: Conv2d,
    dec2: Conv2d,
    up1: Conv2d,
    dec1: Conv2d,
    head: Conv2d,
}
impl UNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        let c1 = Conv2dConfig { padding: 0, ..Default::default() };
        Ok(Self {
            enc1: conv2d(1, 16, 3, c, vb.pp("enc1"))?,
            enc2: conv2d(16, 32, 3, c, vb.pp("enc2"))?,
            bott: conv2d(32, 64, 3, c, vb.pp("bott"))?,
            up2: conv2d(64, 32, 3, c, vb.pp("up2"))?,
            dec2: conv2d(64, 32, 3, c, vb.pp("dec2"))?,
            up1: conv2d(32, 16, 3, c, vb.pp("up1"))?,
            dec1: conv2d(32, 16, 3, c, vb.pp("dec1"))?,
            head: conv2d(16, 2, 1, c1, vb.pp("head"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let e1 = self.enc1.forward(x)?.relu()?;
        let p1 = e1.max_pool2d(2)?;
        let e2 = self.enc2.forward(&p1)?.relu()?;
        let p2 = e2.max_pool2d(2)?;
        let b = self.bott.forward(&p2)?.relu()?;
        let u2 = b.upsample_nearest2d(SZ / 2, SZ / 2)?;
        let u2 = self.up2.forward(&u2)?.relu()?;
        let d2 = self.dec2.forward(&Tensor::cat(&[&u2, &e2], 1)?)?.relu()?;
        let u1 = d2.upsample_nearest2d(SZ, SZ)?;
        let u1 = self.up1.forward(&u1)?.relu()?;
        let d1 = self.dec1.forward(&Tensor::cat(&[&u1, &e1], 1)?)?.relu()?;
        self.head.forward(&d1)
    }
}

// ============================ 路线 B：自编码器（复用第 69 章，只学正常纹理） ============================
struct AutoEnc {
    e1: Conv2d,
    e2: Conv2d,
    efc: Linear,
    dfc: Linear,
    d1: Conv2d,
    d2: Conv2d,
}
impl AutoEnc {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            e1: conv2d(1, 8, 3, c, vb.pp("e1"))?,
            e2: conv2d(8, 16, 3, c, vb.pp("e2"))?,
            efc: linear(16 * (SZ / 4) * (SZ / 4), LATENT, vb.pp("efc"))?, // 16*8*8=1024 → 32
            dfc: linear(LATENT, 16 * (SZ / 4) * (SZ / 4), vb.pp("dfc"))?,
            d1: conv2d(16, 8, 3, c, vb.pp("d1"))?,
            d2: conv2d(8, 1, 3, c, vb.pp("d2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let h = self.e1.forward(x)?.relu()?.max_pool2d(2)?; // 32→16
        let h = self.e2.forward(&h)?.relu()?.max_pool2d(2)?; // 16→8
        let z = self.efc.forward(&h.flatten_from(1)?)?; // 瓶颈
        let x = self.dfc.forward(&z)?.relu()?.reshape((n, 16, SZ / 4, SZ / 4))?;
        let x = x.upsample_nearest2d(SZ / 2, SZ / 2)?; // 8→16
        let x = self.d1.forward(&x)?.relu()?;
        let x = x.upsample_nearest2d(SZ, SZ)?; // 16→32
        let x = self.d2.forward(&x)?;
        candle_nn::ops::sigmoid(&x) // 压回 [0,1]，与输入同域
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

// 类别加权逐像素交叉熵（第 49 章逐像素 CE 的加权版）：缺陷像素只占约 7%，普通 CE 会“偷懒
// 全预测正常”。给缺陷像素更大权重 w_pos，逼网络把稀少的缺陷也认真分出来（工业分割常用的类不平衡对策）。
fn pixel_ce_weighted(logits: &Tensor, target_u32: &Tensor, w_pos: f64) -> Result<Tensor> {
    let (n, c, h, w) = logits.dims4()?;
    let p = n * h * w;
    let flat = logits.permute((0, 2, 3, 1))?.contiguous()?.reshape((p, c))?; // [P,2]
    let logp = candle_nn::ops::log_softmax(&flat, 1)?; // [P,2]
    let picked = logp.gather(&target_u32.reshape((p, 1))?, 1)?.reshape((p,))?; // 真值类的 log 概率
    let tgtf = target_u32.reshape((p,))?.to_dtype(DType::F32)?; // 0/1
    let weight = tgtf.affine(w_pos - 1.0, 1.0)?; // = 1 + (w_pos-1)*缺陷 → 缺陷像素权重 w_pos
    let wsum = weight.sum_all()?;
    picked.neg()?.broadcast_mul(&weight)?.sum_all()?.broadcast_div(&wsum)
}

// ---------- 后处理：连通域（8 邻接 BFS），返回每个缺陷块的掩码/面积/外接框 ----------
struct Comp {
    area: usize,
    bbox: [i32; 4],
    mask: Vec<bool>,
}
fn connected_components(mask: &[bool], min_area: usize) -> Vec<Comp> {
    let mut visited = vec![false; mask.len()];
    let mut comps = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        visited[start] = true;
        let mut cells = Vec::new();
        while let Some(idx) = q.pop_front() {
            cells.push(idx);
            let (x, y) = ((idx % SZ) as i32, (idx / SZ) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= SZ as i32 || ny >= SZ as i32 {
                        continue;
                    }
                    let nidx = ny as usize * SZ + nx as usize;
                    if mask[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        q.push_back(nidx);
                    }
                }
            }
        }
        if cells.len() < min_area {
            continue;
        }
        let mut cm = vec![false; mask.len()];
        let (mut x0, mut y0, mut x1, mut y1) = (SZ as i32, SZ as i32, 0, 0);
        for &c in &cells {
            cm[c] = true;
            let (x, y) = ((c % SZ) as i32, (c / SZ) as i32);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        comps.push(Comp { area: cells.len(), bbox: [x0, y0, x1, y1], mask: cm });
    }
    comps
}

fn mask_iou(a: &[bool], b: &[bool]) -> f32 {
    let (mut i, mut u) = (0usize, 0usize);
    for k in 0..a.len() {
        if a[k] && b[k] {
            i += 1;
        }
        if a[k] || b[k] {
            u += 1;
        }
    }
    if u == 0 {
        0.0
    } else {
        i as f32 / u as f32
    }
}
fn overlaps(a: &[bool], b: &[bool]) -> bool {
    a.iter().zip(b).any(|(&p, &q)| p && q)
}

// 用【正常验证集】的重建误差统计定阈值：τ = 均值 + k·标准差（异常检测的经典无监督定阈）
fn error_map(input: &[f32], recon: &[f32]) -> Vec<f32> {
    // 逐像素平方误差 + 3x3 均值平滑（抑制噪声，突出成片缺陷）
    let raw: Vec<f32> = input.iter().zip(recon).map(|(&a, &b)| (a - b) * (a - b)).collect();
    let mut sm = vec![0f32; SZ * SZ];
    for y in 0..SZ {
        for x in 0..SZ {
            let (mut s, mut c) = (0f32, 0f32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && nx < SZ as i32 && ny < SZ as i32 {
                        s += raw[ny as usize * SZ + nx as usize];
                        c += 1.0;
                    }
                }
            }
            sm[y * SZ + x] = s / c;
        }
    }
    sm
}

// 灰度 [0,1] → RgbImage；把缺陷掩码涂色叠加，画连通域框(红)与 GT 缺陷框(绿)
fn overlay_png(gray: &[f32], mask: &[bool], comps: &[Comp], gts: &[Defect], color: [u8; 3], path: &str) {
    const S: u32 = 8;
    let mut img = RgbImage::new(SZ as u32, SZ as u32);
    for y in 0..SZ {
        for x in 0..SZ {
            let v = (gray[y * SZ + x].clamp(0.0, 1.0) * 255.0) as u8;
            let px = if mask[y * SZ + x] {
                [
                    ((v as u16 + color[0] as u16) / 2) as u8,
                    ((v as u16 + color[1] as u16) / 2) as u8,
                    ((v as u16 + color[2] as u16) / 2) as u8,
                ]
            } else {
                [v, v, v]
            };
            img.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    let mut big = image::imageops::resize(&img, SZ as u32 * S, SZ as u32 * S, image::imageops::FilterType::Nearest);
    let rect = |b: [i32; 4]| Rect::at(b[0] * S as i32, b[1] * S as i32).of_size(((b[2] - b[0] + 1) as u32 * S).max(1), ((b[3] - b[1] + 1) as u32 * S).max(1));
    for d in gts {
        draw_hollow_rect_mut(&mut big, rect(d.bbox), Rgb([60, 220, 90])); // GT 框：绿
    }
    for cp in comps {
        draw_hollow_rect_mut(&mut big, rect(cp.bbox), Rgb([255, 80, 80])); // 检出框：红
    }
    big.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 造数据 =====
    // 监督分割用：带缺陷 + GT 掩码；异常检测用：只用正常图训练。测试集：带缺陷 + GT。
    let mut rng = Lcg::new(7);
    let (n_def, n_normal, n_test) = (80usize, 80usize, 60usize);
    let mut def_imgs = Vec::new();
    let mut def_masks = Vec::new();
    for _ in 0..n_def {
        let (g, m, _) = make_surface(&mut rng, true);
        def_imgs.push(g);
        def_masks.push(m);
    }
    let mut norm_imgs = Vec::new();
    for _ in 0..n_normal {
        let (g, _, _) = make_surface(&mut rng, false);
        norm_imgs.push(g);
    }
    // 一小批正常图专门用于异常检测定阈值
    let mut val_norm = Vec::new();
    for _ in 0..30 {
        let (g, _, _) = make_surface(&mut rng, false);
        val_norm.push(g);
    }
    let mut test_imgs = Vec::new();
    let mut test_masks = Vec::new();
    let mut test_defs = Vec::new();
    for _ in 0..n_test {
        let (g, m, d) = make_surface(&mut rng, true);
        test_imgs.push(g);
        test_masks.push(m);
        test_defs.push(d);
    }
    let def_px: usize = def_masks.iter().map(|m| m.iter().filter(|&&b| b).count()).sum();
    println!("数据：{n_def} 张带缺陷图（缺陷像素占比 {:.1}%）、{n_normal} 张正常图（训异常检测）、{n_test} 张测试图。", 100.0 * def_px as f32 / (n_def * SZ * SZ) as f32);
    let mut kind_cnt = [0usize; 3];
    for ds in &test_defs {
        for d in ds {
            kind_cnt[d.kind] += 1;
        }
    }
    println!("测试集缺陷类型分布：{}={}  {}={}  {}={}", KINDS[0], kind_cnt[0], KINDS[1], kind_cnt[1], KINDS[2], kind_cnt[2]);

    // 目标掩码张量（监督分割用）：[N,SZ,SZ] U32(0/1)
    let mut yt = Vec::with_capacity(n_def * SZ * SZ);
    for m in &def_masks {
        yt.extend(m.iter().map(|&b| if b { 1u32 } else { 0u32 }));
    }
    let xtr = gray_to_tensor(&def_imgs, &dev)?;
    let ytr = Tensor::from_vec(yt, (n_def, SZ, SZ), &dev)?;

    // ===== 路线 A：训练监督 U-Net 分割缺陷 =====
    println!("\n========== 路线 A：监督分割（小 U-Net 逐像素分缺陷）==========");
    let vm_a = VarMap::new();
    let unet = UNet::load(VarBuilder::from_varmap(&vm_a, DType::F32, &dev))?;
    det_init(&vm_a, 42)?;
    let mut opt = AdamW::new(vm_a.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    let epochs_a = 110;
    for epoch in 1..=epochs_a {
        if epoch == epochs_a * 3 / 4 {
            opt.set_learning_rate(5e-4);
        }
        let logits = unet.forward(&xtr)?;
        let loss = pixel_ce_weighted(&logits, &ytr, 4.0)?; // 缺陷像素加权 4×，缓解类不平衡
        opt.backward_step(&loss)?;
        if epoch % 20 == 0 || epoch == 1 {
            println!("  epoch {epoch:>3}  加权逐像素交叉熵 loss = {:.4}", loss.to_scalar::<f32>()?);
        }
    }

    // 评估路线 A：缺陷像素 IoU + 缺陷级检出率/误报（走完整后处理管线）
    let (mut iou_sum_a, mut gt_total, mut gt_hit, mut false_alarm) = (0f32, 0usize, 0usize, 0usize);
    for idx in 0..n_test {
        let x = gray_to_tensor(&test_imgs[idx..idx + 1], &dev)?;
        let pred = unet.forward(&x)?.i(0)?.argmax(0)?.to_vec2::<u32>()?;
        let pm: Vec<bool> = pred.iter().flatten().map(|&v| v == 1).collect();
        iou_sum_a += mask_iou(&pm, &test_masks[idx]);
        let comps = connected_components(&pm, MIN_AREA);
        for d in &test_defs[idx] {
            gt_total += 1;
            if comps.iter().any(|c| overlaps(&c.mask, &d.mask)) {
                gt_hit += 1;
            }
        }
        for c in &comps {
            if !test_defs[idx].iter().any(|d| overlaps(&c.mask, &d.mask)) {
                false_alarm += 1;
            }
        }
        if idx < 3 {
            overlay_png(&test_imgs[idx], &pm, &comps, &test_defs[idx], [80, 120, 255], &format!("defectA_{idx}.png"));
            // 逐缺陷块打印严重度（面积）
            print!("  测试图#{idx} 检出缺陷块: ");
            for c in &comps {
                print!("[面积{}px] ", c.area);
            }
            println!();
        }
    }
    println!("路线 A 结果：缺陷像素 IoU = {:.3}，缺陷级检出率 = {:.1}%（{}/{}），全测试集误报 = {} 个", iou_sum_a / n_test as f32, 100.0 * gt_hit as f32 / gt_total as f32, gt_hit, gt_total, false_alarm);

    // ===== 路线 B：只用正常图训自编码器，重建误差 = 异常 =====
    println!("\n========== 路线 B：异常检测（自编码器只学正常纹理，重建误差=缺陷）==========");
    let xnorm = gray_to_tensor(&norm_imgs, &dev)?;
    let vm_b = VarMap::new();
    let ae = AutoEnc::load(VarBuilder::from_varmap(&vm_b, DType::F32, &dev))?;
    det_init(&vm_b, 43)?;
    let mut opt = AdamW::new(vm_b.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    let epochs_b = 120;
    for epoch in 1..=epochs_b {
        if epoch == epochs_b * 3 / 4 {
            opt.set_learning_rate(5e-4);
        }
        let recon = ae.forward(&xnorm)?;
        let loss = (&recon - &xnorm)?.sqr()?.mean_all()?; // MSE 重建损失
        opt.backward_step(&loss)?;
        if epoch % 20 == 0 || epoch == 1 {
            println!("  epoch {epoch:>3}  重建 MSE = {:.5}", loss.to_scalar::<f32>()?);
        }
    }

    // 用【正常验证集】的重建误差统计定阈值：τ = 均值 + k·标准差
    let mut all_err = Vec::new();
    for g in &val_norm {
        let x = gray_to_tensor(std::slice::from_ref(g), &dev)?;
        let recon = ae.forward(&x)?.i(0)?.flatten_all()?.to_vec1::<f32>()?;
        all_err.extend(error_map(g, &recon));
    }
    let mean: f32 = all_err.iter().sum::<f32>() / all_err.len() as f32;
    let var: f32 = all_err.iter().map(|e| (e - mean) * (e - mean)).sum::<f32>() / all_err.len() as f32;
    let tau = mean + 4.0 * var.sqrt(); // k=4
    println!("  正常验证集重建误差：均值 {mean:.5}，阈值 τ = 均值 + 4σ = {tau:.5}（无需任何缺陷标注）");

    // 评估路线 B：同一套后处理
    let (mut iou_sum_b, mut gt_hit_b, mut false_alarm_b) = (0f32, 0usize, 0usize);
    for idx in 0..n_test {
        let x = gray_to_tensor(&test_imgs[idx..idx + 1], &dev)?;
        let recon = ae.forward(&x)?.i(0)?.flatten_all()?.to_vec1::<f32>()?;
        let err = error_map(&test_imgs[idx], &recon);
        let am: Vec<bool> = err.iter().map(|&e| e > tau).collect(); // 误差超阈 = 异常
        iou_sum_b += mask_iou(&am, &test_masks[idx]);
        let comps = connected_components(&am, MIN_AREA);
        for d in &test_defs[idx] {
            if comps.iter().any(|c| overlaps(&c.mask, &d.mask)) {
                gt_hit_b += 1;
            }
        }
        for c in &comps {
            if !test_defs[idx].iter().any(|d| overlaps(&c.mask, &d.mask)) {
                false_alarm_b += 1;
            }
        }
        if idx < 3 {
            overlay_png(&test_imgs[idx], &am, &comps, &test_defs[idx], [255, 150, 40], &format!("defectB_{idx}.png"));
        }
    }
    println!("路线 B 结果：缺陷像素 IoU = {:.3}，缺陷级检出率 = {:.1}%（{}/{}），全测试集误报 = {} 个", iou_sum_b / n_test as f32, 100.0 * gt_hit_b as f32 / gt_total as f32, gt_hit_b, gt_total, false_alarm_b);

    // ===== 对比小结 =====
    println!("\n========== 对比 ==========");
    println!("  监督分割 A：像素 IoU {:.3}，检出率 {:.1}%  —— 需要缺陷标注，缺陷像素分得最准", iou_sum_a / n_test as f32, 100.0 * gt_hit as f32 / gt_total as f32);
    println!("  异常检测 B：像素 IoU {:.3}，检出率 {:.1}%  —— 【零缺陷标注】，只看正常样本，适合缺陷稀少/未知", iou_sum_b / n_test as f32, 100.0 * gt_hit_b as f32 / gt_total as f32);
    println!("\n已存 defectA_*.png（监督分割）、defectB_*.png（异常检测）：灰度底图 + 缺陷着色 + 红框(连通域)。");
    Ok(())
}
