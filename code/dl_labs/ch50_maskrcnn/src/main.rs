// 第 50 章：教学版 Mask R-CNN——“先检测每个实例、再为每个实例预测掩码”（two-stage 思路）
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现（自造确定性初始化）。
// 运行：cd code/dl_labs/ch50_maskrcnn && cargo run --release
//
// 结构：① 检测阶段：骨干(backbone) + 网格检测头（复用第 46 章：每格 obj+框+类别）
//       ② 掩码阶段：对每个框，把图像裁剪成固定 28×28 小图 → 小 U-Net 掩码头（复用第 49 章）
//                    → 预测该框内“本实例”的前景/背景掩码。
// 训练：检测损失（obj/框/类别）+ 掩码逐像素交叉熵，一起反向。
// 报告：检测命中率 + 命中框平均 IoU + 每实例掩码 IoU（RoI 帧 & 整图帧）。
//
// 🕳️ 一个真实 candle 0.11 坑（本章踩到并绕过）：upsample_nearest2d 的“反向传播”只支持
//    整数且均匀的放大倍数，upsample_bilinear2d 干脆没有反向。所以“把任意大小的特征框
//    resize 到固定尺寸”（真正的 RoIAlign）在训练时会报
//    “backward not supported for non integer upscaling factors”。
//    绕过：RoI 的 resize 放在 Rust 侧对“图像”做（不进 autograd），掩码头内部只用 2× 整数上采样。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const IMG: usize = 48; // 输入边长
const G: usize = 6; // 检测网格边长
const STRIDE: usize = IMG / G; // = 8
const C: usize = 3; // 类别数：圆形 / 菱形 / 三角形
const A: usize = 1 + 4 + C; // 每格属性 = obj + (tx,ty,tw,th) + 类别 = 8
const CLASSES: [&str; C] = ["圆形", "菱形", "三角"];
const M: usize = 28; // 掩码头输入/输出分辨率（呼应真实 Mask R-CNN 的 28×28）
// 每类的基色（叠噪声）：圆=红、菱=绿、三角=蓝
const OBJ_COLOR: [[f32; 3]; C] = [[212.0, 82.0, 72.0], [82.0, 200.0, 110.0], [92.0, 120.0, 228.0]];
// 实例可视化调色板（按实例 id 上色，展示“区分个体”）
const INST_PAL: [[u8; 3]; 6] = [
    [230, 70, 70], [70, 200, 110], [80, 120, 235], [235, 200, 70], [200, 90, 220], [70, 210, 210],
];

// ---------- 确定性随机数发生器 ----------
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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// ---------- 一个实例：类别 + 外接框 + 逐像素掩码（IMG*IMG） ----------
#[derive(Clone)]
struct Inst {
    cls: usize,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    mask: Vec<bool>,
}
impl Inst {
    fn xyxy(&self) -> [f32; 4] {
        [self.cx - self.w / 2.0, self.cy - self.h / 2.0, self.cx + self.w / 2.0, self.cy + self.h / 2.0]
    }
}

// 合成一张多目标场景：2~4 个“非填满外接框”的形状（圆/菱形/三角），各自带实例掩码。
// 形状不填满 bbox（圆≈78%、菱形/三角≈50%），掩码分支才有真活干（否则 mask=box 太平凡）。
fn make_scene(rng: &mut Lcg) -> (RgbImage, Vec<Inst>) {
    let mut base = vec![[22.0f32, 24.0, 30.0]; IMG * IMG]; // 背景暗色
    let n = rng.range(2, 5); // 2..=4 个实例
    let mut insts: Vec<Inst> = Vec::new();
    let mut tries = 0;
    while (insts.len() as i32) < n && tries < 80 {
        tries += 1;
        let cls = rng.range(0, C as i32) as usize;
        let (hw, hh) = match cls {
            0 => {
                let r = rng.range(5, 9) as f32;
                (r, r)
            } // 圆
            1 => {
                let a = rng.range(5, 9) as f32;
                (a, a)
            } // 菱形
            _ => (rng.range(6, 10) as f32, rng.range(6, 10) as f32), // 三角
        };
        let cx = rng.range(hw as i32 + 1, IMG as i32 - hw as i32 - 1) as f32;
        let cy = rng.range(hh as i32 + 1, IMG as i32 - hh as i32 - 1) as f32;
        // 中心间距 >= 16，避免挤进同一/相邻网格格子
        if insts.iter().any(|o| (o.cx - cx).abs() < 16.0 && (o.cy - cy).abs() < 16.0) {
            continue;
        }
        let color = OBJ_COLOR[cls];
        let mut mask = vec![false; IMG * IMG];
        let (x0, y0, x1, y1) = ((cx - hw) as i32, (cy - hh) as i32, (cx + hw) as i32, (cy + hh) as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if x < 0 || y < 0 || x >= IMG as i32 || y >= IMG as i32 {
                    continue;
                }
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let inside = match cls {
                    0 => dx * dx + dy * dy <= hw * hw,          // 圆
                    1 => dx.abs() / hw + dy.abs() / hh <= 1.0,  // 菱形
                    _ => {
                        let ry = (y as f32 - (cy - hh)) / (2.0 * hh); // 顶 0 → 底 1
                        dx.abs() <= hw * ry                            // 三角：半宽随高度增长
                    }
                };
                if inside {
                    mask[y as usize * IMG + x as usize] = true;
                    base[y as usize * IMG + x as usize] = color;
                }
            }
        }
        insts.push(Inst { cls, cx, cy, w: 2.0 * hw, h: 2.0 * hh, mask });
    }
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    for y in 0..IMG {
        for x in 0..IMG {
            let c = base[y * IMG + x];
            let mut px = [0u8; 3];
            for k in 0..3 {
                px[k] = (c[k] + rng.gauss() * 12.0).round().clamp(0.0, 255.0) as u8;
            }
            img.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    (img, insts)
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

// 整数化的框（裁剪用）：返回 (x0,y0,w,h)，钳到图像内、w/h>=1
fn box_to_xywh(b: [f32; 4]) -> (u32, u32, u32, u32) {
    let x0 = b[0].floor().clamp(0.0, IMG as f32 - 1.0) as u32;
    let y0 = b[1].floor().clamp(0.0, IMG as f32 - 1.0) as u32;
    let x1 = (b[2].ceil() as i32).clamp(1, IMG as i32) as u32;
    let y1 = (b[3].ceil() as i32).clamp(1, IMG as i32) as u32;
    let w = (x1.saturating_sub(x0)).max(1).min(IMG as u32 - x0);
    let h = (y1.saturating_sub(y0)).max(1).min(IMG as u32 - y0);
    (x0, y0, w, h)
}

// RoI（教学简化版）：把图像按框裁出来、resize 到 M×M 小图 → CHW /255。
// resize 在 Rust 侧对图像做（不进 autograd），从而回避 candle 非整数上采样无反向的坑。
fn crop_resize_image(img: &RgbImage, b: [f32; 4]) -> Vec<f32> {
    let (x0, y0, w, h) = box_to_xywh(b);
    let sub = image::imageops::crop_imm(img, x0, y0, w, h).to_image();
    let rs = image::imageops::resize(&sub, M as u32, M as u32, image::imageops::FilterType::Triangle);
    let mut data = vec![0f32; 3 * M * M];
    for y in 0..M {
        for x in 0..M {
            let p = rs.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * M * M + y * M + x] = p[c] as f32 / 255.0;
            }
        }
    }
    data
}

// 把 IMG×IMG 实例掩码按框裁剪 + 最近邻 resize 到 M×M（掩码头训练目标）
fn crop_resize_mask(mask: &[bool], b: [f32; 4]) -> Vec<u32> {
    let (x0, y0, w, h) = box_to_xywh(b);
    let mut o = vec![0u32; M * M];
    for oy in 0..M {
        for ox in 0..M {
            let sx = (x0 + ox as u32 * w / M as u32).min(IMG as u32 - 1);
            let sy = (y0 + oy as u32 * h / M as u32).min(IMG as u32 - 1);
            if mask[sy as usize * IMG + sx as usize] {
                o[oy * M + ox] = 1;
            }
        }
    }
    o
}

// 把掩码头预测的 M×M 二值图贴回整图（按框最近邻放大）
fn paste_mask(small: &[u32], b: [f32; 4]) -> Vec<bool> {
    let (x0, y0, w, h) = box_to_xywh(b);
    let mut full = vec![false; IMG * IMG];
    for y in 0..h {
        for x in 0..w {
            let ox = (x * M as u32 / w).min(M as u32 - 1) as usize;
            let oy = (y * M as u32 / h).min(M as u32 - 1) as usize;
            if small[oy * M + ox] == 1 {
                full[(y0 + y) as usize * IMG + (x0 + x) as usize] = true;
            }
        }
    }
    full
}

// ---------- 检测骨干 + 网格检测头（第 46 章）：3 段卷积 → 6×6 特征 → 1×1 头 [N,A,6,6] ----------
struct Detector {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    head: Conv2d,
}
impl Detector {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let cfg0 = Conv2dConfig { padding: 0, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            head: conv2d(64, A, 1, cfg0, vb.pp("head"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 48→24
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 24→12
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?; // 12→6
        self.head.forward(&x) // [N,A,6,6]
    }
}

// ---------- 掩码头：小 U-Net（复用第 49 章）在 M×M 的 RoI 小图上分前景/背景 ----------
// 28→14→7 编码，7→14→28 解码（全部 2× 整数上采样，反向安全），2 条跳跃连接。
struct MaskNet {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    up2: Conv2d,
    dec2: Conv2d,
    up1: Conv2d,
    dec1: Conv2d,
    head: Conv2d,
}
impl MaskNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        let c1 = Conv2dConfig { padding: 0, ..Default::default() };
        Ok(Self {
            enc1: conv2d(3, 8, 3, c, vb.pp("enc1"))?,
            enc2: conv2d(8, 16, 3, c, vb.pp("enc2"))?,
            bott: conv2d(16, 32, 3, c, vb.pp("bott"))?,
            up2: conv2d(32, 16, 3, c, vb.pp("up2"))?,
            dec2: conv2d(32, 16, 3, c, vb.pp("dec2"))?,
            up1: conv2d(16, 8, 3, c, vb.pp("up1"))?,
            dec1: conv2d(16, 8, 3, c, vb.pp("dec1"))?,
            head: conv2d(8, 2, 1, c1, vb.pp("head"))?, // 2 类：背景 / 本实例前景
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let e1 = self.enc1.forward(x)?.relu()?; // [T,8,28,28]
        let p1 = e1.max_pool2d(2)?; // 28→14
        let e2 = self.enc2.forward(&p1)?.relu()?; // [T,16,14,14]
        let p2 = e2.max_pool2d(2)?; // 14→7
        let b = self.bott.forward(&p2)?.relu()?; // [T,32,7,7]
        let u2 = b.upsample_nearest2d(M / 2, M / 2)?; // 7→14（2× 整数）
        let u2 = self.up2.forward(&u2)?.relu()?;
        let c2 = Tensor::cat(&[&u2, &e2], 1)?; // 跳跃连接
        let d2 = self.dec2.forward(&c2)?.relu()?;
        let u1 = d2.upsample_nearest2d(M, M)?; // 14→28（2× 整数）
        let u1 = self.up1.forward(&u1)?.relu()?;
        let c1 = Tensor::cat(&[&u1, &e1], 1)?; // 跳跃连接
        let d1 = self.dec1.forward(&c1)?.relu()?;
        self.head.forward(&d1) // [T,2,28,28]
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

// 数值稳定 BCE：max(x,0) - x*y + log(1+exp(-|x|))（第 44/46 章）
fn bce_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.broadcast_mul(targets)?;
    let sp = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    relu_x.broadcast_sub(&xy)?.broadcast_add(&sp)
}

// 逐像素 2 类交叉熵（第 49 章）：[T,2,M,M] + U32 目标 [T,M,M]
fn pixel_ce(logits: &Tensor, target_u32: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = logits.dims4()?;
    let flat = logits.permute((0, 2, 3, 1))?.contiguous()?.reshape((n * h * w, c))?;
    let tgt = target_u32.reshape((n * h * w,))?;
    candle_nn::loss::cross_entropy(&flat, &tgt)
}

// ---------- 检测目标（第 46 章 build_targets） ----------
fn build_targets(all: &[Vec<Inst>], dev: &Device, pos_w: f32) -> Result<(Tensor, Tensor, Tensor, Tensor, f32)> {
    let n = all.len();
    let mut obj = vec![0f32; n * G * G];
    let mut wgt = vec![1f32; n * G * G];
    let mut boxn = vec![0f32; n * 4 * G * G];
    let mut cls = vec![0f32; n * C * G * G];
    let mut num_pos = 0f32;
    for (ni, insts) in all.iter().enumerate() {
        for it in insts {
            let ci = ((it.cy / STRIDE as f32).floor() as usize).min(G - 1);
            let cj = ((it.cx / STRIDE as f32).floor() as usize).min(G - 1);
            let cell = ci * G + cj;
            obj[ni * G * G + cell] = 1.0;
            wgt[ni * G * G + cell] = pos_w;
            num_pos += 1.0;
            let comps = [it.cx / IMG as f32, it.cy / IMG as f32, it.w / IMG as f32, it.h / IMG as f32];
            for c in 0..4 {
                boxn[(ni * 4 + c) * G * G + cell] = comps[c];
            }
            cls[(ni * C + it.cls) * G * G + cell] = 1.0;
        }
    }
    Ok((
        Tensor::from_vec(obj, (n, 1, G, G), dev)?,
        Tensor::from_vec(wgt, (n, 1, G, G), dev)?,
        Tensor::from_vec(boxn, (n, 4, G, G), dev)?,
        Tensor::from_vec(cls, (n, C, G, G), dev)?,
        num_pos,
    ))
}

#[allow(clippy::too_many_arguments)]
fn det_loss(
    out: &Tensor,
    obj_t: &Tensor,
    wgt: &Tensor,
    boxn_t: &Tensor,
    cls_t: &Tensor,
    num_pos: f32,
    cell_x: &Tensor,
    cell_y: &Tensor,
) -> Result<(Tensor, f32, f32, f32)> {
    let n = out.dim(0)?;
    let obj_logit = out.narrow(1, 0, 1)?;
    let box_raw = out.narrow(1, 1, 4)?;
    let cls_logit = out.narrow(1, 5, C)?;
    let l_obj = bce_elem(&obj_logit, obj_t)?.broadcast_mul(wgt)?.sum_all()?.affine(1.0 / (n * G * G) as f64, 0.0)?;
    let sig = sigmoid(&box_raw)?;
    let (sx, sy) = (sig.narrow(1, 0, 1)?, sig.narrow(1, 1, 1)?);
    let (sw, sh) = (sig.narrow(1, 2, 1)?, sig.narrow(1, 3, 1)?);
    let cx = sx.broadcast_add(cell_x)?.affine(1.0 / G as f64, 0.0)?;
    let cy = sy.broadcast_add(cell_y)?.affine(1.0 / G as f64, 0.0)?;
    let pred_boxn = Tensor::cat(&[&cx, &cy, &sw, &sh], 1)?;
    let l_box = pred_boxn.broadcast_sub(boxn_t)?.abs()?.broadcast_mul(obj_t)?.sum_all()?.affine(1.0 / (num_pos * 4.0) as f64, 0.0)?;
    let l_cls = bce_elem(&cls_logit, cls_t)?.broadcast_mul(obj_t)?.sum_all()?.affine(1.0 / (num_pos * C as f32) as f64, 0.0)?;
    let total = l_obj.broadcast_add(&l_box.affine(5.0, 0.0)?)?.broadcast_add(&l_cls)?;
    Ok((total, l_obj.to_scalar::<f32>()?, l_box.to_scalar::<f32>()?, l_cls.to_scalar::<f32>()?))
}

// ---------- 推理侧：解码 / IoU / NMS（第 46 章） ----------
#[derive(Clone)]
struct Det {
    cls: usize,
    score: f32,
    xyxy: [f32; 4],
}
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn decode(out: &Tensor, conf_thr: f32) -> Result<Vec<Det>> {
    let v = out.i(0)?.to_vec3::<f32>()?;
    let mut dets = Vec::new();
    for i in 0..G {
        for j in 0..G {
            let obj = sigmoidf(v[0][i][j]);
            let (mut best_c, mut best_logit) = (0usize, f32::MIN);
            for c in 0..C {
                if v[5 + c][i][j] > best_logit {
                    best_logit = v[5 + c][i][j];
                    best_c = c;
                }
            }
            let score = obj * sigmoidf(best_logit);
            if score < conf_thr {
                continue;
            }
            let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32;
            let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
            let w = sigmoidf(v[3][i][j]) * IMG as f32;
            let h = sigmoidf(v[4][i][j]) * IMG as f32;
            dets.push(Det { cls: best_c, score, xyxy: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0] });
        }
    }
    Ok(dets)
}
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ua = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ub = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let u = ua + ub - inter;
    if u <= 0.0 {
        0.0
    } else {
        inter / u
    }
}
fn nms(mut dets: Vec<Det>, iou_thr: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut removed = vec![false; dets.len()];
    let mut keep = Vec::new();
    for i in 0..dets.len() {
        if removed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if !removed[j] && dets[j].cls == dets[i].cls && iou(&dets[i].xyxy, &dets[j].xyxy) > iou_thr {
                removed[j] = true;
            }
        }
    }
    keep
}
fn mask_iou(a: &[bool], b: &[bool]) -> f32 {
    let mut inter = 0usize;
    let mut uni = 0usize;
    for i in 0..a.len() {
        if a[i] && b[i] {
            inter += 1;
        }
        if a[i] || b[i] {
            uni += 1;
        }
    }
    if uni == 0 {
        0.0
    } else {
        inter as f32 / uni as f32
    }
}

// 把「输入 | 预测实例掩码(按实例上色) | GT 实例掩码」拼成三联图存 PNG
fn save_triptych(img: &RgbImage, pred_insts: &[Vec<bool>], gt_insts: &[Vec<bool>], path: &str) {
    const S: u32 = 8;
    const GAP: u32 = 4;
    let cell = IMG as u32 * S;
    let mut canvas = RgbImage::from_pixel(cell * 3 + GAP * 2, cell, Rgb([255, 255, 255]));
    let big = image::imageops::resize(img, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &big, 0, 0);
    let paint = |insts: &[Vec<bool>]| -> RgbImage {
        let mut m = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([40, 42, 50]));
        for (id, inst) in insts.iter().enumerate() {
            let col = INST_PAL[id % INST_PAL.len()];
            for y in 0..IMG {
                for x in 0..IMG {
                    if inst[y * IMG + x] {
                        m.put_pixel(x as u32, y as u32, Rgb(col));
                    }
                }
            }
        }
        m
    };
    let pm = image::imageops::resize(&paint(pred_insts), cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &pm, (cell + GAP) as i64, 0);
    let gm = image::imageops::resize(&paint(gt_insts), cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &gm, (cell * 2 + GAP * 2) as i64, 0);
    canvas.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 数据 =====
    let mut rng = Lcg::new(11);
    let (n_train, n_test) = (112usize, 48usize);
    let mut xs = Vec::with_capacity(n_train * 3 * IMG * IMG);
    let mut gts_tr: Vec<Vec<Inst>> = Vec::with_capacity(n_train);
    let mut imgs_tr: Vec<RgbImage> = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        let (img, insts) = make_scene(&mut rng);
        xs.extend_from_slice(&img_to_chw(&img));
        gts_tr.push(insts);
        imgs_tr.push(img);
    }
    let xtr = Tensor::from_vec(xs, (n_train, 3, IMG, IMG), &dev)?;
    let (obj_t, wgt, boxn_t, cls_t, num_pos) = build_targets(&gts_tr, &dev, 6.0)?;
    let n_inst: usize = gts_tr.iter().map(|v| v.len()).sum();
    println!("训练集 {n_train} 张 {IMG}x{IMG} 图，共 {n_inst} 个实例（平均每图 {:.2} 个）", n_inst as f32 / n_train as f32);

    // 掩码阶段训练数据（固定不变，预计算一次）：每个 GT 实例的 RoI 小图 [T,3,M,M] + 目标 [T,M,M]
    let mut mxs = Vec::with_capacity(n_inst * 3 * M * M);
    let mut mts = Vec::with_capacity(n_inst * M * M);
    for (ni, insts) in gts_tr.iter().enumerate() {
        for it in insts {
            mxs.extend_from_slice(&crop_resize_image(&imgs_tr[ni], it.xyxy()));
            mts.extend_from_slice(&crop_resize_mask(&it.mask, it.xyxy()));
        }
    }
    let mask_x = Tensor::from_vec(mxs, (n_inst, 3, M, M), &dev)?;
    let mask_t = Tensor::from_vec(mts, (n_inst, M, M), &dev)?;

    // 网格格子行列索引常量（解码框用）
    let (mut cxv, mut cyv) = (vec![0f32; G * G], vec![0f32; G * G]);
    for i in 0..G {
        for j in 0..G {
            cxv[i * G + j] = j as f32;
            cyv[i * G + j] = i as f32;
        }
    }
    let cell_x = Tensor::from_vec(cxv, (1, 1, G, G), &dev)?;
    let cell_y = Tensor::from_vec(cyv, (1, 1, G, G), &dev)?;

    // ===== 搭网络：检测器（骨干+检测头）+ 掩码头 =====
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let det = Detector::load(vb.pp("det"))?;
    let mnet = MaskNet::load(vb.pp("mask"))?;
    det_init(&vm, 33)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;

    // ===== 训练：检测损失 + 掩码损失 一起反向 =====
    let epochs = 150;
    let lambda_mask = 1.0f64;
    println!("\n开始训练教学版 Mask R-CNN（检测器 + 掩码头，AdamW lr=2e-3→3e-4）...");
    println!("  {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}", "epoch", "total", "obj", "box", "cls", "mask");
    for epoch in 1..=epochs {
        if epoch == epochs * 3 / 4 {
            opt.set_learning_rate(3e-4); // lr 衰减提锐度（第 46/47 章）
        }
        let det_out = det.forward(&xtr)?;
        let (l_det, lo, lb, lc) = det_loss(&det_out, &obj_t, &wgt, &boxn_t, &cls_t, num_pos, &cell_x, &cell_y)?;
        let mlogits = mnet.forward(&mask_x)?; // [T,2,M,M]（用 GT 框裁的正样本 RoI）
        let l_mask = pixel_ce(&mlogits, &mask_t)?;
        let total = l_det.broadcast_add(&l_mask.affine(lambda_mask, 0.0)?)?;
        opt.backward_step(&total)?;
        if epoch % 40 == 0 || epoch == 1 {
            println!("  {epoch:>5} {:>8.4} {lo:>8.4} {lb:>8.4} {lc:>8.4} {:>8.4}", total.to_scalar::<f32>()?, l_mask.to_scalar::<f32>()?);
        }
    }

    // ===== 存 + 加载（与加载真实 Mask R-CNN 权重同一套 API） =====
    vm.save("maskrcnn.safetensors")?;
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["maskrcnn.safetensors"], DType::F32, &dev)? };
    let det = Detector::load(vb2.pp("det"))?;
    let mnet = MaskNet::load(vb2.pp("mask"))?;

    // 测试场景
    let mut te_rng = Lcg::new(999);
    let mut scenes: Vec<(RgbImage, Vec<Inst>)> = Vec::with_capacity(n_test);
    for _ in 0..n_test {
        scenes.push(make_scene(&mut te_rng));
    }

    // ===== 掩码头单独评估（用 GT 框裁 RoI，M×M 帧掩码 IoU，与检测误差解耦）=====
    let mut roi_iou = 0f32;
    let mut roi_cnt = 0usize;
    for (img, insts) in &scenes {
        for it in insts {
            let x = Tensor::from_vec(crop_resize_image(img, it.xyxy()), (1, 3, M, M), &dev)?;
            let pred = mnet.forward(&x)?.i(0)?.argmax(0)?.to_vec2::<u32>()?;
            let pf: Vec<bool> = pred.iter().flatten().map(|&v| v == 1).collect();
            let tf: Vec<bool> = crop_resize_mask(&it.mask, it.xyxy()).iter().map(|&v| v == 1).collect();
            roi_iou += mask_iou(&pf, &tf);
            roi_cnt += 1;
        }
    }
    println!("\n===== 掩码头单独评估（用 GT 框，RoI 帧 {M}×{M}）=====");
    println!("每实例掩码 IoU（RoI 帧）= {:.3}（{} 个实例）", roi_iou / roi_cnt as f32, roi_cnt);

    // ===== 完整实例分割管线：检测 → NMS → 每个检出框裁 RoI 跑掩码头 → 贴回整图 → 匹配 GT =====
    let (conf_thr, nms_thr) = (0.5f32, 0.45f32);
    let (mut tot_gt, mut hit, mut box_iou_sum, mut inst_mask_iou) = (0usize, 0usize, 0f32, 0f32);
    for (idx, (img, insts)) in scenes.iter().enumerate() {
        let x = Tensor::from_vec(img_to_chw(img), (1, 3, IMG, IMG), &dev)?;
        let dets = nms(decode(&det.forward(&x)?, conf_thr)?, nms_thr);
        let mut pred_insts: Vec<Vec<bool>> = Vec::new();
        for d in &dets {
            let rx = Tensor::from_vec(crop_resize_image(img, d.xyxy), (1, 3, M, M), &dev)?;
            let ml = mnet.forward(&rx)?.i(0)?.argmax(0)?.to_vec2::<u32>()?;
            let small: Vec<u32> = ml.iter().flatten().copied().collect();
            pred_insts.push(paste_mask(&small, d.xyxy));
        }
        let mut used = vec![false; insts.len()];
        if idx < 3 {
            println!("测试图 #{idx}：{} 个真值实例，检出 {} 个", insts.len(), dets.len());
        }
        for (di, d) in dets.iter().enumerate() {
            let mut best = (usize::MAX, 0f32);
            for (gi, it) in insts.iter().enumerate() {
                if used[gi] || it.cls != d.cls {
                    continue;
                }
                let v = iou(&d.xyxy, &it.xyxy());
                if v > best.1 {
                    best = (gi, v);
                }
            }
            if best.0 != usize::MAX && best.1 >= 0.5 {
                used[best.0] = true;
                hit += 1;
                box_iou_sum += best.1;
                let miou = mask_iou(&pred_insts[di], &insts[best.0].mask);
                inst_mask_iou += miou;
                if idx < 3 {
                    println!("   ✓ {:<4} score={:.2} 框IoU={:.3} 掩码IoU={:.3}", CLASSES[d.cls], d.score, best.1, miou);
                }
            }
        }
        tot_gt += insts.len();
        if idx < 5 {
            let gt_masks: Vec<Vec<bool>> = insts.iter().map(|it| it.mask.clone()).collect();
            save_triptych(img, &pred_insts, &gt_masks, &format!("inst_{idx}.png"));
        }
    }
    println!("\n===== 完整实例分割管线评估（检测 → 逐实例掩码）=====");
    println!("真值实例 {tot_gt} 个，正确检出 {hit} 个（检出率 {:.1}%）", 100.0 * hit as f32 / tot_gt as f32);
    println!("命中框平均 IoU = {:.3}", box_iou_sum / hit as f32);
    println!("命中实例的整图掩码 IoU = {:.3}", inst_mask_iou / hit as f32);
    println!("\n已存 inst_0.png .. inst_4.png（左=输入，中=预测实例掩码，右=GT；每实例一色 → 区分个体）");
    Ok(())
}
