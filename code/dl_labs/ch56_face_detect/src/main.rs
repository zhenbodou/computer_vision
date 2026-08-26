// 第 56 章：单类“人脸”检测器 + 5 点关键点回归头（复用第 46/47 章检测套路）
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现。
// 运行：cd code/dl_labs/ch56_face_detect && cargo run --release
//
// 管线：合成“人脸”（椭圆脸+两眼+鼻+嘴，随机位置/尺度/背景噪声）
//       → 快速自训练 → VarMap::save 存 safetensors → 加载回来
//       → 预处理 → forward → 解码（obj/框/5关键点）→ 阈值 → NMS
//       → 画框+关键点存 PNG → 报检出率 / 命中框平均 IoU / NMS 前后框数 / 关键点误差。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;

const IMG: usize = 48; // 输入边长（正方形）
const G: usize = 6; // 网格边长：把图切成 6x6 个格子
const STRIDE: usize = IMG / G; // 每格对应的像素数 = 8
const NL: usize = 5; // 关键点数：左眼/右眼/鼻/左嘴角/右嘴角
const A: usize = 1 + 4 + 2 * NL; // 每格属性数 = obj + (tx,ty,tw,th) + 5点(x,y) = 15
// 关键点通道布局（非交错，便于按块广播加上格子坐标）：
//   [0]=obj  [1..5]=box(tx,ty,tw,th)  [5..5+NL]=lmk_x  [5+NL..5+2NL]=lmk_y

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
    // 返回 [lo, hi) 内的整数
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

// ---------- 一张“人脸”的真值：框 + 5 个关键点（绝对像素坐标，交错存 x,y 便于画图） ----------
#[derive(Clone)]
struct Face {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    lmk: [f32; 2 * NL], // [lx0,ly0, lx1,ly1, ...] 顺序：左眼/右眼/鼻/左嘴角/右嘴角
}
impl Face {
    fn xyxy(&self) -> [f32; 4] {
        [
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        ]
    }
}

// 在图上画一个实心椭圆（脸）——手写椭圆方程，避免依赖任何画椭圆 API
fn fill_ellipse(img: &mut RgbImage, cx: f32, cy: f32, a: f32, b: f32, color: Rgb<u8>) {
    let x0 = (cx - a).floor().max(0.0) as i32;
    let x1 = (cx + a).ceil().min(IMG as f32 - 1.0) as i32;
    let y0 = (cy - b).floor().max(0.0) as i32;
    let y1 = (cy + b).ceil().min(IMG as f32 - 1.0) as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = (x as f32 - cx) / a;
            let dy = (y as f32 - cy) / b;
            if dx * dx + dy * dy <= 1.0 {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

// 画一个实心小圆（眼睛/鼻）——手写，半径很小
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

// 合成一张 48x48 图 + 它的人脸真值列表：1~2 张“人脸”，随机位置/尺度/肤色/背景噪声
fn make_image(rng: &mut Lcg) -> (RgbImage, Vec<Face>) {
    // 背景：随机基调 + 逐像素噪声（模拟真实场景的杂乱背景）
    let base = rng.range(12, 46);
    let mut img = RgbImage::new(IMG as u32, IMG as u32);
    for y in 0..IMG as u32 {
        for x in 0..IMG as u32 {
            let v = (base + rng.range(0, 26)).clamp(0, 255) as u8;
            img.put_pixel(x, y, Rgb([v, v, (v as i32 + rng.range(-6, 7)).clamp(0, 255) as u8]));
        }
    }

    let n = rng.range(1, 3); // 1..=2 张脸
    let mut faces: Vec<Face> = Vec::new();
    let mut tries = 0;
    while (faces.len() as i32) < n && tries < 60 {
        tries += 1;
        let w = rng.range(14, 23) as f32; // 脸宽 14~22
        let h = w * 1.25; // 脸略高于宽
        let hw = (w / 2.0) as i32 + 2;
        let hh = (h / 2.0) as i32 + 2;
        let cx = rng.range(hw, IMG as i32 - hw) as f32;
        let cy = rng.range(hh, IMG as i32 - hh) as f32;
        if faces.iter().any(|f| (f.cx - cx).abs() < 16.0 && (f.cy - cy).abs() < 16.0) {
            continue; // 与已有脸保持距离
        }
        let (a, b) = (w / 2.0, h / 2.0);
        // 肤色：暖色调，带随机亮度扰动（同图不同脸亮度不同）
        let sk = rng.range(175, 236);
        let skin = Rgb([sk as u8, (sk - 25).max(0) as u8, (sk - 55).max(0) as u8]);
        fill_ellipse(&mut img, cx, cy, a, b, skin);
        // 5 个关键点的绝对坐标（脸几何决定）
        let le = (cx - 0.40 * a, cy - 0.22 * b); // 左眼
        let re = (cx + 0.40 * a, cy - 0.22 * b); // 右眼
        let no = (cx, cy + 0.08 * b); // 鼻
        let lm = (cx - 0.28 * a, cy + 0.48 * b); // 左嘴角
        let rm = (cx + 0.28 * a, cy + 0.48 * b); // 右嘴角
        // 画五官（深色），位置与关键点严格对齐
        let dark = Rgb([30u8, 28, 32]);
        fill_disk(&mut img, le.0, le.1, 0.14 * a + 0.8, dark);
        fill_disk(&mut img, re.0, re.1, 0.14 * a + 0.8, dark);
        fill_disk(&mut img, no.0, no.1, 0.10 * a + 0.4, dark);
        // 嘴：左右嘴角之间画一条深色粗线（两行像素）
        let my = lm.1.round() as i32;
        for yy in my..=(my + 1) {
            for xx in lm.0.round() as i32..=rm.0.round() as i32 {
                if xx >= 0 && xx < IMG as i32 && yy >= 0 && yy < IMG as i32 {
                    img.put_pixel(xx as u32, yy as u32, dark);
                }
            }
        }
        faces.push(Face {
            cx,
            cy,
            w,
            h,
            lmk: [le.0, le.1, re.0, re.1, no.0, no.1, lm.0, lm.1, rm.0, rm.1],
        });
    }
    (img, faces)
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

// 造一批数据：返回 (输入张量 [N,3,IMG,IMG], 每张图的人脸真值列表)
fn make_dataset(rng: &mut Lcg, n: usize, dev: &Device) -> Result<(Tensor, Vec<Vec<Face>>)> {
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    let mut all = Vec::with_capacity(n);
    for _ in 0..n {
        let (img, faces) = make_image(rng);
        xs.extend_from_slice(&img_to_chw(&img));
        all.push(faces);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    Ok((x, all))
}

// 根据真值构造训练目标张量（都用 [N,*,G,G] 通道优先排布）：
//   obj_t   [N,1,G,G]   负责格 = 1，其余 = 0
//   wgt     [N,1,G,G]   负责格权重 = pos_w，其余 = 1（缓解正负样本极度不平衡）
//   boxn_t  [N,4,G,G]   负责格的归一化框 (cx,cy,w,h)/IMG
//   lmk_t   [N,2NL,G,G] 负责格的归一化关键点：前 NL 通道存 x/IMG，后 NL 通道存 y/IMG
// 返回还带上正样本总数 num_pos。
fn build_targets(
    all: &[Vec<Face>],
    dev: &Device,
    pos_w: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, f32)> {
    let n = all.len();
    let mut obj = vec![0f32; n * G * G];
    let mut wgt = vec![1f32; n * G * G];
    let mut boxn = vec![0f32; n * 4 * G * G];
    let mut lmk = vec![0f32; n * 2 * NL * G * G];
    let mut num_pos = 0f32;
    for (ni, faces) in all.iter().enumerate() {
        for f in faces {
            let ci = ((f.cy / STRIDE as f32).floor() as usize).min(G - 1); // 负责格行
            let cj = ((f.cx / STRIDE as f32).floor() as usize).min(G - 1); // 负责格列
            let cell = ci * G + cj;
            obj[ni * G * G + cell] = 1.0;
            wgt[ni * G * G + cell] = pos_w;
            num_pos += 1.0;
            let comps = [f.cx / IMG as f32, f.cy / IMG as f32, f.w / IMG as f32, f.h / IMG as f32];
            for c in 0..4 {
                boxn[(ni * 4 + c) * G * G + cell] = comps[c];
            }
            for k in 0..NL {
                lmk[(ni * 2 * NL + k) * G * G + cell] = f.lmk[2 * k] / IMG as f32; // x
                lmk[(ni * 2 * NL + NL + k) * G * G + cell] = f.lmk[2 * k + 1] / IMG as f32; // y
            }
        }
    }
    Ok((
        Tensor::from_vec(obj, (n, 1, G, G), dev)?,
        Tensor::from_vec(wgt, (n, 1, G, G), dev)?,
        Tensor::from_vec(boxn, (n, 4, G, G), dev)?,
        Tensor::from_vec(lmk, (n, 2 * NL, G, G), dev)?,
        num_pos,
    ))
}

// ---------- 检测器：三段卷积把 48x48 逐步降到 6x6，再用 1x1 卷积头输出每格 A 个属性 ----------
struct Detector {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    head: Conv2d,
}
impl Detector {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() }; // 3x3 卷积不改变边长
        let cfg0 = Conv2dConfig { padding: 0, ..Default::default() }; // 1x1 卷积头
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            head: conv2d(64, A, 1, cfg0, vb.pp("head"))?, // 输出 [N,A,G,G]
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 48 -> 24
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 24 -> 12
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?; // 12 -> 6
        self.head.forward(&x) // [N,A,6,6]，每个空间位置就是一个网格格子
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

// 数值稳定的逐元素带 logit 的 BCE：max(x,0) - x*y + log(1+exp(-|x|))
fn bce_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.broadcast_mul(targets)?;
    let sp = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?; // log(1+exp(-|x|))
    relu_x.broadcast_sub(&xy)?.broadcast_add(&sp)
}

// 计算检测总损失（objectness 加权 BCE + 框回归 L1 + 关键点回归 L1）
// 返回 (总损失张量, l_obj, l_box, l_lmk) 后三者是打印用的标量
fn compute_loss(
    out: &Tensor,
    obj_t: &Tensor,
    wgt: &Tensor,
    boxn_t: &Tensor,
    lmk_t: &Tensor,
    num_pos: f32,
    cell_x: &Tensor, // [1,1,G,G] 值=列号 j
    cell_y: &Tensor, // [1,1,G,G] 值=行号 i
    lambda_box: f64,
    lambda_lmk: f64,
) -> Result<(Tensor, f32, f32, f32)> {
    let n = out.dim(0)?;
    let obj_logit = out.narrow(1, 0, 1)?; // [N,1,G,G]
    let box_raw = out.narrow(1, 1, 4)?; // [N,4,G,G]
    let lmk_x_raw = out.narrow(1, 5, NL)?; // [N,NL,G,G]
    let lmk_y_raw = out.narrow(1, 5 + NL, NL)?; // [N,NL,G,G]

    // ① objectness：对所有格子做加权 BCE（人脸格极少，故给它更大权重）
    let l_obj = bce_elem(&obj_logit, obj_t)?
        .broadcast_mul(wgt)?
        .sum_all()?
        .affine(1.0 / (n * G * G) as f64, 0.0)?;

    // ② 框回归：把预测解码成归一化 (cx,cy,w,h)，只在负责格上做 L1
    let sig = sigmoid(&box_raw)?;
    let sx = sig.narrow(1, 0, 1)?;
    let sy = sig.narrow(1, 1, 1)?;
    let sw = sig.narrow(1, 2, 1)?;
    let sh = sig.narrow(1, 3, 1)?;
    let cx = sx.broadcast_add(cell_x)?.affine(1.0 / G as f64, 0.0)?; // (col+σ(tx))/G ∈ [0,1]
    let cy = sy.broadcast_add(cell_y)?.affine(1.0 / G as f64, 0.0)?;
    let pred_boxn = Tensor::cat(&[&cx, &cy, &sw, &sh], 1)?; // [N,4,G,G]
    let l_box = pred_boxn
        .broadcast_sub(boxn_t)?
        .abs()?
        .broadcast_mul(obj_t)? // 只在负责格上计损失
        .sum_all()?
        .affine(1.0 / (num_pos * 4.0) as f64, 0.0)?;

    // ③ 关键点回归：预测=格子中心(归一化) + 线性偏移，只在负责格上做 L1
    //    关键：1x1 卷积头是平移不变的，无法直接吐出“绝对图坐标”，
    //    必须像框中心那样注入格子坐标（否则关键点损失学不动）。
    let ccx = cell_x.affine(1.0 / G as f64, 0.5 / G as f64)?; // (col+0.5)/G
    let ccy = cell_y.affine(1.0 / G as f64, 0.5 / G as f64)?; // (row+0.5)/G
    let pred_lx = lmk_x_raw.broadcast_add(&ccx)?; // [N,NL,G,G]
    let pred_ly = lmk_y_raw.broadcast_add(&ccy)?;
    let pred_lmk = Tensor::cat(&[&pred_lx, &pred_ly], 1)?; // [N,2NL,G,G]
    let l_lmk = pred_lmk
        .broadcast_sub(lmk_t)?
        .abs()?
        .broadcast_mul(obj_t)?
        .sum_all()?
        .affine(1.0 / (num_pos * 2.0 * NL as f32) as f64, 0.0)?;

    let total = l_obj
        .broadcast_add(&l_box.affine(lambda_box, 0.0)?)?
        .broadcast_add(&l_lmk.affine(lambda_lmk, 0.0)?)?;
    Ok((
        total,
        l_obj.to_scalar::<f32>()?,
        l_box.to_scalar::<f32>()?,
        l_lmk.to_scalar::<f32>()?,
    ))
}

// ---------- 推理侧：解码 / IoU / NMS ----------
#[derive(Clone)]
struct Det {
    score: f32,
    xyxy: [f32; 4],
    lmk: [f32; 2 * NL], // 交错 [x0,y0,x1,y1,...] 便于画图
}
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// 把一张图的网络输出 [1,A,G,G] 解码成绝对坐标候选框 + 关键点（含置信度阈值过滤）
fn decode(out: &Tensor, conf_thr: f32) -> Result<Vec<Det>> {
    let v = out.i(0)?.to_vec3::<f32>()?; // [A][G][G]
    let mut dets = Vec::new();
    for i in 0..G {
        for j in 0..G {
            let score = sigmoidf(v[0][i][j]); // 单类：objectness 即“人脸”置信度
            if score < conf_thr {
                continue;
            }
            // 解码框：中心=格内偏移，宽高=整图占比，还原成绝对像素
            let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32;
            let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
            let w = sigmoidf(v[3][i][j]) * IMG as f32;
            let h = sigmoidf(v[4][i][j]) * IMG as f32;
            // 解码 5 个关键点：格子中心(归一化) + 线性偏移，再 ×IMG 还原
            let mut lmk = [0f32; 2 * NL];
            for k in 0..NL {
                let lx = ((j as f32 + 0.5) / G as f32 + v[5 + k][i][j]) * IMG as f32;
                let ly = ((i as f32 + 0.5) / G as f32 + v[5 + NL + k][i][j]) * IMG as f32;
                lmk[2 * k] = lx;
                lmk[2 * k + 1] = ly;
            }
            dets.push(Det {
                score,
                xyxy: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                lmk,
            });
        }
    }
    Ok(dets)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let aa = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ab = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let u = aa + ab - inter;
    if u <= 0.0 {
        0.0
    } else {
        inter / u
    }
}

// 单类 NMS：按分数从高到低，删掉与已保留框 IoU 超阈值的框
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
            if !removed[j] && iou(&dets[i].xyxy, &dets[j].xyxy) > iou_thr {
                removed[j] = true;
            }
        }
    }
    keep
}

// 把检测结果画到放大 SCALE 倍的图上并存 PNG：真值框绿/关键点黄，检出框红/关键点青
fn draw_and_save(img: &RgbImage, faces: &[Face], dets: &[Det], path: &str) {
    const SCALE: u32 = 8;
    let mut canvas = image::imageops::resize(
        img,
        IMG as u32 * SCALE,
        IMG as u32 * SCALE,
        image::imageops::FilterType::Nearest,
    );
    let green = Rgb([40u8, 220, 90]);
    let red = Rgb([240u8, 60, 60]);
    let yellow = Rgb([250u8, 220, 40]);
    let cyan = Rgb([40u8, 220, 230]);
    for f in faces {
        rect(&mut canvas, f.xyxy(), SCALE, green);
        for k in 0..NL {
            let (x, y) = ((f.lmk[2 * k] * SCALE as f32) as i32, (f.lmk[2 * k + 1] * SCALE as f32) as i32);
            draw_filled_circle_mut(&mut canvas, (x, y), 4, yellow);
        }
    }
    for d in dets {
        rect(&mut canvas, d.xyxy, SCALE, red);
        for k in 0..NL {
            let (x, y) = ((d.lmk[2 * k] * SCALE as f32) as i32, (d.lmk[2 * k + 1] * SCALE as f32) as i32);
            draw_filled_circle_mut(&mut canvas, (x, y), 3, cyan);
        }
    }
    canvas.save(path).unwrap();
}
fn rect(canvas: &mut RgbImage, b: [f32; 4], scale: u32, color: Rgb<u8>) {
    let x = (b[0] * scale as f32).round() as i32;
    let y = (b[1] * scale as f32).round() as i32;
    let w = ((b[2] - b[0]) * scale as f32).round().max(1.0) as u32;
    let h = ((b[3] - b[1]) * scale as f32).round().max(1.0) as u32;
    draw_hollow_rect_mut(canvas, Rect::at(x, y).of_size(w, h), color);
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let (pos_w, lambda_box, lambda_lmk) = (8.0f32, 5.0f64, 3.0f64);

    // ===== 第 1 步：合成训练数据，构造训练目标 =====
    let mut rng = Lcg::new(7);
    let n_train = 288;
    let (xtr, faces_tr) = make_dataset(&mut rng, n_train, &dev)?;
    let (obj_t, wgt, boxn_t, lmk_t, num_pos) = build_targets(&faces_tr, &dev, pos_w)?;
    println!(
        "训练集：{n_train} 张图，共 {} 张人脸（平均每图 {:.2} 张）",
        num_pos as usize,
        num_pos / n_train as f32
    );

    // 网格格子的行列索引常量张量（解码/关键点用）
    let (mut cxv, mut cyv) = (vec![0f32; G * G], vec![0f32; G * G]);
    for i in 0..G {
        for j in 0..G {
            cxv[i * G + j] = j as f32;
            cyv[i * G + j] = i as f32;
        }
    }
    let cell_x = Tensor::from_vec(cxv, (1, 1, G, G), &dev)?;
    let cell_y = Tensor::from_vec(cyv, (1, 1, G, G), &dev)?;

    // ===== 第 2 步：搭检测器 + 确定性初始化 + 快速自训练 =====
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = Detector::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;

    println!("\n开始训练单类人脸检测器（AdamW, lr=2e-3→4e-4, λ_box={lambda_box}, λ_lmk={lambda_lmk}）...");
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}", "epoch", "total", "obj", "box", "lmk");
    let epochs = 200;
    for epoch in 1..=epochs {
        if epoch == 150 {
            opt.set_learning_rate(4e-4); // 后段降学习率，把 objectness 与回归磨得更利
        }
        let out = net.forward(&xtr)?;
        let (loss, lo, lb, ll) = compute_loss(
            &out, &obj_t, &wgt, &boxn_t, &lmk_t, num_pos, &cell_x, &cell_y, lambda_box, lambda_lmk,
        )?;
        opt.backward_step(&loss)?;
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {epoch:>5}  {:>8.4}  {lo:>8.4}  {lb:>8.4}  {ll:>8.4}", loss.to_scalar::<f32>()?);
        }
    }

    // ===== 第 3 步：VarMap::save 存 safetensors，并打印文件内张量清单 =====
    vm.save("face_detector.safetensors")?;
    let dict = candle_core::safetensors::load("face_detector.safetensors", &dev)?;
    let mut keys: Vec<(String, Vec<usize>)> =
        dict.iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    keys.sort();
    println!("\n已保存 face_detector.safetensors，文件内张量：");
    for (k, shape) in &keys {
        println!("  {k:<13} 形状 {shape:?}");
    }

    // ===== 第 4 步：unsafe from_mmaped_safetensors 加载回来（真实推理入口） =====
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["face_detector.safetensors"], DType::F32, &dev)? };
    let model = Detector::load(vb2)?;
    println!("\n已从 safetensors 重新加载模型，开始推理\n");

    // ===== 第 5 步：全新测试图上跑完整推理管线，报检出率/IoU/NMS前后/关键点误差 =====
    let (conf_thr, nms_thr) = (0.5f32, 0.45f32);
    let mut test_rng = Lcg::new(2025);
    let (mut total_gt, mut total_hit, mut iou_sum) = (0usize, 0usize, 0f32);
    let (mut raw_sum, mut kept_sum) = (0usize, 0usize); // NMS 前 / 后框数
    let (mut lmk_err_sum, mut lmk_err_cnt) = (0f32, 0usize); // 关键点像素误差
    for idx in 0..6 {
        let (img, faces) = make_image(&mut test_rng);
        let input = Tensor::from_vec(img_to_chw(&img), (1, 3, IMG, IMG), &dev)?; // 预处理
        let out = model.forward(&input)?; // 前向 → [1,A,G,G]
        let cands = decode(&out, conf_thr)?; // 解码 + 置信度阈值
        let n_raw = cands.len();
        raw_sum += n_raw;
        let dets = nms(cands, nms_thr); // NMS
        kept_sum += dets.len();

        let mut used = vec![false; faces.len()];
        println!(
            "测试图 #{idx}：{} 张真值人脸，解码候选 {} → NMS 后 {} 个框",
            faces.len(),
            n_raw,
            dets.len()
        );
        for d in &dets {
            let mut best = (usize::MAX, 0f32);
            for (gi, f) in faces.iter().enumerate() {
                if used[gi] {
                    continue;
                }
                let v = iou(&d.xyxy, &f.xyxy());
                if v > best.1 {
                    best = (gi, v);
                }
            }
            let b = d.xyxy;
            if best.0 != usize::MAX && best.1 >= 0.5 {
                used[best.0] = true;
                total_hit += 1;
                iou_sum += best.1;
                let f = &faces[best.0];
                let mut e = 0f32; // 命中脸的 5 点关键点平均欧氏误差（像素）
                for k in 0..NL {
                    let dx = d.lmk[2 * k] - f.lmk[2 * k];
                    let dy = d.lmk[2 * k + 1] - f.lmk[2 * k + 1];
                    e += (dx * dx + dy * dy).sqrt();
                }
                lmk_err_sum += e / NL as f32;
                lmk_err_cnt += 1;
                println!(
                    "   ✓ 人脸 score={:.2}  框=[{:>4.0},{:>4.0},{:>4.0},{:>4.0}]  IoU={:.3}  关键点误差={:.2}px",
                    d.score, b[0], b[1], b[2], b[3], best.1, e / NL as f32
                );
            } else {
                println!(
                    "   ? 人脸 score={:.2}  框=[{:>4.0},{:>4.0},{:>4.0},{:>4.0}]  （误检/重复）",
                    d.score, b[0], b[1], b[2], b[3]
                );
            }
        }
        total_gt += faces.len();
        draw_and_save(&img, &faces, &dets, &format!("face_{idx}.png"));
    }

    println!("\n===== 汇总 =====");
    println!(
        "真值人脸共 {total_gt} 张，正确检出 {total_hit} 张（检出率 {:.1}%）",
        100.0 * total_hit as f32 / total_gt as f32
    );
    if total_hit > 0 {
        println!("命中框的平均 IoU = {:.3}", iou_sum / total_hit as f32);
    }
    if lmk_err_cnt > 0 {
        println!("命中脸的 5 点关键点平均误差 = {:.2} 像素（图边长 {IMG}px）", lmk_err_sum / lmk_err_cnt as f32);
    }
    println!("NMS 前候选框共 {raw_sum} 个 → NMS 后保留 {kept_sum} 个");
    println!("可视化结果已存为 face_0.png .. face_5.png（绿=真值框/黄=真值关键点，红=检出框/青=检出关键点）");
    Ok(())
}
