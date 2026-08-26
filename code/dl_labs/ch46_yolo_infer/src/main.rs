// 第 46 章：极小单尺度网格检测器（YOLO 思路）的完整推理管线
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现。
// 运行：cd code/dl_labs/ch46_yolo_infer && cargo run --release
//
// 管线：合成数据 → 快速自训练几十轮 → VarMap::save 存 safetensors
//       → unsafe from_mmaped_safetensors 加载回来（真实推理入口）
//       → 预处理 → forward → 解码网格输出成绝对坐标框 → 置信度阈值 → NMS
//       → 画框存 PNG → 打印检出框与其 vs 真值(GT)的 IoU。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

const IMG: usize = 48; // 输入边长（正方形）
const G: usize = 6; // 网格边长：把图切成 6x6 个格子
const STRIDE: usize = IMG / G; // 每格对应的像素数 = 8
const C: usize = 2; // 类别数
const A: usize = 1 + 4 + C; // 每格属性数 = obj置信度 + (tx,ty,tw,th) + 类别logits = 7
const CLASSES: [&str; C] = ["方块", "横条"];

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

// ---------- 一个真值框（ground truth） ----------
#[derive(Clone, Copy)]
struct Gt {
    cls: usize,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}
impl Gt {
    fn xyxy(&self) -> [f32; 4] {
        [
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        ]
    }
}

// 合成一张 48x48 图 + 它的真值框列表：1~3 个白色几何形状（方块 / 横条），随机位置
fn make_image(rng: &mut Lcg) -> (RgbImage, Vec<Gt>) {
    let mut img = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    let n = rng.range(1, 4); // 1..=3 个目标
    let mut gts: Vec<Gt> = Vec::new();
    let mut tries = 0;
    while (gts.len() as i32) < n && tries < 80 {
        tries += 1;
        let cls = rng.range(0, C as i32) as usize;
        let (w, h) = if cls == 0 {
            let s = rng.range(10, 15) as f32; // 方块：边长 10~14
            (s, s)
        } else {
            (rng.range(18, 27) as f32, rng.range(6, 9) as f32) // 横条：宽 18~26、高 6~8
        };
        let hw = (w / 2.0) as i32;
        let hh = (h / 2.0) as i32;
        let cx = rng.range(hw + 1, IMG as i32 - hw - 1) as f32;
        let cy = rng.range(hh + 1, IMG as i32 - hh - 1) as f32;
        // 与已有中心保持距离，避免落进同一/相邻格子造成目标挤在一起
        if gts
            .iter()
            .any(|g| (g.cx - cx).abs() < 14.0 && (g.cy - cy).abs() < 14.0)
        {
            continue;
        }
        gts.push(Gt { cls, cx, cy, w, h });
    }
    for g in &gts {
        let x0 = (g.cx - g.w / 2.0).round() as i32;
        let y0 = (g.cy - g.h / 2.0).round() as i32;
        for yy in 0..g.h.round() as i32 {
            for xx in 0..g.w.round() as i32 {
                let (x, y) = (x0 + xx, y0 + yy);
                if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
                    img.put_pixel(x as u32, y as u32, Rgb([235, 235, 235]));
                }
            }
        }
    }
    (img, gts)
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

// 造一批数据：返回 (输入张量 [N,3,IMG,IMG], 每张图的真值框列表)
fn make_dataset(rng: &mut Lcg, n: usize, dev: &Device) -> Result<(Tensor, Vec<Vec<Gt>>)> {
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    let mut all_gts = Vec::with_capacity(n);
    for _ in 0..n {
        let (img, gts) = make_image(rng);
        xs.extend_from_slice(&img_to_chw(&img));
        all_gts.push(gts);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    Ok((x, all_gts))
}

// 根据真值框构造训练目标张量（都用 [N,*,G,G] 通道优先排布，方便与网络输出对齐）：
//   obj_t   [N,1,G,G]  负责格 = 1，其余 = 0
//   wgt     [N,1,G,G]  负责格权重 = pos_w，其余 = 1（缓解正负样本极度不平衡）
//   boxn_t  [N,4,G,G]  负责格的归一化框 (cx,cy,w,h)/IMG（其余无意义，靠掩码屏蔽）
//   cls_t   [N,C,G,G]  负责格的类别 one-hot
// 返回还带上正样本总数 num_pos，用于把框/类别损失按正样本个数归一。
fn build_targets(
    all_gts: &[Vec<Gt>],
    dev: &Device,
    pos_w: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor, f32)> {
    let n = all_gts.len();
    let mut obj = vec![0f32; n * G * G];
    let mut wgt = vec![1f32; n * G * G];
    let mut boxn = vec![0f32; n * 4 * G * G];
    let mut cls = vec![0f32; n * C * G * G];
    let mut num_pos = 0f32;
    for (ni, gts) in all_gts.iter().enumerate() {
        for g in gts {
            let ci = ((g.cy / STRIDE as f32).floor() as usize).min(G - 1); // 负责格行
            let cj = ((g.cx / STRIDE as f32).floor() as usize).min(G - 1); // 负责格列
            let cell = ci * G + cj;
            obj[ni * G * G + cell] = 1.0;
            wgt[ni * G * G + cell] = pos_w;
            num_pos += 1.0;
            let comps = [g.cx / IMG as f32, g.cy / IMG as f32, g.w / IMG as f32, g.h / IMG as f32];
            for c in 0..4 {
                boxn[(ni * 4 + c) * G * G + cell] = comps[c];
            }
            cls[(ni * C + g.cls) * G * G + cell] = 1.0;
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

// 数值稳定的逐元素带 logit 的 BCE：max(x,0) - x*y + log(1+exp(-|x|))
fn bce_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.broadcast_mul(targets)?;
    let sp = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?; // log(1+exp(-|x|))
    relu_x.broadcast_sub(&xy)?.broadcast_add(&sp)
}

// 计算检测总损失（objectness BCE + 类别 BCE + 框回归 L1，加权求和）
// 返回 (总损失张量, l_obj, l_box, l_cls) 后三者是打印用的标量
fn compute_loss(
    out: &Tensor,
    obj_t: &Tensor,
    wgt: &Tensor,
    boxn_t: &Tensor,
    cls_t: &Tensor,
    num_pos: f32,
    cell_x: &Tensor,
    cell_y: &Tensor,
    lambda_box: f64,
    lambda_cls: f64,
) -> Result<(Tensor, f32, f32, f32)> {
    let n = out.dim(0)?;
    let obj_logit = out.narrow(1, 0, 1)?; // [N,1,G,G]
    let box_raw = out.narrow(1, 1, 4)?; // [N,4,G,G]
    let cls_logit = out.narrow(1, 5, C)?; // [N,C,G,G]

    // ① objectness：对所有格子做加权 BCE（正样本极少，故给它更大权重）
    let l_obj = bce_elem(&obj_logit, obj_t)?
        .broadcast_mul(wgt)?
        .sum_all()?
        .affine(1.0 / (n * G * G) as f64, 0.0)?;

    // ② 框回归：把预测解码成归一化 (cx,cy,w,h)，对负责格做 L1
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

    // ③ 类别：对负责格做 BCE（每类独立 logistic）
    let l_cls = bce_elem(&cls_logit, cls_t)?
        .broadcast_mul(obj_t)?
        .sum_all()?
        .affine(1.0 / (num_pos * C as f32) as f64, 0.0)?;

    let total = l_obj
        .broadcast_add(&l_box.affine(lambda_box, 0.0)?)?
        .broadcast_add(&l_cls.affine(lambda_cls, 0.0)?)?;
    Ok((
        total,
        l_obj.to_scalar::<f32>()?,
        l_box.to_scalar::<f32>()?,
        l_cls.to_scalar::<f32>()?,
    ))
}

// ---------- 推理侧：解码 / IoU / NMS ----------
#[derive(Clone)]
struct Det {
    cls: usize,
    score: f32,
    xyxy: [f32; 4],
}
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// 把一张图的网络输出 [1,A,G,G] 解码成绝对坐标候选框（含置信度阈值过滤）
fn decode(out: &Tensor, conf_thr: f32) -> Result<Vec<Det>> {
    let v = out.i(0)?.to_vec3::<f32>()?; // [A][G][G]
    let mut dets = Vec::new();
    for i in 0..G {
        for j in 0..G {
            let obj = sigmoidf(v[0][i][j]); // 该格“有目标”的置信度
            // 取类别 logits 的 argmax
            let mut best_c = 0usize;
            let mut best_logit = f32::MIN;
            for c in 0..C {
                let p = v[5 + c][i][j];
                if p > best_logit {
                    best_logit = p;
                    best_c = c;
                }
            }
            let score = obj * sigmoidf(best_logit); // YOLO 惯例：obj × 类别置信度
            if score < conf_thr {
                continue;
            }
            // 解码框：中心在“格内偏移”，宽高是“整图占比”，全部还原成绝对像素
            let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32;
            let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
            let w = sigmoidf(v[3][i][j]) * IMG as f32;
            let h = sigmoidf(v[4][i][j]) * IMG as f32;
            dets.push(Det {
                cls: best_c,
                score,
                xyxy: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
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

// 类别感知 NMS：按分数从高到低，删掉与已保留框同类且 IoU 超阈值的框
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

// 把检测结果画到放大 SCALE 倍的图上并存 PNG：真值框绿色、检出框红色
fn draw_and_save(img: &RgbImage, gts: &[Gt], dets: &[Det], path: &str) {
    const SCALE: u32 = 6;
    let big = image::imageops::resize(img, IMG as u32 * SCALE, IMG as u32 * SCALE, image::imageops::FilterType::Nearest);
    let mut canvas = big;
    let green = Rgb([40u8, 220, 90]);
    let red = Rgb([240u8, 60, 60]);
    for g in gts {
        let b = g.xyxy();
        rect(&mut canvas, b, SCALE, green);
    }
    for d in dets {
        rect(&mut canvas, d.xyxy, SCALE, red);
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
    let (pos_w, lambda_box, lambda_cls) = (6.0f32, 5.0f64, 1.0f64);

    // ===== 第 1 步：合成训练数据，构造训练目标 =====
    let mut rng = Lcg::new(7);
    let n_train = 384;
    let (xtr, gts_tr) = make_dataset(&mut rng, n_train, &dev)?;
    let (obj_t, wgt, boxn_t, cls_t, num_pos) = build_targets(&gts_tr, &dev, pos_w)?;
    println!("训练集：{n_train} 张图，共 {} 个目标（平均每图 {:.2} 个）", num_pos as usize, num_pos / n_train as f32);

    // 网格格子的行列索引常量张量（解码时用）
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

    println!("\n开始训练小检测器（AdamW, lr=2e-3→4e-4, λ_box={lambda_box}, λ_cls={lambda_cls}）...");
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}", "epoch", "total", "obj", "box", "cls");
    let epochs = 250;
    for epoch in 1..=epochs {
        if epoch == 180 {
            opt.set_learning_rate(4e-4); // 后段降学习率，把 objectness 与框回归“磨”得更利
        }
        let out = net.forward(&xtr)?;
        let (loss, lo, lb, lc) = compute_loss(&out, &obj_t, &wgt, &boxn_t, &cls_t, num_pos, &cell_x, &cell_y, lambda_box, lambda_cls)?;
        opt.backward_step(&loss)?;
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {epoch:>5}  {:>8.4}  {lo:>8.4}  {lb:>8.4}  {lc:>8.4}", loss.to_scalar::<f32>()?);
        }
    }

    // ===== 第 3 步：VarMap::save 存 safetensors，并打印文件内张量清单 =====
    vm.save("detector.safetensors")?;
    let dict = candle_core::safetensors::load("detector.safetensors", &dev)?;
    let mut keys: Vec<(String, Vec<usize>)> = dict.iter().map(|(k, v)| (k.clone(), v.dims().to_vec())).collect();
    keys.sort();
    println!("\n已保存 detector.safetensors，文件内张量：");
    for (k, shape) in &keys {
        println!("  {k:<13} 形状 {shape:?}");
    }

    // ===== 第 4 步：unsafe from_mmaped_safetensors 加载回来（这就是真实推理入口） =====
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["detector.safetensors"], DType::F32, &dev)? };
    let model = Detector::load(vb2)?;
    println!("\n已从 safetensors 重新加载模型，开始推理\n");

    // ===== 第 5 步：在全新测试图上跑完整推理管线，逐图打印检出框与 IoU，并画框存 PNG =====
    let (conf_thr, nms_thr) = (0.5f32, 0.45f32);
    let mut test_rng = Lcg::new(2025);
    let (mut total_gt, mut total_hit, mut iou_sum) = (0usize, 0usize, 0f32);
    for idx in 0..5 {
        let (img, gts) = make_image(&mut test_rng);
        let input = Tensor::from_vec(img_to_chw(&img), (1, 3, IMG, IMG), &dev)?; // 预处理：CHW + /255 + 加 batch 维
        let out = model.forward(&input)?; // 前向 → [1,A,G,G]
        let cands = decode(&out, conf_thr)?; // 解码 + 置信度阈值
        let dets = nms(cands, nms_thr); // NMS

        // 把每个检出框贪心匹配到同类、IoU 最大的真值框，统计命中与 IoU
        let mut used = vec![false; gts.len()];
        println!("测试图 #{idx}：{} 个真值目标，检出 {} 个框", gts.len(), dets.len());
        for d in &dets {
            let mut best = (usize::MAX, 0f32);
            for (gi, g) in gts.iter().enumerate() {
                if used[gi] || g.cls != d.cls {
                    continue;
                }
                let v = iou(&d.xyxy, &g.xyxy());
                if v > best.1 {
                    best = (gi, v);
                }
            }
            let b = d.xyxy;
            if best.0 != usize::MAX && best.1 >= 0.5 {
                used[best.0] = true;
                total_hit += 1;
                iou_sum += best.1;
                println!("   ✓ {:<3} score={:.2}  框=[{:>4.0},{:>4.0},{:>4.0},{:>4.0}]  IoU={:.3}", CLASSES[d.cls], d.score, b[0], b[1], b[2], b[3], best.1);
            } else {
                println!("   ? {:<3} score={:.2}  框=[{:>4.0},{:>4.0},{:>4.0},{:>4.0}]  （误检/重复）", CLASSES[d.cls], d.score, b[0], b[1], b[2], b[3]);
            }
        }
        total_gt += gts.len();
        draw_and_save(&img, &gts, &dets, &format!("det_{idx}.png"));
    }

    println!("\n===== 汇总 =====");
    println!("真值目标共 {total_gt} 个，正确检出 {total_hit} 个（检出率 {:.1}%）", 100.0 * total_hit as f32 / total_gt as f32);
    if total_hit > 0 {
        println!("命中框的平均 IoU = {:.3}", iou_sum / total_hit as f32);
    }
    println!("可视化结果已存为 det_0.png .. det_4.png（绿=真值，红=检出）");
    Ok(())
}
