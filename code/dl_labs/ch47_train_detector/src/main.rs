// 第 47 章：从零训练小检测器 —— 损失三件套、mAP 评估、数据闭环与受控实验
// candle 0.11.0，纯 CPU，无需下载任何数据/模型；结果可复现。
// 运行：cd code/dl_labs/ch47_train_detector && cargo run --release
//
// 本章跑三次训练，做两个“一次只改一个变量”的受控实验：
//   实验一（划分泄漏）：同一批“场景 + 近似重复帧”，分别用【随机划分】和【按场景分组划分】
//                      得到 train/val，比较 val 上的 mAP@0.5 —— 看泄漏如何虚高指标。
//   实验二（消融）：在分组划分（诚实）上，只把框回归损失权重 λ_box 从 1 改到 5，
//                  比较 mAP@0.5 与平均 IoU 的真实变化。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};

const IMG: usize = 48;
const G: usize = 6;
const STRIDE: usize = IMG / G;
const C: usize = 2;
const A: usize = 1 + 4 + C;

// ---------- 确定性随机数 ----------
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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

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
        [self.cx - self.w / 2.0, self.cy - self.h / 2.0, self.cx + self.w / 2.0, self.cy + self.h / 2.0]
    }
}

// 一个“场景”＝一组固定布局的目标（模拟同一监控画面/同一段视频里的固定内容）。
// 尺寸范围故意放宽、含小目标与细横条，让“未见过的新场景”真的更难 —— 这样划分泄漏才咬得动。
fn make_scene(rng: &mut Lcg) -> Vec<Gt> {
    let n = rng.range(2, 5); // 2~4 个目标，画面更拥挤
    let mut gts: Vec<Gt> = Vec::new();
    let mut tries = 0;
    while (gts.len() as i32) < n && tries < 120 {
        tries += 1;
        let cls = rng.range(0, C as i32) as usize;
        let (w, h) = if cls == 0 {
            let s = rng.range(8, 17) as f32; // 方块：8~16（含难的小目标）
            (s, s)
        } else {
            (rng.range(12, 31) as f32, rng.range(5, 10) as f32) // 横条：宽 12~30、高 5~9（含难的细条）
        };
        let (hw, hh) = ((w / 2.0) as i32, (h / 2.0) as i32);
        let cx = rng.range(hw + 1, IMG as i32 - hw - 1) as f32;
        let cy = rng.range(hh + 1, IMG as i32 - hh - 1) as f32;
        if gts.iter().any(|g| (g.cx - cx).abs() < 13.0 && (g.cy - cy).abs() < 13.0) {
            continue;
        }
        gts.push(Gt { cls, cx, cy, w, h });
    }
    gts
}

// 从一个场景渲染出一“帧”：给每个目标加 ±2px 的位置抖动 + 随机亮度 + 像素噪声。
// 这样同场景的多帧是“近似重复”，正是相邻视频帧的写照 —— 划分泄漏的温床。
fn render_frame(scene: &[Gt], rng: &mut Lcg) -> (Vec<f32>, Vec<Gt>) {
    let mut px = vec![0f32; IMG * IMG];
    let mut gts = Vec::with_capacity(scene.len());
    for g in scene {
        let cx = (g.cx + rng.range(-2, 3) as f32).clamp(g.w / 2.0, IMG as f32 - g.w / 2.0);
        let cy = (g.cy + rng.range(-2, 3) as f32).clamp(g.h / 2.0, IMG as f32 - g.h / 2.0);
        let bright = 0.55 + rng.unit() * 0.4; // 每个目标亮度随机（外观多样，进一步拉开新旧场景难度）
        let jg = Gt { cls: g.cls, cx, cy, w: g.w, h: g.h };
        let b = jg.xyxy();
        for y in b[1].round() as i32..b[3].round() as i32 {
            for x in b[0].round() as i32..b[2].round() as i32 {
                if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
                    px[y as usize * IMG + x as usize] = bright;
                }
            }
        }
        gts.push(jg);
    }
    for p in px.iter_mut() {
        *p = (*p + 0.06 + (rng.unit() - 0.5) * 0.08).clamp(0.0, 1.0);
    }
    let mut chw = Vec::with_capacity(3 * IMG * IMG);
    for _ in 0..3 {
        chw.extend_from_slice(&px);
    }
    (chw, gts)
}

// 把若干帧打包成输入张量 [N,3,IMG,IMG] + 每帧真值
fn pack(frames: &[(Vec<f32>, Vec<Gt>)], dev: &Device) -> Result<(Tensor, Vec<Vec<Gt>>)> {
    let n = frames.len();
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    let mut gts = Vec::with_capacity(n);
    for (chw, g) in frames {
        xs.extend_from_slice(chw);
        gts.push(g.clone());
    }
    Ok((Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?, gts))
}

fn build_targets(all_gts: &[Vec<Gt>], dev: &Device, pos_w: f32) -> Result<(Tensor, Tensor, Tensor, Tensor, f32)> {
    let n = all_gts.len();
    let mut obj = vec![0f32; n * G * G];
    let mut wgt = vec![1f32; n * G * G];
    let mut boxn = vec![0f32; n * 4 * G * G];
    let mut cls = vec![0f32; n * C * G * G];
    let mut num_pos = 0f32;
    for (ni, gts) in all_gts.iter().enumerate() {
        for g in gts {
            let ci = ((g.cy / STRIDE as f32).floor() as usize).min(G - 1);
            let cj = ((g.cx / STRIDE as f32).floor() as usize).min(G - 1);
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
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?;
        self.head.forward(&x)
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

fn bce_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.broadcast_mul(targets)?;
    let sp = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    relu_x.broadcast_sub(&xy)?.broadcast_add(&sp)
}

// 检测损失三件套：objectness BCE + 类别 BCE + 框回归 L1，加权求和
fn compute_loss(
    out: &Tensor, obj_t: &Tensor, wgt: &Tensor, boxn_t: &Tensor, cls_t: &Tensor,
    num_pos: f32, cell_x: &Tensor, cell_y: &Tensor, lambda_box: f64, lambda_cls: f64,
) -> Result<(Tensor, f32, f32, f32)> {
    let n = out.dim(0)?;
    let obj_logit = out.narrow(1, 0, 1)?;
    let box_raw = out.narrow(1, 1, 4)?;
    let cls_logit = out.narrow(1, 5, C)?;

    let l_obj = bce_elem(&obj_logit, obj_t)?.broadcast_mul(wgt)?.sum_all()?.affine(1.0 / (n * G * G) as f64, 0.0)?;

    let sig = sigmoid(&box_raw)?;
    let sx = sig.narrow(1, 0, 1)?;
    let sy = sig.narrow(1, 1, 1)?;
    let sw = sig.narrow(1, 2, 1)?;
    let sh = sig.narrow(1, 3, 1)?;
    let cx = sx.broadcast_add(cell_x)?.affine(1.0 / G as f64, 0.0)?;
    let cy = sy.broadcast_add(cell_y)?.affine(1.0 / G as f64, 0.0)?;
    let pred_boxn = Tensor::cat(&[&cx, &cy, &sw, &sh], 1)?;
    let l_box = pred_boxn.broadcast_sub(boxn_t)?.abs()?.broadcast_mul(obj_t)?.sum_all()?.affine(1.0 / (num_pos * 4.0) as f64, 0.0)?;

    let l_cls = bce_elem(&cls_logit, cls_t)?.broadcast_mul(obj_t)?.sum_all()?.affine(1.0 / (num_pos * C as f32) as f64, 0.0)?;

    let total = l_obj.broadcast_add(&l_box.affine(lambda_box, 0.0)?)?.broadcast_add(&l_cls.affine(lambda_cls, 0.0)?)?;
    Ok((total, l_obj.to_scalar::<f32>()?, l_box.to_scalar::<f32>()?, l_cls.to_scalar::<f32>()?))
}

fn cell_grids(dev: &Device) -> Result<(Tensor, Tensor)> {
    let (mut cxv, mut cyv) = (vec![0f32; G * G], vec![0f32; G * G]);
    for i in 0..G {
        for j in 0..G {
            cxv[i * G + j] = j as f32;
            cyv[i * G + j] = i as f32;
        }
    }
    Ok((Tensor::from_vec(cxv, (1, 1, G, G), dev)?, Tensor::from_vec(cyv, (1, 1, G, G), dev)?))
}

// ---------- 推理侧解码 / IoU / NMS / 评估 ----------
#[derive(Clone)]
struct DetE {
    img: usize,
    cls: usize,
    score: f32,
    bbox: [f32; 4],
}
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// 解码单张图的输出 [A,G,G] → 候选框
fn decode(v: &[Vec<Vec<f32>>], img: usize, conf_thr: f32) -> Vec<DetE> {
    let mut dets = Vec::new();
    for i in 0..G {
        for j in 0..G {
            let obj = sigmoidf(v[0][i][j]);
            let (mut bc, mut bl) = (0usize, f32::MIN);
            for c in 0..C {
                if v[5 + c][i][j] > bl {
                    bl = v[5 + c][i][j];
                    bc = c;
                }
            }
            let score = obj * sigmoidf(bl);
            if score < conf_thr {
                continue;
            }
            let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32;
            let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
            let w = sigmoidf(v[3][i][j]) * IMG as f32;
            let h = sigmoidf(v[4][i][j]) * IMG as f32;
            dets.push(DetE { img, cls: bc, score, bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0] });
        }
    }
    dets
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let aa = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ab = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let u = aa + ab - inter;
    if u <= 0.0 {
        0.0
    } else {
        inter / u
    }
}

fn nms(mut dets: Vec<DetE>, iou_thr: f32) -> Vec<DetE> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut removed = vec![false; dets.len()];
    let mut keep = Vec::new();
    for i in 0..dets.len() {
        if removed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if !removed[j] && dets[j].cls == dets[i].cls && iou(&dets[i].bbox, &dets[j].bbox) > iou_thr {
                removed[j] = true;
            }
        }
    }
    keep
}

// 单类 AP@iou_thr（连续/all-points 积分法）
fn average_precision(dets: &[DetE], gts: &[Vec<Gt>], class: usize, iou_thr: f32) -> Option<f32> {
    let n_gt = gts.iter().flatten().filter(|g| g.cls == class).count();
    if n_gt == 0 {
        return None;
    }
    let mut cd: Vec<&DetE> = dets.iter().filter(|d| d.cls == class).collect();
    cd.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut used: Vec<Vec<bool>> = gts.iter().map(|g| vec![false; g.len()]).collect();
    let (mut prec, mut rec) = (Vec::new(), Vec::new());
    let (mut ctp, mut cfp) = (0f32, 0f32);
    for d in &cd {
        let mut best = (usize::MAX, iou_thr);
        for (gi, g) in gts[d.img].iter().enumerate() {
            if g.cls != class || used[d.img][gi] {
                continue;
            }
            let v = iou(&d.bbox, &g.xyxy());
            if v >= best.1 {
                best = (gi, v);
            }
        }
        if best.0 != usize::MAX {
            ctp += 1.0;
            used[d.img][best.0] = true;
        } else {
            cfp += 1.0;
        }
        prec.push(ctp / (ctp + cfp));
        rec.push(ctp / n_gt as f32);
    }
    if prec.is_empty() {
        return Some(0.0);
    }
    for i in (0..prec.len() - 1).rev() {
        prec[i] = prec[i].max(prec[i + 1]); // 精确率包络（单调不增）
    }
    let (mut ap, mut prev_r) = (0f32, 0f32);
    for i in 0..prec.len() {
        ap += (rec[i] - prev_r) * prec[i];
        prev_r = rec[i];
    }
    Some(ap)
}

struct Metrics {
    map: f32,
    ap: [f32; C],
    recall: f32,
    precision: f32,
    mean_iou: f32,
}

// 在验证集上评估：mAP@0.5（全体候选）+ 固定阈值处的召回/精确率/平均 IoU（工作点）
fn evaluate(model: &Detector, xval: &Tensor, gts: &[Vec<Gt>], nms_thr: f32, op_conf: f32) -> Result<Metrics> {
    let out = model.forward(xval)?; // [Nval,A,G,G]
    let n = out.dim(0)?;
    let mut all_dets: Vec<DetE> = Vec::new(); // 低阈值全体候选（算 AP 用）
    let mut op_dets: Vec<Vec<DetE>> = Vec::new(); // 工作点检出（算召回/IoU 用）
    for k in 0..n {
        let v = out.i(k)?.to_vec3::<f32>()?;
        let cands = decode(&v, k, 0.05);
        let kept = nms(cands, nms_thr);
        op_dets.push(kept.iter().filter(|d| d.score >= op_conf).cloned().collect());
        all_dets.extend(kept);
    }
    // mAP@0.5
    let mut ap = [0f32; C];
    let (mut sum, mut cnt) = (0f32, 0);
    for c in 0..C {
        if let Some(a) = average_precision(&all_dets, gts, c, 0.5) {
            ap[c] = a;
            sum += a;
            cnt += 1;
        }
    }
    let map = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
    // 工作点：贪心匹配算召回、精确率、平均 IoU
    let (mut tp, mut fp, mut iou_sum) = (0usize, 0usize, 0f32);
    let total_gt: usize = gts.iter().map(|g| g.len()).sum();
    for (k, dets) in op_dets.iter().enumerate() {
        let mut dd = dets.clone();
        dd.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let mut used = vec![false; gts[k].len()];
        for d in &dd {
            let mut best = (usize::MAX, 0.5f32);
            for (gi, g) in gts[k].iter().enumerate() {
                if used[gi] || g.cls != d.cls {
                    continue;
                }
                let v = iou(&d.bbox, &g.xyxy());
                if v >= best.1 {
                    best = (gi, v);
                }
            }
            if best.0 != usize::MAX {
                used[best.0] = true;
                tp += 1;
                iou_sum += best.1;
            } else {
                fp += 1;
            }
        }
    }
    Ok(Metrics {
        map,
        ap,
        recall: tp as f32 / total_gt.max(1) as f32,
        precision: tp as f32 / (tp + fp).max(1) as f32,
        mean_iou: if tp > 0 { iou_sum / tp as f32 } else { 0.0 },
    })
}

// 从零训练一个检测器，返回训好的 VarMap（供加载出模型评估）
fn train(
    xtr: &Tensor, gts_tr: &[Vec<Gt>], dev: &Device,
    epochs: usize, lambda_box: f64, init_seed: u64, verbose: bool,
) -> Result<VarMap> {
    let (obj_t, wgt, boxn_t, cls_t, num_pos) = build_targets(gts_tr, dev, 6.0)?;
    let (cell_x, cell_y) = cell_grids(dev)?;
    let vm = VarMap::new();
    let net = Detector::load(VarBuilder::from_varmap(&vm, DType::F32, dev))?;
    det_init(&vm, init_seed)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    for epoch in 1..=epochs {
        if epoch == (epochs * 7 / 10) {
            opt.set_learning_rate(4e-4);
        }
        let out = net.forward(xtr)?;
        let (loss, lo, lb, lc) = compute_loss(&out, &obj_t, &wgt, &boxn_t, &cls_t, num_pos, &cell_x, &cell_y, lambda_box, 1.0)?;
        opt.backward_step(&loss)?;
        if verbose && (epoch % 30 == 0 || epoch == 1) {
            println!("    epoch {epoch:>3}  total={:.4}  obj={lo:.4}  box={lb:.4}  cls={lc:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok(vm)
}

// 训练 + 从 VarMap 直接构模型 + 在 val 上评估（一条龙）
fn run(name: &str, xtr: &Tensor, gts_tr: &[Vec<Gt>], xval: &Tensor, gts_val: &[Vec<Gt>], dev: &Device, epochs: usize, lambda_box: f64, verbose: bool) -> Result<Metrics> {
    println!("[{name}] 训练中（{} 帧，λ_box={lambda_box}）...", gts_tr.len());
    let vm = train(xtr, gts_tr, dev, epochs, lambda_box, 42, verbose)?;
    let model = Detector::load(VarBuilder::from_varmap(&vm, DType::F32, dev))?;
    evaluate(&model, xval, gts_val, 0.45, 0.5)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let epochs = 200;

    // ===== 造“场景 + 近似重复帧” =====
    let n_scenes = 16;
    let frames_per_scene = 12;
    let mut rng = Lcg::new(7);
    let scenes: Vec<Vec<Gt>> = (0..n_scenes).map(|_| make_scene(&mut rng)).collect();
    // 每个场景渲染 frames_per_scene 帧（带抖动+噪声），记录它属于哪个场景
    let mut frames: Vec<(Vec<f32>, Vec<Gt>)> = Vec::new();
    let mut frame_scene: Vec<usize> = Vec::new();
    for (si, sc) in scenes.iter().enumerate() {
        for _ in 0..frames_per_scene {
            frames.push(render_frame(sc, &mut rng));
            frame_scene.push(si);
        }
    }
    let total = frames.len();
    println!("共 {n_scenes} 个场景 × {frames_per_scene} 帧 = {total} 帧\n");

    // ===== 划分 A：按场景分组（正确）—— 前 12 场景做 train，后 4 场景做 val（val 是全新布局） =====
    let val_scene_start = 12;
    let (mut gtr_idx, mut gval_idx) = (Vec::new(), Vec::new());
    for (fi, &si) in frame_scene.iter().enumerate() {
        if si < val_scene_start {
            gtr_idx.push(fi);
        } else {
            gval_idx.push(fi);
        }
    }
    // ===== 划分 B：随机划分（泄漏）—— 打乱所有帧后取同样大小的 val（近似重复帧会混进 val） =====
    let mut order: Vec<usize> = (0..total).collect();
    let mut shuf = Lcg::new(999);
    for i in (1..order.len()).rev() {
        let j = (shuf.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let n_val = gval_idx.len();
    let rval_idx: Vec<usize> = order[..n_val].to_vec();
    let rtr_idx: Vec<usize> = order[n_val..].to_vec();

    let subset = |idx: &[usize]| -> Result<(Tensor, Vec<Vec<Gt>>)> {
        let sel: Vec<(Vec<f32>, Vec<Gt>)> = idx.iter().map(|&i| frames[i].clone()).collect();
        pack(&sel, &dev)
    };

    // —— 泄漏诊断：随机划分里，有多少 val 帧的场景也出现在 train？——
    let train_scene_set: std::collections::HashSet<usize> = rtr_idx.iter().map(|&i| frame_scene[i]).collect();
    let leaked = rval_idx.iter().filter(|&&i| train_scene_set.contains(&frame_scene[i])).count();
    println!("实验一 · 划分泄漏");
    println!("  随机划分：{n_val} 个 val 帧里，有 {leaked} 帧的场景也在 train 中（近似重复→泄漏）");
    println!("  分组划分：val 全部来自 train 从未见过的 {} 个场景（无泄漏）\n", n_scenes - val_scene_start);

    let (xr_tr, gr_tr) = subset(&rtr_idx)?;
    let (xr_val, gr_val) = subset(&rval_idx)?;
    let (xg_tr, gg_tr) = subset(&gtr_idx)?;
    let (xg_val, gg_val) = subset(&gval_idx)?;

    let m_leak = run("随机划分", &xr_tr, &gr_tr, &xr_val, &gr_val, &dev, epochs, 5.0, false)?;
    let m_group = run("分组划分", &xg_tr, &gg_tr, &xg_val, &gg_val, &dev, epochs, 5.0, true)?;

    println!("\n  {:<10}{:>10}{:>10}{:>10}{:>10}", "划分方式", "mAP@0.5", "召回率", "精确率", "平均IoU");
    println!("  {:<10}{:>10.3}{:>10.3}{:>10.3}{:>10.3}", "随机(泄漏)", m_leak.map, m_leak.recall, m_leak.precision, m_leak.mean_iou);
    println!("  {:<10}{:>10.3}{:>10.3}{:>10.3}{:>10.3}", "分组(诚实)", m_group.map, m_group.recall, m_group.precision, m_group.mean_iou);
    println!("  → 随机划分把 mAP 虚高了 {:.1} 个百分点（泄漏的代价）", (m_leak.map - m_group.map) * 100.0);

    // ===== 实验二：消融 —— 只改 λ_box（1 → 5），在分组划分上比较 =====
    println!("\n实验二 · 消融（只改框回归损失权重 λ_box，其余全部不变）");
    let m_lb1 = run("λ_box=1", &xg_tr, &gg_tr, &xg_val, &gg_val, &dev, epochs, 1.0, false)?;
    // λ_box=5 复用实验一的分组划分结果 m_group
    println!("\n  {:<10}{:>10}{:>10}{:>10}{:>10}", "λ_box", "mAP@0.5", "召回率", "精确率", "平均IoU");
    println!("  {:<10}{:>10.3}{:>10.3}{:>10.3}{:>10.3}", "1", m_lb1.map, m_lb1.recall, m_lb1.precision, m_lb1.mean_iou);
    println!("  {:<10}{:>10.3}{:>10.3}{:>10.3}{:>10.3}", "5", m_group.map, m_group.recall, m_group.precision, m_group.mean_iou);
    println!("  分类别 AP@0.5（λ_box=1 → 5）：方块 {:.3}→{:.3}，横条 {:.3}→{:.3}", m_lb1.ap[0], m_group.ap[0], m_lb1.ap[1], m_group.ap[1]);
    println!("  → 框回归权重从 1 提到 5，平均 IoU {:.3}→{:.3}，mAP@0.5 {:.3}→{:.3}", m_lb1.mean_iou, m_group.mean_iou, m_lb1.map, m_group.map);
    Ok(())
}
