//! 第 48 章 检测评估、标注质量与统计不确定性
//! 运行：cargo run --example ch48_metrics_annotation
//! 依赖：仅用 std（合成 预测框 + GT，无需任何模型，确定性可复现）。
//!
//! 全套纯手写检测评估，每个数字都是真实运行结果：
//!   ① PR 曲线：按分数排序，预测与 GT 按 IoU≥阈值 一对一匹配，判 TP/FP/FN
//!   ② AP：11 点插值 与 全点积分 两种都实现，多类取 mAP
//!   ③ mAP@0.5 与 mAP@[.5:.95] 的区别
//!   ④ 混淆矩阵 + 每类召回
//!   ⑤ bootstrap 95% 置信区间（对图像重采样，固定种子可复现）
//!   ⑥ 标注质量：注入 10% 漏标后 mAP 掉多少

// ============ 基础：框、IoU、确定性伪随机 ============

#[derive(Clone, Copy)]
struct Xyxy {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}
impl Xyxy {
    fn area(self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
    fn iou(self, o: Xyxy) -> f32 {
        let ix1 = self.x1.max(o.x1);
        let iy1 = self.y1.max(o.y1);
        let ix2 = self.x2.min(o.x2);
        let iy2 = self.y2.min(o.y2);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let uni = self.area() + o.area() - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

/// 由中心宽高造框
fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Xyxy {
    Xyxy { x1: cx - w / 2.0, y1: cy - h / 2.0, x2: cx + w / 2.0, y2: cy + h / 2.0 }
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// [0,1) 均匀
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// [-1,1) 均匀
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

// ============ 数据结构 ============

#[derive(Clone, Copy)]
struct Gt {
    cls: usize,
    b: Xyxy,
}
#[derive(Clone, Copy)]
struct Pred {
    cls: usize,
    score: f32,
    b: Xyxy,
}
#[derive(Clone)]
struct Sample {
    gts: Vec<Gt>,
    preds: Vec<Pred>,
}

const N_CLS: usize = 3; // 三类：C0/C1/C2

/// 合成一个数据集：n_images 张图，每张 2~4 个 GT。
/// 「模拟检测器」按 85% 概率命中每个 GT（框带抖动、分数偏高），
/// 偶尔预测错类别（类别混淆），并额外撒少量低分背景误检。其余 GT 漏检。
fn synth_dataset(seed: u64, n_images: usize) -> Vec<Sample> {
    let mut rng = Lcg::new(seed);
    let mut imgs = Vec::new();
    for _ in 0..n_images {
        let n_gt = 2 + (rng.unit() * 3.0) as usize; // 2..=4
        let mut gts = Vec::new();
        let mut preds = Vec::new();
        for _ in 0..n_gt {
            let cls = (rng.unit() * N_CLS as f32) as usize;
            let (cx, cy) = (30.0 + rng.unit() * 200.0, 30.0 + rng.unit() * 200.0);
            let (w, h) = (30.0 + rng.unit() * 40.0, 30.0 + rng.unit() * 40.0);
            gts.push(Gt { cls, b: from_cxcywh(cx, cy, w, h) });
            // 是否命中该 GT
            if rng.unit() < 0.85 {
                let pb = from_cxcywh(
                    cx + rng.signed() * 6.0,
                    cy + rng.signed() * 6.0,
                    w + rng.signed() * 6.0,
                    h + rng.signed() * 6.0,
                );
                // 12% 概率预测成邻类（类别混淆）
                let pcls = if rng.unit() < 0.12 { (cls + 1) % N_CLS } else { cls };
                let score = 0.55 + rng.unit() * 0.44; // TP 分数偏高
                preds.push(Pred { cls: pcls, score, b: pb });
            }
        }
        // 背景误检：0~2 个低分随机框
        let n_fp = (rng.unit() * 2.5) as usize;
        for _ in 0..n_fp {
            let cls = (rng.unit() * N_CLS as f32) as usize;
            let (cx, cy) = (30.0 + rng.unit() * 200.0, 30.0 + rng.unit() * 200.0);
            let (w, h) = (30.0 + rng.unit() * 40.0, 30.0 + rng.unit() * 40.0);
            let score = 0.30 + rng.unit() * 0.40; // FP 分数偏低
            preds.push(Pred { cls, score, b: from_cxcywh(cx, cy, w, h) });
        }
        imgs.push(Sample { gts, preds });
    }
    imgs
}

fn all_refs(imgs: &[Sample]) -> Vec<&Sample> {
    imgs.iter().collect()
}

// ============ ① PR 曲线：匹配 + 累计 precision/recall ============

/// 对某个类、某个 IoU 阈值做匹配，返回按分数降序累计的 (precision[], recall[], n_gt)。
/// 规则：每个预测在其所在图找同类、未匹配、IoU 最大且 ≥阈值的 GT → TP 并占用该 GT；
/// 否则 FP。一个 GT 只能被匹配一次（后来的高分重复框只能算 FP —— 这正是 NMS 影响 AP 的原因）。
fn pr_curve(images: &[&Sample], cls: usize, iou_t: f32) -> (Vec<f32>, Vec<f32>, usize) {
    // 收集该类全部预测：(score, 图下标, 框)
    let mut dets: Vec<(f32, usize, Xyxy)> = Vec::new();
    let mut n_gt = 0usize;
    for (ii, im) in images.iter().enumerate() {
        for g in &im.gts {
            if g.cls == cls {
                n_gt += 1;
            }
        }
        for p in &im.preds {
            if p.cls == cls {
                dets.push((p.score, ii, p.b));
            }
        }
    }
    if n_gt == 0 {
        return (vec![], vec![], 0);
    }
    // 关键：按分数从高到低（决定 PR 曲线的扫描顺序）
    dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // 每图该类 GT 的「已被占用」标记
    let mut matched: Vec<Vec<bool>> = images
        .iter()
        .map(|im| vec![false; im.gts.len()])
        .collect();

    let mut prec = Vec::with_capacity(dets.len());
    let mut rec = Vec::with_capacity(dets.len());
    let (mut cum_tp, mut cum_fp) = (0.0f32, 0.0f32);
    for &(_, ii, pb) in &dets {
        let mut best_iou = 0.0f32;
        let mut best_j: Option<usize> = None;
        for (j, g) in images[ii].gts.iter().enumerate() {
            if g.cls != cls || matched[ii][j] {
                continue;
            }
            let iou = pb.iou(g.b);
            if iou > best_iou {
                best_iou = iou;
                best_j = Some(j);
            }
        }
        if best_iou >= iou_t {
            cum_tp += 1.0;
            matched[ii][best_j.unwrap()] = true; // 占用该 GT
        } else {
            cum_fp += 1.0;
        }
        prec.push(cum_tp / (cum_tp + cum_fp));
        rec.push(cum_tp / n_gt as f32);
    }
    (prec, rec, n_gt)
}

// ============ ② AP：全点积分 与 11 点插值 ============

/// 全点积分（VOC2010 / COCO 风格）：先把 precision 从右往左取「后缀最大值」变单调，
/// 再按 recall 的每一次跳变累加矩形面积。
fn ap_all_points(prec: &[f32], rec: &[f32]) -> f32 {
    if prec.is_empty() {
        return 0.0;
    }
    // 两端补哨兵：recall 0→1，precision 两端补 0
    let mut mrec = Vec::with_capacity(prec.len() + 2);
    let mut mpre = Vec::with_capacity(prec.len() + 2);
    mrec.push(0.0);
    mpre.push(0.0);
    mrec.extend_from_slice(rec);
    mpre.extend_from_slice(prec);
    mrec.push(1.0);
    mpre.push(0.0);
    // precision 单调不增（从后往前取 max）
    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }
    // recall 有跳变处累加面积
    let mut ap = 0.0;
    for i in 1..mrec.len() {
        if (mrec[i] - mrec[i - 1]).abs() > 1e-12 {
            ap += (mrec[i] - mrec[i - 1]) * mpre[i];
        }
    }
    ap
}

/// 11 点插值（VOC2007 风格）：在 recall = 0,0.1,...,1.0 处取「召回≥该值的最大 precision」，取平均。
fn ap_11_points(prec: &[f32], rec: &[f32]) -> f32 {
    if prec.is_empty() {
        return 0.0;
    }
    let mut ap = 0.0;
    for t in 0..=10 {
        let level = t as f32 / 10.0;
        let mut p_max = 0.0f32;
        for i in 0..rec.len() {
            if rec[i] >= level && prec[i] > p_max {
                p_max = prec[i];
            }
        }
        ap += p_max / 11.0;
    }
    ap
}

/// 单类 AP：返回 (全点 AP, 11 点 AP, n_gt)
fn class_ap(images: &[&Sample], cls: usize, iou_t: f32) -> (f32, f32, usize) {
    let (prec, rec, n_gt) = pr_curve(images, cls, iou_t);
    (ap_all_points(&prec, &rec), ap_11_points(&prec, &rec), n_gt)
}

/// mAP：对有 GT 的类取平均。use_11=true 用 11 点插值，否则全点积分。
fn map_at(images: &[&Sample], iou_t: f32, use_11: bool) -> f32 {
    let mut sum = 0.0;
    let mut cnt = 0;
    for c in 0..N_CLS {
        let (ap_all, ap_11, n_gt) = class_ap(images, c, iou_t);
        if n_gt > 0 {
            sum += if use_11 { ap_11 } else { ap_all };
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f32 }
}

/// mAP@[.5:.95]：IoU 从 0.50 到 0.95 步长 0.05 共 10 档，全点 AP 平均。
fn map_5_95(images: &[&Sample]) -> f32 {
    let mut sum = 0.0;
    for k in 0..10 {
        let t = 0.5 + 0.05 * k as f32;
        sum += map_at(images, t, false);
    }
    sum / 10.0
}

// ============ ④ 混淆矩阵（class-agnostic 贪心匹配 + 置信度阈值）============

/// (N_CLS+1) x (N_CLS+1)，最后一行/列为 background。
/// 行=真值类别，列=预测类别；对角=判对，非对角=类别混淆，
/// [c][BG]=该类漏检，[BG][c]=背景误报成 c。
fn confusion_matrix(images: &[&Sample], iou_t: f32, conf_t: f32) -> Vec<Vec<u32>> {
    let bg = N_CLS;
    let mut cm = vec![vec![0u32; N_CLS + 1]; N_CLS + 1];
    for im in images {
        let preds: Vec<&Pred> = im.preds.iter().filter(|p| p.score >= conf_t).collect();
        let gts = &im.gts;
        // 所有 IoU≥阈值 的候选对，class-agnostic，按 IoU 降序贪心占用
        let mut cand: Vec<(f32, usize, usize)> = Vec::new();
        for (pi, p) in preds.iter().enumerate() {
            for (gi, g) in gts.iter().enumerate() {
                let iou = p.b.iou(g.b);
                if iou >= iou_t {
                    cand.push((iou, pi, gi));
                }
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut p_used = vec![false; preds.len()];
        let mut g_used = vec![false; gts.len()];
        for (_, pi, gi) in cand {
            if !p_used[pi] && !g_used[gi] {
                p_used[pi] = true;
                g_used[gi] = true;
                cm[gts[gi].cls][preds[pi].cls] += 1;
            }
        }
        for (gi, g) in gts.iter().enumerate() {
            if !g_used[gi] {
                cm[g.cls][bg] += 1; // 漏检
            }
        }
        for (pi, p) in preds.iter().enumerate() {
            if !p_used[pi] {
                cm[bg][p.cls] += 1; // 背景误报
            }
        }
    }
    cm
}

// ============ ⑤ bootstrap 置信区间 ============

/// 对图像做有放回重采样 B 次，每次重算 mAP@0.5（全点），
/// 得到 mAP 的分布，取 2.5% / 97.5% 分位数作为 95% 置信区间。种子固定 → 可复现。
fn bootstrap_map(imgs: &[Sample], b_iter: usize, seed: u64) -> (f32, f32, f32, f32) {
    let n = imgs.len();
    let mut rng = Lcg::new(seed);
    let mut vals = Vec::with_capacity(b_iter);
    for _ in 0..b_iter {
        // 有放回抽 n 张图（同一张可能被抽中多次，各算作独立样本）
        let sample: Vec<&Sample> = (0..n)
            .map(|_| {
                let idx = (rng.unit() * n as f32) as usize % n;
                &imgs[idx]
            })
            .collect();
        vals.push(map_at(&sample, 0.5, false));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = vals.iter().sum::<f32>() / b_iter as f32;
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / b_iter as f32).sqrt();
    let lo = vals[(0.025 * b_iter as f32) as usize];
    let hi = vals[(0.975 * b_iter as f32) as usize];
    (mean, std, lo, hi)
}

// ============ ⑥ 标注质量：注入漏标 ============

/// 复制数据集，**恰好**删除 drop_rate 比例的 GT（模拟漏标）。返回 (新数据集, 实删数, 原总数)。
/// 用确定性 Fisher-Yates 洗牌选出被删的 GT，保证「10% 就是 10%」且可复现。
fn inject_missing_labels(imgs: &[Sample], drop_rate: f32, seed: u64) -> (Vec<Sample>, usize, usize) {
    // 收集所有 GT 的全局定位 (图下标, 图内下标)
    let mut all: Vec<(usize, usize)> = Vec::new();
    for (i, im) in imgs.iter().enumerate() {
        for j in 0..im.gts.len() {
            all.push((i, j));
        }
    }
    let total = all.len();
    let k = (drop_rate * total as f32).round() as usize; // 恰好丢这么多个
    // Fisher-Yates 洗牌（确定性），取前 k 个作为「被漏标」
    let mut rng = Lcg::new(seed);
    for i in (1..all.len()).rev() {
        let j = (rng.unit() * (i + 1) as f32) as usize % (i + 1);
        all.swap(i, j);
    }
    let drop_set: std::collections::HashSet<(usize, usize)> = all.into_iter().take(k).collect();
    let mut out = Vec::with_capacity(imgs.len());
    for (i, im) in imgs.iter().enumerate() {
        let mut gts = Vec::new();
        for (j, g) in im.gts.iter().enumerate() {
            if !drop_set.contains(&(i, j)) {
                gts.push(*g); // 未被选中漏标的 GT 保留
            }
        }
        out.push(Sample { gts, preds: im.preds.clone() });
    }
    (out, k, total)
}

// ============ main ============

fn main() {
    let imgs = synth_dataset(48, 24);
    let refs = all_refs(&imgs);
    let n_gt_total: usize = imgs.iter().map(|s| s.gts.len()).sum();
    let n_pred_total: usize = imgs.iter().map(|s| s.preds.len()).sum();
    println!(
        "合成数据集：{} 张图，共 {} 个 GT、{} 个预测框，{} 类。\n",
        imgs.len(),
        n_gt_total,
        n_pred_total,
        N_CLS
    );

    // ① PR 曲线（展示 C0 在 IoU=0.5 的部分扫描点）
    println!("== ① PR 曲线（类别 C0, IoU≥0.5, 按分数从高到低扫描）==");
    let (prec, rec, ngt0) = pr_curve(&refs, 0, 0.5);
    println!("C0 共 {} 个 GT，{} 个预测；随阈值下降的 (recall, precision) 轨迹：", ngt0, prec.len());
    for i in 0..prec.len() {
        // 每隔几点打印一次，避免刷屏
        if i % 2 == 0 || i == prec.len() - 1 {
            println!("  第{:>2}个预测: recall={:.3}  precision={:.3}", i + 1, rec[i], prec[i]);
        }
    }
    println!();

    // ② AP / mAP：两种算法对比
    println!("== ② AP（全点积分 vs 11 点插值）@ IoU=0.5 ==");
    for c in 0..N_CLS {
        let (ap_all, ap_11, ng) = class_ap(&refs, c, 0.5);
        println!("  C{c}: 全点 AP={ap_all:.4}  11点 AP={ap_11:.4}  (GT={ng})");
    }
    let map50_all = map_at(&refs, 0.5, false);
    let map50_11 = map_at(&refs, 0.5, true);
    println!("  → mAP@0.5 全点={map50_all:.4}   11点={map50_11:.4}");
    println!();

    // ③ mAP@0.5 vs mAP@[.5:.95]
    println!("== ③ mAP@0.5 vs mAP@[.5:.95] ==");
    println!("  mAP@0.5      = {:.4}  （框只要「大致对」，IoU≥0.5 即算命中）", map50_all);
    println!("  mAP@[.5:.95] = {:.4}  （IoU 0.50→0.95 十档平均，对定位精度苛刻得多）", map_5_95(&refs));
    println!();

    // ④ 混淆矩阵 + 每类召回
    println!("== ④ 混淆矩阵（IoU=0.5, conf≥0.5；行=真值, 列=预测；BG=背景）==");
    let cm = confusion_matrix(&refs, 0.5, 0.5);
    print!("        ");
    for c in 0..N_CLS {
        print!("预测C{c} ");
    }
    println!("预测BG");
    for r in 0..=N_CLS {
        let name = if r < N_CLS { format!("真值C{r}") } else { "真值BG".to_string() };
        print!("{name}   ");
        for c in 0..=N_CLS {
            print!("{:>5} ", cm[r][c]);
        }
        println!();
    }
    println!("  每类召回（对角/该行 GT 总数）:");
    for c in 0..N_CLS {
        let row_sum: u32 = cm[c].iter().sum(); // 该类 GT 总数（含漏检列）
        let recall = if row_sum > 0 { cm[c][c] as f32 / row_sum as f32 } else { 0.0 };
        let col_sum: u32 = (0..=N_CLS).map(|r| cm[r][c]).sum(); // 预测为该类的总数
        let prec = if col_sum > 0 { cm[c][c] as f32 / col_sum as f32 } else { 0.0 };
        println!("    C{c}: 召回={recall:.3}  精确率={prec:.3}");
    }
    println!();

    // ⑤ bootstrap 95% CI
    println!("== ⑤ mAP@0.5 的 bootstrap 95% 置信区间（对图像有放回重采样 2000 次）==");
    let (mean, std, lo, hi) = bootstrap_map(&imgs, 2000, 12345);
    println!("  点估计 mAP@0.5 = {map50_all:.4}");
    println!("  bootstrap 均值={mean:.4}  标准误={std:.4}  95% CI=[{lo:.4}, {hi:.4}]");
    println!("  → 结论：真实 mAP 大概落在 [{lo:.3}, {hi:.3}]，报数字不带这个区间是不专业的。");
    println!();

    // ⑥ 标注质量：注入漏标（10% 主实验 + 递增趋势）
    println!("== ⑥ 标注质量实验：注入漏标后 mAP 怎么变 ==");
    let (corrupt, dropped, total) = inject_missing_labels(&imgs, 0.10, 999);
    let crefs = all_refs(&corrupt);
    let map_corrupt = map_at(&crefs, 0.5, false);
    println!("  原始    mAP@0.5 = {map50_all:.4}（GT 总数 {total}）");
    println!(
        "  漏标10% mAP@0.5 = {map_corrupt:.4}（删掉 {dropped} 个 GT）",
    );
    let delta = map50_all - map_corrupt;
    println!(
        "  → mAP 掉了 {delta:.4}（相对下降 {:.1}%）。模型没变，只是标注变差，指标就被毒化了。",
        100.0 * delta / map50_all
    );
    println!("  漏标比例递增趋势（每档独立随机删除，固定种子）：");
    for pct in [0u32, 5, 10, 20, 30] {
        let (c, d, _) = inject_missing_labels(&imgs, pct as f32 / 100.0, 999);
        let m = map_at(&all_refs(&c), 0.5, false);
        println!("    漏标 {pct:>2}%（删 {d:>2} 个）: mAP@0.5 = {m:.4}");
    }
    println!("    原因：被漏标的真实目标仍被模型检出，却因「查无此 GT」被算成误报(FP)，拉低 precision。");
}
