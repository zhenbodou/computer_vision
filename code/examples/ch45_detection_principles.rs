//! 第 45 章 目标检测原理：框表示、IoU、网格解码与 NMS
//! 运行：cargo run --example ch45_detection_principles
//! 依赖：仅用 std（合成检测框，无需任何模型/图片，确定性可复现）。
//!
//! 本例覆盖检测「后处理」的几块基石，每个数字都是真实运行结果：
//!   ① 框的三种表示 xyxy / xywh / cxcywh 及互转（附往返自检）
//!   ② IoU 交并比
//!   ③ 单阶段检测器的「网格 + anchor」解码：合成一份 raw 输出 → 解码成绝对坐标框
//!   ④ 贪心 NMS 非极大值抑制：在一堆合成重叠框上跑，打印抑制前后框数
//!   ⑤ 一对「anchor↔GT」匹配后的分类/定位损失数值示例（不训练，只算一次）

// ============ ① 框的三种表示与互转 ============

/// xyxy：左上 + 右下（半开区间 [x1,x2)），最适合算 IoU / 裁剪
#[derive(Clone, Copy, Debug)]
struct Xyxy {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

/// xywh：左上 + 宽高，最适合画框、也是 COCO 标注的存储格式
#[derive(Clone, Copy, Debug)]
struct Xywh {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// cxcywh：中心 + 宽高，检测器「回归」最常用（相对中心的偏移更好学）
#[derive(Clone, Copy, Debug)]
struct Cxcywh {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

impl Xyxy {
    fn to_xywh(self) -> Xywh {
        Xywh {
            x: self.x1,
            y: self.y1,
            w: self.x2 - self.x1,
            h: self.y2 - self.y1,
        }
    }
    fn to_cxcywh(self) -> Cxcywh {
        let w = self.x2 - self.x1;
        let h = self.y2 - self.y1;
        Cxcywh {
            cx: self.x1 + w / 2.0,
            cy: self.y1 + h / 2.0,
            w,
            h,
        }
    }
    /// 面积；宽或高为负时截 0，避免退化框给出负面积
    fn area(self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
    /// 交并比 IoU ∈ [0,1]：交集面积 / 并集面积
    fn iou(self, o: Xyxy) -> f32 {
        let ix1 = self.x1.max(o.x1);
        let iy1 = self.y1.max(o.y1);
        let ix2 = self.x2.min(o.x2);
        let iy2 = self.y2.min(o.y2);
        let iw = (ix2 - ix1).max(0.0); // 无交集时宽/高取 0
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let uni = self.area() + o.area() - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

impl Xywh {
    fn to_xyxy(self) -> Xyxy {
        Xyxy {
            x1: self.x,
            y1: self.y,
            x2: self.x + self.w,
            y2: self.y + self.h,
        }
    }
}

impl Cxcywh {
    fn to_xyxy(self) -> Xyxy {
        Xyxy {
            x1: self.cx - self.w / 2.0,
            y1: self.cy - self.h / 2.0,
            x2: self.cx + self.w / 2.0,
            y2: self.cy + self.h / 2.0,
        }
    }
}

fn demo_box_repr() {
    println!("== ① 框表示与互转（往返自检）==");
    // 一个左上角 (30,40)、宽 60 高 80 的框
    let a = Xyxy { x1: 30.0, y1: 40.0, x2: 90.0, y2: 120.0 };
    let xywh = a.to_xywh();
    let cxcywh = a.to_cxcywh();
    println!("xyxy   = [{:.0},{:.0},{:.0},{:.0}]", a.x1, a.y1, a.x2, a.y2);
    println!("xywh   = [x={:.0}, y={:.0}, w={:.0}, h={:.0}]", xywh.x, xywh.y, xywh.w, xywh.h);
    println!("cxcywh = [cx={:.0}, cy={:.0}, w={:.0}, h={:.0}]", cxcywh.cx, cxcywh.cy, cxcywh.w, cxcywh.h);
    // 转回 xyxy 应与原始完全一致（往返无损）
    let back1 = xywh.to_xyxy();
    let back2 = cxcywh.to_xyxy();
    println!(
        "xywh→xyxy 往返: [{:.0},{:.0},{:.0},{:.0}]  cxcywh→xyxy 往返: [{:.0},{:.0},{:.0},{:.0}]",
        back1.x1, back1.y1, back1.x2, back1.y2, back2.x1, back2.y1, back2.x2, back2.y2
    );
    println!();
}

fn demo_iou() {
    println!("== ② IoU 交并比（几组真实数字）==");
    let base = Xyxy { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 }; // 100x100 参考框
    let cases = [
        ("完全重合", Xyxy { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 }),
        ("平移一半 ", Xyxy { x1: 50.0, y1: 0.0, x2: 150.0, y2: 100.0 }),
        ("小框内含 ", Xyxy { x1: 25.0, y1: 25.0, x2: 75.0, y2: 75.0 }),
        ("完全不沾 ", Xyxy { x1: 200.0, y1: 200.0, x2: 300.0, y2: 300.0 }),
    ];
    for (name, b) in cases {
        println!("  {name}: IoU = {:.4}", base.iou(b));
    }
    println!();
}

// ============ ③ 网格 + anchor 解码 ============

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 单阶段检测器的一格「原始输出」：偏移 + 宽高对数 + objectness 打分
#[derive(Clone, Copy)]
struct RawCell {
    tx: f32, // 中心 x 偏移（解码要过 sigmoid，落在本格内）
    ty: f32,
    tw: f32, // 宽的对数缩放（解码要 exp，再乘 anchor）
    th: f32,
    obj: f32, // objectness 原始 logit（解码过 sigmoid 成置信度）
}

/// 网格解码：把每格的 raw 输出还原成「绝对坐标框 + 置信度」。
/// 只有 objectness 过阈值的格子才输出——这正是「密集预测 → 稀疏候选」的第一步。
fn decode_grid(grid: &[Vec<RawCell>], stride: f32, anchor: (f32, f32), conf_thresh: f32) -> Vec<(Xyxy, f32)> {
    let (aw, ah) = anchor;
    let mut out = Vec::new();
    for (row, line) in grid.iter().enumerate() {
        for (col, c) in line.iter().enumerate() {
            let score = sigmoid(c.obj);
            if score < conf_thresh {
                continue; // 低置信格子直接丢弃
            }
            // 中心 = (格内偏移 + 格坐标) * stride，落回模型输入图的像素坐标
            let cx = (sigmoid(c.tx) + col as f32) * stride;
            let cy = (sigmoid(c.ty) + row as f32) * stride;
            // 宽高 = anchor 尺寸 * exp(对数缩放)
            let w = aw * c.tw.exp();
            let h = ah * c.th.exp();
            out.push((Cxcywh { cx, cy, w, h }.to_xyxy(), score));
        }
    }
    out
}

fn demo_decode() {
    println!("== ③ 网格 + anchor 解码（4x4 网格, stride=32, 模型输入 128x128）==");
    const S: usize = 4;
    let stride = 32.0;
    let anchor = (50.0, 50.0); // 每格一个 50x50 的 anchor
    // 默认所有格 objectness 很低（obj=-5 → sigmoid≈0.007），只有两格「点亮」
    let mut grid = vec![vec![RawCell { tx: 0.0, ty: 0.0, tw: 0.0, th: 0.0, obj: -5.0 }; S]; S];
    // 目标 A：第 1 行第 1 列，居中、宽高用 anchor 原尺寸
    grid[1][1] = RawCell { tx: 0.0, ty: 0.0, tw: 0.0, th: 0.0, obj: 2.2 };
    // 目标 B：第 2 行第 3 列，宽略大、高略小
    grid[2][3] = RawCell { tx: 0.3, ty: -0.2, tw: 0.2, th: -0.1, obj: 1.8 };

    let boxes = decode_grid(&grid, stride, anchor, 0.5);
    println!("网格共 {} 格，过 conf>0.5 阈值后解出 {} 个候选框：", S * S, boxes.len());
    for (b, s) in &boxes {
        let c = b.to_cxcywh();
        println!(
            "  框 [x1={:.1},y1={:.1},x2={:.1},y2={:.1}]  中心({:.1},{:.1}) 宽高({:.1},{:.1})  conf={:.3}",
            b.x1, b.y1, b.x2, b.y2, c.cx, c.cy, c.w, c.h, s
        );
    }
    println!();
}

// ============ 确定性伪随机（造重叠框用，不依赖系统时钟）============

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 返回 [-1,1) 的伪随机数
    fn signed(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
    /// 返回 [0,1) 的伪随机数
    fn unit(&mut self) -> f32 {
        (self.signed() + 1.0) / 2.0
    }
}

// ============ ④ 贪心 NMS ============

/// 贪心 NMS：按分数从高到低，保留当前最高分框，抑制与它 IoU 超阈值的其余框。
/// 返回保留下来的框（已按分数降序）。
fn nms(mut boxes: Vec<(Xyxy, f32)>, iou_thresh: f32) -> Vec<(Xyxy, f32)> {
    // 按 score 降序排序（分数高的先「占坑」）
    boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let n = boxes.len();
    let mut suppressed = vec![false; n];
    let mut keep = Vec::new();
    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        keep.push(boxes[i]); // 当前最高分且未被抑制 → 保留
        for j in (i + 1)..n {
            if !suppressed[j] && boxes[i].0.iou(boxes[j].0) > iou_thresh {
                suppressed[j] = true; // 与保留框重叠过多 → 抑制
            }
        }
    }
    keep
}

/// 合成「一堆重叠框」：3 个真实目标，每个被检测器重复预测 4 次（中心/尺寸带抖动、
/// 分数各异），共 12 个框。这正是解码后同一目标被相邻网格重复命中的典型情形。
fn synth_overlapping_boxes() -> Vec<(Xyxy, f32)> {
    let mut rng = Lcg::new(45);
    // 3 个相距较远的目标（中心、宽高）
    let objects = [(60.0f32, 60.0f32, 56.0f32, 56.0f32), (180.0, 80.0, 60.0, 48.0), (110.0, 180.0, 50.0, 70.0)];
    let mut boxes = Vec::new();
    for (cx, cy, w, h) in objects {
        for k in 0..4 {
            // 中心抖动 ±6px、宽高抖动 ±5px：框互相高度重叠但不完全相同
            let b = Cxcywh {
                cx: cx + rng.signed() * 6.0,
                cy: cy + rng.signed() * 6.0,
                w: w + rng.signed() * 5.0,
                h: h + rng.signed() * 5.0,
            };
            let score = 0.55 + rng.unit() * 0.4 - k as f32 * 0.02; // 0.5~0.95 区间
            boxes.push((b.to_xyxy(), score));
        }
    }
    boxes
}

fn demo_nms() {
    println!("== ④ 贪心 NMS 非极大值抑制 ==");
    let boxes = synth_overlapping_boxes();
    // 抑制前：同一目标附近有一簇高分框，互相 IoU 很高
    let mut sample_iou = 0.0f32;
    if boxes.len() >= 2 {
        sample_iou = boxes[0].0.iou(boxes[1].0); // 同目标的两个重复框
    }
    println!("抑制前共 {} 个框（同目标框对示例 IoU={:.3}）", boxes.len(), sample_iou);
    let kept = nms(boxes, 0.5);
    println!("NMS(IoU>0.5) 后剩 {} 个框：", kept.len());
    for (b, s) in &kept {
        let c = b.to_cxcywh();
        println!("  保留框 中心({:.1},{:.1}) 宽高({:.1},{:.1})  score={:.3}", c.cx, c.cy, c.w, c.h, s);
    }
    println!();
}

// ============ ⑤ 匹配后的损失数值示例（不训练）============

fn demo_loss() {
    println!("== ⑤ 一对「匹配 anchor↔GT」的损失数值（仅算一次，不训练）==");
    // 某 anchor 被匹配为正样本：它的预测框 vs 对应 GT 框
    let pred = Cxcywh { cx: 62.0, cy: 58.0, w: 52.0, h: 60.0 };
    let gt = Cxcywh { cx: 60.0, cy: 60.0, w: 56.0, h: 56.0 };

    // (a) 分类损失：objectness 用二元交叉熵 BCE。正样本目标=1，
    //     设网络输出的 objectness logit = 1.5 → p=sigmoid(1.5)
    let obj_logit = 1.5f32;
    let p = sigmoid(obj_logit);
    let bce_pos = -(p.max(1e-9)).ln(); // 目标为 1 时 BCE = -log(p)
    println!("  分类(objectness) p=sigmoid({obj_logit})={p:.4}, BCE(正样本)= {bce_pos:.4}");

    // (b) 定位损失-L1：中心/宽高的绝对误差之和（此处用像素单位，实际会先归一化）
    let l1 = (pred.cx - gt.cx).abs() + (pred.cy - gt.cy).abs() + (pred.w - gt.w).abs() + (pred.h - gt.h).abs();
    println!("  定位 L1(cx,cy,w,h 绝对误差之和) = {l1:.4} px");

    // (c) 定位损失-IoU：1 - IoU，直接对齐最终指标
    let iou = pred.to_xyxy().iou(gt.to_xyxy());
    println!("  定位 IoU = {iou:.4} → IoU loss = 1-IoU = {:.4}", 1.0 - iou);

    // 总损失 = 加权和（权重是超参，这里演示 λ_box=5, λ_obj=1）
    let total = 1.0 * bce_pos + 5.0 * (1.0 - iou);
    println!("  加权总损失(λ_obj=1, λ_box=5, 用 IoU loss) = {total:.4}");
    println!();
}

fn main() {
    demo_box_repr();
    demo_iou();
    demo_decode();
    demo_nms();
    demo_loss();
}
