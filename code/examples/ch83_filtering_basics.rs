//! 第 83 章 检测结果清洗：置信度 / 类别 / 尺寸 / 长宽比过滤
//! 运行：cargo run --example ch83_filtering_basics
//! 依赖：无（只用 std）——事件规则层示例统一用合成的 Vec<Frame> 当输入，自包含、可复现。
//!
//! 本文件同时原样给出第十八部分的共享数据模型（见 _authoring/EVENTS_MODEL.md）：
//! BBox / Detection / Frame / Track / Event / Stage。ch84–ch93 都沿用这套类型。

// 共享模型里的 center/bottom_center/iou、Track/Event/Stage 本章还用不到，
// 但为了让读者一次看清完整模型，仍原样保留；用 allow 关掉"未使用"告警。
#![allow(dead_code)]

// ========================= 第十八部分 共享数据模型 =========================
// 坐标约定：原点在左上角，x 向右、y 向下（与 image 库一致）。
// 时间戳统一用 u64 毫秒（ts_ms）；合成数据自己造递增值，不读系统时钟。

/// 轴对齐包围盒：左上角 (x, y) + 宽高 (w, h)，单位像素。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        BBox { x, y, w, h }
    }

    /// 中心点
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    /// 底边中点（脚点）——过线 / 入区域判定常用它而非中心（见第 84 章）。
    pub fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }

    /// 面积（像素平方），负宽高按 0 处理。
    pub fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }

    /// 交并比 IoU ∈ [0,1]，用于同帧去重 / 跨帧关联（见第 86 章）。
    pub fn iou(&self, o: &BBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let iw = (x2 - x1).max(0.0);
        let ih = (y2 - y1).max(0.0);
        let inter = iw * ih;
        let uni = self.area() + o.area() - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

/// 一个检测框：来自检测器（第九部分）。事件层只消费它，不关心它怎么来的。
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String, // 类别名，如 "person" / "car"
    pub score: f32,    // 置信度 [0,1]
}

/// 一帧的检测结果（带时间戳）。事件层的输入通常是 Vec<Frame> 或帧流。
#[derive(Clone, Debug)]
pub struct Frame {
    pub ts_ms: u64,
    pub detections: Vec<Detection>,
}

/// 跟踪后的目标：跟踪器（第十三部分）给每个目标分配稳定 id。
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub label: String,
    pub history: Vec<(u64, BBox)>, // (ts_ms, bbox) 按时间递增
}

/// 事件层的输出：一条业务事件 / 告警。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String, // "roi" | "tripwire" | "intrusion" | "dwell" | ...
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}

/// 事件层的统一心智模型：流水线上一个可插拔的"环"。
/// 有状态的规则（时间窗、cooldown、绊线方向）把状态放进 &mut self。
pub trait Stage {
    fn process(&mut self, frame: &Frame) -> Vec<Event>;
}
// ======================= 共享数据模型结束 =======================

/// 一条检测被剔除的原因——用来统计"到底为什么被清掉"。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reject {
    LowScore,  // 置信度太低
    BadClass,  // 类别不在白名单
    TooSmall,  // 框面积过小（多半是噪点 / 碎块）
    TooLarge,  // 框面积过大（多半是误检 / 整屏框）
    BadAspect, // 长宽比不合理（形状不像目标）
}

/// 清洗过滤器的配置。真实项目里这些阈值通常从配置文件读，且可按类别分别设定。
struct DetectionCleaner {
    score_thresh: f32,  // 置信度阈值：低于它直接丢
    allow: Vec<String>, // 类别白名单：只保留这些 label
    min_area: f32,      // 最小框面积（像素^2）
    max_area: f32,      // 最大框面积
    min_aspect: f32,    // 最小长宽比 w/h
    max_aspect: f32,    // 最大长宽比 w/h
}

impl DetectionCleaner {
    /// 判断单个检测：Ok(()) 表示保留，Err(原因) 表示剔除。
    /// 检查有先后：先做便宜的（分数、类别对比），再做几何的（尺寸、形状）。
    fn check(&self, d: &Detection) -> Result<(), Reject> {
        if d.score < self.score_thresh {
            return Err(Reject::LowScore);
        }
        if !self.allow.iter().any(|c| c == &d.label) {
            return Err(Reject::BadClass);
        }
        let area = d.bbox.area();
        if area < self.min_area {
            return Err(Reject::TooSmall);
        }
        if area > self.max_area {
            return Err(Reject::TooLarge);
        }
        // 长宽比 = 宽 / 高；高为 0 视为无穷大（必然出局）。
        let aspect = if d.bbox.h > 0.0 {
            d.bbox.w / d.bbox.h
        } else {
            f32::INFINITY
        };
        if aspect < self.min_aspect || aspect > self.max_aspect {
            return Err(Reject::BadAspect);
        }
        Ok(())
    }

    /// 清洗一帧：返回只留下"干净"检测的新 Frame（下游拿到的就是它）。
    fn clean_frame(&self, f: &Frame) -> Frame {
        let kept = f
            .detections
            .iter()
            .filter(|d| self.check(d).is_ok())
            .cloned()
            .collect();
        Frame {
            ts_ms: f.ts_ms,
            detections: kept,
        }
    }
}

/// 剔除原因的计数器。
#[derive(Default)]
struct Stats {
    total: u32,
    kept: u32,
    low_score: u32,
    bad_class: u32,
    too_small: u32,
    too_large: u32,
    bad_aspect: u32,
}

impl Stats {
    fn record(&mut self, r: Result<(), Reject>) {
        self.total += 1;
        match r {
            Ok(()) => self.kept += 1,
            Err(Reject::LowScore) => self.low_score += 1,
            Err(Reject::BadClass) => self.bad_class += 1,
            Err(Reject::TooSmall) => self.too_small += 1,
            Err(Reject::TooLarge) => self.too_large += 1,
            Err(Reject::BadAspect) => self.bad_aspect += 1,
        }
    }
    fn rejected(&self) -> u32 {
        self.total - self.kept
    }
}

/// 造一个检测的小助手：按 label 顺手编个 class_id。
fn det(x: f32, y: f32, w: f32, h: f32, label: &str, score: f32) -> Detection {
    let class_id = match label {
        "person" => 0,
        "car" => 1,
        "bird" => 2,
        "cat" => 3,
        _ => 99,
    };
    Detection {
        bbox: BBox::new(x, y, w, h),
        class_id,
        label: label.to_string(),
        score,
    }
}

/// 合成一批"脏"检测：混入低分、无关类别、碎框、巨框、畸形框。时间戳自己造递增值。
fn make_frames() -> Vec<Frame> {
    vec![
        Frame {
            ts_ms: 1000,
            detections: vec![
                det(100.0, 80.0, 40.0, 110.0, "person", 0.92), // 站立行人：留
                det(200.0, 90.0, 38.0, 100.0, "person", 0.35), // 分数太低：剔
                det(300.0, 150.0, 120.0, 70.0, "car", 0.88),   // 汽车：留
                det(50.0, 40.0, 25.0, 20.0, "bird", 0.80),     // 无关类别：剔
            ],
        },
        Frame {
            ts_ms: 1040,
            detections: vec![
                det(105.0, 82.0, 42.0, 112.0, "person", 0.95), // 留
                det(400.0, 300.0, 6.0, 8.0, "person", 0.90),   // 小碎框（噪点）：剔
                det(0.0, 0.0, 600.0, 360.0, "person", 0.91),   // 几乎整屏的巨框：剔
                det(280.0, 140.0, 130.0, 66.0, "car", 0.75),   // 留
            ],
        },
        Frame {
            ts_ms: 1080,
            detections: vec![
                det(110.0, 85.0, 41.0, 109.0, "person", 0.89), // 留
                det(150.0, 100.0, 120.0, 30.0, "person", 0.93), // 太扁（横条）：剔
                det(150.0, 100.0, 10.0, 120.0, "person", 0.60), // 太细（竖条）：剔
                det(500.0, 200.0, 30.0, 28.0, "cat", 0.70),    // 无关类别：剔
            ],
        },
    ]
}

fn main() {
    let frames = make_frames();

    let cleaner = DetectionCleaner {
        score_thresh: 0.5,
        allow: vec!["person".to_string(), "car".to_string()],
        min_area: 300.0,
        max_area: 100_000.0,
        min_aspect: 0.15,
        max_aspect: 3.5,
    };

    let mut stats = Stats::default();

    println!("== 逐帧清洗 ==");
    for f in &frames {
        let cleaned = cleaner.clean_frame(f);
        for d in &f.detections {
            stats.record(cleaner.check(d));
        }
        println!(
            "帧 ts={:>5}ms：{} 个检测 -> 保留 {} 个",
            f.ts_ms,
            f.detections.len(),
            cleaned.detections.len()
        );
    }

    println!("\n== 汇总 ==");
    println!("清洗前检测总数：{}", stats.total);
    println!("清洗后检测总数：{}", stats.kept);
    println!("被剔除总数：{}", stats.rejected());
    println!("  低置信度  LowScore : {}", stats.low_score);
    println!("  类别不符  BadClass : {}", stats.bad_class);
    println!("  框太小    TooSmall : {}", stats.too_small);
    println!("  框太大    TooLarge : {}", stats.too_large);
    println!("  长宽比坏  BadAspect: {}", stats.bad_aspect);
}
