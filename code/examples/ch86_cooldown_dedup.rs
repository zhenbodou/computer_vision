//! 第 86 章 冷却（cooldown）、跨帧去重、告警聚合与节流
//! 运行：cargo run --example ch86_cooldown_dedup
//! 依赖：无（只用 std）——合成一段「持续命中」的帧流当输入，自包含、可复现。
//!
//! BBox / Detection / Track / Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md）。
//! 本章核心：同帧 NMS 去重（复用 BBox::iou）+ 基于 track_id 的 cooldown 跨帧去重 + 告警聚合。

#![allow(dead_code)]

use std::collections::HashMap;

// ========================= 用到的共享数据模型（第 83 章） =========================
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
    pub fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }
    /// 交并比 IoU ∈ [0,1]：同帧去重和跨帧关联的通用度量。
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

/// 检测框（同第 83 章）。检测器只给框，本身不含稳定 id。
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String,
    pub score: f32,
}

/// 跟踪后的目标（同第 83 章）：跟踪器（第 62 章）给每个目标分配稳定 track_id。
/// 稳定 id 正是「跨帧去重」的钥匙——同一个 id 的反复命中，就是同一个人。
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub label: String,
    pub history: Vec<(u64, BBox)>,
}

/// 事件层的输出：一条业务事件 / 告警（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String,
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 跟踪器把稳定 track_id 附到框上后的「带 id 检测」。
/// 共享模型里 id 记在 Track 上；逐帧做 cooldown 时，(track_id, Detection) 配对最顺手。
#[derive(Clone, Debug)]
struct TrackedDet {
    track_id: u64,
    det: Detection,
}

/// 一帧「带 id 的检测」结果（跟踪之后的形态）。
#[derive(Clone, Debug)]
struct TrackedFrame {
    ts_ms: u64,
    dets: Vec<TrackedDet>,
}

/// ① 同帧去重：贪心 NMS。按分数从高到低，保留一个就丢掉与它 IoU 过高的其余框。
/// 处理的是「一个物理目标被检测器吐成好几个重叠框」的情况——复用 BBox::iou。
fn nms_same_frame(mut dets: Vec<TrackedDet>, iou_thr: f32) -> Vec<TrackedDet> {
    dets.sort_by(|a, b| b.det.score.partial_cmp(&a.det.score).unwrap());
    let mut keep: Vec<TrackedDet> = Vec::new();
    'outer: for d in dets {
        for k in &keep {
            if d.det.bbox.iou(&k.det.bbox) > iou_thr {
                continue 'outer; // 与已保留的高分框重叠过多 → 判为重复，丢弃
            }
        }
        keep.push(d);
    }
    keep
}

/// ② cooldown 冷却闸门：同一个 key 一旦放行，冷却期 cd_ms 内不再放行。
/// key 用什么，决定了「按目标冷却」还是「按区域冷却」——见 main 里的两种玩法。
struct CooldownGate {
    cd_ms: u64,
    last_fire: HashMap<u64, u64>, // key -> 上次放行的 ts_ms
}

impl CooldownGate {
    fn new(cd_ms: u64) -> Self {
        CooldownGate { cd_ms, last_fire: HashMap::new() }
    }
    /// 冷却期内返回 false（压制），否则返回 true 并刷新计时。
    fn allow(&mut self, key: u64, ts_ms: u64) -> bool {
        match self.last_fire.get(&key) {
            Some(&t) if ts_ms.saturating_sub(t) < self.cd_ms => false,
            _ => {
                self.last_fire.insert(key, ts_ms);
                true
            }
        }
    }
}

/// 合成一段帧流：3 个目标先后（部分并存）在布防区里持续命中，且检测器时不时多吐一个重复框。
/// 时间戳自己造递增值（10 fps），不读系统时钟，保证可复现。
fn make_stream() -> Vec<TrackedFrame> {
    // (track_id, 锚框, 出现起始帧, 出现结束帧)。三块位置分开，避免不同目标被误当成同一个。
    let tracks = [
        (1u64, BBox::new(300.0, 150.0, 50.0, 150.0), 0usize, 179usize), // 门口那个人：站了 18 秒
        (2u64, BBox::new(120.0, 160.0, 45.0, 140.0), 60, 110),          // 左侧路过者：与 1 号并存一段
        (3u64, BBox::new(480.0, 150.0, 50.0, 150.0), 210, 280),         // 稍后来的第三个人
    ];
    let n = 300usize;
    let mut frames = Vec::new();
    for i in 0..n {
        let ts_ms = 1000 + i as u64 * 100; // 10 fps：每帧 +100ms
        let mut dets = Vec::new();
        for &(id, a, s, e) in &tracks {
            if i >= s && i <= e {
                let jx = ((i * 3) % 7) as f32 - 3.0; // -3..+3 确定性抖动
                // 主框
                dets.push(TrackedDet {
                    track_id: id,
                    det: Detection {
                        bbox: BBox::new(a.x + jx, a.y, a.w, a.h),
                        class_id: 0,
                        label: "person".to_string(),
                        score: 0.90,
                    },
                });
                // 每 5 帧检测器多吐一个高度重叠的重复框（同一个人两个框）——交给同帧 NMS 合并
                if i % 5 == 0 {
                    dets.push(TrackedDet {
                        track_id: id,
                        det: Detection {
                            bbox: BBox::new(a.x + jx + 8.0, a.y + 6.0, a.w, a.h),
                            class_id: 0,
                            label: "person".to_string(),
                            score: 0.82,
                        },
                    });
                }
            }
        }
        frames.push(TrackedFrame { ts_ms, dets });
    }
    frames
}

fn main() {
    let stream = make_stream();
    let total_frames = stream.len();
    let raw_boxes: usize = stream.iter().map(|f| f.dets.len()).sum();

    println!("== 原始检测流（10 fps，共 {} 帧 = {} 秒）==", total_frames, total_frames / 10);
    println!("原始检测框（含同帧重复框）：{}", raw_boxes);

    // ---------- ① 同帧去重：NMS（复用 BBox::iou） ----------
    let deduped: Vec<TrackedFrame> = stream
        .iter()
        .map(|f| TrackedFrame {
            ts_ms: f.ts_ms,
            dets: nms_same_frame(f.dets.clone(), 0.5),
        })
        .collect();
    let after_nms: usize = deduped.iter().map(|f| f.dets.len()).sum();
    println!("\n== ① 同帧去重：NMS（IoU>0.5）==");
    println!("去重前：{} 框", raw_boxes);
    println!("去重后：{} 框（合并掉 {} 个重复框）", after_nms, raw_boxes - after_nms);

    // ---------- ② 跨帧去重 + cooldown（按 track_id） ----------
    // 去重后，每个「目标在区域内」的帧都算一次命中。不做 cooldown 就会每帧一条告警。
    let hits_after_dedup = after_nms; // 每个去重后的框都是一次「某目标在布防区」的命中
    let cd_ms = 60_000u64; // 冷却 60 秒：一个目标进入后 60s 内只报一次

    let mut gate_by_track = CooldownGate::new(cd_ms);
    let mut alarms: Vec<Event> = Vec::new();
    for f in &deduped {
        for td in &f.dets {
            // 按 track_id 冷却：同一个人反复命中，只在冷却窗口外才再报
            if gate_by_track.allow(td.track_id, f.ts_ms) {
                alarms.push(Event {
                    kind: "intrusion".to_string(),
                    ts_ms: f.ts_ms,
                    track_id: Some(td.track_id),
                    message: "有人进入布防区".to_string(),
                });
            }
        }
    }
    println!("\n== ② 跨帧去重 + cooldown（按 track_id，冷却 {}s）==", cd_ms / 1000);
    println!("若不做 cooldown：每个「目标在区域内」的帧都报 → {} 条告警", hits_after_dedup);
    println!("实际发出的告警（每个目标进入后 {}s 内只报一次）：", cd_ms / 1000);
    for e in &alarms {
        println!(
            "  [告警] t={:>4.1}s  track#{}  {}  {}",
            e.ts_ms as f64 / 1000.0,
            e.track_id.unwrap(),
            e.kind,
            e.message
        );
    }
    println!("命中帧（去重后仍是「某目标在区域内」）：{}  →  实际告警：{}", hits_after_dedup, alarms.len());

    // ---------- cooldown 按目标 vs 按区域 的对比 ----------
    // 换成「按区域」冷却：整个布防区共用一个 key，冷却期内任何目标进入都被压掉。
    let mut gate_by_region = CooldownGate::new(cd_ms);
    let region_key = 999u64; // 整个区域一个固定 key
    let mut region_alarms = 0usize;
    for f in &deduped {
        // 一帧里只要有目标在区域内，就尝试触发一次「区域级」告警
        if !f.dets.is_empty() && gate_by_region.allow(region_key, f.ts_ms) {
            region_alarms += 1;
        }
    }
    println!("\n== cooldown 按目标 vs 按区域 ==");
    println!("按 track_id 冷却：{} 条告警（每个不同的人各报一次，能抓住并存/后到的人）", alarms.len());
    println!("按 区域   冷却：{} 条告警（冷却期内 track#2、track#3 进入被一并压掉，会漏报不同目标）", region_alarms);

    // ---------- ③ 告警聚合 / 节流：把同类事件聚成一条 ----------
    // 真实项目常再加一层：一个窗口内的同类告警聚合成一条摘要（如「5 分钟内 N 次闯入」）。
    // 这里窗口就取整段 30 秒，演示 3 条 intrusion → 1 条摘要。
    let intrusions: Vec<&Event> = alarms.iter().filter(|e| e.kind == "intrusion").collect();
    println!("\n== ③ 告警聚合 / 节流（把同类事件聚成一条）==");
    if let (Some(first), Some(last)) = (intrusions.first(), intrusions.last()) {
        println!("{}s 窗口内 intrusion 类事件：{} 次 → 聚合 1 条摘要", total_frames / 10, intrusions.len());
        println!(
            "  摘要：{}秒内共发生 {} 次闯入（首次 t={:.1}s，末次 t={:.1}s）",
            total_frames / 10,
            intrusions.len(),
            first.ts_ms as f64 / 1000.0,
            last.ts_ms as f64 / 1000.0,
        );
    }

    // ---------- 总览：一条检测流被层层收敛 ----------
    println!("\n== 总览：告警是怎样被一层层收敛的 ==");
    println!("原始框 {} → 同帧NMS {} → cooldown后告警 {} → 聚合摘要 1 条", raw_boxes, after_nms, alarms.len());
}
