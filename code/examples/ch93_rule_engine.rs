//! 第 93 章 可配置的规则引擎：把第 83–92 章的规则串成一条可插拔、可配置的流水线
//! 运行：cargo run --example ch93_rule_engine
//! 依赖：serde + serde_json（纯 Rust，已在主 code/Cargo.toml，勿重复添加）。
//!
//! BBox / Detection / Frame / Event / Stage 就是第 83 章给出的共享数据模型。
//! 本章把前面各章的规则（阈值/类别/ROI/cooldown/计数…）统一成 `trait Stage`，
//! 用一个 `Pipeline` 串起来；再用 serde_json 让整条流水线**由 JSON 配置驱动**：
//! 同一批合成帧，改配置（调阈值 / 换 ROI）→ 产出事件数随之变化。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ========================= 第 83 章共享数据模型 =========================
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
    /// 底边中点（脚点）——ROI 判定用它而非中心（见第 84 章）。
    pub fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub ts_ms: u64,
    pub detections: Vec<Detection>,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String,
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 射线法判断点是否在多边形内——原样取自第 84 章。
fn point_in_polygon(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if (yi > py) != (yj > py) {
            let x_at = xi + (py - yi) / (yj - yi) * (xj - xi);
            if px < x_at {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

// ============================================================================
// 统一的规则环 Stage
// ============================================================================
/// 一个规则环：处理一帧，可以**就地筛掉**帧里的一些检测（过滤类规则），
/// **并/或**往 `out` 里**吐出事件**（告警类规则）。有状态的规则（cooldown）把状态放进 self。
///
/// 这是把第 83 章那个 `process(&Frame) -> Vec<Event>` 心智模型稍作扩展——
/// 让「过滤环」和「事件环」能共用同一条链：过滤环改 `frame` 不吐事件，事件环读 `frame` 吐事件。
pub trait Stage {
    fn name(&self) -> &str;
    fn process(&mut self, frame: &mut Frame, out: &mut Vec<Event>);
}

/// 过滤：置信度阈值（第 83 章）
struct ScoreThreshold {
    min_score: f32,
}
impl Stage for ScoreThreshold {
    fn name(&self) -> &str {
        "score_threshold"
    }
    fn process(&mut self, f: &mut Frame, _out: &mut Vec<Event>) {
        f.detections.retain(|d| d.score >= self.min_score);
    }
}

/// 过滤：类别白名单（第 83 章）
struct ClassFilter {
    allow: Vec<String>,
}
impl Stage for ClassFilter {
    fn name(&self) -> &str {
        "class_filter"
    }
    fn process(&mut self, f: &mut Frame, _out: &mut Vec<Event>) {
        f.detections.retain(|d| self.allow.iter().any(|c| c == &d.label));
    }
}

/// 过滤：ROI 多边形，只保留脚点落在区域内的检测（第 84 章）
struct RoiFilter {
    polygon: Vec<(f32, f32)>,
}
impl Stage for RoiFilter {
    fn name(&self) -> &str {
        "roi_filter"
    }
    fn process(&mut self, f: &mut Frame, _out: &mut Vec<Event>) {
        f.detections.retain(|d| {
            let (fx, fy) = d.bbox.bottom_center();
            point_in_polygon(fx, fy, &self.polygon)
        });
    }
}

/// 事件：区域内有目标就报「闯入」，按区域级 cooldown 去重（第 84 + 86 章）
struct PresenceAlarm {
    cooldown_ms: u64,
    message: String,
    last_fire: Option<u64>, // 上次报警时间戳（状态）
}
impl Stage for PresenceAlarm {
    fn name(&self) -> &str {
        "presence_alarm"
    }
    fn process(&mut self, f: &mut Frame, out: &mut Vec<Event>) {
        if f.detections.is_empty() {
            return;
        }
        let cool = matches!(self.last_fire, Some(t) if f.ts_ms.saturating_sub(t) < self.cooldown_ms);
        if !cool {
            self.last_fire = Some(f.ts_ms);
            out.push(Event {
                kind: "intrusion".to_string(),
                ts_ms: f.ts_ms,
                track_id: None,
                message: self.message.clone(),
            });
        }
    }
}

/// 事件：区域内目标数 ≥ 阈值就报「拥挤」，同样按 cooldown 去重（第 88 章）
struct CrowdAlarm {
    min_count: usize,
    cooldown_ms: u64,
    last_fire: Option<u64>,
}
impl Stage for CrowdAlarm {
    fn name(&self) -> &str {
        "crowd_alarm"
    }
    fn process(&mut self, f: &mut Frame, out: &mut Vec<Event>) {
        if f.detections.len() < self.min_count {
            return;
        }
        let cool = matches!(self.last_fire, Some(t) if f.ts_ms.saturating_sub(t) < self.cooldown_ms);
        if !cool {
            self.last_fire = Some(f.ts_ms);
            out.push(Event {
                kind: "crowd".to_string(),
                ts_ms: f.ts_ms,
                track_id: None,
                message: format!("区内目标数达到 {} 人", f.detections.len()),
            });
        }
    }
}

/// 流水线：把若干 Stage 串起来。每一帧克隆一份，依次流过各环。
struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
}
impl Pipeline {
    fn run(&mut self, frames: &[Frame]) -> Vec<Event> {
        let mut all = Vec::new();
        for fr in frames {
            let mut f = fr.clone(); // 过滤环会改帧，不能动原始输入
            let mut out = Vec::new();
            for s in &mut self.stages {
                s.process(&mut f, &mut out);
            }
            all.extend(out);
        }
        all
    }
    fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

// ============================================================================
// 配置层：serde 可序列化/反序列化的「规则规格」，与运行期的有状态 Stage 分开
// ============================================================================
/// 一条规则的配置规格。`#[serde(tag = "type")]` → JSON 里用 "type" 字段区分是哪种规则。
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StageSpec {
    ScoreThreshold { min_score: f32 },
    ClassFilter { allow: Vec<String> },
    RoiFilter { polygon: Vec<[f32; 2]> },
    PresenceAlarm { cooldown_ms: u64, message: String },
    CrowdAlarm { min_count: usize, cooldown_ms: u64 },
}

/// 整条流水线的配置：就是一串规则规格。
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PipelineConfig {
    stages: Vec<StageSpec>,
}

/// 把「配置规格」实例化成「运行期的有状态 Stage」。
fn build_stage(spec: &StageSpec) -> Box<dyn Stage> {
    match spec {
        StageSpec::ScoreThreshold { min_score } => Box::new(ScoreThreshold { min_score: *min_score }),
        StageSpec::ClassFilter { allow } => Box::new(ClassFilter { allow: allow.clone() }),
        StageSpec::RoiFilter { polygon } => Box::new(RoiFilter {
            polygon: polygon.iter().map(|p| (p[0], p[1])).collect(),
        }),
        StageSpec::PresenceAlarm { cooldown_ms, message } => Box::new(PresenceAlarm {
            cooldown_ms: *cooldown_ms,
            message: message.clone(),
            last_fire: None,
        }),
        StageSpec::CrowdAlarm { min_count, cooldown_ms } => Box::new(CrowdAlarm {
            min_count: *min_count,
            cooldown_ms: *cooldown_ms,
            last_fire: None,
        }),
    }
}

fn build_pipeline(cfg: &PipelineConfig) -> Pipeline {
    Pipeline {
        stages: cfg.stages.iter().map(build_stage).collect(),
    }
}

// ============================================================================
// 合成输入 + 运行报告
// ============================================================================
/// 造一个 person 检测：给定脚点 (fx, fy)，反推左上角（框 w=40, h=100）。
fn person(fx: f32, fy: f32, score: f32) -> Detection {
    Detection {
        bbox: BBox::new(fx - 20.0, fy - 100.0, 40.0, 100.0),
        class_id: 0,
        label: "person".to_string(),
        score,
    }
}
fn car(fx: f32, fy: f32, score: f32) -> Detection {
    Detection {
        bbox: BBox::new(fx - 60.0, fy - 60.0, 120.0, 60.0),
        class_id: 1,
        label: "car".to_string(),
        score,
    }
}

/// 合成一段 2fps 的帧流（ts=1000+i*500）。安排好几种目标，好让「改配置→变结果」看得清：
///  - P1 脚点(300,300) 0.90：i=4..15 在右侧 ROI 内长期停留
///  - P2 脚点(380,320) 0.85：i=8..12 也在右侧 ROI 内（与 P1、P3 并存）
///  - P3 脚点(250,300) 0.70：i=8..10 在右侧 ROI 内（把区内人数顶到 3）
///  - P4 脚点(100,300) 0.95：i=2..15 在**左侧**（右侧 ROI 外；换成左侧 ROI 才算数）
///  - 低分 person 脚点(310,305) 0.40：i=6,7（会被 0.5 阈值滤掉；换 0.8 阈值更是必死）
///  - car 脚点(300,300) 0.92：全程（会被类别白名单滤掉）
fn make_frames() -> Vec<Frame> {
    let mut frames = Vec::new();
    for i in 0..16u64 {
        let ts_ms = 1000 + i * 500;
        let mut dets = Vec::new();
        dets.push(car(300.0, 300.0, 0.92));
        if (2..=15).contains(&i) {
            dets.push(person(100.0, 300.0, 0.95)); // P4 左侧
        }
        if (4..=15).contains(&i) {
            dets.push(person(300.0, 300.0, 0.90)); // P1
        }
        if (8..=12).contains(&i) {
            dets.push(person(380.0, 320.0, 0.85)); // P2
        }
        if (8..=10).contains(&i) {
            dets.push(person(250.0, 300.0, 0.70)); // P3
        }
        if (6..=7).contains(&i) {
            dets.push(person(310.0, 305.0, 0.40)); // 低分
        }
        frames.push(Frame { ts_ms, detections: dets });
    }
    frames
}

fn count_by_kind(events: &[Event]) -> BTreeMap<String, u32> {
    let mut m = BTreeMap::new();
    for e in events {
        *m.entry(e.kind.clone()).or_insert(0) += 1;
    }
    m
}

fn run_and_report(title: &str, cfg: &PipelineConfig, frames: &[Frame]) -> Vec<Event> {
    let mut p = build_pipeline(cfg);
    println!("── {} ──", title);
    println!("   规则链：{:?}", p.stage_names());
    let events = p.run(frames);
    println!("   产出事件 {} 条，按 kind：{:?}", events.len(), count_by_kind(&events));
    for e in &events {
        println!("     [{:<9}] t={:.1}s  {}", e.kind, e.ts_ms as f64 / 1000.0, e.message);
    }
    events
}

fn main() {
    let frames = make_frames();
    let total_dets: usize = frames.iter().map(|f| f.detections.len()).sum();
    println!("合成 {} 帧（2fps），原始检测框共 {} 个（含 car、低分、区外目标等脏货）\n", frames.len(), total_dets);

    // ---------- ① 手写结构体配置，跑通并给真实数字 ----------
    let roi_right = vec![[200.0, 150.0], [440.0, 150.0], [440.0, 330.0], [200.0, 330.0]];
    let config_a = PipelineConfig {
        stages: vec![
            StageSpec::ScoreThreshold { min_score: 0.5 },
            StageSpec::ClassFilter { allow: vec!["person".to_string()] },
            StageSpec::RoiFilter { polygon: roi_right.clone() },
            StageSpec::PresenceAlarm { cooldown_ms: 3000, message: "有人进入布防区".to_string() },
            StageSpec::CrowdAlarm { min_count: 3, cooldown_ms: 10000 },
        ],
    };
    println!("=== ① 手写结构体驱动流水线 ===");
    let events_a = run_and_report("baseline（阈值0.5 / person / 右侧ROI）", &config_a, &frames);

    // ---------- 手写结构体 == 一段 JSON：序列化出来给你看 ----------
    let json_a = serde_json::to_string_pretty(&config_a).unwrap();
    println!("\n=== 这套配置序列化成 JSON 就是（serde_json）===\n{}\n", json_a);

    // 把这段 JSON 再反序列化回来跑，验证「JSON 驱动的是同一条流水线」
    let cfg_from_json: PipelineConfig = serde_json::from_str(&json_a).unwrap();
    let events_roundtrip = build_pipeline(&cfg_from_json).run(&frames);
    println!(
        "JSON 往返一致性检查：结构体产出 {} 条，JSON 产出 {} 条 → {}\n",
        events_a.len(),
        events_roundtrip.len(),
        if events_a.len() == events_roundtrip.len() { "一致 ✓" } else { "不一致 ✗" }
    );

    // ---------- ② 改 JSON 配置：把置信度阈值 0.5 → 0.8 ----------
    println!("=== ② 只改 JSON 里的阈值 0.5 → 0.8（不改一行代码）===");
    let json_b = r#"{
      "stages": [
        { "type": "score_threshold", "min_score": 0.8 },
        { "type": "class_filter", "allow": ["person"] },
        { "type": "roi_filter", "polygon": [[200,150],[440,150],[440,330],[200,330]] },
        { "type": "presence_alarm", "cooldown_ms": 3000, "message": "有人进入布防区" },
        { "type": "crowd_alarm", "min_count": 3, "cooldown_ms": 10000 }
      ]
    }"#;
    let cfg_b: PipelineConfig = serde_json::from_str(json_b).unwrap();
    let events_b = run_and_report("阈值抬到 0.8（P3=0.70 被滤掉，区内不足 3 人）", &cfg_b, &frames);

    // ---------- ③ 改 JSON 配置：把 ROI 从右侧换到左侧 ----------
    println!("\n=== ③ 只改 JSON 里的 ROI 多边形：右侧 → 左侧（不改一行代码）===");
    let json_c = r#"{
      "stages": [
        { "type": "score_threshold", "min_score": 0.5 },
        { "type": "class_filter", "allow": ["person"] },
        { "type": "roi_filter", "polygon": [[50,150],[200,150],[200,330],[50,330]] },
        { "type": "presence_alarm", "cooldown_ms": 3000, "message": "有人进入左侧布防区" },
        { "type": "crowd_alarm", "min_count": 3, "cooldown_ms": 10000 }
      ]
    }"#;
    let cfg_c: PipelineConfig = serde_json::from_str(json_c).unwrap();
    let events_c = run_and_report("ROI 换到左侧（改由 P4 触发，右侧那几个人反而没了）", &cfg_c, &frames);

    // ---------- 总览 ----------
    println!("\n=== 总览：同一批帧，配置一改，产出就变 ===");
    println!("  ① baseline（阈值0.5, 右ROI）  → {} 条：{:?}", events_a.len(), count_by_kind(&events_a));
    println!("  ② 阈值抬到 0.8               → {} 条：{:?}", events_b.len(), count_by_kind(&events_b));
    println!("  ③ ROI 换到左侧               → {} 条：{:?}", events_c.len(), count_by_kind(&events_c));
    println!("\n可插拔规则链 + 配置化：改行为不必改代码、不必重新编译，这正是工业落地的核心。");
}
