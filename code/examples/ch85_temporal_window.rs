//! 第 85 章 时间窗口过滤与状态机（N/M 投票、去抖）
//! 运行：cargo run --example ch85_temporal_window
//! 依赖：无（只用 std）——事件规则层示例统一用合成的 Vec<Frame> 当输入，自包含、可复现。
//!
//! BBox / Detection / Frame / Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md），
//! 这里为了能独立运行再抄用到的那几个。核心新内容：N/M 滑动窗口投票 + 四态状态机去抖。

#![allow(dead_code)]

use std::collections::VecDeque;

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
    /// 交并比 IoU ∈ [0,1]：没有 track_id 时用它把「这一帧的框」和「上一帧/锚框」关联起来。
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

/// 检测框（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String,
    pub score: f32,
}

/// 一帧的检测结果（带时间戳）。
#[derive(Clone, Debug)]
pub struct Frame {
    pub ts_ms: u64,
    pub detections: Vec<Detection>,
}

/// 事件层的输出：一条业务事件 / 告警。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String,
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 目标在门口的「锚框」：真实目标大致站在这里。没有跟踪器给 id 时，
/// 我们靠「这一帧有没有框和锚框 IoU 够大」来判断本帧是不是命中（跨帧关联的最朴素做法）。
fn anchor() -> BBox {
    BBox::new(280.0, 120.0, 60.0, 180.0)
}

/// 判断一帧是否「命中」锚框：有 person 检测且与锚框 IoU ≥ 阈值即算命中。
fn hits_anchor(frame: &Frame, anchor: &BBox, iou_thr: f32) -> bool {
    frame
        .detections
        .iter()
        .any(|d| d.label == "person" && d.bbox.iou(anchor) >= iou_thr)
}

/// 四态状态机：空闲 → 候选 → 确认 → 消失（Lost 是「确认后暂时丢」的过渡态）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Idle,      // 空闲：什么都没有
    Candidate, // 候选：见到零星命中，还没凑够票数
    Confirmed, // 确认：最近 M 帧里够 N 次命中，认定真目标
    Lost,      // 疑似消失：确认后开始漏检，但还在容忍的 gap 之内
}

impl State {
    fn name(&self) -> &'static str {
        match self {
            State::Idle => "Idle 空闲",
            State::Candidate => "Candidate 候选",
            State::Confirmed => "Confirmed 确认",
            State::Lost => "Lost 疑失",
        }
    }
}

/// N/M 滑动窗口投票 + 状态机去抖器。
struct Debouncer {
    window_m: usize,     // 滑动窗口长度 M：只看最近 M 帧
    need_n: usize,       // 确认阈值 N：窗口里 ≥ N 次命中才确认
    gap_tolerance: u32,  // 确认后能容忍的连续漏检数（短暂遮挡不算消失）
    cand_patience: u32,  // 候选态最多容忍几帧连续漏检，超了就放弃回空闲
    window: VecDeque<bool>,
    consec_miss: u32,
    state: State,
}

impl Debouncer {
    fn new(window_m: usize, need_n: usize, gap_tolerance: u32, cand_patience: u32) -> Self {
        Debouncer {
            window_m,
            need_n,
            gap_tolerance,
            cand_patience,
            window: VecDeque::new(),
            consec_miss: 0,
            state: State::Idle,
        }
    }

    fn votes(&self) -> usize {
        self.window.iter().filter(|&&h| h).count()
    }

    /// 回到空闲时清空窗口与计数——「放弃了就从头再来」，别让旧票污染下一个目标。
    fn reset(&mut self) {
        self.window.clear();
        self.consec_miss = 0;
    }

    /// 喂进本帧的命中与时间戳，返回可能产生的事件（确认 / 消失）。
    fn update(&mut self, hit: bool, ts_ms: u64) -> Option<Event> {
        // 1) 维护最近 M 帧的滑动窗口
        self.window.push_back(hit);
        if self.window.len() > self.window_m {
            self.window.pop_front();
        }
        let votes = self.votes();
        // 2) 维护连续漏检计数
        if hit {
            self.consec_miss = 0;
        } else {
            self.consec_miss += 1;
        }
        // 3) 状态转移
        let mut event = None;
        match self.state {
            State::Idle => {
                if hit {
                    self.state = State::Candidate; // 见到第一次命中，进入观察
                }
            }
            State::Candidate => {
                if votes >= self.need_n {
                    self.state = State::Confirmed; // 票数够 → 确认，吐出事件
                    event = Some(Event {
                        kind: "confirm".to_string(),
                        ts_ms,
                        track_id: None, // 本章没有跟踪器，暂无稳定 id（第 86 章用 track_id）
                        message: format!("目标确认（最近{}帧命中{}次）", self.window.len(), votes),
                    });
                } else if self.consec_miss >= self.cand_patience {
                    self.state = State::Idle; // 只是零星误检，没凑够票 → 放弃
                    self.reset();
                }
            }
            State::Confirmed => {
                if self.consec_miss >= 1 {
                    self.state = State::Lost; // 刚开始漏检，先进疑失（不急着报消失）
                }
            }
            State::Lost => {
                if hit {
                    self.state = State::Confirmed; // 又出现了：短暂遮挡而已，不重复报确认
                } else if self.consec_miss > self.gap_tolerance {
                    self.state = State::Idle; // 连续漏检超过容忍 → 真的走了
                    event = Some(Event {
                        kind: "gone".to_string(),
                        ts_ms,
                        track_id: None,
                        message: format!("目标消失（连续漏检{}帧）", self.consec_miss),
                    });
                    self.reset();
                }
            }
        }
        event
    }
}

/// 用命中序列合成一段带「抖动」的检测流：
/// H=真的有 person 站在门口（含个别单帧误检），·=这一帧没有命中（含个别漏检/短暂遮挡）。
/// 剧情：单帧误检#1 → 真目标出现(中途几次漏检) → 真目标离开 → 单帧误检#2。
fn make_frames() -> (Vec<Frame>, Vec<bool>) {
    // 索引:0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    // 命中:·  H  ·  ·  H  H  H  H  ·  H  H  ·  ·  H  H  H  ·  ·  ·  ·  ·  H  ·  ·
    let pattern = [
        false, true, false, false, true, true, true, true, false, true, true, false, false,
        true, true, true, false, false, false, false, false, true, false, false,
    ];
    let a = anchor();
    let mut frames = Vec::new();
    for (i, &hit) in pattern.iter().enumerate() {
        let ts_ms = 1000 + i as u64 * 40; // 25 fps：每帧 +40ms，纯合成不读系统时钟
        let mut dets = Vec::new();
        if hit {
            // 命中帧：造一个和锚框高度重叠的 person 框（带一点确定性抖动，模拟框的漂移）
            let jx = ((i * 7) % 11) as f32 - 5.0; // -5..+5 的确定性偏移
            let jy = ((i * 5) % 9) as f32 - 4.0;
            dets.push(Detection {
                bbox: BBox::new(a.x + jx, a.y + jy, a.w, a.h),
                class_id: 0,
                label: "person".to_string(),
                score: 0.85,
            });
        }
        frames.push(Frame { ts_ms, detections: dets });
    }
    (frames, pattern.to_vec())
}

// ---------- 几种「不去抖 / 弱去抖」的基线，用来和状态机对比 ----------

/// 基线一：每帧只要命中就报一次（最原始，等于没去抖）。
fn naive_per_frame(hits: &[bool]) -> usize {
    hits.iter().filter(|&&h| h).count()
}

/// 基线二：只在「上升沿」（从没命中变命中）报一次——把每次「重新出现」都当成新目标。
fn naive_rising_edge(hits: &[bool]) -> usize {
    let mut count = 0;
    let mut prev = false;
    for &h in hits {
        if h && !prev {
            count += 1;
        }
        prev = h;
    }
    count
}

/// 基线三：要求「连续 N 帧」命中才确认，且一旦漏检立刻掉线、需重新连续 N 帧。
/// 比每帧报强，但一次短暂遮挡就会把同一个目标「重新确认」一遍。
fn strict_consecutive(hits: &[bool], n: usize) -> usize {
    let mut count = 0;
    let mut streak = 0usize;
    let mut confirmed = false;
    for &h in hits {
        if h {
            streak += 1;
            if streak >= n && !confirmed {
                count += 1; // 刚凑够连续 N 帧 → 确认
                confirmed = true;
            }
        } else {
            streak = 0;
            confirmed = false; // 一漏检就掉线
        }
    }
    count
}

fn main() {
    let (frames, hits) = make_frames();
    let a = anchor();

    // 用 N/M=3/5 投票 + 状态机跑一遍，逐帧打印过程
    let mut deb = Debouncer::new(5, 3, 2, 2);
    let mut events: Vec<Event> = Vec::new();

    println!("== 逐帧过程：N/M = 3/5 投票 + 四态状态机（gap 容忍=2）==");
    println!("帧  ts     命中  票数  连漏  状态            事件");
    for (i, f) in frames.iter().enumerate() {
        let hit = hits_anchor(f, &a, 0.3); // 没有 track_id：靠与锚框的 IoU 判断本帧命中
        let ev = deb.update(hit, f.ts_ms);
        let votes = deb.votes();
        let note = match &ev {
            Some(e) if e.kind == "confirm" => "  <<< 确认目标！".to_string(),
            Some(e) if e.kind == "gone" => "  <<< 判定消失".to_string(),
            _ => String::new(),
        };
        println!(
            "{:>2}  {:>5}  {:^4}  {}/{}   {}    {:<14}{}",
            i,
            f.ts_ms,
            if hit { "H" } else { "·" },
            votes,
            deb.window_m,
            deb.consec_miss,
            deb.state.name(),
            note,
        );
        if let Some(e) = ev {
            events.push(e);
        }
    }

    println!("\n== 状态机吐出的事件 ==");
    for e in &events {
        println!("  [{}] ts={}ms  {}", e.kind, e.ts_ms, e.message);
    }

    // ---------- 对比：去抖前 vs 去抖后 ----------
    let confirms = events.iter().filter(|e| e.kind == "confirm").count();
    let gones = events.iter().filter(|e| e.kind == "gone").count();

    println!("\n== 对比：这段序列里真实目标只有 1 个 ==");
    println!("总帧数：{}", hits.len());
    println!("有命中的帧数：{}", naive_per_frame(&hits));
    println!(
        "[不去抖] 每帧命中即报           → {} 条告警（含 2 次单帧误检 + 同一目标反复报）",
        naive_per_frame(&hits)
    );
    println!(
        "[弱去抖] 仅「重新出现」即确认   → {} 次确认（每次遮挡后都当新目标）",
        naive_rising_edge(&hits)
    );
    println!(
        "[弱去抖] 连续 3 帧确认(漏检即掉线) → {} 次确认（遮挡后重复确认同一人）",
        strict_consecutive(&hits, 3)
    );
    println!(
        "[去抖]   N/M 投票 + 状态机       → {} 次确认 + {} 次消失（干净利落）",
        confirms, gones
    );
}
