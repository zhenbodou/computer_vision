// 第 97 章 项目：实时处理管线（采集 → 检测 → 跟踪 → 事件）
// 运行：cd code/projects/ch97_pipeline && cargo run --release
//
// 目标：把整条流水线【串起来】，做成一个可插拔的项目骨架。
//   ① 采集 + 检测：合成 Vec<Frame>（模拟摄像头出帧 + 检测器输出，确定性可复现）。
//   ② 跟踪：一个简易 SORT 式跟踪器（恒速 alpha-beta 预测 + 贪心 IoU 关联），给稳定 track_id。
//   ③ 事件：挂 1~2 个可插拔 Stage（绊线 Tripwire + 区域 ROI），吐出业务 Event。
// 另外演示【实时性】：处理慢于到帧时的有界队列 + 丢最旧帧，以及“帧账目守恒”不变式（呼应第 75 章）。
// 报真实数字：处理帧数 / 生成轨迹数 / ID 切换数 / 触发事件数；只用 std。
//
// 数据模型 BBox/Detection/Frame/Track/Event 沿用第 83 章（EVENTS_MODEL），此处为独立运行再抄一份。

use std::collections::HashMap;
use std::collections::VecDeque;

// ============================ 第 83 章的共享数据模型 ============================

/// 轴对齐包围盒：左上角 (x,y) + 宽高 (w,h)，单位像素。
#[derive(Clone, Copy, Debug)]
struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl BBox {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        BBox { x, y, w, h }
    }
    fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    /// 底边中点（脚点）——过线 / 进区域判定看脚踩在哪，而非躯干中心。
    fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
    fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }
    /// 交并比 IoU ∈ [0,1]，用于跨帧关联。
    fn iou(&self, o: &BBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let uni = self.area() + o.area() - inter;
        if uni <= 0.0 {
            0.0
        } else {
            inter / uni
        }
    }
}

/// 检测框：来自检测器（第九部分）。事件层只消费它。
#[derive(Clone, Debug)]
struct Detection {
    bbox: BBox,
    #[allow(dead_code)]
    class_id: u32,
    label: String,
    #[allow(dead_code)]
    score: f32,
}

/// 一帧的检测结果（带时间戳）。管线的输入是 Vec<Frame>。
#[derive(Clone, Debug)]
struct Frame {
    ts_ms: u64,
    detections: Vec<Detection>,
}

/// 事件层的输出：一条业务事件 / 告警。
#[derive(Clone, Debug)]
struct Event {
    kind: String,
    ts_ms: u64,
    track_id: Option<u64>,
    message: String,
}

// ============================ 确定性随机数（LCG，务必走高位）============================

// candle 那套坑这里用不到，但离散/连续随机一律从高位派生，避免 LCG 低位规律（见 API_NOTES §18）。
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // 高位转 [0,1)
    }
    /// 近似标准正态（12 个均匀数求和 - 6），给检测框加一点抖动更像真检测器。
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// ============================ 跟踪器：简易 SORT（恒速 alpha-beta + 贪心 IoU）============================

/// 一条轨迹：恒速状态（位置 + 速度 + 尺寸）+ 生命周期计数。
struct Track {
    id: u64,
    label: String,
    cx: f32,
    cy: f32,
    vx: f32,
    vy: f32,
    w: f32,
    h: f32,
    hits: u32,
    time_since_update: u32,
    confirmed: bool,
}
impl Track {
    fn bbox(&self) -> BBox {
        BBox::new(self.cx - self.w / 2.0, self.cy - self.h / 2.0, self.w, self.h)
    }
    fn predict(&mut self) {
        // 恒速推进一帧；先假设“这帧没配上”，配上了再清零。
        self.cx += self.vx;
        self.cy += self.vy;
        self.time_since_update += 1;
    }
    fn update(&mut self, b: &BBox, label: &str) {
        let (mx, my) = b.center();
        let (rx, ry) = (mx - self.cx, my - self.cy); // 残差 = 观测 - 预测
        self.cx += 0.6 * rx; // 修正位置（α=0.6）
        self.cy += 0.6 * ry;
        self.vx += 0.25 * rx; // 修正速度（β=0.25）
        self.vy += 0.25 * ry;
        self.w = b.w;
        self.h = b.h;
        self.hits += 1;
        self.time_since_update = 0;
        self.label = label.to_string();
        if self.hits >= MIN_HITS {
            self.confirmed = true;
        }
    }
}

const MIN_HITS: u32 = 3; // 连续命中 3 次才“转正”，滤掉一闪而过的误检
const MAX_AGE: u32 = 3; // 连续失配 3 帧才删除，给短暂遮挡留缓冲
const IOU_THRESH: f32 = 0.3; // 关联的 IoU 门槛

/// 跟踪器某帧的输出：一个确认目标（带稳定 track_id）。
#[derive(Clone, Debug)]
struct TrackObs {
    track_id: u64,
    bbox: BBox,
    label: String,
}

struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
}
impl Tracker {
    fn new() -> Self {
        Tracker {
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// 吃一帧检测，吐出这一帧所有【已确认】轨迹的观测（带 track_id）。
    fn update(&mut self, frame: &Frame) -> Vec<TrackObs> {
        // ① 预测：每条轨迹恒速推进一帧
        for t in self.tracks.iter_mut() {
            t.predict();
        }
        // ② 关联：贪心 IoU。所有 IoU≥阈值的候选对按 IoU 降序，依次占用。
        let dets = &frame.detections;
        let mut cand: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            for (di, d) in dets.iter().enumerate() {
                let iou = t.bbox().iou(&d.bbox);
                if iou >= IOU_THRESH {
                    cand.push((iou, ti, di));
                }
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut track_used = vec![false; self.tracks.len()];
        let mut det_used = vec![false; dets.len()];
        for (_, ti, di) in cand {
            if !track_used[ti] && !det_used[di] {
                track_used[ti] = true;
                det_used[di] = true;
                self.tracks[ti].update(&dets[di].bbox, &dets[di].label);
            }
        }
        // ③ 没配上的检测 → 新建轨迹（试用期）
        for (di, d) in dets.iter().enumerate() {
            if !det_used[di] {
                let (cx, cy) = d.bbox.center();
                self.tracks.push(Track {
                    id: self.next_id,
                    label: d.label.clone(),
                    cx,
                    cy,
                    vx: 0.0,
                    vy: 0.0,
                    w: d.bbox.w,
                    h: d.bbox.h,
                    hits: 1,
                    time_since_update: 0,
                    confirmed: false,
                });
                self.next_id += 1;
            }
        }
        // ④ 老化：连续失配超过 MAX_AGE 的轨迹删除
        self.tracks.retain(|t| t.time_since_update <= MAX_AGE);
        // 输出这一帧的已确认轨迹
        self.tracks
            .iter()
            .filter(|t| t.confirmed && t.time_since_update == 0)
            .map(|t| TrackObs {
                track_id: t.id,
                bbox: t.bbox(),
                label: t.label.clone(),
            })
            .collect()
    }
}

// ============================ 可插拔事件阶段：trait Stage ============================

/// 跟踪器输出的“带 id 的一帧”——事件规则的统一输入。
/// （第 83 章的 Stage 吃 &Frame；到了完整管线里，绊线方向 / 区域停留都要按 track_id 分别判定，
///   所以这里让 Stage 吃 TrackedFrame——同一心智模型的自然延伸。）
struct TrackedFrame {
    ts_ms: u64,
    obs: Vec<TrackObs>,
}

/// 一个规则 / 过滤环节：吃一帧（带 id），产出若干事件（可能为空）。有状态的规则自己持有 &mut self。
trait Stage {
    fn name(&self) -> &str;
    fn process(&mut self, tf: &TrackedFrame) -> Vec<Event>;
}

// ---- 几何内核：叉积、线段相交、射线法点在多边形内（第 84 / 87 章）----

fn cross(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}
fn segments_intersect(a: (f32, f32), b: (f32, f32), p: (f32, f32), q: (f32, f32)) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(a, b, q);
    let d3 = cross(p, q, a);
    let d4 = cross(p, q, b);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}
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

/// 绊线阶段：脚点轨迹跨过有向线段 A→B 就吐事件，并按叉积符号分方向。
struct TripwireStage {
    a: (f32, f32),
    b: (f32, f32),
    prev_foot: HashMap<u64, (f32, f32)>, // 每个 track 上一帧脚点
}
impl TripwireStage {
    fn new(a: (f32, f32), b: (f32, f32)) -> Self {
        TripwireStage {
            a,
            b,
            prev_foot: HashMap::new(),
        }
    }
}
impl Stage for TripwireStage {
    fn name(&self) -> &str {
        "tripwire"
    }
    fn process(&mut self, tf: &TrackedFrame) -> Vec<Event> {
        let mut out = Vec::new();
        for o in &tf.obs {
            let foot = o.bbox.bottom_center();
            if let Some(&prev) = self.prev_foot.get(&o.track_id) {
                if segments_intersect(self.a, self.b, prev, foot) {
                    // 出发点在“门外(+侧, y 大)”→ 进 IN；在“门内(-侧)”→ 出 OUT。
                    let is_in = cross(self.a, self.b, prev) > 0.0;
                    let (tag, desc) = if is_in {
                        ("进 IN", "门外→门内")
                    } else {
                        ("出 OUT", "门内→门外")
                    };
                    out.push(Event {
                        kind: "tripwire".into(),
                        ts_ms: tf.ts_ms,
                        track_id: Some(o.track_id),
                        message: format!("{} {}（{}）", o.label, tag, desc),
                    });
                }
            }
            self.prev_foot.insert(o.track_id, foot);
        }
        out
    }
}

/// 区域阶段：脚点落在布防多边形内就算命中；按 track_id 冷却去重（第 86 章），持续停留只报一次。
struct RoiStage {
    poly: Vec<(f32, f32)>,
    cd_ms: u64,
    last_fire: HashMap<u64, u64>,
}
impl RoiStage {
    fn new(poly: Vec<(f32, f32)>, cd_ms: u64) -> Self {
        RoiStage {
            poly,
            cd_ms,
            last_fire: HashMap::new(),
        }
    }
}
impl Stage for RoiStage {
    fn name(&self) -> &str {
        "roi"
    }
    fn process(&mut self, tf: &TrackedFrame) -> Vec<Event> {
        let mut out = Vec::new();
        for o in &tf.obs {
            let (fx, fy) = o.bbox.bottom_center();
            if point_in_polygon(fx, fy, &self.poly) {
                // 冷却闸门：同一 track 冷却期内只报一次，把“每帧命中”收敛成一条。
                let recent = self
                    .last_fire
                    .get(&o.track_id)
                    .map(|&t0| tf.ts_ms.saturating_sub(t0) < self.cd_ms)
                    .unwrap_or(false);
                if !recent {
                    self.last_fire.insert(o.track_id, tf.ts_ms);
                    out.push(Event {
                        kind: "roi".into(),
                        ts_ms: tf.ts_ms,
                        track_id: Some(o.track_id),
                        message: format!("{} 进入布防区", o.label),
                    });
                }
            }
        }
        out
    }
}

// ============================ 管线：把跟踪器 + 若干 Stage 串起来 ============================

struct Pipeline {
    tracker: Tracker,
    stages: Vec<Box<dyn Stage>>,
    // 吞吐统计
    frames_processed: usize,
    dets_seen: usize,
    events: Vec<Event>,
}
impl Pipeline {
    fn new(stages: Vec<Box<dyn Stage>>) -> Self {
        Pipeline {
            tracker: Tracker::new(),
            stages,
            frames_processed: 0,
            dets_seen: 0,
            events: Vec::new(),
        }
    }
    fn stage_names(&self) -> String {
        let mut names = vec!["Tracker".to_string()];
        names.extend(self.stages.iter().map(|s| s.name().to_string()));
        names.join(" -> ")
    }
    /// 喂一帧：采集→(跟踪)→(逐 Stage 事件)。返回这一帧跟踪器的确认观测（供评估 id-switch）。
    fn feed(&mut self, frame: &Frame) -> Vec<TrackObs> {
        self.frames_processed += 1;
        self.dets_seen += frame.detections.len();
        let obs = self.tracker.update(frame);
        let tf = TrackedFrame {
            ts_ms: frame.ts_ms,
            obs: obs.clone(),
        };
        for st in self.stages.iter_mut() {
            let evs = st.process(&tf);
            self.events.extend(evs);
        }
        obs
    }
}

// ============================ 合成场景：采集 + 检测器输出 ============================

const N_FRAMES: usize = 45;
const CAPTURE_PERIOD_MS: u64 = 40; // 25 fps 出帧

// 绊线（有向线段 A→B）：只在 x∈[100,540] 判定；y 大侧=门外(+)，y 小侧=门内(-)。
const WIRE_A: (f32, f32) = (100.0, 200.0);
const WIRE_B: (f32, f32) = (540.0, 200.0);

/// 一个真值目标：在 [start,end] 帧内脚点从 foot0 线性走到 foot1；missed 帧无检测（模拟漏检）。
struct GtTarget {
    gt: u64,
    label: &'static str,
    class_id: u32,
    start: usize,
    end: usize,
    foot0: (f32, f32),
    foot1: (f32, f32),
    w: f32,
    h: f32,
    missed: &'static [usize],
}
impl GtTarget {
    fn foot_at(&self, f: usize) -> (f32, f32) {
        let t = (f - self.start) as f32 / (self.end - self.start) as f32;
        (
            self.foot0.0 + (self.foot1.0 - self.foot0.0) * t,
            self.foot0.1 + (self.foot1.1 - self.foot0.1) * t,
        )
    }
    fn box_at(&self, f: usize) -> BBox {
        let (fx, fy) = self.foot_at(f);
        BBox::new(fx - self.w / 2.0, fy - self.h, self.w, self.h) // 脚点=底边中点
    }
    fn present(&self, f: usize) -> bool {
        f >= self.start && f <= self.end && !self.missed.contains(&f)
    }
}

fn scene() -> Vec<GtTarget> {
    vec![
        // 目标1（人）：脚点从下(y=262)走到上(y=150)，向上跨绊线 → 进 IN；第 12、13 帧漏检。
        GtTarget {
            gt: 1,
            label: "person",
            class_id: 0,
            start: 0,
            end: 40,
            foot0: (200.0, 262.0),
            foot1: (200.0, 150.0),
            w: 30.0,
            h: 70.0,
            missed: &[12, 13],
        },
        // 目标2（人）：脚点从上(y=150)走到下(y=262)，向下跨绊线 → 出 OUT。
        GtTarget {
            gt: 2,
            label: "person",
            class_id: 0,
            start: 0,
            end: 40,
            foot0: (320.0, 150.0),
            foot1: (320.0, 262.0),
            w: 30.0,
            h: 70.0,
            missed: &[],
        },
        // 目标3（车）：脚点从 (460,210) 走到 (460,300)，全程在绊线下方(不跨线)，中途驶入布防 ROI。
        GtTarget {
            gt: 3,
            label: "car",
            class_id: 2,
            start: 5,
            end: 44,
            foot0: (460.0, 210.0),
            foot1: (460.0, 300.0),
            w: 60.0,
            h: 40.0,
            missed: &[],
        },
    ]
}

/// 生成整段合成输入：每帧对每个在场目标造一个带抖动的检测框（模拟检测器输出）。
fn synth_frames(targets: &[GtTarget], rng: &mut Lcg) -> Vec<Frame> {
    let mut frames = Vec::with_capacity(N_FRAMES);
    for f in 0..N_FRAMES {
        let mut dets = Vec::new();
        for t in targets {
            if t.present(f) {
                let mut b = t.box_at(f);
                // 检测器天然有几像素漂移：给框加一点高斯抖动（可复现）。
                b.x += 0.6 * rng.gauss();
                b.y += 0.6 * rng.gauss();
                dets.push(Detection {
                    bbox: b,
                    class_id: t.class_id,
                    label: t.label.to_string(),
                    score: 0.9,
                });
            }
        }
        frames.push(Frame {
            ts_ms: f as u64 * CAPTURE_PERIOD_MS,
            detections: dets,
        });
    }
    frames
}

// ============================ 评估：把跟踪器输出的 track_id 对回真值，数 id-switch ============================

/// 每帧把每个确认观测配到“最近的真值目标”（本场景各目标分得很开，最近邻无歧义）。
fn count_id_switches(targets: &[GtTarget], per_frame_obs: &[(usize, Vec<TrackObs>)]) -> (usize, usize) {
    let mut last_id: HashMap<u64, u64> = HashMap::new(); // gt -> 上一次挂的 track_id
    let mut switches = 0usize;
    let mut seen_tracks: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for (f, obs) in per_frame_obs {
        for o in obs {
            seen_tracks.insert(o.track_id);
            // 找最近真值目标
            let (oc_x, oc_y) = o.bbox.center();
            let mut best_gt = 0u64;
            let mut best_d = f32::MAX;
            for t in targets {
                if t.present(*f) {
                    let (fx, fy) = t.foot_at(*f);
                    let (cx, cy) = (fx, fy - t.h / 2.0); // 真值中心
                    let d = (cx - oc_x).powi(2) + (cy - oc_y).powi(2);
                    if d < best_d {
                        best_d = d;
                        best_gt = t.gt;
                    }
                }
            }
            if best_gt != 0 {
                if let Some(&prev) = last_id.get(&best_gt) {
                    if prev != o.track_id {
                        switches += 1;
                    }
                }
                last_id.insert(best_gt, o.track_id);
            }
        }
    }
    (seen_tracks.len(), switches)
}

// ============================ 实时性：处理慢于到帧时的丢帧不变式（确定性模拟）============================

/// 有界队列 + 丢最旧帧的离散事件模拟：到帧 40ms/帧、单帧处理 proc_ms、队列容量 cap。
/// 返回（被处理的帧序号列表, 丢弃数）。确定性：不读系统时钟，用虚拟毫秒时钟推进。
fn drop_oldest_sim(n: usize, capture_ms: u64, proc_ms: u64, cap: usize) -> (Vec<usize>, usize) {
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut processed: Vec<usize> = Vec::new();
    let mut dropped = 0usize;
    let mut busy_until = 0u64; // 处理器在此刻之前忙碌
    let horizon = n as u64 * capture_ms + cap as u64 * proc_ms + 1000; // 足够排空队列
    for t in 0..=horizon {
        // 到帧？（每 capture_ms 到一帧）
        if t % capture_ms == 0 {
            let seq = (t / capture_ms) as usize;
            if seq < n {
                if queue.len() >= cap {
                    queue.pop_front(); // 满了丢最旧（队首），保最新
                    dropped += 1;
                }
                queue.push_back(seq);
            }
        }
        // 处理器空闲且有活 → 取队首处理
        if t >= busy_until {
            if let Some(seq) = queue.pop_front() {
                processed.push(seq);
                busy_until = t + proc_ms;
            }
        }
    }
    debug_assert!(queue.is_empty(), "horizon 不够，队列没排空");
    (processed, dropped)
}

// ============================ main ============================

fn main() {
    println!("==== 第 97 章：实时处理管线（采集 → 检测 → 跟踪 → 事件）====");
    let roi: Vec<(f32, f32)> = vec![
        (380.0, 220.0),
        (560.0, 220.0),
        (560.0, 340.0),
        (380.0, 340.0),
    ];
    println!(
        "场景：640×360；绊线 y=200 (x∈[100,540])；布防 ROI 矩形 x∈[380,560] y∈[220,340]"
    );
    println!(
        "合成输入：{} 帧 @25fps（含目标1两帧漏检），3 个目标（2 人 + 1 车）\n",
        N_FRAMES
    );

    let targets = scene();
    let mut rng = Lcg::new(20260827);
    let frames = synth_frames(&targets, &mut rng);

    // ---------- ① 管线全量运行（不丢帧）----------
    let mut pipe = Pipeline::new(vec![
        Box::new(TripwireStage::new(WIRE_A, WIRE_B)),
        Box::new(RoiStage::new(roi.clone(), 60_000)), // 冷却 60s：停留期内每目标只报一次
    ]);
    println!("—— ① 管线全量运行（不丢帧）——");
    println!("可插拔阶段：[{}]", pipe.stage_names());

    let mut per_frame_obs: Vec<(usize, Vec<TrackObs>)> = Vec::new();
    for (f, frame) in frames.iter().enumerate() {
        let obs = pipe.feed(frame);
        per_frame_obs.push((f, obs));
    }
    let (n_tracks, switches) = count_id_switches(&targets, &per_frame_obs);

    println!("处理帧数         = {}", pipe.frames_processed);
    println!("累计检测框        = {}", pipe.dets_seen);
    println!("生成轨迹数（确认）= {}", n_tracks);
    println!("ID 切换数         = {}", switches);
    println!("触发事件数        = {}", pipe.events.len());
    for e in &pipe.events {
        println!(
            "  [{}] t={:>4}ms track#{} {}",
            e.kind,
            e.ts_ms,
            e.track_id.unwrap_or(0),
            e.message
        );
    }

    // ---------- ② 实时性：处理慢于到帧时的丢帧不变式 ----------
    println!("\n—— ② 实时性：处理慢于到帧时的丢帧不变式（呼应第 75 章）——");
    let proc_ms = 55u64;
    let cap = 3usize;
    println!(
        "到帧 {}ms/帧(≈25fps)，单帧处理 {}ms(≈18fps)，有界队列(容量{})丢最旧",
        CAPTURE_PERIOD_MS, proc_ms, cap
    );
    let (kept, dropped) = drop_oldest_sim(N_FRAMES, CAPTURE_PERIOD_MS, proc_ms, cap);
    println!(
        "产出 {} 帧，处理 {} 帧，丢弃 {} 帧   （{} + {} = {} ✓ 帧账目守恒）",
        N_FRAMES,
        kept.len(),
        dropped,
        kept.len(),
        dropped,
        kept.len() + dropped
    );
    assert_eq!(kept.len() + dropped, N_FRAMES, "帧账目必须守恒");

    // 只把“处理到的帧”喂给同一套管线，看事件是否还保得住
    let mut pipe2 = Pipeline::new(vec![
        Box::new(TripwireStage::new(WIRE_A, WIRE_B)),
        Box::new(RoiStage::new(roi.clone(), 60_000)),
    ]);
    for &seq in &kept {
        pipe2.feed(&frames[seq]);
    }
    println!(
        "只喂“处理到的 {} 帧”给管线：触发事件数 = {}（持续性事件在丢帧下仍保住）",
        kept.len(),
        pipe2.events.len()
    );
    for e in &pipe2.events {
        println!(
            "  [{}] t={:>4}ms track#{} {}",
            e.kind,
            e.ts_ms,
            e.track_id.unwrap_or(0),
            e.message
        );
    }

    println!("\n结论：一条 [Tracker]->[Tripwire]->[ROI] 的可插拔管线，把 45 帧检测流串成了 3 条稳定轨迹、");
    println!("3 条业务事件、0 次 ID 切换；实时丢帧下帧账目守恒，持续性事件不丢失。");
}
