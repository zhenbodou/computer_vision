// 第 104 章 项目：客流 / 车流计数（绊线方向 + 跟踪去重 + 分类）
// 运行：cd code/projects/ch104_counting && cargo run --release
//
// 目标：在门口/路口拉一条【计数绊线】，统计“进 / 出”的人数、车数。
//   ① 绊线方向（第 87 章，叉积定侧）：脚点跨过有向线 A→B，叉积符号翻转 = 进 or 出。
//   ② 跟踪去重（第 62 / 97 章）：靠稳定 track_id，让同一个目标只被数一次——
//      否则检测框在线附近抖动会反复穿线，把 1 个人算成好几个。
//   ③ 分类计数：按 label(person/car) 分别累加，得到客流 / 车流。
// 报真实数字：进 / 出计数 vs 真值、去重前后对比、以及“只跨延长线不算数”的边界。
// 只用 std，合成带 track_id 的轨迹，确定性可复现。
//
// 数据模型 BBox/Track/Event 沿用第 83 章（EVENTS_MODEL）；这里从跟踪器输出（Track）接手。

use std::collections::HashMap;

// ============================ 第 83 章共享模型（节选：BBox / Track / Event）============================

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
    /// 底边中点（脚点）——判断“越没越线”看脚踩在哪。
    fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
}

/// 跟踪后的目标：跟踪器（第 62 / 97 章）给的稳定 id + 按时间递增的轨迹历史。
#[derive(Clone, Debug)]
struct Track {
    id: u64,
    label: String,
    history: Vec<(u64, BBox)>, // (ts_ms, bbox)
}

/// 事件层输出：一次过线计数。
#[derive(Clone, Debug)]
struct Event {
    kind: String, // "in" / "out"
    ts_ms: u64,
    track_id: Option<u64>,
    label: String,
}

// ============================ 几何内核：叉积、线段相交（第 87 章）============================

/// 叉积 z 分量：符号 = P 在有向线 A→B 的哪一侧。
fn cross(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// 段 AB 与段 PQ 是否“真正相交”（两组叉积同时异号）——天然排除“延长线相交”。
fn segments_intersect(a: (f32, f32), b: (f32, f32), p: (f32, f32), q: (f32, f32)) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(a, b, q);
    let d3 = cross(p, q, a);
    let d4 = cross(p, q, b);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// 一次绊线跨越方向：Some(true)=进(IN)，Some(false)=出(OUT)，None=没跨（含只跨延长线）。
fn crossing_dir(a: (f32, f32), b: (f32, f32), p_prev: (f32, f32), p_curr: (f32, f32)) -> Option<bool> {
    if !segments_intersect(a, b, p_prev, p_curr) {
        return None;
    }
    // 约定：线上方(y 小, cross<0)=门内；下方(y 大, cross>0)=门外。
    // 出发点在门外(cross>0) → 走向门内 = 进(IN)。
    Some(cross(a, b, p_prev) > 0.0)
}

// ============================ 计数器：绊线 + 分类，含“去重 / 不去重”两种口径 ============================

/// 一条计数绊线。
struct Counter {
    a: (f32, f32),
    b: (f32, f32),
    dedup_ms: u64, // 同一 track_id 的去重窗口：窗口内的重复穿线不再计数
}

/// 分类计数结果：按 label 分别累加进 / 出。
#[derive(Default)]
struct Tally {
    // label -> (in, out)
    map: HashMap<String, (u32, u32)>,
}
impl Tally {
    fn add(&mut self, label: &str, is_in: bool) {
        let e = self.map.entry(label.to_string()).or_insert((0, 0));
        if is_in {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }
    fn get(&self, label: &str) -> (u32, u32) {
        *self.map.get(label).unwrap_or(&(0, 0))
    }
}

/// 朴素计数：每一次几何穿线都算一次（不做任何去重）。
/// 抖动 / 徘徊会被反复计数——这正是需要跟踪去重的原因。
fn count_naive(c: &Counter, tracks: &[Track]) -> (Tally, Vec<Event>) {
    let mut tally = Tally::default();
    let mut events = Vec::new();
    for t in tracks {
        for w in t.history.windows(2) {
            let prev = w[0].1.bottom_center();
            let curr = w[1].1.bottom_center();
            if let Some(is_in) = crossing_dir(c.a, c.b, prev, curr) {
                tally.add(&t.label, is_in);
                events.push(Event {
                    kind: if is_in { "in".into() } else { "out".into() },
                    ts_ms: w[1].0,
                    track_id: Some(t.id),
                    label: t.label.clone(),
                });
            }
        }
    }
    (tally, events)
}

/// 去重计数：靠 track_id + 去重窗口，让同一目标在窗口内只被计一次。
/// 每条轨迹独立维护“上次计数时刻”，窗口内的重复穿线（抖动/徘徊）被吞掉。
fn count_dedup(c: &Counter, tracks: &[Track]) -> (Tally, Vec<Event>) {
    let mut tally = Tally::default();
    let mut events = Vec::new();
    for t in tracks {
        let mut last_fire: Option<u64> = None;
        for w in t.history.windows(2) {
            let prev = w[0].1.bottom_center();
            let curr = w[1].1.bottom_center();
            if let Some(is_in) = crossing_dir(c.a, c.b, prev, curr) {
                let ts = w[1].0;
                let recent = last_fire.map(|t0| ts.saturating_sub(t0) < c.dedup_ms).unwrap_or(false);
                if !recent {
                    last_fire = Some(ts);
                    tally.add(&t.label, is_in);
                    events.push(Event {
                        kind: if is_in { "in".into() } else { "out".into() },
                        ts_ms: ts,
                        track_id: Some(t.id),
                        label: t.label.clone(),
                    });
                }
            }
        }
    }
    (tally, events)
}

// ============================ 合成场景：带 track_id 的多条轨迹（含抖动者、跨延长线者）============================

const STEP_MS: u64 = 40; // 25 fps

/// 用逐帧脚点序列造一条 Track：feet[i] 是第 (start_frame+i) 帧的脚点。
fn track_from_feet(id: u64, label: &str, w: f32, h: f32, start_frame: usize, feet: &[(f32, f32)]) -> Track {
    let mut history = Vec::new();
    for (i, &(fx, fy)) in feet.iter().enumerate() {
        let f = (start_frame + i) as u64;
        let bbox = BBox::new(fx - w / 2.0, fy - h, w, h); // 脚点=底边中点
        history.push((f * STEP_MS, bbox));
    }
    Track {
        id,
        label: label.to_string(),
        history,
    }
}

/// 造一条“直线匀速穿过”的轨迹：脚点从 (x,y0) 线性到 (x,y1)，n 帧。
fn straight(id: u64, label: &str, w: f32, h: f32, x: f32, y0: f32, y1: f32, n: usize, start: usize) -> Track {
    let feet: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let s = i as f32 / (n - 1) as f32;
            (x, y0 + (y1 - y0) * s)
        })
        .collect();
    track_from_feet(id, label, w, h, start, &feet)
}

fn scene() -> Vec<Track> {
    vec![
        // track#1 person：门外(y=318)向上走进门内(y=178)。→ 真值 进 1。
        straight(1, "person", 30.0, 70.0, 150.0, 318.0, 178.0, 20, 0),
        // track#2 person：门外→门内。→ 真值 进 1。
        straight(2, "person", 30.0, 70.0, 250.0, 322.0, 182.0, 20, 5),
        // track#3 person：门内(y=178)向下走出门外(y=318)。→ 真值 出 1。
        straight(3, "person", 30.0, 70.0, 350.0, 178.0, 318.0, 20, 3),
        // track#4 car：门外→门内（大框、更快）。→ 真值 进 1。
        straight(4, "car", 64.0, 44.0, 450.0, 338.0, 158.0, 24, 0),
        // track#5 car：门内→门外。→ 真值 出 1。
        straight(5, "car", 64.0, 44.0, 520.0, 158.0, 338.0, 24, 6),
        // track#6 person：从门外走进门内，但检测框在线上剧烈抖动，来回穿线多次。
        // 真值 进 1（净方向向内）；朴素口径会数成 进 3 / 出 2。→ 去重后 进 1。
        track_from_feet(
            6,
            "person",
            30.0,
            70.0,
            2,
            &[
                (550.0, 270.0), // 门外
                (550.0, 262.0),
                (550.0, 247.0), // ↑ 穿线：进
                (550.0, 256.0), // ↓ 穿线：出
                (550.0, 244.0), // ↑ 进
                (550.0, 258.0), // ↓ 出
                (550.0, 238.0), // ↑ 进
                (550.0, 225.0),
                (550.0, 210.0),
                (550.0, 200.0), // 稳定在门内
            ],
        ),
        // track#7 person：在绊线【延长线】外侧穿过（x=600 > 端点 B 的 x=560）。
        // 几何上没跨过“线段”，→ 计数 0（提醒：绊线是线段不是无限长直线）。
        straight(7, "person", 30.0, 70.0, 600.0, 318.0, 178.0, 16, 0),
    ]
}

fn main() {
    println!("==== 第 104 章：客流 / 车流计数（绊线方向 + 跟踪去重 + 分类）====");
    let c = Counter {
        a: (80.0, 250.0),
        b: (560.0, 250.0),
        dedup_ms: 1500, // 同一 track_id 的去重窗口
    };
    println!("计数绊线 A→B = {:?}→{:?}（向上穿=进 IN，向下穿=出 OUT）", c.a, c.b);
    let tracks = scene();
    println!("合成 {} 条轨迹（跟踪器已给稳定 id）\n", tracks.len());

    // —— 真值（人工标注的“净流量”）——
    println!("—— 真值（各目标的净穿线方向）——");
    println!("  person：进 3（track#1,#2,#6）/ 出 1（track#3）；track#7 在延长线外，不计");
    println!("  car   ：进 1（track#4）/ 出 1（track#5）");
    let gt_person = (3u32, 1u32);
    let gt_car = (1u32, 1u32);

    // —— 朴素口径：每次几何穿线都算 ——
    println!("\n—— 朴素计数（不去重：每次穿线都算一次）——");
    let (naive, naive_ev) = count_naive(&c, &tracks);
    let (np_in, np_out) = naive.get("person");
    let (nc_in, nc_out) = naive.get("car");
    println!("  几何穿线事件共 {} 次", naive_ev.len());
    println!("  person：进 {} / 出 {}", np_in, np_out);
    println!("  car   ：进 {} / 出 {}", nc_in, nc_out);
    println!("  → track#6 在线上抖动，被数成 进 3 / 出 2，客流被夸大。");

    // —— 去重口径：靠 track_id + 去重窗口 ——
    println!("\n—— 去重计数（按 track_id + {}ms 窗口，同一目标只数一次）——", c.dedup_ms);
    let (dedup, dedup_ev) = count_dedup(&c, &tracks);
    for e in &dedup_ev {
        let dir = if e.kind == "in" { "进" } else { "出" };
        println!("  [{}] t={:>4}ms track#{} {} {}", e.kind, e.ts_ms, e.track_id.unwrap_or(0), e.label, dir);
    }
    let (dp_in, dp_out) = dedup.get("person");
    let (dc_in, dc_out) = dedup.get("car");
    println!("  计数事件共 {} 次", dedup_ev.len());
    println!("  person：进 {} / 出 {}", dp_in, dp_out);
    println!("  car   ：进 {} / 出 {}", dc_in, dc_out);

    // —— 对比表：朴素 vs 去重 vs 真值 ——
    println!("\n—— 对比：朴素 vs 去重 vs 真值 ——");
    println!("  类别    方向   朴素   去重   真值");
    println!("  person  进     {:>3}    {:>3}    {:>3}", np_in, dp_in, gt_person.0);
    println!("  person  出     {:>3}    {:>3}    {:>3}", np_out, dp_out, gt_person.1);
    println!("  car     进     {:>3}    {:>3}    {:>3}", nc_in, dc_in, gt_car.0);
    println!("  car     出     {:>3}    {:>3}    {:>3}", nc_out, dc_out, gt_car.1);

    let ok = (dp_in, dp_out) == gt_person && (dc_in, dc_out) == gt_car;
    println!("\n  去重口径与真值一致：{}", if ok { "是 ✓" } else { "否 ✗" });

    // —— 净流量与在场估计 ——
    println!("\n—— 净流量（进 - 出，可估计“区域内当前人数/车数”）——");
    println!("  person 净流入 = {} - {} = {}", dp_in, dp_out, dp_in as i32 - dp_out as i32);
    println!("  car    净流入 = {} - {} = {}", dc_in, dc_out, dc_in as i32 - dc_out as i32);

    println!("\n—— 边界说明 ——");
    println!("  track#7 脚点在绊线【延长线】外侧(x=600 越过端点 B 的 x=560)穿过 → 计数 0（绊线是线段不是无限直线）。");
}
