//! 第 87 章 绊线检测（line-crossing）与方向计数
//! 运行：cargo run --example ch87_tripwire
//! 依赖：无（只用 std）——用合成的、带 track_id 的轨迹（Vec<Track>）当输入，自包含、可复现。
//!
//! BBox / Track / Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md）。
//! 核心新内容：有向线段 AB + 脚点「位移段」的线段相交判定 + 叉积定方向 + 按 track_id 的去抖计数。

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
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    /// 底边中点（脚点）——过线判定用它而非中心（人站在地面上，脚点才是真正“踩没踩线”）。
    pub fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
}

/// 跟踪后的目标（同第 83 章）：跟踪器（第 62 章）给每个目标分配稳定 track_id。
/// 绊线要看「前一帧脚点 → 当前帧脚点」这一小段位移，跨帧连起来靠的正是稳定 id。
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub label: String,
    pub history: Vec<(u64, BBox)>, // (ts_ms, bbox) 按时间递增
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

/// 叉积的 z 分量：cross(A, B, P) = (B - A) × (P - A)。
/// 它的**符号**告诉我们点 P 在有向线 A→B 的哪一侧：
///   > 0 一侧、 < 0 另一侧、 = 0 恰好在直线上。
fn cross(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// 两条线段是否“真正相交”（proper intersection）：段 AB 与段 PQ 相交。
/// 判据：P、Q 分居直线 AB 两侧（d1·d2 < 0），且 A、B 分居直线 PQ 两侧（d3·d4 < 0）。
/// 用严格 <0 排除了「共线 / 端点正好压线」的退化情形；这也天然处理了
/// **“绊线端点之外的延长线不算”**——交点必须同时落在两条线段内部。
fn segments_intersect(a: (f32, f32), b: (f32, f32), p: (f32, f32), q: (f32, f32)) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(a, b, q);
    let d3 = cross(p, q, a);
    let d4 = cross(p, q, b);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// 一条有向绊线：线段 A→B。约定 A→B 的方向定义“正侧/负侧”：
///   side = cross(A, B, P) > 0 记为 **门外**（本例里画面下方、y 大的一侧）；
///   side < 0 记为 **门内**（画面上方、y 小的一侧）。
/// 于是「门外→门内」= 进（IN），「门内→门外」= 出（OUT）。
struct Tripwire {
    a: (f32, f32),
    b: (f32, f32),
}

impl Tripwire {
    fn new(a: (f32, f32), b: (f32, f32)) -> Self {
        Tripwire { a, b }
    }

    /// 判断脚点从 p_prev 走到 p_curr 这一步是否跨过绊线，并给出方向。
    /// 返回：Some(true)=进(IN)，Some(false)=出(OUT)，None=没跨（或只跨了延长线）。
    ///
    /// 关键：**必须用「连续两帧的位移段」判相交**，而不能只看当前脚点在线的哪一侧——
    /// 只看单点无法区分“它是刚跨过来”还是“它本来就在这侧”。
    fn crossing_dir(&self, p_prev: (f32, f32), p_curr: (f32, f32)) -> Option<bool> {
        if !segments_intersect(self.a, self.b, p_prev, p_curr) {
            return None; // 位移段没跨过绊线线段（含只跨了端点外延长线的情况）
        }
        // 跨了：看出发点 p_prev 在哪一侧。门外(+) 出发 → 进；门内(-) 出发 → 出。
        Some(cross(self.a, self.b, p_prev) > 0.0)
    }
}

/// 双向计数器。
#[derive(Default, Clone, Copy)]
struct Counter {
    in_cnt: u32,
    out_cnt: u32,
}
impl Counter {
    fn add(&mut self, is_in: bool) {
        if is_in {
            self.in_cnt += 1;
        } else {
            self.out_cnt += 1;
        }
    }
}

/// 由一串脚点造一条轨迹：固定框大小 w=30,h=70，从脚点反推左上角。
/// dt_ms 为帧间隔（合成的递增时间戳，不读系统时钟）。
fn track_from_feet(id: u64, label: &str, feet: &[(f32, f32)], dt_ms: u64) -> Track {
    let (w, h) = (30.0f32, 70.0f32);
    let history = feet
        .iter()
        .enumerate()
        .map(|(i, &(fx, fy))| (i as u64 * dt_ms, BBox::new(fx - w / 2.0, fy - h, w, h)))
        .collect();
    Track {
        id,
        label: label.to_string(),
        history,
    }
}

fn main() {
    let dt = 100u64; // 10 fps，帧间隔 100ms

    // 绊线：地铁闸口那条横线，A=(100,300) → B=(540,300)，x 只在 [100,540] 之间算数。
    let wire = Tripwire::new((100.0, 300.0), (540.0, 300.0));

    // ---------- 合成 4 条带 track_id 的轨迹（脚点序列） ----------
    // ① 正常进站：x≈200 处从下(门外,y大)往上(门内,y小)走，干净跨线一次 → 进。
    let t1 = track_from_feet(1, "person", &[(200.0, 380.0), (200.0, 330.0), (200.0, 280.0), (200.0, 230.0)], dt);
    // ② 正常出站：x≈420 处从上往下走，干净跨线一次 → 出。
    let t2 = track_from_feet(2, "person", &[(420.0, 230.0), (420.0, 280.0), (420.0, 330.0), (420.0, 380.0)], dt);
    // ③ 越线但在绊线**端点之外**（x=610 > 540）：跨了“延长线”，不该计数。
    let t3 = track_from_feet(3, "person", &[(610.0, 380.0), (610.0, 330.0), (610.0, 280.0), (610.0, 230.0)], dt);
    // ④ 贴着线来回抖动（检测框在线附近漂移）：净效果是进了一次，但原始位移段跨线 5 次。
    let t4 = track_from_feet(4, "person", &[(300.0, 306.0), (300.0, 294.0), (300.0, 305.0), (300.0, 295.0), (300.0, 307.0), (300.0, 293.0)], dt);

    let tracks = vec![t1, t2, t3, t4];

    // ---------- A) 朴素计数：几何判定，但不做任何去抖 ----------
    let mut naive = Counter::default();
    for t in &tracks {
        for w in t.history.windows(2) {
            let p_prev = w[0].1.bottom_center();
            let p_curr = w[1].1.bottom_center();
            if let Some(is_in) = wire.crossing_dir(p_prev, p_curr) {
                naive.add(is_in);
            }
        }
    }

    // ---------- B) 去抖计数：同一 track_id 触发后冷却 cooldown_ms 内不再计数 ----------
    // 冷却是「按 track_id 各记各的」：一个人抖动不影响别人。（冷却思想见第 86 章）
    let cooldown_ms = 800u64;
    let mut last_fire: HashMap<u64, u64> = HashMap::new();
    let mut deb = Counter::default();
    let mut events: Vec<Event> = Vec::new();

    for t in &tracks {
        for w in t.history.windows(2) {
            let (ts_curr, b_curr) = w[1];
            let p_prev = w[0].1.bottom_center();
            let p_curr = b_curr.bottom_center();
            let Some(is_in) = wire.crossing_dir(p_prev, p_curr) else {
                continue;
            };
            // 冷却门：距上次该 track 触发不足 cooldown_ms 就压掉（抑制贴线抖动的反复计数）
            let recent = last_fire
                .get(&t.id)
                .map(|&t0| ts_curr.saturating_sub(t0) < cooldown_ms)
                .unwrap_or(false);
            if recent {
                continue;
            }
            last_fire.insert(t.id, ts_curr);
            deb.add(is_in);
            events.push(Event {
                kind: "tripwire".to_string(),
                ts_ms: ts_curr,
                track_id: Some(t.id),
                message: format!(
                    "track {} {}（{}）",
                    t.id,
                    if is_in { "进 IN" } else { "出 OUT" },
                    if is_in { "门外→门内" } else { "门内→门外" }
                ),
            });
        }
    }

    // ---------- 打印 ----------
    println!("绊线 A=({:.0},{:.0}) → B=({:.0},{:.0})，只在这条线段上判定（端点外的延长线不算）", wire.a.0, wire.a.1, wire.b.0, wire.b.1);
    println!("规定：门外(+侧,y大) → 门内(-侧,y小) = 进；反之 = 出\n");

    // 逐条轨迹跑一遍“几何判定”，直观看每条轨迹跨了几次
    println!("== 逐条轨迹的原始跨线（仅几何，不去抖）==");
    for t in &tracks {
        let mut c = Counter::default();
        for w in t.history.windows(2) {
            if let Some(is_in) = wire.crossing_dir(w[0].1.bottom_center(), w[1].1.bottom_center()) {
                c.add(is_in);
            }
        }
        let note = match t.id {
            3 => "  (在端点外，跨的是延长线 → 不计数)",
            4 => "  (贴线抖动，原始跨线次数虚高)",
            _ => "",
        };
        println!("  track#{} {}: 进 {} / 出 {}{}", t.id, t.label, c.in_cnt, c.out_cnt, note);
    }

    println!("\n== 去抖后吐出的跨线事件（按 track_id 冷却 {}ms）==", cooldown_ms);
    for e in &events {
        println!("  [{}] t={:>4}ms {}", e.kind, e.ts_ms, e.message);
    }

    println!("\n== 双向计数对比 ==");
    println!("[不去抖] 进 {} 人 / 出 {} 人（被 track#4 的贴线抖动灌了水）", naive.in_cnt, naive.out_cnt);
    println!("[去抖]   进 {} 人 / 出 {} 人（干净结果）", deb.in_cnt, deb.out_cnt);
    println!("净流量（进 - 出）= {}", deb.in_cnt as i32 - deb.out_cnt as i32);
}
