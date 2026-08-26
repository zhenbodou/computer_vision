// 第 103 章 项目：周界入侵报警（ROI 多边形 + 绊线方向 + cooldown）
// 运行：cd code/projects/ch103_perimeter && cargo run --release
//
// 把三件“积木”组装成一个周界报警系统：
//   ① ROI 多边形（第 84 章，射线法）：只有脚点落在【布防区】内才可能报警。
//   ② 绊线方向（第 87 章，叉积定侧）：越过【警戒线】进入院内=入侵方向，出院=离开（非报警）。
//   ③ cooldown 冷却（第 86 章）：同一目标持续停留只报一次，避免告警风暴。
// 核心是【两因子】：既要越警戒线【进入方向】、又要落在布防区内，才算真入侵——比只用 ROI 少报“离场保安”。
// 报真实数字：入侵事件数、cooldown 去重前后对比、绊线方向判定。只用 std，合成带 track_id 的轨迹，可复现。
//
// 数据模型 BBox/Track/Event 沿用第 83 章（EVENTS_MODEL）；Detection/Frame 是跟踪器的领域，这里从跟踪器输出（Track）接手。

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
    /// 底边中点（脚点）——判断“进没进区域 / 越没越线”看脚踩在哪。
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

/// 事件层的输出：一条业务事件 / 告警。
#[derive(Clone, Debug)]
struct Event {
    kind: String,
    ts_ms: u64,
    track_id: Option<u64>,
    message: String,
}

// ============================ 几何内核：叉积、线段相交、射线法（第 84 / 87 章）============================

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

/// 射线法：点 (px,py) 是否在多边形 poly 内（凹多边形也适用）。
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

// ============================ 周界报警系统：ROI + 绊线方向 + cooldown ============================

/// 布防配置：一块布防多边形 + 一条有向警戒绊线 + 冷却时长。
struct Perimeter {
    poly: Vec<(f32, f32)>, // 布防区（脚点落在其中才可能报警）
    wire_a: (f32, f32),    // 警戒线端点 A
    wire_b: (f32, f32),    // 警戒线端点 B（越过 A→B 进入院内 = 入侵方向）
    cd_ms: u64,            // 同一 track_id 的报警冷却时长
}

/// 一次绊线跨越：Some(true)=入侵方向(进院)，Some(false)=离开方向(出院)，None=没跨（含只跨延长线）。
fn crossing_dir(a: (f32, f32), b: (f32, f32), p_prev: (f32, f32), p_curr: (f32, f32)) -> Option<bool> {
    if !segments_intersect(a, b, p_prev, p_curr) {
        return None;
    }
    // 约定：警戒线上方(y 小, cross<0)=院外/街面；下方(y 大, cross>0)=院内。
    // 出发点在院外(cross<0) → 进院 = 入侵方向。
    Some(cross(a, b, p_prev) < 0.0)
}

/// 统计一条轨迹对警戒线的跨越方向：(入侵方向次数, 离开方向次数)。
fn count_directions(peri: &Perimeter, t: &Track) -> (u32, u32) {
    let (mut inward, mut outward) = (0u32, 0u32);
    for w in t.history.windows(2) {
        let prev = w[0].1.bottom_center();
        let curr = w[1].1.bottom_center();
        match crossing_dir(peri.wire_a, peri.wire_b, prev, curr) {
            Some(true) => inward += 1,
            Some(false) => outward += 1,
            None => {}
        }
    }
    (inward, outward)
}

/// 统计一条轨迹脚点落在布防区内的帧数（“原始命中帧”）。
fn count_inside_frames(peri: &Perimeter, t: &Track) -> usize {
    t.history
        .iter()
        .filter(|(_, b)| {
            let (fx, fy) = b.bottom_center();
            point_in_polygon(fx, fy, &peri.poly)
        })
        .count()
}

/// 方案 A（仅 ROI + cooldown，不看方向）：任何目标脚点落在布防区内即报，按 track_id 冷却。
/// 缺点：把“本来就在院里 / 正在离场”的目标也报了。
fn alarms_roi_only(peri: &Perimeter, tracks: &[Track]) -> Vec<Event> {
    let mut last_fire: HashMap<u64, u64> = HashMap::new();
    let mut out = Vec::new();
    for t in tracks {
        for (ts, b) in &t.history {
            let (fx, fy) = b.bottom_center();
            if point_in_polygon(fx, fy, &peri.poly) {
                let recent = last_fire
                    .get(&t.id)
                    .map(|&t0| ts.saturating_sub(t0) < peri.cd_ms)
                    .unwrap_or(false);
                if !recent {
                    last_fire.insert(t.id, *ts);
                    out.push(Event {
                        kind: "intrusion".into(),
                        ts_ms: *ts,
                        track_id: Some(t.id),
                        message: format!("{} 出现在布防区内", t.label),
                    });
                }
            }
        }
    }
    out
}

/// 方案 B（两因子：越警戒线【进入方向】且落在布防区内；再按 track_id 冷却）。
/// 返回 (报警事件, 两因子命中帧数)。
fn alarms_two_factor(peri: &Perimeter, tracks: &[Track]) -> (Vec<Event>, usize) {
    let mut out = Vec::new();
    let mut hit_frames = 0usize;
    for t in tracks {
        let mut entered_inward = false; // 该目标是否曾越警戒线进院
        let mut last_fire: Option<u64> = None;
        let mut prev_foot: Option<(f32, f32)> = None;
        for (ts, b) in &t.history {
            let foot = b.bottom_center();
            // 先更新“是否越线进院”
            if let Some(pf) = prev_foot {
                if let Some(true) = crossing_dir(peri.wire_a, peri.wire_b, pf, foot) {
                    entered_inward = true;
                }
            }
            prev_foot = Some(foot);
            // 两因子：进入方向来的 + 此刻在布防区内
            let inside = point_in_polygon(foot.0, foot.1, &peri.poly);
            if entered_inward && inside {
                hit_frames += 1;
                let recent = last_fire
                    .map(|t0| ts.saturating_sub(t0) < peri.cd_ms)
                    .unwrap_or(false);
                if !recent {
                    last_fire = Some(*ts);
                    out.push(Event {
                        kind: "intrusion".into(),
                        ts_ms: *ts,
                        track_id: Some(t.id),
                        message: format!("{} 翻越警戒线闯入布防区", t.label),
                    });
                }
            }
        }
    }
    (out, hit_frames)
}

// ============================ 合成场景：4 条带 track_id 的轨迹 ============================

const STEP_MS: u64 = 40; // 25 fps

/// 用“分段脚点轨迹”造一条 Track：segs 是若干 (起始帧, 结束帧, foot0, foot1) 段，线性插值。
fn build_track(id: u64, label: &str, w: f32, h: f32, segs: &[(usize, usize, (f32, f32), (f32, f32))]) -> Track {
    let mut history = Vec::new();
    for &(f0, f1, p0, p1) in segs {
        for f in f0..=f1 {
            let denom = if f1 > f0 { (f1 - f0) as f32 } else { 1.0 };
            let s = (f - f0) as f32 / denom;
            let fx = p0.0 + (p1.0 - p0.0) * s;
            let fy = p0.1 + (p1.1 - p0.1) * s;
            let bbox = BBox::new(fx - w / 2.0, fy - h, w, h); // 脚点=底边中点
            history.push((f as u64 * STEP_MS, bbox));
        }
    }
    Track {
        id,
        label: label.to_string(),
        history,
    }
}

fn scene() -> Vec<Track> {
    vec![
        // track#1 翻墙者：街面(y=150)翻入院内(y=300)，随后在院里逗留 100 帧。→ 入侵。
        build_track(
            1,
            "person",
            30.0,
            70.0,
            &[
                (0, 30, (300.0, 145.0), (300.0, 300.0)), // 翻入
                (31, 130, (300.0, 300.0), (300.0, 300.0)), // 逗留
            ],
        ),
        // track#2 翻墙者：稍晚从另一处(x=250)翻入并逗留。→ 入侵（cooldown 按 id，不会被 #1 吞掉）。
        build_track(
            2,
            "person",
            30.0,
            70.0,
            &[
                (20, 50, (250.0, 150.0), (250.0, 290.0)),
                (51, 120, (250.0, 290.0), (250.0, 290.0)),
            ],
        ),
        // track#3 保安：本就在院内(y=300)，向上走出院子(y=120)。→ 越线【离开】方向，非报警。
        build_track(3, "person", 30.0, 70.0, &[(0, 40, (400.0, 300.0), (400.0, 120.0))]),
        // track#4 路人：沿街面(y=110)平行走过，从不越线、从不进区。→ 0 报警。
        build_track(4, "person", 30.0, 70.0, &[(0, 40, (100.0, 110.0), (540.0, 110.0))]),
    ]
}

fn main() {
    println!("==== 第 103 章：周界入侵报警（ROI 多边形 + 绊线方向 + cooldown）====");
    let peri = Perimeter {
        poly: vec![(150.0, 180.0), (490.0, 180.0), (520.0, 330.0), (120.0, 330.0)],
        wire_a: (150.0, 180.0),
        wire_b: (490.0, 180.0),
        cd_ms: 10_000, // 冷却 10s
    };
    println!("布防多边形（梯形院子）：{:?}", peri.poly);
    println!(
        "警戒绊线 A→B = {:?}→{:?}（越过它进入院内 = 入侵方向；出院 = 离开）",
        peri.wire_a, peri.wire_b
    );
    let tracks = scene();
    println!("合成 {} 条轨迹（跟踪器已给稳定 id）\n", tracks.len());

    // —— 绊线方向判定（叉积定侧）——
    println!("—— 绊线方向判定（叉积定侧，进院 / 出院）——");
    for t in &tracks {
        let (inward, outward) = count_directions(&peri, t);
        let note = match (inward, outward) {
            (i, _) if i > 0 => "翻墙进院（入侵方向）",
            (_, o) if o > 0 => "走出院子（离开方向，非报警）",
            _ => "未越警戒线（街面路过）",
        };
        println!(
            "  track#{} {}：进院 {} 次 / 出院 {} 次  —— {}",
            t.id, t.label, inward, outward, note
        );
    }

    // —— ROI 命中：脚点落在布防区内的帧数 ——
    println!("\n—— ROI 命中（脚点落在布防区内的帧数）——");
    let mut raw_inside = 0usize;
    for t in &tracks {
        let c = count_inside_frames(&peri, t);
        raw_inside += c;
        println!("  track#{} {}：{} 帧在区内", t.id, t.label, c);
    }
    println!("  合计原始命中帧 = {}", raw_inside);

    // —— 方案 A：仅 ROI + cooldown（不看方向）——
    println!("\n—— 方案 A：仅 ROI + cooldown（不看方向）——");
    let a = alarms_roi_only(&peri, &tracks);
    for e in &a {
        println!("  [{}] t={:>4}ms track#{} {}", e.kind, e.ts_ms, e.track_id.unwrap_or(0), e.message);
    }
    println!("  方案 A 报警数 = {}（把离场保安 track#3 也报了）", a.len());

    // —— 方案 B：两因子（越警戒线进入方向 且 在布防区内）+ cooldown ——
    println!("\n—— 方案 B：两因子（越警戒线【进入方向】且在布防区内）+ cooldown ——");
    let (b, two_factor_frames) = alarms_two_factor(&peri, &tracks);
    println!(
        "  不做 cooldown：每个“入侵后仍在区内”的帧都报 → {} 条",
        two_factor_frames
    );
    println!("  实际报警（按 track_id 冷却 {}s 去重）：", peri.cd_ms / 1000);
    for e in &b {
        println!("  [{}] t={:>4}ms track#{} {}", e.kind, e.ts_ms, e.track_id.unwrap_or(0), e.message);
    }
    println!("  入侵事件数 = {}", b.len());

    // —— 去重 / 过滤前后对比 ——
    println!("\n—— 去重 / 过滤前后对比 ——");
    println!(
        "原始命中帧 {} → 两因子过滤(进入方向且在区内) {} → cooldown 后报警 {} 条",
        raw_inside, two_factor_frames, b.len()
    );
    println!("方案 A（仅 ROI）{} 条  vs  方案 B（两因子）{} 条", a.len(), b.len());
    println!("差异来自：track#3 保安在区内但方向是“离开”，两因子直接排除；track#4 从未进区，ROI 直接排除。");
}
