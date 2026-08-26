//! 第 88 章 区域入侵、停留（dwell）与目标计数
//! 运行：cargo run --example ch88_intrusion_dwell
//! 依赖：无（只用 std）——用合成的、带 track_id 的轨迹（Vec<Track>）当输入，自包含、可复现。
//!
//! BBox / Track / Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md）；
//! 判「脚点是否在 ROI 多边形内」用第 84 章的射线法 point_in_polygon。
//! 三条规则：① 入侵（进入沿触发）② 停留 dwell（在区内持续 ≥ T 秒）③ 区内实时计数。

#![allow(dead_code)]

use std::collections::BTreeMap;

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
    /// 底边中点（脚点）——判「进没进区域」用脚点而非中心（见第 84 章）。
    pub fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
}

/// 跟踪后的目标（同第 83 章）：跟踪器（第 62 章）给每个目标分配稳定 track_id。
/// dwell「基于 track 历史算时长」——就是扫 history 里这段 (ts_ms, bbox) 求进/出区的时间差。
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub label: String,
    pub history: Vec<(u64, BBox)>, // (ts_ms, bbox) 按时间递增
}

/// 事件层的输出：一条业务事件 / 告警（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String, // "intrusion" | "dwell" | "crowd"
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 射线法（ray casting）判断点是否在多边形内——原样取自第 84 章。
/// 半开区间约定 (yi > py) != (yj > py)，天然处理凹多边形与顶点退化。
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

/// 目标脚点是否落在 ROI 内。
fn foot_in_roi(b: &BBox, roi: &[(f32, f32)]) -> bool {
    let (fx, fy) = b.bottom_center();
    point_in_polygon(fx, fy, roi)
}

/// 把一条 track 的历史切成若干次「在 ROI 内的停留」visit：(进入 ts, 最后在区内 ts)。
/// 相邻两个「在区内」的采样点，只要时间间隔 ≤ gap_ms，就算同一次停留（gap 容忍：
/// **短暂离开又回来、或几帧漏检，不该把 dwell 计时清零**）。
/// gap_ms 传帧间隔 dt 就相当于「不容忍」（缺一帧即断开），传大值即容忍。
fn roi_visits(t: &Track, roi: &[(f32, f32)], gap_ms: u64) -> Vec<(u64, u64)> {
    let mut visits = Vec::new();
    let mut cur: Option<(u64, u64)> = None; // (enter_ts, last_in_ts)
    for &(ts, b) in &t.history {
        if !foot_in_roi(&b, roi) {
            continue; // 出区的采样点不参与，间隔按“区内点之间”的时间差来算
        }
        match cur {
            Some((enter, last)) if ts.saturating_sub(last) <= gap_ms => {
                cur = Some((enter, ts)); // 间隔够小 → 延续本次停留
            }
            _ => {
                if let Some(v) = cur {
                    visits.push(v); // 收尾上一次
                }
                cur = Some((ts, ts)); // 开一次新的停留
            }
        }
    }
    if let Some(v) = cur {
        visits.push(v);
    }
    visits
}

/// 从 Vec<Track> 跑出全部事件（入侵 + 停留），并返回。dwell_ms 为停留阈值 T。
fn detect_events(tracks: &[Track], roi: &[(f32, f32)], dwell_ms: u64, gap_ms: u64) -> Vec<Event> {
    let mut events = Vec::new();
    for t in tracks {
        for (enter, last) in roi_visits(t, roi, gap_ms) {
            // ① 入侵：每次「进入沿」（新的一次停留开始）报一次
            events.push(Event {
                kind: "intrusion".to_string(),
                ts_ms: enter,
                track_id: Some(t.id),
                message: format!("track#{} 进入 ROI", t.id),
            });
            // ② 停留：本次停留时长 = last - enter；≥ T 才告警，告警“发生”在跨过阈值那一刻
            let dwell = last - enter;
            if dwell >= dwell_ms {
                let fire_ts = enter + dwell_ms; // 停留计时第一次达到 T 的时刻
                events.push(Event {
                    kind: "dwell".to_string(),
                    ts_ms: fire_ts,
                    track_id: Some(t.id),
                    message: format!(
                        "track#{} 停留超时：本次共停留 {:.1}s（≥ 阈值 {:.0}s）",
                        t.id,
                        dwell as f64 / 1000.0,
                        dwell_ms as f64 / 1000.0
                    ),
                });
            }
        }
    }
    events.sort_by_key(|e| e.ts_ms);
    events
}

/// 造一条 track：给定「present 的帧索引 → 脚点」列表，dt_ms 帧间隔，反推左上角。
/// 固定框 w=30,h=70。时间戳 ts = frame_idx * dt_ms + base_ms（合成递增值，不读系统时钟）。
fn track_from_frames(id: u64, label: &str, frames: &[(usize, (f32, f32))], dt_ms: u64, base_ms: u64) -> Track {
    let (w, h) = (30.0f32, 70.0f32);
    let history = frames
        .iter()
        .map(|&(fi, (fx, fy))| (base_ms + fi as u64 * dt_ms, BBox::new(fx - w / 2.0, fy - h, w, h)))
        .collect();
    Track { id, label: label.to_string(), history }
}

/// 小工具：把「帧区间 [s,e] 内每帧同一脚点」展开成 (帧号, 脚点) 列表，可挖掉 skip 里的帧（模拟漏检 / 短暂离开）。
fn span(s: usize, e: usize, foot: (f32, f32), skip: &[usize]) -> Vec<(usize, (f32, f32))> {
    (s..=e).filter(|i| !skip.contains(i)).map(|i| (i, foot)).collect()
}

fn main() {
    let dt = 500u64; // 2 fps，帧间隔 500ms
    let base = 1000u64; // 起始时间戳（合成值）
    let dwell_t = 30_000u64; // 停留阈值 T = 30 秒
    let gap_tol = 3_000u64; // gap 容忍：短暂离开 ≤ 3s 不清零 dwell 计时
    let crowd_n = 3u32; // 区内目标数 ≥ 3 触发“拥挤/排队超限”

    // ROI：保险柜 / 排队区，一块矩形（用多边形表示，射线法通用）。x∈(200,440), y∈(150,330)。
    let roi: Vec<(f32, f32)> = vec![(200.0, 150.0), (440.0, 150.0), (440.0, 330.0), (200.0, 330.0)];

    // ---------- 合成 5 条带 track_id 的轨迹 ----------
    // track#7：在保险柜前长时间逗留（帧 4..69），中途帧 30、31 短暂漏检（人被遮了一下）。
    let t7 = track_from_frames(7, "person", &span(4, 69, (300.0, 300.0), &[30, 31]), dt, base);
    // track#1：先在区外（帧 9），再进区快速穿过（帧 10..16）——演示“区外→区内”的进入沿。
    let mut f1 = vec![(9usize, (150.0, 300.0))]; // 区外（x=150<200）
    f1.extend(span(10, 16, (250.0, 280.0), &[]));
    let t1 = track_from_frames(1, "person", &f1, dt, base);
    // track#8：中等停留（帧 20..59，共 19.5s，卡在阈值以下，用来和 T 做区分）。
    let t8 = track_from_frames(8, "person", &span(20, 59, (380.0, 300.0), &[]), dt, base);
    // track#2：短暂经过（帧 40..52），把区内人数在这段时间顶到峰值。
    let t2 = track_from_frames(2, "person", &span(40, 52, (330.0, 250.0), &[]), dt, base);
    // track#3：全程在区外走动（x=120），一条负样本——不该产生任何事件、也不该被计数。
    let t3 = track_from_frames(3, "person", &span(0, 79, (120.0, 300.0), &[]), dt, base);

    let tracks = vec![t7, t1, t8, t2, t3];

    println!("ROI = 矩形 x∈(200,440) y∈(150,330)；停留阈值 T={}s；gap 容忍={}s；拥挤阈值={} 人\n", dwell_t / 1000, gap_tol / 1000, crowd_n);

    // ---------- 规则 ①②：入侵 + 停留（gap 容忍开启） ----------
    let events = detect_events(&tracks, &roi, dwell_t, gap_tol);
    println!("== ①② 入侵 & 停留事件（gap 容忍 {}s）==", gap_tol / 1000);
    for e in &events {
        println!("  [{:<9}] t={:>5.1}s track={:?}  {}", e.kind, e.ts_ms as f64 / 1000.0, e.track_id.unwrap(), e.message);
    }
    let n_intr = events.iter().filter(|e| e.kind == "intrusion").count();
    let n_dwell = events.iter().filter(|e| e.kind == "dwell").count();
    println!("  小计：入侵 {} 次，停留告警 {} 次", n_intr, n_dwell);

    // ---------- 逐条 track 的停留时长小结（基于 track.history 算） ----------
    println!("\n== 各 track 在 ROI 内的停留时长（基于 history）==");
    for t in &tracks {
        let visits = roi_visits(t, &roi, gap_tol);
        if visits.is_empty() {
            println!("  track#{} {}: 从未进入 ROI", t.id, t.label);
            continue;
        }
        for (enter, last) in &visits {
            let dwell = (last - enter) as f64 / 1000.0;
            let flag = if (last - enter) >= dwell_t { " → 超过阈值，触发 dwell 告警 ✔" } else { "" };
            println!("  track#{} {}: 进入 t={:.1}s，停留 {:.1}s{}", t.id, t.label, *enter as f64 / 1000.0, dwell, flag);
        }
    }

    // ---------- 坑演示：关掉 gap 容忍（gap=dt，缺一帧即断），track#7 的停留被“打断” ----------
    let events_strict = detect_events(&tracks, &roi, dwell_t, dt);
    let strict_dwell = events_strict.iter().filter(|e| e.kind == "dwell").count();
    let strict_intr_t7 = events_strict.iter().filter(|e| e.kind == "intrusion" && e.track_id == Some(7)).count();
    println!("\n== 坑：gap 容忍的重要性（track#7 中途漏检 2 帧）==");
    println!("  开启 gap 容忍({}s)：track#7 视作一次停留 32.5s → 停留告警 {} 次，入侵 1 次", gap_tol / 1000, n_dwell);
    println!("  关闭 gap 容忍     ：漏检把停留切成两段(12.5s+18.5s)，都不够 30s → 停留告警 {} 次，track#7 入侵被拆成 {} 次", strict_dwell, strict_intr_t7);

    // ---------- 规则 ③：区内实时目标计数（每个时间戳数一次） ----------
    // 按 ts 汇总「这一时刻有几个不同 track 的脚点在 ROI 内」。
    let mut count_at: BTreeMap<u64, u32> = BTreeMap::new();
    for i in 0..80u64 {
        count_at.insert(base + i * dt, 0); // 先把每一帧都补上 0，方便看清人数升降
    }
    for t in &tracks {
        for &(ts, b) in &t.history {
            if foot_in_roi(&b, &roi) {
                *count_at.get_mut(&ts).unwrap() += 1;
            }
        }
    }
    let peak = count_at.values().copied().max().unwrap_or(0);
    println!("\n== ③ 区内实时人数（峰值 {} 人）==", peak);
    print!("  人数变化时间线：");
    let mut prev = u32::MAX;
    for (&ts, &c) in &count_at {
        if c != prev {
            print!("{:.1}s:{}人  ", ts as f64 / 1000.0, c);
            prev = c;
        }
    }
    println!();

    // 拥挤告警：人数第一次达到阈值时报一次（上升沿）。
    let mut crowd_events = Vec::new();
    let mut prev_c = 0u32;
    for (&ts, &c) in &count_at {
        if c >= crowd_n && prev_c < crowd_n {
            crowd_events.push(Event {
                kind: "crowd".to_string(),
                ts_ms: ts,
                track_id: None,
                message: format!("区内目标数达到 {} 人（≥ 阈值 {}）", c, crowd_n),
            });
        }
        prev_c = c;
    }
    println!("  拥挤告警（人数 ≥ {}）：", crowd_n);
    for e in &crowd_events {
        println!("    [crowd] t={:.1}s  {}", e.ts_ms as f64 / 1000.0, e.message);
    }
}
