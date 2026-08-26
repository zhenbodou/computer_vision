//! 第 77 章 视频目标检测与跟踪落地（纯 Rust 最小闭环）
//! 运行：cargo run --example ch77_video_detection
//! 依赖：仅用 std。用合成的逐帧检测结果（Vec<Frame>）跑「检测 + IoU 关联跟踪 + 跨帧 ID」
//!       最小闭环，并对比工程上最关键的一个权衡：**逐帧全检测 vs 关键帧检测+跟踪补帧**。
//!
//! 数据模型 BBox / Detection / Frame 沿用第十八部分的共享心智（见第 83 章 EVENTS_MODEL），
//! 此处按本部分「局部最小自定义」原则重写，不跨部分依赖。跟踪器是第 62 章简化 SORT 的落地版：
//! 用「有限差分速度」估计运动（对变化的关键帧间隔 dt 正确）+ IoU 关联 + 生命周期，
//! 把它接到「检测节奏 K」上，量化省算力与掉精度的真实代价。

#![allow(dead_code)]

use std::collections::HashMap;

// ============ 共享数据模型（心智同第 83 章）============
#[derive(Clone, Copy, Debug)]
struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}
impl BBox {
    fn from_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        BBox {
            x: cx - w / 2.0,
            y: cy - h / 2.0,
            w,
            h,
        }
    }
    fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    fn iou(&self, o: &BBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let uni = self.w * self.h + o.w * o.h - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

/// 一个检测框。gt 是隐藏真值 id（真实检测器没有它），仅用于事后统计 ID 切换与轨迹误差。
#[derive(Clone, Copy)]
struct Detection {
    bbox: BBox,
    gt: u32,
}

/// 一帧的检测结果。真实系统里 detections 来自检测器（YOLO 等，见第 46 章 / 第 94 章）。
struct Frame {
    ts_ms: u64,
    detections: Vec<Detection>,
}

// ============ 确定性伪随机：给检测框加可复现抖动 ============
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn signed(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

// ============ 跟踪器：有限差分速度 + IoU 关联 + 生命周期（简化 SORT，详见第 62 章）============
const IOU_THRESH: f32 = 0.2;
const MAX_AGE: u32 = 10; // 允许连续多少帧没被检测校正（关键帧模式要靠它撑过补帧区间）
const MIN_HITS: u32 = 2; // 命中够这么多次才「确认」、才对外输出稳定 id

struct Track {
    id: u64,
    cx: f32,
    cy: f32,
    vx: f32,
    vy: f32,
    w: f32,
    h: f32,
    last_obs: (f32, f32), // 上一次匹配到的观测中心（算有限差分速度用）
    hits: u32,
    time_since_update: u32, // 距上次匹配上的帧数（即刚 coast 的间隔 dt）
    confirmed: bool,
}
impl Track {
    fn new(id: u64, b: &BBox) -> Self {
        let (cx, cy) = b.center();
        Track {
            id,
            cx,
            cy,
            vx: 0.0,
            vy: 0.0,
            w: b.w,
            h: b.h,
            last_obs: (cx, cy),
            hits: 1,
            time_since_update: 0,
            confirmed: MIN_HITS <= 1,
        }
    }
    /// 预测：按当前速度把中心推进一帧（非关键帧也照走，靠它补帧）。
    fn predict(&mut self) {
        self.cx += self.vx;
        self.cy += self.vy;
        self.time_since_update += 1;
    }
    fn bbox(&self) -> BBox {
        BBox::from_center(self.cx, self.cy, self.w, self.h)
    }
    /// 用一个检测框校正：有限差分估速度（除以真实间隔 dt，对变化的 K 正确），位置贴到观测。
    fn update(&mut self, b: &BBox) {
        let (mx, my) = b.center();
        let dt = self.time_since_update.max(1) as f32; // 刚 coast 了几帧
        let nvx = (mx - self.last_obs.0) / dt;
        let nvy = (my - self.last_obs.1) / dt;
        if self.hits == 1 {
            self.vx = nvx; // 第 2 个观测：直接定速度，避免冷启动欠速
            self.vy = nvy;
        } else {
            self.vx = 0.7 * self.vx + 0.3 * nvx; // 之后轻度平滑
            self.vy = 0.7 * self.vy + 0.3 * nvy;
        }
        self.cx = mx; // 位置贴到观测（检测可信）
        self.cy = my;
        self.w += 0.5 * (b.w - self.w);
        self.h += 0.5 * (b.h - self.h);
        self.last_obs = (mx, my);
        self.hits += 1;
        self.time_since_update = 0;
        if self.hits >= MIN_HITS {
            self.confirmed = true;
        }
    }
}

/// IoU + 贪心关联：返回 (匹配对[(轨迹下标,检测下标)], 未匹配检测下标)。
fn associate(tracks: &[Track], dets: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut cand: Vec<(f32, usize, usize)> = Vec::new();
    for (ti, t) in tracks.iter().enumerate() {
        let tb = t.bbox();
        for (di, d) in dets.iter().enumerate() {
            let iou = tb.iou(&d.bbox);
            if iou >= IOU_THRESH {
                cand.push((iou, ti, di));
            }
        }
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut t_used = vec![false; tracks.len()];
    let mut d_used = vec![false; dets.len()];
    let mut matched = Vec::new();
    for (_, ti, di) in cand {
        if !t_used[ti] && !d_used[di] {
            t_used[ti] = true;
            d_used[di] = true;
            matched.push((ti, di));
        }
    }
    let un_d = (0..dets.len()).filter(|&i| !d_used[i]).collect();
    (matched, un_d)
}

// ============ 合成场景：3 个目标沿「缓弯」运动，含一次短暂漏检 ============
// 缓弯（水平匀速 + 垂直正弦）→ 恒速补帧会在弯上漂移，K 越大漂移越大（真实目标很少匀速直线）。
struct Scene {
    frames: Vec<Frame>,
    gt_center: Vec<HashMap<u32, (f32, f32)>>, // 每帧每个真值目标的真实中心（评估用）
}

fn synth_scene() -> Scene {
    let n = 40;
    let mut rng = Lcg::new(2025);
    let mut frames = Vec::new();
    let mut gt_center = Vec::new();
    for f in 0..n {
        let ff = f as f32;
        // (id, cx, cy)：水平匀速，垂直缓慢正弦摆（不同相位），box 32×56
        let objs = [
            (1u32, 40.0 + 4.0 * ff, 80.0 + 14.0 * (0.22 * ff).sin()),
            (2u32, 360.0 - 4.0 * ff, 150.0 + 14.0 * (0.22 * ff + 2.0).sin()),
            (3u32, 30.0 + 3.6 * ff, 230.0 + 12.0 * (0.20 * ff + 1.0).sin()),
        ];
        let mut dets = Vec::new();
        let mut gt = HashMap::new();
        for &(id, cx, cy) in &objs {
            gt.insert(id, (cx, cy));
            // 目标2 在第 18、19 帧漏检（模拟短暂遮挡）
            if id == 2 && (f == 18 || f == 19) {
                continue;
            }
            let jx = rng.signed() * 1.2; // ±1.2px 检测抖动
            let jy = rng.signed() * 1.2;
            dets.push(Detection {
                bbox: BBox::from_center(cx + jx, cy + jy, 32.0, 56.0),
                gt: id,
            });
        }
        frames.push(Frame {
            ts_ms: 1000 + 40 * f as u64,
            detections: dets,
        });
        gt_center.push(gt);
    }
    Scene { frames, gt_center }
}

// ============ 以「检测节奏 K」跑一遍跟踪，返回评估指标 ============
struct Metrics {
    detector_calls: u32,        // 检测器被调用次数（算力代价代理）
    id_switches: u32,           // 真值目标的 id 变更次数（含断轨后被重建）
    coverage: f32,              // 有确认轨迹覆盖的「真值目标·帧」占比
    mean_traj_err: f32,         // 被覆盖处，跟踪中心与真值中心的平均误差（px）
    final_ids: Vec<(u32, u64)>, // 每个真值目标最终挂在哪个 id 上
}

/// 记录/更新「真值目标 gt 当前挂在哪个 track id」，变了就计一次 id-switch（含断轨重建）。
fn assign(gt_to_tid: &mut HashMap<u32, u64>, gt: u32, tid: u64, sw: &mut u32) {
    if let Some(&prev) = gt_to_tid.get(&gt) {
        if prev != tid {
            *sw += 1;
        }
    }
    gt_to_tid.insert(gt, tid);
}

/// K=1 即逐帧全检测；K>1 即每 K 帧才跑一次检测，其余帧跟踪器只预测（补帧），省算力。
fn run(scene: &Scene, keyframe_k: usize) -> Metrics {
    let mut tracks: Vec<Track> = Vec::new();
    let mut next_id: u64 = 1;
    let mut gt_to_tid: HashMap<u32, u64> = HashMap::new();
    let mut id_switches = 0u32;
    let mut detector_calls = 0u32;
    let (mut covered, mut gt_frames) = (0u32, 0u32);
    let (mut err_sum, mut err_cnt) = (0f32, 0u32);

    for (f, frame) in scene.frames.iter().enumerate() {
        // ① 预测：所有轨迹按当前速度推进一帧
        for t in tracks.iter_mut() {
            t.predict();
        }
        // ② 关键帧才跑检测（第 0 帧必跑以起步）；其余帧只靠上面的 predict() 补帧
        if f % keyframe_k == 0 {
            detector_calls += 1;
            let (matched, un_d) = associate(&tracks, &frame.detections);
            for &(ti, di) in &matched {
                tracks[ti].update(&frame.detections[di].bbox);
                assign(&mut gt_to_tid, frame.detections[di].gt, tracks[ti].id, &mut id_switches);
            }
            for &di in &un_d {
                tracks.push(Track::new(next_id, &frame.detections[di].bbox));
                assign(&mut gt_to_tid, frame.detections[di].gt, next_id, &mut id_switches);
                next_id += 1;
            }
        }
        // ③ 老化删除
        tracks.retain(|t| t.time_since_update <= MAX_AGE);

        // ④ 评估：每个真值目标当前是否有「确认轨迹」覆盖，覆盖处比中心误差
        for (&gt_id, &(gx, gy)) in &scene.gt_center[f] {
            gt_frames += 1;
            if let Some(&tid) = gt_to_tid.get(&gt_id) {
                if let Some(t) = tracks.iter().find(|t| t.id == tid && t.confirmed) {
                    covered += 1;
                    err_sum += ((t.cx - gx).powi(2) + (t.cy - gy).powi(2)).sqrt();
                    err_cnt += 1;
                }
            }
        }
    }

    let mut final_ids: Vec<(u32, u64)> = gt_to_tid.iter().map(|(&g, &t)| (g, t)).collect();
    final_ids.sort();
    Metrics {
        detector_calls,
        id_switches,
        coverage: covered as f32 / gt_frames as f32,
        mean_traj_err: if err_cnt > 0 { err_sum / err_cnt as f32 } else { 0.0 },
        final_ids,
    }
}

fn main() {
    let scene = synth_scene();
    let n_frames = scene.frames.len();
    println!(
        "合成场景：{} 帧，3 个沿缓弯运动的目标（目标2 在第 18、19 帧漏检模拟短暂遮挡）。\n",
        n_frames
    );

    // ---- 逐帧全检测（K=1）：先把最小闭环跑通，打印稳定轨迹与 ID 切换 ----
    println!("== 模式 A：逐帧全检测（K=1）——检测+IoU关联+跨帧ID 最小闭环 ==");
    let m1 = run(&scene, 1);
    for (gt, tid) in &m1.final_ids {
        println!("  真值目标 gt{gt} -> 稳定跟踪 id{tid}");
    }
    println!(
        "  检测器调用 {} 次 | ID 切换 = {}（理想 0）| 覆盖率 {:.0}% | 平均轨迹误差 {:.2}px",
        m1.detector_calls,
        m1.id_switches,
        m1.coverage * 100.0,
        m1.mean_traj_err
    );

    // ---- 关键帧检测 + 跟踪补帧：不同 K 的算力/精度权衡 ----
    println!("\n== 模式 B：关键帧检测 + 跟踪补帧——省算力 vs 掉精度 ==");
    println!(
        "{:<8}{:<10}{:<9}{:<9}{:<12}{:<8}",
        "K", "检测调用", "覆盖率", "ID切换", "平均轨迹误差", "省算力"
    );
    for &k in &[1usize, 2, 4, 6, 10] {
        let m = run(&scene, k);
        let save = 100.0 * (1.0 - m.detector_calls as f32 / n_frames as f32);
        println!(
            "{:<8}{:<10}{:<9}{:<9}{:<12}{:<7.0}%",
            format!("每{k}帧"),
            format!("{}次", m.detector_calls),
            format!("{:.0}%", m.coverage * 100.0),
            m.id_switches,
            format!("{:.2}px", m.mean_traj_err),
            save
        );
    }

    // ---- 实时性预算：把「检测节奏」换算成能否喂饱帧率 ----
    println!("\n== 实时性预算（设检测器单次耗时 D=30ms，跟踪+关联≈1ms/帧）==");
    let d_ms = 30.0f32;
    let track_ms = 1.0f32;
    let budget_ms = 1000.0 / 25.0; // 25fps 每帧预算 40ms
    println!("帧预算（25fps）= {:.0}ms/帧。检测器 D={:.0}ms/次。", budget_ms, d_ms);
    for &k in &[1usize, 2, 4] {
        let amortized = d_ms / k as f32 + track_ms; // 检测摊到 K 帧 + 每帧跟踪
        let max_fps = 1000.0 / amortized;
        let ok = if amortized <= budget_ms { "✓ 喂得饱" } else { "✗ 追不上" };
        println!(
            "  K={k}: 平均 {:.1}ms/帧（检测摊薄 {:.1} + 跟踪 {:.0}）-> 理论上限 {:.0}fps  {}",
            amortized,
            d_ms / k as f32,
            track_ms,
            max_fps,
            ok
        );
    }

    println!("\n结论：K 越大越省算力（检测调用线性下降），但恒速补帧在缓弯上漂移、轨迹误差随 K 上升；");
    println!("K 过大到「一次都接不上」时会断轨、重建轨迹 → 覆盖率骤降、ID 切换飙升。中庸的 K 才是甜点。");
}
