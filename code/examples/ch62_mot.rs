//! 第 62 章 多目标跟踪（SORT 简化版）
//! 运行：cargo run --example ch62_mot
//! 依赖：仅用 std。合成多帧、多目标、含交叉与偶发漏检的检测框序列，
//!       确定性可复现（LCG 造抖动，不用系统时钟）。
//!
//! 跟踪相关类型在本部分「局部最小自定义」：框用 (x,y,w,h) f32，IoU 自己写十来行，
//! 不依赖第十八部分 ch83 的数据模型。轨迹将在第十八部分被业务规则消费。
//!
//! SORT 三件套：
//!   ① 恒速模型（alpha-beta 稳态卡尔曼）预测每条轨迹下一帧位置（思想见第 60 章）
//!   ② IoU 代价 + 贪心分配做「检测↔轨迹」关联（匈牙利算法更优，见正文）
//!   ③ 轨迹生命周期：新建 / 命中确认 min_hits / 失配老化 max_age / 删除

use std::collections::HashMap;

// ============ 最小包围盒 + IoU ============

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
    /// 由中心点造框，方便描述目标运动
    fn from_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        BBox::new(cx - w / 2.0, cy - h / 2.0, w, h)
    }
    fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    /// 交并比 IoU ∈ [0,1]：交集面积 / 并集面积
    fn iou(&self, o: &BBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let iw = (x2 - x1).max(0.0);
        let ih = (y2 - y1).max(0.0);
        let inter = iw * ih;
        let uni = self.w * self.h + o.w * o.h - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

/// 合成的「检测」：跟踪器只能看到 bbox；gt 是隐藏真值 id，仅用于事后统计 id-switch
#[derive(Clone, Copy)]
struct Det {
    bbox: BBox,
    gt: u32,
}

// ============ 确定性伪随机（给检测框加可复现抖动） ============

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 返回 [-1,1) 的伪随机数
    fn signed(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

// ============ 一条轨迹：恒速模型 + alpha-beta 稳态卡尔曼 ============

const IOU_THRESH: f32 = 0.3; // 允许匹配的最小 IoU
const MAX_AGE: u32 = 3; // 失配超过这么多帧就删除
const MIN_HITS: u32 = 3; // 命中够这么多次才「确认」（正式对外输出 id）
const ALPHA: f32 = 0.6; // 位置修正增益
const BETA: f32 = 0.25; // 速度修正增益

struct Track {
    id: u64,
    // 状态：中心位置 (cx,cy) + 速度 (vx,vy)；宽高单独平滑
    cx: f32,
    cy: f32,
    vx: f32,
    vy: f32,
    w: f32,
    h: f32,
    hits: u32,              // 累计命中次数
    time_since_update: u32, // 距上次匹配上的帧数（0 表示本帧刚更新）
    confirmed: bool,
    first_frame: usize,
    last_frame: usize,
}

impl Track {
    fn new(id: u64, b: &BBox, frame: usize) -> Self {
        let (cx, cy) = b.center();
        Track {
            id,
            cx,
            cy,
            vx: 0.0,
            vy: 0.0,
            w: b.w,
            h: b.h,
            hits: 1,
            time_since_update: 0,
            confirmed: MIN_HITS <= 1,
            first_frame: frame,
            last_frame: frame,
        }
    }
    /// 预测：按恒速把中心推进一步；失配计数 +1
    fn predict(&mut self) {
        self.cx += self.vx;
        self.cy += self.vy;
        self.time_since_update += 1;
    }
    /// 当前（预测后的）估计框
    fn bbox(&self) -> BBox {
        BBox::from_center(self.cx, self.cy, self.w, self.h)
    }
    /// 用一个检测框校正状态（alpha-beta：用残差同时修正位置与速度）
    fn update(&mut self, b: &BBox, frame: usize) {
        let (mx, my) = b.center();
        let rx = mx - self.cx; // 残差 = 观测 − 预测
        let ry = my - self.cy;
        self.cx += ALPHA * rx;
        self.cy += ALPHA * ry;
        self.vx += BETA * rx; // 帧间隔 dt=1，速度增益直接乘残差
        self.vy += BETA * ry;
        self.w += 0.5 * (b.w - self.w); // 宽高做简单指数平滑
        self.h += 0.5 * (b.h - self.h);
        self.hits += 1;
        self.time_since_update = 0;
        self.last_frame = frame;
        if self.hits >= MIN_HITS {
            self.confirmed = true;
        }
    }
}

// ============ 关联：IoU 代价矩阵 + 贪心分配 ============

/// 返回 (匹配对[(轨迹下标,检测下标)], 未匹配轨迹下标, 未匹配检测下标)。
/// 贪心：把所有 IoU≥阈值 的候选对按 IoU 从大到小排序，依次占用。
/// 匈牙利算法能求「全局总代价最优」的分配，贪心是它的近似（见正文对比）。
fn associate(tracks: &[Track], dets: &[Det]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
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
    let un_t = (0..tracks.len()).filter(|&i| !t_used[i]).collect();
    let un_d = (0..dets.len()).filter(|&i| !d_used[i]).collect();
    (matched, un_t, un_d)
}

// ============ 合成检测序列 ============

/// 30 帧，三个目标：
///  gt1 中间车道左→右，gt2 下方车道右→左（两者在中途「交叉」而过，车道相差 24px）
///  gt3 顶部车道左→右。gt1 在第 8、9 帧漏检（模拟短暂遮挡），gt3 在第 20 帧漏检。
fn synth_frames() -> Vec<Vec<Det>> {
    let n_frames = 30;
    let mut rng = Lcg::new(2024);
    let mut jit = |a: f32| a * rng.signed() * 1.5; // ±1.5px 抖动
    let mut frames = Vec::new();
    for f in 0..n_frames {
        let ff = f as f32;
        let d1 = BBox::from_center(44.0 + 8.0 * ff + jit(1.0), 76.0 + jit(1.0), 28.0, 54.0);
        let d2 = BBox::from_center(276.0 - 8.0 * ff + jit(1.0), 100.0 + jit(1.0), 28.0, 54.0);
        let d3 = BBox::from_center(60.0 + 7.0 * ff + jit(1.0), 30.0 + jit(1.0), 24.0, 40.0);
        let mut dets = Vec::new();
        if f != 8 && f != 9 {
            dets.push(Det { bbox: d1, gt: 1 });
        }
        dets.push(Det { bbox: d2, gt: 2 });
        if f != 20 {
            dets.push(Det { bbox: d3, gt: 3 });
        }
        frames.push(dets);
    }
    frames
}

fn main() {
    let frames = synth_frames();
    let mut tracks: Vec<Track> = Vec::new();
    let mut next_id: u64 = 1;

    // id-switch 统计：记录每个真值目标「上一次」被分配到的 track id，变了就 +1
    let mut gt_to_tid: HashMap<u32, u64> = HashMap::new();
    let mut gt_len: HashMap<u32, u32> = HashMap::new(); // 每个目标被成功跟踪的帧数
    let mut idsw = 0u32;

    println!("== 逐帧 id 分配（gtN=真值目标，idN=跟踪器分配的 id）==");
    for (f, dets) in frames.iter().enumerate() {
        // ① 预测：把每条轨迹按恒速推进一步
        for t in tracks.iter_mut() {
            t.predict();
        }
        // ② 关联：IoU + 贪心
        let (matched, _un_t, un_d) = associate(&tracks, dets);

        // 记录每个检测最终拿到的 id（用于打印）
        let mut det_id = vec![0u64; dets.len()];

        // ③ 更新匹配上的轨迹 + 统计 id-switch
        for &(ti, di) in &matched {
            tracks[ti].update(&dets[di].bbox, f);
            let (gt, tid) = (dets[di].gt, tracks[ti].id);
            det_id[di] = tid;
            if let Some(&prev) = gt_to_tid.get(&gt) {
                if prev != tid {
                    idsw += 1;
                    println!("  ⚠️ 帧{f:>2} 目标 gt{gt} 的 id 从 {prev} 跳成 {tid}（id-switch）");
                }
            }
            gt_to_tid.insert(gt, tid);
            *gt_len.entry(gt).or_insert(0) += 1;
        }
        // ④ 新建：未匹配的检测各起一条新轨迹
        for &di in &un_d {
            let t = Track::new(next_id, &dets[di].bbox, f);
            det_id[di] = next_id;
            gt_to_tid.insert(dets[di].gt, next_id);
            *gt_len.entry(dets[di].gt).or_insert(0) += 1;
            tracks.push(t);
            next_id += 1;
        }
        // ⑤ 老化：失配超过 MAX_AGE 的轨迹删除
        let before = tracks.len();
        tracks.retain(|t| t.time_since_update <= MAX_AGE);
        let deleted = before - tracks.len();

        // 打印本帧：检测→id 映射 + 当前确认轨迹（含靠预测「续命」的）
        let map: Vec<String> = dets
            .iter()
            .enumerate()
            .map(|(i, d)| format!("gt{}->id{}", d.gt, det_id[i]))
            .collect();
        let live: Vec<String> = tracks
            .iter()
            .filter(|t| t.confirmed)
            .map(|t| {
                let coast = if t.time_since_update > 0 { "*" } else { "" };
                format!("id{}@({:.0},{:.0}){}", t.id, t.cx, t.cy, coast)
            })
            .collect();
        let extra = if deleted > 0 {
            format!("  [删除{deleted}条]")
        } else {
            String::new()
        };
        println!(
            "帧{f:>2} | 检测{}: {:<28} | 确认轨迹: {}{}",
            dets.len(),
            map.join(" "),
            live.join(" "),
            extra
        );
    }

    // ---------- 汇总 ----------
    println!("\n== 汇总 ==");
    println!("总共创建轨迹（分配过的 id 数）: {}", next_id - 1);
    println!("存活到最后的轨迹: {}", tracks.len());
    let mut gts: Vec<_> = gt_len.keys().copied().collect();
    gts.sort();
    for gt in gts {
        // 该目标现在挂在哪个 id 上、跟踪了多少帧
        let tid = gt_to_tid[&gt];
        println!("目标 gt{gt}: 最终 id={tid}，被成功跟踪 {} 帧", gt_len[&gt]);
    }
    println!("\n各轨迹寿命（命中次数 / 首末帧）:");
    let mut ts: Vec<&Track> = tracks.iter().collect();
    ts.sort_by_key(|t| t.id);
    for t in ts {
        println!(
            "  id{}: 命中 {} 次，帧 {}..={}（跨度 {} 帧）",
            t.id,
            t.hits,
            t.first_frame,
            t.last_frame,
            t.last_frame - t.first_frame + 1
        );
    }
    println!("\nid-switch 次数 = {idsw}（理想为 0）");
    println!("注: 标 * 的轨迹是当帧漏检、靠恒速预测续命的（如 gt1 在第 8、9 帧被遮挡）。");
}
