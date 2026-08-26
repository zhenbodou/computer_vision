//! 第 89 章 轨迹与行为分析：速度、方向、逆行、徘徊、热力图
//! 运行：cargo run --example ch89_trajectory
//! 依赖：核心逻辑仅用 std；image 0.25 只用来把热力图存成 PNG（主工程已有依赖）。
//!
//! 事件层数据模型见第 83 章。这里原样沿用 BBox / Track / Event，
//! 用合成轨迹（递增 ts_ms，不用系统时钟）做行为分析，自包含可复现。

use std::f32::consts::PI;

// ============ 第 83 章定义的共享数据模型（此处原样沿用其中三个） ============

#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// 底边中点（脚点）——判过线/入区域/走向都用这个而非中心
    fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
}

#[derive(Clone, Debug)]
struct Track {
    id: u64,
    label: String,
    history: Vec<(u64, BBox)>, // (ts_ms, bbox)，按时间递增
}

#[derive(Clone, Debug)]
struct Event {
    kind: String,
    ts_ms: u64,
    track_id: Option<u64>,
    message: String,
}

// ============ 合成数据：std 内无随机数，自带一个确定性 LCG 造可复现抖动 ============

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 返回 [-1, 1) 的伪随机数，用来给检测框加抖动
    fn signed(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

/// 由一串脚点(底边中点)造一条轨迹：固定框大小 w=20,h=40，反推左上角。
fn track_from_feet(id: u64, label: &str, feet: &[(f32, f32)], dt_ms: u64) -> Track {
    let (w, h) = (20.0f32, 40.0f32);
    let history = feet
        .iter()
        .enumerate()
        .map(|(i, &(fx, fy))| {
            let ts = i as u64 * dt_ms;
            (ts, BBox::new(fx - w / 2.0, fy - h, w, h))
        })
        .collect();
    Track {
        id,
        label: label.to_string(),
        history,
    }
}

// ============ 行为分析：全部基于 Track.history 的脚点序列 ============

/// 相邻两点的瞬时速度序列（像素/秒）
fn seg_speeds(t: &Track) -> Vec<f32> {
    t.history
        .windows(2)
        .map(|w| {
            let (t0, b0) = w[0];
            let (t1, b1) = w[1];
            let (x0, y0) = b0.bottom_center();
            let (x1, y1) = b1.bottom_center();
            let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
            let dt = (t1 - t0) as f32 / 1000.0; // 毫秒 -> 秒
            d / dt
        })
        .collect()
}

/// 轨迹总路程（把每一小段长度加起来）
fn path_length(t: &Track) -> f32 {
    t.history
        .windows(2)
        .map(|w| {
            let (x0, y0) = w[0].1.bottom_center();
            let (x1, y1) = w[1].1.bottom_center();
            ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
        })
        .sum()
}

/// 净位移：首点到末点的直线距离
fn net_disp(t: &Track) -> f32 {
    let (x0, y0) = t.history.first().unwrap().1.bottom_center();
    let (x1, y1) = t.history.last().unwrap().1.bottom_center();
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

fn duration_ms(t: &Track) -> u64 {
    t.history.last().unwrap().0 - t.history.first().unwrap().0
}

/// 净位移方向角（度）。图像坐标 y 向下：向右=0°，向下=+90°，向左=±180°，向上=-90°。
fn net_direction_deg(t: &Track) -> f32 {
    let (x0, y0) = t.history.first().unwrap().1.bottom_center();
    let (x1, y1) = t.history.last().unwrap().1.bottom_center();
    (y1 - y0).atan2(x1 - x0) * 180.0 / PI
}

/// 两角之差，归一化到 [-180,180]
fn ang_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d < -180.0 {
        d += 360.0;
    }
    d
}

/// 平滑方向：把一个窗口内的位移向量相加再取角度（向量平均，天然避开角度绕环问题）
fn smoothed_dir_deg(t: &Track, i: usize, win: usize) -> f32 {
    let n = t.history.len();
    let lo = i.saturating_sub(win);
    let hi = (i + win).min(n - 1);
    let (x0, y0) = t.history[lo].1.bottom_center();
    let (x1, y1) = t.history[hi].1.bottom_center();
    (y1 - y0).atan2(x1 - x0) * 180.0 / PI
}

fn main() {
    // ---------- 合成三条不同行为的轨迹（脚点，帧间隔 100ms=10fps） ----------
    let dt = 100u64;

    // ① 正常通过：从左向右匀速走，带 ±2px 检测抖动
    let mut j1 = Lcg::new(1);
    let feet1: Vec<(f32, f32)> = (0..21)
        .map(|i| {
            (
                20.0 + 12.0 * i as f32 + 2.0 * j1.signed(),
                120.0 + 2.0 * j1.signed(),
            )
        })
        .collect();

    // ② 逆行：从右向左（规定顺行方向是向右），带抖动
    let mut j2 = Lcg::new(7);
    let feet2: Vec<(f32, f32)> = (0..21)
        .map(|i| {
            (
                300.0 - 12.0 * i as f32 + 2.0 * j2.signed(),
                90.0 + 2.0 * j2.signed(),
            )
        })
        .collect();

    // ③ 徘徊：在 (150,150) 附近来回打转，走得久但没走远
    let feet3: Vec<(f32, f32)> = (0..40)
        .map(|i| {
            let a = i as f32 * 0.5;
            (150.0 + 8.0 * a.sin(), 150.0 + 8.0 * (a * 0.7).cos())
        })
        .collect();

    let tracks = vec![
        track_from_feet(1, "person", &feet1, dt),
        track_from_feet(2, "person", &feet2, dt),
        track_from_feet(3, "person", &feet3, dt),
    ];

    // 规定顺行方向：向右 = 0°；判逆行需目标确实在“赶路”（净速度够大）
    let designated_deg = 0.0f32;
    let travel_speed_min = 40.0f32; // 净速度阈值（像素/秒）
    let wrong_way_tol = 120.0f32; // 与规定方向夹角超过它就算逆行
    let dwell_min_ms = 3000u64; // 徘徊：停留时长门槛
    let net_disp_max = 40.0f32; // 徘徊：净位移上限（走来走去但没走远）

    let mut events: Vec<Event> = Vec::new();

    // ---------- 速度抖动与平滑：拿正常轨迹举例 ----------
    let sp1 = seg_speeds(&tracks[0]);
    let raw_min = sp1.iter().cloned().fold(f32::INFINITY, f32::min);
    let raw_max = sp1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("== 速度与抖动（轨迹 1 正常通过）==");
    print!("前 6 段瞬时速度(px/s): ");
    for s in sp1.iter().take(6) {
        print!("{s:.0} ");
    }
    println!();
    println!("瞬时速度范围: {raw_min:.0} ~ {raw_max:.0} px/s（受检测框抖动影响，跳动明显）");
    let avg_path = path_length(&tracks[0]) / (duration_ms(&tracks[0]) as f32 / 1000.0);
    println!("按总路程算的平均速度: {avg_path:.1} px/s（平滑后稳定得多）\n");

    // ---------- 方向平滑：瞬时方向 vs 窗口平滑方向 ----------
    println!("== 方向与平滑（轨迹 1）==");
    print!("逐段瞬时方向(度): ");
    for w in tracks[0].history.windows(2).take(6) {
        let (x0, y0) = w[0].1.bottom_center();
        let (x1, y1) = w[1].1.bottom_center();
        print!("{:.0} ", (y1 - y0).atan2(x1 - x0) * 180.0 / PI);
    }
    println!("（慢速/抖动时逐帧方向很不稳）");
    println!(
        "窗口(±3帧)平滑方向@中点: {:.1} 度；整段净方向: {:.1} 度\n",
        smoothed_dir_deg(&tracks[0], 10, 3),
        net_direction_deg(&tracks[0])
    );

    // ---------- 逐条轨迹：算指标 + 判行为 ----------
    println!("== 逐条轨迹的行为判定 ==");
    for t in &tracks {
        let dur = duration_ms(t);
        let path = path_length(t);
        let net = net_disp(t);
        let dir = net_direction_deg(t);
        let net_vel = net / (dur as f32 / 1000.0); // 净速度（真正的“赶路”速度）
        let wander = path / net.max(1.0); // 徘徊比：来回打转时远大于 1

        // 逆行：确实在赶路 + 净方向与规定方向夹角过大
        let is_wrong_way =
            net_vel > travel_speed_min && ang_diff(dir, designated_deg).abs() > wrong_way_tol;
        // 徘徊：停留够久 + 没走远
        let is_loiter = dur >= dwell_min_ms && net < net_disp_max;

        let verdict = if is_wrong_way {
            "逆行!"
        } else if is_loiter {
            "徘徊!"
        } else {
            "正常"
        };

        println!(
            "轨迹#{} {:<7} 时长{:>4}ms 路程{:>5.0} 净位移{:>5.0} 净方向{:>6.1}° 净速度{:>5.0} 徘徊比{:>4.1} => {}",
            t.id, t.label, dur, path, net, dir, net_vel, wander, verdict
        );

        let last_ts = t.history.last().unwrap().0;
        if is_wrong_way {
            events.push(Event {
                kind: "wrong_way".into(),
                ts_ms: last_ts,
                track_id: Some(t.id),
                message: format!(
                    "目标 {} 逆行：净方向 {:.0}° 与规定 {:.0}° 相反",
                    t.id, dir, designated_deg
                ),
            });
        }
        if is_loiter {
            events.push(Event {
                kind: "loiter".into(),
                ts_ms: last_ts,
                track_id: Some(t.id),
                message: format!(
                    "目标 {} 徘徊：{:.1}s 内净位移仅 {:.0}px",
                    t.id,
                    dur as f32 / 1000.0,
                    net
                ),
            });
        }
    }

    // ---------- 热力图：把所有脚点累加到二维网格 ----------
    const W: u32 = 320;
    const H: u32 = 180;
    const CELL: u32 = 20; // 每格 20px -> 16x9 网格
    let gw = (W / CELL) as usize;
    let gh = (H / CELL) as usize;
    let mut grid = vec![vec![0u32; gw]; gh];
    for t in &tracks {
        for (_, b) in &t.history {
            let (fx, fy) = b.bottom_center();
            let cx = (fx.max(0.0).min((W - 1) as f32) as u32 / CELL) as usize;
            let cy = (fy.max(0.0).min((H - 1) as f32) as u32 / CELL) as usize;
            grid[cy][cx] += 1;
        }
    }
    // 找峰值格子
    let (mut px, mut py, mut pv) = (0usize, 0usize, 0u32);
    for y in 0..gh {
        for x in 0..gw {
            if grid[y][x] > pv {
                px = x;
                py = y;
                pv = grid[y][x];
            }
        }
    }
    println!("\n== 热力图 ({}x{} 网格) ==", gw, gh);
    println!(
        "峰值格子: (列{}, 行{})，累计 {} 个点，对应画面像素中心 ≈ ({}, {})",
        px,
        py,
        pv,
        px as u32 * CELL + CELL / 2,
        py as u32 * CELL + CELL / 2
    );

    // 导出成灰度 PNG（每格放大成 CELL×CELL 的色块，按峰值归一化）
    let maxv = pv.max(1);
    let mut heat = image::GrayImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let gx = (x / CELL) as usize;
            let gy = (y / CELL) as usize;
            let v = (grid[gy][gx] * 255 / maxv) as u8;
            heat.put_pixel(x, y, image::Luma([v]));
        }
    }
    heat.save("out_ch89_heatmap.png").unwrap();
    println!("热力图已存 out_ch89_heatmap.png（越白越热）");

    // ---------- 汇总产生的事件 ----------
    println!("\n== 产生的事件 ({} 条) ==", events.len());
    for e in &events {
        println!("[{}] @{}ms track={:?} {}", e.kind, e.ts_ms, e.track_id, e.message);
    }
}
