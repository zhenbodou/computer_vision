# 第 89 章 轨迹与行为分析（速度、方向、逆行、越界、热力图）

> **本章导读**
> - 学完你能：把跟踪器给的一串轨迹点，变成**速度、运动方向、逆行、徘徊**这些能直接报警的"行为"。
> - 亲手实现：相邻点算速度、位移向量定方向、多帧平均去抖、逆行/徘徊判定，以及把成千上万个轨迹点烧成一张**热力图**。
> - 全程只用 `std`（热力图存 PNG 时才用一下 `image`），输入是**合成轨迹**（自己造递增 `ts_ms`，不看系统时钟），跑一次结果完全可复现。

## 场景：光有框、有 id，还差最后一层"看懂"

前面几章我们把检测框越洗越干净：置信度过滤、ROI、时间窗投票、cooldown、绊线、停留。到 [第 88 章](ch88-intrusion-dwell.md) 结束时，画面里每个目标都已经有了稳定的 `track_id`，我们也能算它"在某个区域待了多久"。

可是真实项目里，甲方还会提一堆更"行为化"的需求：

- "这条单行道，有没有车**逆行**？"
- "出口闸机，有没有人**往里走**（逆流进人）？"
- "这个墙角，有没有人**长时间徘徊**、鬼鬼祟祟？"
- "一整天下来，画面里**哪块区域最'热'**（人来人往最密）？"

这些都不是"某一帧里有没有框"能回答的——它们关乎目标**怎么动的**：多快、朝哪、走没走远、来回打转没有。而"目标怎么动"这件事，全都写在跟踪器交给我们的那串轨迹点里。这一章，我们就把这串点榨干。

## 一切从 Track.history 说起

回忆 [第 83 章](ch83-filtering-basics.md) 定义的 `Track`：跟踪器（[第 62 章](../p13-tracking/ch62-mot.md) 的 SORT/DeepSORT 一类）给每个目标分配一个稳定 `id`，并把它历次出现的位置记进 `history`——一串按时间递增的 `(ts_ms, BBox)`：

```rust
struct Track {
    id: u64,
    label: String,
    history: Vec<(u64, BBox)>, // (ts_ms, bbox)，按时间递增
}
```

分析走位时，我们不用整个框，只取**脚点**（`bbox.bottom_center()`，底边中点）。原因 [第 83 章](ch83-filtering-basics.md) 说过：人站在地面上，脚点比中心更能代表他"在地面的哪个位置"，也更贴近"过线""进区域"这类判定。于是一条轨迹就抽象成了平面上一串带时间戳的点：

```text
   ts=0     100    200    300  ...   (毫秒)
   ●────────●──────●──────●──── →   脚点随时间移动的折线
 (20,120)        (56,120)      (向右走)
```

有了这串点，速度、方向、行为，全是几何题。

## 速度：位移除以时间，但小心抖动

最朴素的速度：相邻两个脚点的直线距离，除以两帧的时间差。记两点为 \\( (x_0,y_0) \\) 在 \\( t_0 \\)、\\( (x_1,y_1) \\) 在 \\( t_1 \\)（毫秒），则瞬时速度（像素/秒）：

\\[ v = \frac{\sqrt{(x_1-x_0)^2 + (y_1-y_0)^2}}{(t_1 - t_0)/1000} \\]

公式简单，坑在后面。**检测框每一帧都在轻微抖动**——同一个静止的人，框可能这帧 `(56,120)`、下帧 `(58,119)`，纯粹是检测器的量化噪声。这点抖动除以 0.1 秒的时间差会被**放大 10 倍**，让瞬时速度上蹿下跳。等下实测你会看到，一个匀速走的人，逐帧瞬时速度在 `89~139 px/s` 之间乱跳，而他真实的平均速度稳稳是 `120` 左右。

> 🕳️ **坑**：**别拿单帧瞬时速度去做判断。** 帧间隔越小、放大越狠。稳妥的做法有两种：① 用**较长基线**——拿总路程除以总时长，或隔 N 帧再算；② 做**滑动平均**平滑。本章代码用"总路程 ÷ 总时长"得到平滑速度，你会看到它稳定在 `120.6 px/s`。

还有一个更根本的坑：**像素速度不等于真实速度**。同样是 `120 px/s`，画面近处的人（框大）可能只是慢慢踱步，远处的人（框小）却是在狂奔——透视让"每像素代表多少米"随位置变化。要换算成真实的"米/秒"，需要相机标定（[第 64 章](../p14-3d/ch64-calibration.md)）给出地平面的像素-米比例 \\( s \\)，再 \\( v_{\text{real}} = v_{\text{px}} \times s \\)。没标定时，像素速度只能做**相对比较**（谁比谁快），别当成物理速度上报。

## 方向：位移向量的角度

方向就是位移向量 \\( (dx, dy) = (x_1-x_0,\ y_1-y_0) \\) 的角度。用 `atan2`：

\\[ \theta = \operatorname{atan2}(dy,\ dx) \times \frac{180}{\pi} \\]

**注意图像坐标 y 是向下的**（和 `image` 库一致），所以角度盘长这样：

```text
              -90° (向上)
                │
   ±180° ───────┼─────── 0°  (向右)
   (向左)       │
              +90° (向下)
```

向右 `0°`、向下 `+90°`、向左 `±180°`、向上 `-90°`。方向也有它的坑：

> ⚠️ **注意**：**慢速时方向极不稳。** 目标几乎不动时，位移向量的长度接近 0，方向几乎全由抖动决定——这帧朝东、下帧朝西都有可能。所以**判方向前先看速度**：只有目标确实在"赶路"（净速度够大）时，方向才可信。徘徊、静止的目标根本不该去谈"它朝哪走"。

平滑方向不能直接对角度求平均（`350°` 和 `10°` 平均成 `180°` 就闹笑话了，它俩其实都在"向右偏上"附近）。正确做法是**对位移向量求和再取角度**（向量平均），天然绕过了 `360°` 环的接缝。本章用一个 `±3` 帧的窗口做向量平滑。

## 逆行：方向和"规定方向"拧着来

有了可信的方向，逆行就好办了：给定一个**规定的顺行方向**（比如单行道规定向右 = `0°`），算目标净方向和它的**夹角**，超过某个容差（如 `120°`）就是逆行。夹角要归一化到 `[-180°, 180°]`，别忘了绕环：

```rust
fn ang_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 { d -= 360.0; }
    if d < -180.0 { d += 360.0; }
    d
}
```

再叠加上一节的"速度门槛"，逆行判据就是：**净速度 > 阈值 且 |夹角| > 容差**。这样徘徊的目标（净速度极低）不会被误判成逆行。

> 💡 **提示**：**逆行常和绊线搭配用。** [第 87 章](ch87-tripwire.md) 的绊线能判目标是 A→B 还是 B→A 过线（用叉积定侧）。把"绊线方向"和这里的"轨迹方向"结合，就能实现"出口只报逆向进入""闸机反向闯入"这类精准规则——这也是标题里"越界"的落点：越界=过了某条线，逆行=过线的方向不对。

## 徘徊：走得久，却没走远

徘徊（loitering）的直觉特征是"**在小范围里磨蹭很久**"。量化成两个数：

- **停留时长** `duration`：末点时间戳减首点，够长（如 ≥ 3 秒）。
- **净位移** `net_disp`：首点到末点的直线距离，很小（如 < 40 px）——人绕了半天又回到原地附近。

只用净位移还不够刻画"来回打转"，再加一个**徘徊比** = 总路程 ÷ 净位移。正常通过的人这个比值 ≈ 1（走的基本是直线），而原地打转的人路程很长、净位移很小，比值会飙到十几、几十。等下实测里徘徊者的徘徊比是 `19.8`，一眼就和正常人的 `1.0` 拉开了。

> 📖 **术语**：**徘徊 / 逗留（loitering）**。安防里指目标在敏感区域附近长时间无目的停留，常是可疑行为的前兆。它和 [第 88 章](ch88-intrusion-dwell.md) 的**停留（dwell）**是近亲：dwell 强调"在某 ROI 内待够时长"，loitering 更强调"待得久 + 走位来回没进展"。两者都靠 `Track.history` 的时间跨度来算。

## 热力图：把千万个点烧成一张图

最后一个需求："哪块区域最热"。做法朴素得可爱：在画面上铺一张**二维网格**（每格比如 20×20 像素），把**每一个轨迹点**都往它所在的格子里 `+1`。累积一整天，格子里的数字就是"这块地方被踩过多少次"。数据结构一个 `Vec<Vec<u32>>`（或 `ndarray::Array2<u32>`）就够。

导出成灰度图看一眼最直观：把每个格子的计数**按峰值归一化**到 `0~255`，越白越热，存成 PNG。两个坑：

- **分辨率**：格子太大（如 100px）丢细节，太小（如 2px）又太碎、单点噪声明显。按目标大小取，一般设成目标框的一半到一倍。
- **归一化**：一定要除以峰值（或某个分位数）再映射到 `0~255`，否则要么全黑要么一片死白。个别超热点会把其他区域压暗时，可改用对数或分位数拉伸。

## 动手写代码：三条轨迹 + 一张热力图

我们合成三条典型轨迹——正常通过、逆行、徘徊——把上面所有分析跑一遍，并累积一张热力图。仅 `std`（存 PNG 用 `image`）。

`Cargo.toml`（主工程已配好，仅列相关项）：

```toml
[dependencies]
image = "0.25"
```

完整代码（`examples/ch89_trajectory.rs`）：

```rust
use std::f32::consts::PI;

// ===== 第 83 章定义的共享数据模型（此处原样沿用其中三个）=====

#[derive(Clone, Copy, Debug, PartialEq)]
struct BBox { x: f32, y: f32, w: f32, h: f32 }

impl BBox {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Self { BBox { x, y, w, h } }
    /// 底边中点（脚点）——判过线/入区域/走向都用这个而非中心
    fn bottom_center(&self) -> (f32, f32) { (self.x + self.w / 2.0, self.y + self.h) }
}

#[derive(Clone, Debug)]
struct Track { id: u64, label: String, history: Vec<(u64, BBox)> }

#[derive(Clone, Debug)]
struct Event { kind: String, ts_ms: u64, track_id: Option<u64>, message: String }

// ===== std 内无随机数：自带确定性 LCG 造可复现抖动 =====
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed) }
    /// 返回 [-1,1) 的伪随机数，用来给检测框加抖动
    fn signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 2.0 - 1.0
    }
}

/// 由一串脚点造轨迹：固定框 w=20,h=40，反推左上角
fn track_from_feet(id: u64, label: &str, feet: &[(f32, f32)], dt_ms: u64) -> Track {
    let (w, h) = (20.0f32, 40.0f32);
    let history = feet.iter().enumerate()
        .map(|(i, &(fx, fy))| (i as u64 * dt_ms, BBox::new(fx - w / 2.0, fy - h, w, h)))
        .collect();
    Track { id, label: label.to_string(), history }
}

// ===== 行为分析：全部基于脚点序列 =====

/// 相邻两点的瞬时速度序列（像素/秒）
fn seg_speeds(t: &Track) -> Vec<f32> {
    t.history.windows(2).map(|w| {
        let (t0, b0) = w[0]; let (t1, b1) = w[1];
        let (x0, y0) = b0.bottom_center(); let (x1, y1) = b1.bottom_center();
        let d = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        d / ((t1 - t0) as f32 / 1000.0) // 毫秒 -> 秒
    }).collect()
}

/// 轨迹总路程
fn path_length(t: &Track) -> f32 {
    t.history.windows(2).map(|w| {
        let (x0, y0) = w[0].1.bottom_center(); let (x1, y1) = w[1].1.bottom_center();
        ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
    }).sum()
}

/// 净位移：首点到末点直线距离
fn net_disp(t: &Track) -> f32 {
    let (x0, y0) = t.history.first().unwrap().1.bottom_center();
    let (x1, y1) = t.history.last().unwrap().1.bottom_center();
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

fn duration_ms(t: &Track) -> u64 { t.history.last().unwrap().0 - t.history.first().unwrap().0 }

/// 净位移方向角（度），图像坐标 y 向下：右=0，下=+90，左=±180，上=-90
fn net_direction_deg(t: &Track) -> f32 {
    let (x0, y0) = t.history.first().unwrap().1.bottom_center();
    let (x1, y1) = t.history.last().unwrap().1.bottom_center();
    (y1 - y0).atan2(x1 - x0) * 180.0 / PI
}

/// 两角之差，归一化到 [-180,180]
fn ang_diff(a: f32, b: f32) -> f32 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 { d -= 360.0; }
    if d < -180.0 { d += 360.0; }
    d
}

/// 平滑方向：把窗口内位移向量相加再取角度（向量平均，避开角度绕环）
fn smoothed_dir_deg(t: &Track, i: usize, win: usize) -> f32 {
    let n = t.history.len();
    let lo = i.saturating_sub(win);
    let hi = (i + win).min(n - 1);
    let (x0, y0) = t.history[lo].1.bottom_center();
    let (x1, y1) = t.history[hi].1.bottom_center();
    (y1 - y0).atan2(x1 - x0) * 180.0 / PI
}

fn main() {
    let dt = 100u64; // 帧间隔 100ms = 10fps

    // ① 正常通过：从左向右匀速，带 ±2px 检测抖动
    let mut j1 = Lcg::new(1);
    let feet1: Vec<(f32, f32)> = (0..21)
        .map(|i| (20.0 + 12.0 * i as f32 + 2.0 * j1.signed(), 120.0 + 2.0 * j1.signed()))
        .collect();
    // ② 逆行：从右向左（规定顺行是向右）
    let mut j2 = Lcg::new(7);
    let feet2: Vec<(f32, f32)> = (0..21)
        .map(|i| (300.0 - 12.0 * i as f32 + 2.0 * j2.signed(), 90.0 + 2.0 * j2.signed()))
        .collect();
    // ③ 徘徊：在 (150,150) 附近来回打转
    let feet3: Vec<(f32, f32)> = (0..40)
        .map(|i| { let a = i as f32 * 0.5; (150.0 + 8.0 * a.sin(), 150.0 + 8.0 * (a * 0.7).cos()) })
        .collect();

    let tracks = vec![
        track_from_feet(1, "person", &feet1, dt),
        track_from_feet(2, "person", &feet2, dt),
        track_from_feet(3, "person", &feet3, dt),
    ];

    let designated_deg = 0.0f32;   // 规定顺行方向：向右
    let travel_speed_min = 40.0f32; // 判逆行需净速度大于它
    let wrong_way_tol = 120.0f32;   // 与规定方向夹角超过它算逆行
    let dwell_min_ms = 3000u64;     // 徘徊：停留时长门槛
    let net_disp_max = 40.0f32;     // 徘徊：净位移上限

    let mut events: Vec<Event> = Vec::new();

    // 速度抖动与平滑：拿正常轨迹举例
    let sp1 = seg_speeds(&tracks[0]);
    let raw_min = sp1.iter().cloned().fold(f32::INFINITY, f32::min);
    let raw_max = sp1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("== 速度与抖动（轨迹 1 正常通过）==");
    print!("前 6 段瞬时速度(px/s): ");
    for s in sp1.iter().take(6) { print!("{s:.0} "); }
    println!();
    println!("瞬时速度范围: {raw_min:.0} ~ {raw_max:.0} px/s（受检测框抖动影响，跳动明显）");
    let avg_path = path_length(&tracks[0]) / (duration_ms(&tracks[0]) as f32 / 1000.0);
    println!("按总路程算的平均速度: {avg_path:.1} px/s（平滑后稳定得多）\n");

    // 方向平滑：瞬时 vs 窗口平滑
    println!("== 方向与平滑（轨迹 1）==");
    print!("逐段瞬时方向(度): ");
    for w in tracks[0].history.windows(2).take(6) {
        let (x0, y0) = w[0].1.bottom_center(); let (x1, y1) = w[1].1.bottom_center();
        print!("{:.0} ", (y1 - y0).atan2(x1 - x0) * 180.0 / PI);
    }
    println!("（慢速/抖动时逐帧方向很不稳）");
    println!("窗口(±3帧)平滑方向@中点: {:.1} 度；整段净方向: {:.1} 度\n",
        smoothed_dir_deg(&tracks[0], 10, 3), net_direction_deg(&tracks[0]));

    // 逐条轨迹：算指标 + 判行为
    println!("== 逐条轨迹的行为判定 ==");
    for t in &tracks {
        let dur = duration_ms(t);
        let path = path_length(t);
        let net = net_disp(t);
        let dir = net_direction_deg(t);
        let net_vel = net / (dur as f32 / 1000.0); // 净速度（真正“赶路”的速度）
        let wander = path / net.max(1.0);          // 徘徊比：打转时远大于 1

        let is_wrong_way = net_vel > travel_speed_min
            && ang_diff(dir, designated_deg).abs() > wrong_way_tol;
        let is_loiter = dur >= dwell_min_ms && net < net_disp_max;

        let verdict = if is_wrong_way { "逆行!" }
            else if is_loiter { "徘徊!" } else { "正常" };

        println!("轨迹#{} {:<7} 时长{:>4}ms 路程{:>5.0} 净位移{:>5.0} 净方向{:>6.1}° 净速度{:>5.0} 徘徊比{:>4.1} => {}",
            t.id, t.label, dur, path, net, dir, net_vel, wander, verdict);

        let last_ts = t.history.last().unwrap().0;
        if is_wrong_way {
            events.push(Event { kind: "wrong_way".into(), ts_ms: last_ts, track_id: Some(t.id),
                message: format!("目标 {} 逆行：净方向 {:.0}° 与规定 {:.0}° 相反", t.id, dir, designated_deg) });
        }
        if is_loiter {
            events.push(Event { kind: "loiter".into(), ts_ms: last_ts, track_id: Some(t.id),
                message: format!("目标 {} 徘徊：{:.1}s 内净位移仅 {:.0}px", t.id, dur as f32 / 1000.0, net) });
        }
    }

    // 热力图：把所有脚点累加到二维网格
    const W: u32 = 320; const H: u32 = 180; const CELL: u32 = 20; // 16x9 网格
    let gw = (W / CELL) as usize; let gh = (H / CELL) as usize;
    let mut grid = vec![vec![0u32; gw]; gh];
    for t in &tracks {
        for (_, b) in &t.history {
            let (fx, fy) = b.bottom_center();
            let cx = (fx.max(0.0).min((W - 1) as f32) as u32 / CELL) as usize;
            let cy = (fy.max(0.0).min((H - 1) as f32) as u32 / CELL) as usize;
            grid[cy][cx] += 1;
        }
    }
    let (mut px, mut py, mut pv) = (0usize, 0usize, 0u32);
    for y in 0..gh { for x in 0..gw {
        if grid[y][x] > pv { px = x; py = y; pv = grid[y][x]; }
    }}
    println!("\n== 热力图 ({}x{} 网格) ==", gw, gh);
    println!("峰值格子: (列{}, 行{})，累计 {} 个点，对应画面像素中心 ≈ ({}, {})",
        px, py, pv, px as u32 * CELL + CELL / 2, py as u32 * CELL + CELL / 2);

    // 导出灰度 PNG：每格放大成 CELL×CELL 色块，按峰值归一化
    let maxv = pv.max(1);
    let mut heat = image::GrayImage::new(W, H);
    for y in 0..H { for x in 0..W {
        let v = (grid[(y / CELL) as usize][(x / CELL) as usize] * 255 / maxv) as u8;
        heat.put_pixel(x, y, image::Luma([v]));
    }}
    heat.save("out_ch89_heatmap.png").unwrap();
    println!("热力图已存 out_ch89_heatmap.png（越白越热）");

    println!("\n== 产生的事件 ({} 条) ==", events.len());
    for e in &events {
        println!("[{}] @{}ms track={:?} {}", e.kind, e.ts_ms, e.track_id, e.message);
    }
}
```

运行：

```bash
cargo run --example ch89_trajectory
```

终端输出（数字真实可复现）：

```text
== 速度与抖动（轨迹 1 正常通过）==
前 6 段瞬时速度(px/s): 129 126 112 132 117 122 
瞬时速度范围: 89 ~ 139 px/s（受检测框抖动影响，跳动明显）
按总路程算的平均速度: 120.6 px/s（平滑后稳定得多）

== 方向与平滑（轨迹 1）==
逐段瞬时方向(度): -2 2 -9 2 7 -4 （慢速/抖动时逐帧方向很不稳）
窗口(±3帧)平滑方向@中点: -2.2 度；整段净方向: 0.1 度

== 逐条轨迹的行为判定 ==
轨迹#1 person  时长2000ms 路程  241 净位移  239 净方向   0.1° 净速度  120 徘徊比 1.0 => 正常
轨迹#2 person  时长2000ms 路程  244 净位移  242 净方向-179.3° 净速度  121 徘徊比 1.0 => 逆行!
轨迹#3 person  时长3900ms 路程  127 净位移    6 净方向 -41.3° 净速度    2 徘徊比19.8 => 徘徊!

== 热力图 (16x9 网格) ==
峰值格子: (列7, 行7)，累计 40 个点，对应画面像素中心 ≈ (150, 150)
热力图已存 out_ch89_heatmap.png（越白越热）

== 产生的事件 (2 条) ==
[wrong_way] @2000ms track=Some(2) 目标 2 逆行：净方向 -179° 与规定 0° 相反
[loiter] @3900ms track=Some(3) 目标 3 徘徊：3.9s 内净位移仅 6px
```

几个结论对上号了：

1. **瞬时速度在 `89~139` 之间乱跳，平均却是 `120.6`**——这就是"别信单帧速度、要平滑"的活教材。同理方向：逐段瞬时方向从 `-9°` 到 `+7°` 抖，整段净方向却是干净的 `0.1°`（正东，向右）。
2. **轨迹 2 净方向 `-179.3°`**（正西，向左），和规定的 `0°` 几乎正相反、净速度高达 `121`——判为**逆行**。
3. **轨迹 3 净速度只有 `2 px/s`、徘徊比 `19.8`**——它走了 `127` 像素的路程，净位移却只有 `6` 像素，典型的"原地打转 3.9 秒"，判为**徘徊**。注意它净方向 `-41.3°` 看着像"向右上"，但净速度这么低，这个方向根本没意义——正好印证了"慢速别谈方向"。
4. **热力图峰值落在格子 `(列7,行7)`、累计 40 个点**，换算到像素中心 `(150,150)`——正是徘徊者磨蹭的地方。打开 `out_ch89_heatmap.png`，那一格是最亮的白块。

## 常见坑与提示

> 🕳️ **坑**：**帧率不稳时，一定要用真实 `ts_ms` 算速度，别默认"每帧 1/25 秒"。** 真实视频会丢帧、会卡顿，相邻两帧的时间差可能是 40ms 也可能是 200ms。速度公式里的分母必须是 `(t1 - t0)`，不能写死帧率——这也是我们把时间戳一路带在 `history` 里的原因。

> ⚠️ **注意**：**像素速度 ≠ 真实速度，透视会骗你。** 画面近大远小，同样 `px/s` 在近处慢、远处快。做"超速""异常奔跑"这类报警前，务必用 [第 64 章](../p14-3d/ch64-calibration.md) 的标定把像素换算成米；没标定就只做相对比较。

> 🕳️ **坑**：**逆行/方向阈值要按场景标定，别用一个魔法数走天下。** 十字路口和单行道对"逆行"的容忍角度完全不同；`travel_speed_min` 设太低会把徘徊者误判成逆行，设太高又会漏掉慢速逆行的自行车。先跑一批真实轨迹，看正常/异常两类的净速度、夹角分布，再定阈值。

> 💡 **提示**：**热力图别只累加一次就完事。** 长时间运行时，用**带衰减的累加**（每隔一段时间整体乘 `0.95`）能得到"最近一段时间"的热度，而不是从开机到现在的总和——这对"实时找当前最挤的区域"更有用。分辨率和归一化方式（线性/对数/分位数）也要按画面调。

## 本章小结

- 行为分析的原料就是 `Track.history` 里那串 `(ts_ms, 脚点)`；速度、方向、逆行、徘徊、热力图，全是在这串点上做几何。
- **速度** = 位移 / 时间差（px/s）；单帧瞬时速度受抖动放大，务必用长基线或滑动平均**平滑**；像素速度要靠标定才能变真实速度。
- **方向** = 位移向量的 `atan2` 角度（注意 y 向下）；慢速时方向不可信，判方向前先过速度门槛；平滑用**向量求和**而非角度平均。
- **逆行** = 在"确实赶路"的前提下，净方向与规定方向夹角过大；常与 [第 87 章](ch87-tripwire.md) 绊线方向结合成"越界+逆向"规则。
- **徘徊** = 停留时长够久 + 净位移小 + 徘徊比（路程/净位移）大。
- **热力图** = 把每个轨迹点撒进二维网格 `+1`，归一化后存灰度图；注意分辨率、归一化、以及可选的时间衰减。

## 动手练习

1. **加一条"闯红灯"轨迹**：合成一条先停（净速度≈0）后突然加速冲出去的轨迹，观察它的瞬时速度序列，试着写个"启动检测"（连续几帧速度从 0 跃升到某阈值以上）。
2. **换规定方向**：把 `designated_deg` 改成 `90.0`（规定向下为顺行），看轨迹 1（向右）会不会被判成逆行；体会阈值 `wrong_way_tol` 怎么影响"斜着走"的判定。
3. **热力图加衰减**：让程序把三条轨迹的点分"两个时段"喂入，第二时段前先把 `grid` 整体乘 `0.5`，对比衰减前后峰值格子的计数变化。
4. **进阶——速度直方图**：把三条轨迹所有瞬时速度收集起来，用 [第 12 章](../p02-image-basics/ch12-histogram.md) 的思路画个速度直方图，直观看出"大多数人走多快、有没有异常的快/慢目标"。

下一章我们换个视角：报警不能一天 24 小时无脑响——该**按时段布防**（仓库只在夜里布防），还要能发现**摄像头被人遮挡、挪动、搞失焦**。👉 [第 90 章 布防排班、相机遮挡与场景鲁棒性](ch90-arming-tamper.md)
