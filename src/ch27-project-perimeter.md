# 第 27 章 · 实战二：周界入侵报警系统

> **本章导读**：第 26 章我们把检测器打通成了一个端到端程序，但它盯着的还是**单张图**——喂一张进去，画一批框出来。这一章跨出真正的一步：处理**一路视频流**。这是全书第一个像样的**视频分析项目**：把前面几章的零件——第 26 章的**检测**、第 21 章的**跟踪**、第 22 章的 **ROI**、第 23 章的**绊线**、第 24 章的**冷却**——拼成一个能落地的系统：**只要有人闯进禁区，就报警**。报警可以是打印日志、存一张抓拍图、回调一个函数。真正的难点不在"检测到人"，而在**怎么不误报、不刷屏**：只报人不报车、同一个人只报一次、短暂被遮挡了别当成新人。读完这一章，你就有了一套可以对着 RTSP 跑起来的周界报警骨架，也理解了"从单帧检测到视频事件"这中间隔着的一整层工程。

## 27.1 需求与效果：从"检测到人"到"有人闯入禁区"

先把需求说清楚。一个典型的周界入侵场景是这样的：摄像头对着一个院子、一段铁轨、一片变电站围栏。运维在监控画面上**手点几个点，圈出一块禁区多边形**（也可能只拉一条**警戒线**）。系统的任务是：

> 一旦有**人**踏进这块禁区（或越过警戒线），就报警——打印一条带时间戳的日志、存一张画了标注的抓拍图，或者回调一个上层函数去推消息。

听起来"检测到人 + 判断在不在多边形里"就完事了，其实远不止。真实项目里，光靠单帧判定会被三件事按在地上摩擦：

| 需求 | 单帧做不到，为什么 | 靠哪一章的工具 |
| --- | --- | --- |
| **只报人，不报车** | 检测器会把画面里的车、猫、狗全报出来 | 类别过滤（本章 27.5） |
| **同一个人只报一次** | 一个人站在禁区里，每帧都判"在区内"，会连报几百条 | 跟踪拿稳定 ID（第 21 章）+ 冷却（第 24 章） |
| **短暂遮挡不算新人** | 人被电线杆挡一下、检测漏了两帧，回来后不能当成"新入侵" | 跟踪的生命周期 `max_age`（第 21 章） |
| **偶尔一帧误检别报** | 树影、反光被误当成人，只闪一帧 | N-of-M 连续确认（第 24 章） |

所以周界报警不是一个"判断题"，而是一条**流水线**：每一帧的数据从帧源出发，依次穿过检测、跟踪、空间判定、事件判定、去重，最后才可能变成一条报警。数据流长这样：

```text
┌────────┐    ┌────────┐    ┌────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐    ┌────────┐
│  帧源   │──→ │  检测   │──→ │  跟踪   │──→ │ ROI判定  │──→ │ 绊线判定 │──→ │  冷却   │──→ │  报警   │
│ 第 6 章 │帧  │ 第26章  │框  │ 第21章  │ID  │ 第22章   │候  │ 第23章   │事  │ 第24章  │  │存图/日志 │
└────────┘    └────────┘    └────────┘    └─────────┘    └─────────┘    └────────┘    └────────┘
  RgbImage    Detection      Track        脚点在区内?    穿越警戒线?     去重刷屏      ⚠ 入侵!
```

- **帧源**：一路视频的取帧。RTSP 摄像头、本地文件都行，解码细节第 6 章讲过（`opencv::videoio::VideoCapture`，或 ffmpeg 抽帧）。本章不纠结解码，统一把它抽象成一个**帧迭代器** `impl Iterator<Item = image::RgbImage>`——每 `next()` 一次吐一帧原图。
- **检测 / 跟踪**：直接复用第 26 章的 `Detector` 和第 21 章的 `Tracker`，拿到带稳定 `id` 的 `Track` 列表。
- **ROI / 绊线 / 冷却**：本章的主角，把"一堆带 ID 的目标"翻译成"值得上报的入侵事件"。

这一章的核心，就是把最右边那四个方框（ROI 判定 → 绊线 → 冷却 → 报警）用工程手法拼扎实。

## 27.2 配置驱动：把业务参数从代码里拆出来

动手写逻辑之前，先立一条工程规矩：**凡是"这个项目才有、换个现场就要改"的参数，都不许写死在代码里，一律进配置文件。** 禁区画在哪、警戒线两端在哪、关注哪些类别、冷却几秒、检测阈值多少——这些都是**业务参数**，不是程序逻辑。写死了，换一路摄像头就要改代码重编译；配置化了，运维改个 JSON 重启即可。

用 `serde` 定义配置结构体，字段名就是 JSON 的键：

```rust,ignore
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerimeterConfig {
    /// 禁区多边形顶点，按边相邻顺序排列（原图像素坐标）
    pub zone: Vec<(f32, f32)>,
    /// 警戒线两端点；为 None 时只做区域入侵，不做绊线
    pub tripwire: Option<[(f32, f32); 2]>,
    /// 关注的 COCO 类别号，只报这些类；[0] = 只报 person
    pub classes: Vec<usize>,
    /// 检测置信度阈值，低于此分数的检测直接丢弃
    pub score_thresh: f32,
    /// 同一个 track 两次报警的最小间隔（秒），防刷屏
    pub cooldown_secs: u64,
    /// 连续多少帧确认在区内，才真正算「闯入」（N-of-M 的 N）
    pub confirm_frames: u32,
    /// 标定时用的分辨率 (宽, 高)——上面那些顶点是在这个尺寸下点的
    pub calib_size: (u32, u32),
}
```

对应的示例 JSON（注意 `serde` 把 Rust 的二元组 `(f32, f32)` 序列化成 2 元素数组 `[x, y]`，正好符合直觉）：

```json
{
  "zone": [[320, 180], [960, 180], [1040, 620], [260, 620]],
  "tripwire": [[300, 400], [1000, 400]],
  "classes": [0],
  "score_thresh": 0.35,
  "cooldown_secs": 30,
  "confirm_frames": 3,
  "calib_size": [1280, 720]
}
```

加载就是一行：

```rust,ignore
let cfg: PerimeterConfig =
    serde_json::from_reader(std::fs::File::open("perimeter.json")?)?;
```

这里 `calib_size` 字段先埋个伏笔：禁区顶点是运维在 **1280×720** 的预览画面上点的，如果实际跑的是 **1920×1080**，顶点坐标就得按比例放大，否则整块禁区会缩在画面左上角——这是 27.8 节要重点敲的坑。把标定分辨率也写进配置，就是为了运行时能自动做这个缩放。

兑现这个伏笔只要一个方法：加载配置后、一旦知道视频流的真实分辨率，就把所有顶点从"标定坐标系"原地搬到"运行坐标系"。

```rust,ignore
impl PerimeterConfig {
    /// 把在 calib_size 分辨率下标定的顶点，缩放到实际运行分辨率（原地改写）。
    /// 加载配置后、确定视频流真实尺寸时调用一次即可。
    pub fn scale_to(&mut self, run_w: u32, run_h: u32) {
        let (cw, ch) = self.calib_size;
        // 分辨率一致、或标定尺寸缺失（0），都无需缩放，直接返回
        if cw == 0 || ch == 0 || (cw, ch) == (run_w, run_h) {
            return;
        }
        let (sx, sy) = (run_w as f32 / cw as f32, run_h as f32 / ch as f32);
        for p in &mut self.zone {                 // 禁区每个顶点
            p.0 *= sx;
            p.1 *= sy;
        }
        if let Some(wire) = &mut self.tripwire {   // 警戒线两端（若配置了）
            for end in wire.iter_mut() {
                end.0 *= sx;
                end.1 *= sy;
            }
        }
    }
}
```

**为什么是加载后立刻缩放，而不是每帧判定时再换算？** 因为顶点是**静态**的——标定完就不动了，缩放一次就永久对齐到运行分辨率；之后每帧的 `point_in_polygon`、`segments_intersect` 都在同一个坐标系里比，既省算力又不会漏掉哪一处忘了换算。这正是"坐标全程统一在原图像素系"（27.8）最省心的落地方式。

## 27.3 为什么必须先跟踪、再判定

新手最容易问的一句：既然检测每帧都能给出"人在哪"，为什么非要多一步跟踪？直接判断"这一帧有没有人的脚在禁区里"不就行了？

因为**入侵是一个跨帧的"事件"，不是单帧的"状态"**。"有人在区内"是状态，每帧都成立；"有人**踏入**区内"是事件，只在**从区外变到区内的那一瞬间**发生一次。要捕捉"变化"，你必须知道"这个人上一帧在哪"——而这需要**把同一个人在连续帧之间对应起来**，也就是给他一个**稳定的身份 ID**。这正是第 21 章跟踪做的事。

有了稳定 ID，两个核心需求才有着落：

- **"同一个人只报一次"**：靠 `track.id` 去重。没有 ID，你根本分不清这一帧的人和上一帧的人是不是同一个，只能帧帧都报。
- **"短暂遮挡不算新人"**：跟踪器的 `max_age`（第 21 章）允许一个目标丢几帧后用同一个 ID 复活，被电线杆挡一下不会变成"新入侵"。

**判定用哪个点？** 第 22.4 节论证过：地面场景要用**脚底落地点**，也就是框的底边中点 `(cx, y2)`，而不是框中心。原因很直白——业务关心的是"这个人的**脚**踩没踩进禁区"，不是"他的躯干中心飘在哪"。一个高个子站在禁区边缘外，脚已经进去了、框中心却还在线外，用中心点就会漏报。直接复用第 22 章给 `BBox` 加的方法：

```rust,ignore
impl BBox {
    /// 底边中点：目标「脚下」落地的位置 (cx, y2)
    pub fn bottom_center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, self.y2)
    }
}
```

于是"这个 track 现在在不在禁区里"就是一句话：

```rust,ignore
let foot = track.bbox.bottom_center();          // (cx, y2)，脚点
let inside = point_in_polygon(foot, &cfg.zone);  // 第 22 章的射线法
```

`point_in_polygon` 是第 22 章造好的射线法判定，直接拿来用，不重复推导。到这里我们能回答"某 track 此刻在不在区内"，但还差最关键的一步——**捕捉"进入"这个动作**。

## 27.4 两种触发逻辑：区域入侵与绊线穿越

周界报警有两种经典的触发方式，对应两种业务语义：

| 触发方式 | 语义 | 用什么判 | 需要记住的历史 |
| --- | --- | --- | --- |
| **区域入侵** | 脚点**从区外进到区内** | 脚点 + `point_in_polygon` | 每个 track **上一帧在不在区内** |
| **绊线穿越** | 目标**跨过一条警戒线** | 中心点连线 vs 线段相交 | 每个 track **上一帧的中心点** |

两者都要"上一帧的信息"，这再次印证了 27.3：**没有跟踪的稳定 ID，就没法把"这一帧"和"上一帧"对应起来。**

### ① 区域入侵：一个极简状态机

核心是一句话：**只在"从区外踏入区内"的那一帧报警。** 用一个 `HashMap<u64, bool>` 记住每个 track 上一帧在不在区内，把当前帧和上一帧一比，只有 `false → true` 这个跳变才是"闯入"：

```rust,ignore
use std::collections::HashMap;

/// 区域入侵状态机：返回本帧「刚踏入禁区」的 track id 列表。
/// last_inside: 每个 track 上一帧是否在禁区内（跨帧记忆，外部持有）
fn detect_intrusion(
    tracks: &[Track],
    zone: &[(f32, f32)],
    last_inside: &mut HashMap<u64, bool>,
) -> Vec<u64> {
    let mut fired = Vec::new();
    for t in tracks {
        let foot = t.bbox.bottom_center();
        let now_in = point_in_polygon(foot, zone);
        let was_in = *last_inside.get(&t.id).unwrap_or(&false); // 没记录过 = 上一帧不在
        // ★ 核心：只有 false → true 这一次跳变，才算「闯入」
        if !was_in && now_in {
            fired.push(t.id);
        }
        last_inside.insert(t.id, now_in); // 更新状态，供下一帧比较
    }
    fired
}
```

看清楚这个状态机做对了什么：

- 一个人**一直站在**禁区里：第一帧 `false→true` 报一次，之后每帧都是 `true→true`，**不再触发**。天然满足"只报一次"。
- 一个人**从区内走到区外**：`true→false`，不触发（离开不是入侵；想报"离开"就反过来判，见练习）。
- `unwrap_or(&false)`：新出现的 track 默认"上一帧不在区内"。所以一个人**第一次出现时就已经站在区内**，也会被正确地报一次。

这段就是区域入侵的完整逻辑骨架。但它有个软肋——**只要有一帧误检把脚点算进了区内，就会 `false→true` 误报一次**。这个洞留到 27.5 用 N-of-M 补。

### ② 绊线穿越：中心点连线与警戒线段求交

第二种触发是**绊线**：目标的运动轨迹跨过一条设定的线段就报警。判定方法第 23 章推导过——用**叉积符号** `side()` 判断点在有向线段的哪一侧，再用"两条线段互相跨在对方两侧"判相交。这两个几何原语第 23 章已经写好、还专门做了**鲁棒版**（显式判符号，而不是 `side()*side()<0` 那种乘积写法，后者在退化/边界情形不稳），这里**直接 `use` 进来，不重复定义**：

```rust,ignore
// 复用第 23 章的两个几何原语（不再本地重写，避免与第 23 章实现漂移）：
//   side(a, b, p)                —— 叉积符号：点 p 在有向线段 a→b 的哪一侧
//   segments_intersect(a,b,c,d)  —— 两线段是否真正相交（四次叉积的鲁棒版）
use crate::{side, segments_intersect};
```

绊线判定和区域入侵结构一样，只是"记住的历史"从 `bool` 变成了"上一帧的中心点"。用这一帧和上一帧的中心点连成一条短线段，去和警戒线求交：

```rust,ignore
// last_center: HashMap<u64, (f32,f32)>，记录每个 track 上一帧的中心点
let cur = t.bbox.center();                     // 第 21 章已在用的 center()
if let Some(&prev) = last_center.get(&t.id) {
    if let Some(wire) = cfg.tripwire {
        if segments_intersect(prev, cur, wire[0], wire[1]) {
            // 这一帧 track 的轨迹跨过了警戒线 → 触发
        }
    }
}
last_center.insert(t.id, cur);
```

绊线还能顺便判**方向**：穿越前后 `side()` 的符号从正变负还是从负变正，就是"从 A 侧进"还是"从 B 侧出"——这正是第 28 章人流统计"进/出计数"的基础，本章先不展开。

本章接下来的骨架以**区域入侵**为主线（更贴合"禁区"语义），绊线作为可选的第二触发并行挂上。

## 27.5 抑制误报与刷屏：把状态机武装起来

27.4 的裸状态机有两个致命问题，都会让运维被报警淹没：

1. **单帧误检 → 误报**：树影、反光被误当成人，只在区内闪一帧，裸状态机照报不误。
2. **同一事件反复报 → 刷屏**：一个人在禁区边缘来回蹭，脚点在线内线外反复横跳，`false→true` 一次次触发；或者跟踪掉了 ID 重分配，同一个人被当成新人再报一次。

对应三道防线，全是前面章节造好的工具：

**（a）类别 + 阈值过滤**：在检测出来的第一时间就把不关心的类别和低分框扔掉。只留 `person`、只留分数够高的，从源头减少后面所有环节的噪声：

```rust,ignore
dets.retain(|d| cfg.classes.contains(&d.class_id) && d.score >= cfg.score_thresh);
```

**（b）N-of-M 连续确认**：不再对"裸的当前帧在不在区内"做状态机，而是要求**连续 `N` 帧脚点都在区内**，才认定"确实进来了"。用一个 `HashMap<u64, u32>` 数连续在区内的帧数，攒够 `N` 帧才把"确认在区内"这个信号翻成 `true`。第 24 章的 N-of-M 窗口（M 帧里有 N 帧命中就算数）是更宽松的版本，这里用最简单的"连续 N 帧"够用：

```rust,ignore
let streak = inside_streak.entry(t.id).or_insert(0);
*streak = if raw_in { *streak + 1 } else { 0 }; // 一出区就清零
let confirmed_in = *streak >= cfg.confirm_frames; // 攒够 N 帧才算「确认在区内」
```

关键手法：**把状态机喂的信号从"裸的 `raw_in`"换成"去抖后的 `confirmed_in`"**。这样偶尔一帧误检（`streak` 刚到 1 就清零）永远够不到确认线，自然被过滤，而状态机"只报一次"的好处原封不动保留。

**（c）Cooldown 冷却去重**：用第 24 章的冷却器，按 `track_id` 做键。同一个 track 报过一次后，`cooldown_secs` 秒内再触发也直接吞掉。这一层专治"边缘反复横跳"和"ID 短时间抖动"带来的重复报警：

```rust,ignore
if cooldown.allow(t.id, now) {
    // 距上次报警已超过冷却时长，放行；allow 内部会记下这次时间
    // （冷却时长在构造 Cooldown 时就传入，allow 只收 (key, now)——见第 24 章）
}
```

三道防线叠起来，最终的判定条件是：**类别对 + 分数够 + 连续 N 帧确认在区内 + 状态从"未确认"跳到"确认" + 没在冷却期**——五个条件同时满足，才发一条报警。下一节把它们塞进主循环。

## 27.6 报警动作：抓拍存图 + 结构化日志

判定成立后要"做点什么"。最实用的两个动作：**存一张画了标注的抓拍图**、**打印一条带时间戳和 track_id 的日志**。

抓拍图要一眼能看懂"谁、在哪闯的"，所以画三样东西：禁区多边形、闯入者的框、"⚠ 入侵"文字。用 `imageproc` 画：

```rust,ignore
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;

/// 报警：在一帧的副本上画标注，存图 + 打日志。
impl PerimeterAlarm {
fn raise_alarm(&self, frame: &RgbImage, hit: &Track) {
    let mut canvas = frame.clone();        // 拷贝，别污染原始帧
    let red = Rgb([255, 0, 0]);
    let yellow = Rgb([255, 220, 0]);

    // 1) 画禁区多边形（顶点首尾相连）
    let poly = &self.config.zone;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()]; // 最后一点连回第一点
        draw_line_segment_mut(&mut canvas, a, b, yellow);
    }
    // 2) 画闯入者的框（红）
    let bb = &hit.bbox;
    draw_hollow_rect_mut(
        &mut canvas,
        Rect::at(bb.x1 as i32, bb.y1 as i32)
            .of_size((bb.x2 - bb.x1).max(1.0) as u32, (bb.y2 - bb.y1).max(1.0) as u32),
        red,
    );
    // 3) 标注文字「⚠ 入侵 #id」：draw_text_mut 需自带 CJK 字体，加载略（见第 26 章）

    // 4) 存图 + 结构化日志（时间戳用 chrono）
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = format!("alarms/intrude_{}_id{}.jpg", ts, hit.id);
    let _ = std::fs::create_dir_all("alarms");
    let _ = canvas.save(&path);
    println!(
        "[{}] ⚠ 周界入侵  track#{}  脚点={:?}  抓拍={}",
        ts, hit.id, hit.bbox.bottom_center(), path
    );
}
}
```

几个工程细节：`frame.clone()` 是为了在副本上画，不弄脏送去存档的原始帧；文字 `draw_text_mut` 要渲染中文/emoji 得自带一份 CJK 字体（第 26 章画类别标签时讲过字体加载），这里省略；日志带上**时间戳和 track_id**，是为了事后能对着抓拍图和录像回溯。

> 这里的"报警动作"只是最朴素的落地版。生产系统里，报警通常不是存本地图，而是**推消息队列（MQ）、回调 webhook、写数据库**，让上层平台去弹窗、派单、留证。这套"事件产出后如何可靠地送出去"（重试、去重、下游宕机不丢）属于生产运维范畴，**第 32 章会专门讲**。本章把 `raise_alarm` 写成一个方法，就是留好这个扩展点——换成推 MQ 只改这一个函数。

## 27.7 完整骨架：`PerimeterAlarm`

把 27.3~27.6 串成一个结构体。字段就是"逻辑 + 跨帧记忆 + 去重器"：

```rust,ignore
use std::collections::HashMap;
use std::time::{Duration, Instant};
use image::RgbImage;

pub struct PerimeterAlarm {
    detector: Detector,                  // 第 26 章：单帧检测器
    tracker: Tracker,                    // 第 21 章：多目标跟踪器
    config: PerimeterConfig,             // 27.2 的配置
    last_inside: HashMap<u64, bool>,     // 每个 track 的「确认在区内」状态（状态机）
    inside_streak: HashMap<u64, u32>,    // 连续在区内帧数（N-of-M 去抖）
    cooldown: Cooldown<u64>,             // 第 24 章：按 track_id 冷却，防刷屏
}

impl PerimeterAlarm {
    pub fn new(detector: Detector, tracker: Tracker, config: PerimeterConfig) -> Self {
        // 冷却时长在构造时定死（第 24 章 Cooldown 约定）：先把秒数读出来，再 move config
        let cooldown = Cooldown::new(Duration::from_secs(config.cooldown_secs));
        Self {
            detector,
            tracker,
            config,
            last_inside: HashMap::new(),
            inside_streak: HashMap::new(),
            cooldown,
        }
    }

    /// 处理一帧，返回本帧新触发报警的 track id 列表。
    /// 检测可能失败（模型推理出错），所以返回 Result；主循环负责「记一笔、跳过这帧」，
    /// 绝不让一帧坏数据用 `?` 把 24/7 服务整个带崩。
    pub fn process_frame(&mut self, frame: &RgbImage, now: Instant) -> anyhow::Result<Vec<u64>> {
        // 1) 检测：一帧原图 → 一批 Detection（detect 返回 Result，用 ? 上抛给主循环）
        let mut dets = self.detector.detect(frame)?;

        // 2) 类别 + 阈值过滤（27.5-a）：只留 person、只留够分的
        dets.retain(|d| {
            self.config.classes.contains(&d.class_id) && d.score >= self.config.score_thresh
        });

        // 3) 跟踪（第 21 章）：拿到带稳定 id 的 Track
        let tracks = self.tracker.update(&dets);

        // 4) 逐 track 做区域入侵判定：状态机 + N-of-M + 冷却
        let mut alarms = Vec::new();
        for t in &tracks {
            let foot = t.bbox.bottom_center();                       // 27.3：脚点
            let raw_in = point_in_polygon(foot, &self.config.zone);  // 第 22 章：在区内？

            // N-of-M 去抖：连续 N 帧在区内，才算「确认在区内」
            let streak = self.inside_streak.entry(t.id).or_insert(0);
            *streak = if raw_in { *streak + 1 } else { 0 };
            let confirmed_in = *streak >= self.config.confirm_frames;

            // 状态机：喂给它的是去抖后的 confirmed_in，不是裸 raw_in。
            // 冷却时长已在构造 Cooldown 时定死，allow 只收 (key, now)。
            let was_in = *self.last_inside.get(&t.id).unwrap_or(&false);
            if !was_in && confirmed_in && self.cooldown.allow(t.id, now) {
                self.raise_alarm(frame, t);      // 27.6：存图 + 日志
                alarms.push(t.id);
            }
            self.last_inside.insert(t.id, confirmed_in);
        }

        // 5) 清理已消失 track 的状态，防止三个 HashMap 只涨不消（见 27.8 坑）
        let alive: std::collections::HashSet<u64> = tracks.iter().map(|t| t.id).collect();
        self.last_inside.retain(|id, _| alive.contains(id));
        self.inside_streak.retain(|id, _| alive.contains(id));

        Ok(alarms)
    }
}
```

驱动它的主循环，就是把帧源迭代一遍——这就是一个能对着 RTSP 或视频文件跑起来的完整程序：

```rust,ignore
fn main() -> anyhow::Result<()> {
    // 加载配置、检测器、跟踪器（各自构造细节见对应章节）
    let mut cfg: PerimeterConfig =
        serde_json::from_reader(std::fs::File::open("perimeter.json")?)?;
    let detector = Detector::load("yolov8n.onnx")?;   // 第 26 章：构造器是 load，不是 new
    let tracker = Tracker::new();                      // 第 21 章：无参，内部默认已适配一般场景

    // 帧源：一个解码器句柄——既能迭代取帧，也能查到流的真实分辨率（解码见第 6 章）
    let frame_source = open_frame_source("rtsp://...")?; // 迭代产出 RgbImage
    // ★ 兑现 27.2 的伏笔：用流的真实分辨率把禁区/警戒线顶点缩放到运行坐标系。
    //   漏了这一步，1280×720 标定的禁区跑在 1920×1080 上会缩在左上角——本章最高频翻车点。
    cfg.scale_to(frame_source.width(), frame_source.height());
    let mut alarm = PerimeterAlarm::new(detector, tracker, cfg);

    for frame in frame_source {
        // 逐帧喂进去；报警在 process_frame 内部产生（存图/日志）。
        // 关键：单帧检测失败只记一笔、跳过这帧，绝不用 ? 让一帧坏数据杀死 24/7 服务。
        if let Err(e) = alarm.process_frame(&frame, Instant::now()) {
            eprintln!("处理帧失败: {:#}", e);
            continue;
        }
    }
    Ok(())
}
```

整套逻辑一目了然：**检测 → 过滤 → 跟踪 → 状态机(去抖) → 冷却 → 报警**，正好对应 27.1 的数据流图。想加绊线，就在第 4 步循环里并行挂上 27.4-② 的 `segments_intersect` 判定（再加一个 `last_center` 字段）即可。

> **一个易踩的时间坑**：上面用 `Instant::now()`（单调递增的真实经过时间）驱动冷却，对**实时 RTSP** 没问题。但如果帧源是**离线文件**、你在用比实时快得多的速度回放，`Instant::now()` 的"30 秒冷却"就成了程序实际运行的 30 秒，而不是"视频里的 30 秒"，冷却行为会失真。离线回放时应该用**帧号 / 帧率**换算出的"视频时间"来喂冷却器。

## 27.8 工程坑位清单

- **坐标必须全程在原图像素系**。禁区顶点、警戒线端点、Track 的框、脚点——统统是**原图像素坐标**（呼应第 21、22 章）。检测器内部的 letterbox 坐标、跟踪器的归一化坐标，出了各自的模块都要还原到原图，否则和禁区一比就是错位。
- **标定分辨率 ≠ 运行分辨率就要缩放顶点**。这是本章最高频的翻车点。运维在 1280×720 的预览上圈的禁区，拿到 1920×1080 的实际流上跑，禁区会缩在左上角一小块。所以 `config.calib_size` 要存下来，加载后按 `实际宽/标定宽`、`实际高/标定高` 把所有顶点缩放一遍再用——就是 27.2 的 `PerimeterConfig::scale_to`，`main` 里拿到流分辨率后调一次。
- **ID 跳变会导致重复报警或漏报**。跟踪器把一个人跟丢又重分了新 ID，他的 `last_inside`/`inside_streak` 清零，重新走一遍"确认 → 报警"，于是同一个人被报第二次；反过来，两个人的 ID 被张冠李戴，也可能漏报。**报警系统的上限，就是跟踪的稳定性**（第 21 章）。冷却是最后一道兜底，但治标不治本。
- **夜间 / 逆光会漏检**。周界报警往往 24 小时值守，而检测器在夜间、逆光、大雨下的召回会明显下降——漏检就没有 Track，再完美的判定逻辑也无米下锅。工程上要么上红外/补光，要么针对夜间场景单独调阈值、换模型。
- **多路视频的状态必须隔离**。一台机器同时跑十几路摄像头时，**每一路都要有自己独立的一套** `tracker` / `last_inside` / `inside_streak` / `cooldown`。绝不能几路共用——A 路的 track#5 和 B 路的 track#5 是两个完全不相干的目标，状态串了就全乱套。最干净的做法是每路视频一个独立的 `PerimeterAlarm` 实例。
- **HashMap 会内存泄漏**。`last_inside` 等三个表若只加不删，跑几天就塞满早已消失的 track id。骨架里第 5 步用 `retain` 按"本帧还活着的 id"清理，就是防这个（和第 22 章 `MotionHistory` 的泄漏是同一类坑）。

## 27.9 小结

- 周界入侵报警是全书第一个**视频流分析项目**：不再处理单张图，而是把一路视频**逐帧**送过"检测 → 跟踪 → ROI → 绊线 → 冷却 → 报警"这条流水线。
- **入侵是跨帧"事件"而非单帧"状态"**，所以必须**先跟踪拿稳定 ID**，才能捕捉"从区外踏入区内"的跳变，并实现"同一个人只报一次""短暂遮挡不算新人"。
- 判定点用**脚底中点** `(cx, y2)`，因为业务关心"脚踩没踩进禁区"；调第 22 章 `point_in_polygon`、第 23 章 `side()`/线段相交，不重复造轮子。
- **区域入侵**是个极简状态机：`HashMap<u64,bool>` 记上一帧在否，只在 `false→true` 触发；**绊线穿越**则用上一帧到这一帧的中心点连线去和警戒线求交。
- 三道防线压误报和刷屏：**类别+阈值**过滤只留 person、**N-of-M 连续确认**滤掉单帧误检、**Cooldown** 按 track_id 去重；全部串进 `PerimeterAlarm::process_frame`。
- 上线要盯死的坑：坐标全程原图像素、标定/运行分辨率不一致要缩放顶点、ID 跳变、夜间漏检、多路状态隔离、HashMap 清理。生产里报警动作通常改成推 MQ/webhook/DB（可靠投递见第 32 章）。

## 27.10 练习

1. **加"逗留超过 T 秒才报警"**。现在是"踏入即报"。改成：一个人**连续在禁区内停留超过 T 秒**才报警（防止只是路过蹭一下边缘）。提示：给每个 track 记一个"确认进入的时间戳/帧号"，在 `confirmed_in` 为真时算已停留时长，够 T 秒再放行报警。想清楚它和 `confirm_frames` 去抖的区别——一个防误检，一个是业务规则。

2. **支持多个禁区**。把 `config.zone` 从单个多边形改成 `Vec` of 多边形（比如 `zones: Vec<ZoneCfg>`，每个带自己的名字和冷却）。`process_frame` 里对每个 track、每个禁区分别判定，报警日志带上"闯入了哪个区"。注意 `last_inside` 的键要从 `u64` 变成 `(zone_id, track_id)`。

3. **把"闯入"改成"离开禁区"报警**。有的场景要反过来：目标**不许离开**某个区域（比如工人不得走出安全区）。只需把状态机的触发条件从 `!was_in && now_in` 改成 `was_in && !now_in`。动手改完，想想"一个人第一次出现时就在区外"会不会误报，该怎么处理初始状态。

4. **双绊线判进出方向**。用两条平行的警戒线，判断目标是"进"还是"出"：先穿过 A 线、再穿过 B 线算"进"，反之算"出"。提示：给每个 track 记录它最近穿过的是哪条线，用第 23 章 `side()` 的符号变化定方向。这正是**第 28 章人流统计**的核心——那里会把它做成一个完整的进出计数器。
