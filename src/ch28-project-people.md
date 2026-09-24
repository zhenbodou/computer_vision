# 第 28 章 · 实战三：人流统计与人脸抓拍

> **本章导读**：这是本部分综合最难的一个项目。它同时压上两条要求都很硬的子任务：一边要用**绊线**统计进出人数——计数必须**准，不重、不漏**；一边要给视野里的每个人**抓拍一张最清晰、最正的脸**存档，而人脸框和人体框来自两个不同模型，得**跨模型绑定**（人体框 ↔ 人脸框）。这一章把第七部分几乎所有过滤器都用上：ROI 之外的**绊线**（第 23 章）、**去重/冷却**（第 24 章）、**脸人绑定 containment**（第 25 章），再叠上检测（第 26 章）与跟踪（第 21 章）。读完你会得到一个能上线的 `PeopleSystem`：一个主循环把"检测 → 跟踪 → 计数 → 人脸绑定 → 抓拍 → 落盘"串成一条流水线。

---

## 28.1 需求拆解：两条独立的子任务

先把系统要干的事说清楚。它其实是**两件互不依赖的事**，共用同一条"检测 + 跟踪"的上游：

- **子任务①：过线计数。** 在画面里画一条**虚拟计数线**，规定"从上往下穿过算**进**、从下往上穿过算**出**"，屏幕实时显示 `进 X / 出 Y`。核心诉求是**计数准**：同一个人穿一次只能 +1，不能因为他在画面里出现了 30 帧就数 30 次。
- **子任务②：人脸抓拍。** 每个人从进入视野到离开，给他挑**一张**最清晰、最正的脸存档（不是每帧都存，那样又多又糊）。等他走出视野时，把这张"最佳脸"落盘，（可选）算特征入库或查重。

两件事**都建立在同一条上游**——人体检测器（第 26 章，只取 `person=0`）+ 多目标跟踪（第 21 章，给每个人一个稳定的 `track_id`）——然后各走各的分支。整个系统的数据流是这样：

```text
                    视频帧流 (RTSP / 文件, 第6章)
                         │  一帧 RgbImage
                         ▼
              ┌───────────────────────────┐
              │  人体检测器 Detector         │ 第16/26章：只留 person(class_id=0)
              │  detect() → Vec<Detection>  │
              └──────────────┬──────────────┘
                             ▼
              ┌───────────────────────────┐
              │  多目标跟踪 Tracker          │ 第21章：给每个人一个稳定 track_id
              │  update() → Vec<Track>      │
              └───────┬───────────┬─────────┘
                      │           │
        ┌─────────────┘           └───────────────┐
        ▼  子任务① 过线计数              子任务② 人脸抓拍 ▼
   ┌────────────────┐                 ┌──────────────────────┐
   │ 上帧中心↔本帧中心 │                 │ detect_faces()        │ 第20章
   │ 连线与线相交?    │ 第23章            │ containment 绑到人体   │ 第25章
   │ side() 定方向    │                 │ 质量分择优, 存"最佳脸"  │
   │ GateCounter 去重 │ 第24章            └──────────┬───────────┘
   └───────┬────────┘                             │ track 消失(max_age)
           ▼                                       ▼
      进 X / 出 Y（实时刷新）              最佳脸落盘 +（可选）算特征入库/查重
```

看清这张图很关键：**计数和抓拍是两条平行支路**，唯一的公共祖先是"带 `track_id` 的人体轨迹"。下面两节先把计数支路彻底讲透，再回头讲抓拍支路。

## 28.2 计数原理：为什么必须基于跟踪 ID

新手最容易想到的计数法是：**逐帧检测，每帧数一下框，看有没有跨过线**。这条路一定翻车。视频是每秒几十帧的连续画面，一个人走过计数线要花十几帧，如果你逐帧判断"这一帧有没有人在线上"，同一个人会被反复命中，一次穿越被数成十几次。

问题的根子是：**逐帧检测里的框是"无记忆"的**——这一帧的框和上一帧的框，系统并不知道是不是同一个人。要计数不重不漏，就必须先回答"这是同一个人吗"，而这正是第 21 章**多目标跟踪**给出的东西：`track_id`。同一个人在画面里走，`track_id` 从头到尾不变。**有了稳定的身份，计数才从"数框"变成"数人"**。

那怎么判定一个 track "穿过了线"？靠一帧的位置不行（他可能一直站在线附近）。正确判据是看**运动**：把这个 track **上一帧的中心点**和**本帧的中心点**连成一条小线段，如果这条**运动线段**和**计数线段**相交（第 23 章的 `segments_intersect`），就说明他在这两帧之间跨过去了。

```text
   计数线  A ●──────────────────────● B
                       ╱
          上帧中心 ●───╱──● 本帧中心   ← 两帧中心连线与计数线相交 = 穿越
                     ╳ 交点
```

方向怎么定？用第 23 章的 `side()`——它算的是叉积符号，告诉你一个点在有向线段 `A→B` 的**哪一侧**（正/负）。同一个点，穿线前在一侧、穿线后到了另一侧，`side` 的符号必然**翻转**。于是：

- `side` 从**负变正** → 本例约定为"从上往下" → **进**
- `side` 从**正变负** → "从下往上" → **出**

（符号到"进/出"的映射，取决于你怎么画线、以及图像 y 轴朝下这个约定，**上线前校准一次**即可，见 28.3 最后一行。）

把"记录每个 track 的上一帧中心 → 判相交 → 定方向 → 去重计数"串成完整逻辑：

```rust,ignore
use std::collections::HashMap;
use crate::{Track, side, segments_intersect}; // 第21章 Track，第23章 side/相交
use crate::GateCounter;                        // 第24章：按 (track_id, 方向) 去重的闸机计数器

// 方向约定：这里不自定义枚举，直接用第 24 章 GateCounter::on_cross 认的 i8——
//   +1 = 进（本例「从上往下」）， -1 = 出（「从下往上」）。
// 于是几何层用第 23 章的 side / segments_intersect，计数层用第 24 章的 i8，一路无缝对接。

type P = (f32, f32);

/// 判断一个 track 从 prev 走到 cur，是否穿过计数线；穿了就返回方向（+1 进 / -1 出）
fn cross_direction(line: (P, P), prev: P, cur: P) -> Option<i8> {
    let (a, b) = line;
    // 1) 两帧中心的「运动线段」是否真的和计数线段相交（第 23 章）
    if !segments_intersect(a, b, prev, cur) {
        return None; // 没穿过——他可能只是在线附近走动
    }
    // 2) 用叉积符号判方向：中心点从线的哪一侧到了哪一侧（第 23 章 side）
    let s_prev = side(a, b, prev); // 上帧在线的哪侧
    let s_cur = side(a, b, cur);   // 本帧在线的哪侧
    if s_prev < 0.0 && s_cur > 0.0 {
        Some(1) // 负→正：本例约定「从上往下 = 进」
    } else if s_prev > 0.0 && s_cur < 0.0 {
        Some(-1) // 正→负：从下往上 = 出
    } else {
        None // 符号没真正翻转（擦着线、停在线上），不计
    }
}

/// 对本帧所有 track 做一次计数，同时更新「每个 track 的上一帧中心」
fn count_crossings(
    tracks: &[Track],
    line: (P, P),
    last_center: &mut HashMap<u64, P>, // track_id -> 上一帧中心
    counter: &mut GateCounter,
) {
    for tk in tracks {
        let cur = tk.bbox.center(); // 复用 BBox::center()
        if let Some(&prev) = last_center.get(&tk.id) {
            if let Some(dir) = cross_direction(line, prev, cur) {
                // GateCounter 内部按 (track_id, dir) 去重：同一人同方向只计一次
                counter.on_cross(tk.id, dir);
            }
        }
        last_center.insert(tk.id, cur); // 本帧中心存起来，供下一帧当「上一帧」
    }
}
```

**去重那一步是灵魂**。`cross_direction` 只判断"这两帧之间有没有穿"，但一个人穿线那一刻，可能连着两三帧的运动线段都压在计数线上，触发多次。所以最后交给第 24 章的 `GateCounter`：它内部维护一张"**这个 `track_id` 这个方向是不是已经计过**"的记录（一个 `HashSet<(u64, i8)>`），`on_cross` 时若已存在就忽略。这样"同一个人同一个方向"永远只 +1。显示时读 `counter` 的字段即可（`entered` 字段=进、`exited` 字段=出）：

```rust,ignore
println!("进 {} / 出 {}", counter.entered, counter.exited);
```

## 28.3 计数的坑：为什么"看起来对"却数不准

计数看着简单，真上线才发现处处是坑。它们大多不是代码 bug，而是**几何位置和跟踪质量**的问题。集中列一遍：

| 坑 | 现象 | 根因 | 对策 |
| --- | --- | --- | --- |
| **ID 跳变** | 一个人被数两次，或漏数 | 遮挡、检测丢帧导致 track 断开又以**新 ID** 重开（第 21 章）；新旧 ID 各算各的 | 调好跟踪 `max_age` / IoU 关联阈值；用 ByteTrack 保住低分框；把线放在跟得最稳的区域 |
| **线位置太差** | 多人挤在线上，数不清 | 线画在出入口、闸机口等**拥挤处**，人体框互相重叠、中心点乱 | 线画在**画面中部、人流已经分散**的地方，且尽量**垂直于主流向** |
| **斜穿 / 擦线** | 时数时不数 | 中心轨迹**擦着**线走，`side` 符号在 0 附近抖动 | 用 `segments_intersect` 判**真穿越**；要求符号**真正翻转**（`< 0` 到 `> 0`）才算 |
| **线上横跳** | 一个人反复 +1 | 徘徊者在线两侧来回蹭，反复触发穿越 | 第 24 章**冷却/状态机**：一个 track 计过一次后进入冷却期，短时间内不再计 |
| **双向对穿** | 进出方向记反了 | `side` 符号与"进/出"的映射没校准 | 按画面 y 轴朝向**校准一次**：本例图像 y 朝下、线左端为 A，负→正即"向下=进" |

前两条呼应第 21 章：**计数准不准，七成看跟踪稳不稳**。ID 频繁跳变时，再漂亮的穿越判定也救不回来——所以工程上常把计数线**刻意放在画面中段**，那里人已经走开、框不重叠，跟踪最稳、中心点最干净。

"线上横跳"这条尤其阴险：一个人站在线边刷手机，身体轻微晃动，中心点在线两侧来回，每次都翻转符号。光靠 `GateCounter` 的"同方向只计一次"还不够（因为他是一进一出交替）。这时要请出第 24 章的**冷却/状态机去抖**：给每个 track 一个"刚计过 → 冷却 N 帧"的状态，冷却期内的穿越一律忽略，等他真的走远了才解冻。

## 28.4 人脸抓拍：绑定 + 质量分 + 只留最好

现在切到第二条支路。目标：**每个人在整段视野期间，只留一张最好的脸**。它分三步——绑定、打分、择优替换。

### 第一步：把脸绑到人体 track（为什么不能只按 IoU）

每帧我们对画面跑 `detect_faces`，拿到一堆人脸框。但人脸框来自人脸模型，人体框来自人体检测器，**是两套框**，得判定"这张脸属于哪个人体 track"。这正是第 25 章的活儿，判据是**包含比例 `containment`**（脸∩人体 / 脸面积），而**不是 IoU**：

人体框比人脸框大得多，就算脸 100% 落在人体框里，IoU 也可能只有 0.07（第 25 章那组数字：脸面积 2400、人体面积 35000，IoU ≈ 2400/35000），阈值根本没法设。而 `containment` 分母换成脸自己的面积，脸完全在人体里就是 `1.0`，范围永远 `0~1`，阈值好设（`≥ 0.6` 才认）。所以脸人绑定**必须用 containment**。

### 第二步：给脸打一个"质量分"

一个人走过来，会被拍到几十张脸：远处的小脸、侧过去的脸、糊掉的脸、正对镜头的清晰脸。我们要能**比较**它们，挑出最好那张。定义**人脸质量分**：

$$
\text{quality} = \underbrace{\text{脸框面积}}_{\text{越近越大越清晰}} \times \underbrace{\text{检测分}}_{\text{越像脸越高}} \times \underbrace{\text{正脸程度}}_{\text{越正越好}}
$$

三项相乘的意思是"**一票否决**"：脸再大，只要是侧脸（正脸程度低）总分也上不去；正得再标准，只要太小太糊也不行。前两项现成（`face.bbox.area()` 和 `face.score`），"正脸程度"用第 17 章那样的 **5 点关键点**估计左右偏转（yaw）：正脸时鼻尖大致在两眼的正中间，侧脸时鼻尖会偏向一边。

```rust,ignore
use crate::{BBox, Face}; // 第20章 Face{ bbox, score, landmarks:[(f32,f32);5] }

/// 正脸程度：用 5 点关键点估计左右偏转(yaw)，正脸≈1，侧脸→0
/// landmarks 约定顺序：[左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
fn frontal_score(lm: &[(f32, f32); 5]) -> f32 {
    let (lex, _) = lm[0];                    // 左眼 x
    let (rex, _) = lm[1];                    // 右眼 x
    let (nx, _) = lm[2];                     // 鼻尖 x
    let eye_cx = (lex + rex) * 0.5;          // 两眼中点 x
    let eye_dist = (rex - lex).abs().max(1e-3); // 两眼间距，用来做尺度归一化
    let off = (nx - eye_cx).abs() / eye_dist;   // 鼻尖偏离两眼中点的程度：正脸≈0，侧脸变大
    (1.0 - off).clamp(0.0, 1.0)              // 偏得越少越接近 1
}

/// 人脸质量分 = 面积 × 检测分 × 正脸程度（任一项差，总分就被拉低）
fn face_quality(face: &Face) -> f32 {
    face.bbox.area() * face.score * frontal_score(&face.landmarks)
}
```

### 第三步：每个 track 维护"目前见过的最佳脸"

给每个 track 存一份快照，只有新来的脸**质量分更高**才替换。这就是要落盘的东西：

```rust,ignore
/// 一个 track 到目前为止「最好的那张脸」的快照
pub struct FaceSnapshot {
    pub track_id: u64,
    pub best_score: f32,       // 目前最佳脸的质量分
    pub crop: image::RgbImage, // 抠出来的人脸图，等 track 结束时落盘
}

/// 从整帧里把人脸框区域抠出来（注意夹到图像边界内，别越界 panic）
fn crop_bbox(frame: &image::RgbImage, b: &BBox) -> image::RgbImage {
    let (fw, fh) = (frame.width() as f32, frame.height() as f32);
    let x = b.x1.clamp(0.0, fw) as u32;
    let y = b.y1.clamp(0.0, fh) as u32;
    let w = ((b.x2 - b.x1).max(1.0) as u32).min(frame.width() - x);
    let h = ((b.y2 - b.y1).max(1.0) as u32).min(frame.height() - y);
    image::imageops::crop_imm(frame, x, y, w, h).to_image()
}

/// 用本帧检测到的脸，更新某个 track 的最佳脸快照
fn update_best_face(
    track_id: u64,
    face: &Face,
    frame: &image::RgbImage,
    best_face: &mut std::collections::HashMap<u64, FaceSnapshot>,
) {
    let q = face_quality(face);
    // 没存过 → 直接存；存过 → 只有更好才替换（map_or 处理「没存过」的情况）
    let better = best_face.get(&track_id).map_or(true, |snap| q > snap.best_score);
    if better {
        best_face.insert(track_id, FaceSnapshot {
            track_id,
            best_score: q,
            crop: crop_bbox(frame, &face.bbox), // 只有更好时才抠图，省内存拷贝
        });
    }
}
```

把绑定和择优连起来：每帧对每个 person track，在"落进它人体框（`containment ≥ 0.6`）"的脸里挑质量分最高的那张，再拿去更新最佳脸。

## 28.5 生命周期：人离开视野时才落盘

抓拍有个和计数很不一样的**时间点问题**：到底什么时候把最佳脸写到磁盘？

答案是——**等这个 track 结束（离开视野）时**，而不是每帧写、也不是刚绑到脸就写。道理很简单：**在他还在画面里的时候，最佳脸随时可能被更好的一张刷新**。他刚进画面时是个远处小侧脸，走到中间变成清晰大正脸——如果你一绑到脸就落盘，存下来的就是那张糊的。只有等他走出视野、确定不会再有更好的脸了，此刻手上的 `best_face` 才是**这一趟真正的最佳**。

那"离开视野"由谁判定？由第 21 章的跟踪器：一个 track 连续多帧没匹配上检测，达到 `max_age` 阈值，跟踪器就认定它**彻底消失**并移除。我们**订阅这个事件**：跟踪器报告某 track 消失时，触发一次落盘。这就是"**抓拍在 track 结束时提交**"的生命周期配合——计数在"穿越那一刻"提交，抓拍在"track 结束那一刻"提交，两者节奏不同。

```rust,ignore
/// track 消失（max_age 到）时调用：把它这一趟的最佳脸落盘并清理状态
fn on_track_lost(
    track_id: u64,
    last_center: &mut std::collections::HashMap<u64, (f32, f32)>,
    best_face: &mut std::collections::HashMap<u64, FaceSnapshot>,
) {
    last_center.remove(&track_id); // 计数用的上一帧中心，随 track 一起清掉（防内存泄漏）
    if let Some(snap) = best_face.remove(&track_id) {
        // 1) 落盘：文件名带上 track_id 和质量分，方便回溯
        let path = format!("snapshots/track_{}_{:.0}.jpg", snap.track_id, snap.best_score);
        let _ = snap.crop.save(&path); // 工程上要处理 Err（磁盘满/无权限）

        // 2)（可选）算特征入库 / 查重：指路第 20 章人脸库、第 25 章
        //    - 把 crop 当一张小图，重新 detect_faces + face_embedding 得到 512 维向量
        //    - 与人脸库逐一 cosine 比对：命中已知人→打标签；没命中→作为新人入库
        //    let faces = detect_faces(&snap.crop);
        //    if let Some(f0) = faces.first() {
        //        let emb = face_embedding(&snap.crop, f0); // L2 归一化 512 维
        //        // gallery.match_or_insert(&emb, 0.5) —— cosine 阈值查重
        //    }
    }
    // best_face 里若没有该 track，说明它整趟都没绑到过清晰脸（一直背对镜头），
    // 那就什么也不存——这是对的，别硬存一张糊脸。
}
```

注意最后那句注释：**没绑到脸就不存**，是合理行为，不是 bug。一个全程背对镜头的人，我们照样把他计进人流数，但没有他的脸可抓——诚实地留空，比硬存一张模糊侧脸有用。

## 28.6 整合：PeopleSystem 主循环

把两条支路和上游拼进一个结构体，就是可以逐帧喂视频的完整系统。

```rust,ignore
use std::collections::HashMap;
use crate::{Detector, Tracker, Track, GateCounter};

type P = (f32, f32);

pub struct PeopleSystem {
    detector: Detector,                       // 人体检测器（第26章）
    tracker: Tracker,                         // 多目标跟踪（第21章）
    line: (P, P),                             // 计数线两端点（原图像素坐标）
    counter: GateCounter,                     // 进/出计数器（第24章，内部去重）
    last_center: HashMap<u64, P>,             // track_id -> 上一帧中心（计数用）
    best_face: HashMap<u64, FaceSnapshot>,    // track_id -> 目前最佳脸（抓拍用）
    frame_idx: u64,                           // 帧序号，用于隔帧跑人脸（见 28.7）
}

impl PeopleSystem {
    pub fn new(detector: Detector, tracker: Tracker, line: (P, P)) -> Self {
        Self {
            detector, tracker, line,
            counter: GateCounter::default(),
            last_center: HashMap::new(),
            best_face: HashMap::new(),
            frame_idx: 0,
        }
    }

    /// 处理一帧：检测 → 跟踪 → 计数 → 人脸绑定 → 抓拍 → （track 结束）落盘。
    /// detect 可能失败（推理出错），故返回 Result；主循环记一笔、跳过坏帧，不崩整条流水线。
    pub fn process_frame(&mut self, frame: &image::RgbImage) -> anyhow::Result<()> {
        // ── 上游：检测 + 跟踪 ───────────────────────────────
        let mut dets = self.detector.detect(frame)?;
        dets.retain(|d| d.class_id == 0); // 只要 person（COCO 类别约定：person=0）
        let tracks: Vec<Track> = self.tracker.update(&dets); // 每个人一个稳定 track_id

        // ── 子任务①：过线计数 ──────────────────────────────
        for tk in &tracks {
            let cur = tk.bbox.center();
            if let Some(&prev) = self.last_center.get(&tk.id) {
                if let Some(dir) = cross_direction(self.line, prev, cur) {
                    self.counter.on_cross(tk.id, dir); // 同一 track 同方向只计一次
                }
            }
            self.last_center.insert(tk.id, cur); // 更新「上一帧中心」
        }

        // ── 子任务②：人脸抓拍 ──────────────────────────────
        // 性能门控（28.7）：隔帧跑一次，且只在有 track 需要脸时才跑人脸检测
        if self.frame_idx % FACE_EVERY_N == 0 && self.need_faces(&tracks) {
            let faces = detect_faces(frame); // 整帧所有脸（第20章）
            for tk in &tracks {
                // 在「落进这个人体框」的脸里，挑质量分最高的那张（第25章 containment 绑定）
                let best = faces.iter()
                    .filter(|f| containment(&f.bbox, &tk.bbox) >= 0.6) // 用包含比，别用 IoU
                    .max_by(|x, y| face_quality(x).total_cmp(&face_quality(y)));
                if let Some(face) = best {
                    update_best_face(tk.id, face, frame, &mut self.best_face); // 更好才替换
                }
            }
        }

        // ── 生命周期：跟踪器报告哪些 track 到达 max_age 彻底消失（第21章）──
        // just_removed() 返回本帧被清理的 id（第 21 章为此暴露的接口，见接口契约）；
        // 返回 owned Vec，好让循环体里能接着 &mut 借 self 的其他字段而不冲突。
        for gone in self.tracker.just_removed() {
            on_track_lost(gone, &mut self.last_center, &mut self.best_face);
        }

        self.frame_idx += 1;
        Ok(())
    }

    /// 是否还有 track 需要跑人脸：只要有人「还没攒到足够好的脸」就跑（28.7）
    fn need_faces(&self, tracks: &[Track]) -> bool {
        tracks.iter().any(|tk| {
            self.best_face.get(&tk.id).map_or(true, |s| s.best_score < GOOD_ENOUGH)
        })
    }

    /// 实时读数，画到屏幕上（entered=进、exited=出）
    pub fn readout(&self) -> (u64, u64) {
        (self.counter.entered, self.counter.exited) // (进, 出)
    }
}

const FACE_EVERY_N: u64 = 3;     // 每 3 帧跑一次人脸检测
const GOOD_ENOUGH: f32 = 5.0e4;  // 质量分够高就不再为这个 track 跑人脸（数值按实测标定）
```

这段代码把本章所有零件串了起来，`process_frame` 一帧只需一次遍历就完成计数与抓拍两件事。上层拿到解码后的帧（第 6 章）循环喂进来即可：

```rust,ignore
let mut sys = PeopleSystem::new(detector, tracker, ((0.0, 540.0), (1920.0, 540.0)));
for frame in video_frames { // 第6章解码出的 RgbImage 流
    // 单帧检测失败只记一笔、跳过这帧，别让一帧坏数据杀死长跑服务
    if let Err(e) = sys.process_frame(&frame) {
        eprintln!("处理帧失败: {:#}", e);
        continue;
    }
    let (into, out) = sys.readout();
    // draw_line + draw_text：把计数线和「进 into / 出 out」叠加到画面上
}
```

## 28.7 性能考量：人脸检测比人体检测贵得多

最后一节谈工程账。上面主循环里 `FACE_EVERY_N` 和 `need_faces` 那两处门控不是可有可无的优化，而是**能不能跑实时**的关键。

现实是：**人脸检测通常比人体检测更贵**——它往往要在整帧上找很多个小脸，还常配对齐、关键点，算力吃得多。如果每帧、对每个人都跑一遍人脸检测，帧率会被拖垮。人流场景又恰恰是"很多人同时在画面里"，更扛不住。所以工程上有三条省算力的常规操作，本章代码已经埋了两条：

| 策略 | 做法 | 为什么有效 |
| --- | --- | --- |
| **隔 N 帧跑一次** | `frame_idx % FACE_EVERY_N == 0` | 抓拍要的是"整段最好一张"，不必逐帧；隔几帧完全够挑出好脸 |
| **只对"还没拿到好脸"的 track 跑** | `need_faces`：所有人都已攒到 `GOOD_ENOUGH` 就跳过 | 一个人拿到清晰正脸后，就没必要再为他花人脸算力 |
| **只在框足够大时跑** | 人体框太小（远处）时不跑人脸 | 远处小脸本来就识别不了、质量分也低，跑了白跑 |

第三条留给读者按需加（在绑定前 `filter` 掉 `tk.bbox.area()` 过小的 track 即可）。

还有更进一步的做法：**把人脸检测放到另一个线程/另一个推理流**，与人体检测、跟踪**并发**，让 CPU/GPU 别互相等。这属于系统层的工程话题——**详见第 29 章**，那里会系统讲并发、降频、批处理与部署，把本章这个单线程主循环改造成能扛高并发的服务。

## 28.8 小结

- 本系统是**两条平行子任务**共用"检测 + 跟踪"上游：①绊线**过线计数**，②每人一张**最佳脸抓拍**；先看懂 28.1 那张数据流图再看代码。
- 计数**必须基于 `track_id`**（第 21 章），不能逐帧数框——否则一次穿越被数很多次。穿越判定 = **上帧中心↔本帧中心**的运动线段与**计数线段相交**（`segments_intersect`），方向由 **`side()` 符号翻转**（负→正=进 / 正→负=出）决定，再用 **`GateCounter` 按 (track_id, 方向) 去重**保证同人同向只 +1。
- 计数的坑几乎都在"几何 + 跟踪质量"：**ID 跳变**（调好跟踪、线放稳区）、**线画在拥挤处**（改画到画面中部人流分散处）、**擦线/斜穿**（要求符号真翻转）、**线上横跳**（第 24 章冷却/状态机去抖）、**方向记反**（按 y 轴校准一次）。
- 人脸抓拍三步：用第 25 章 **`containment` 把脸绑到人体 track**（脸框小、必须用包含比而非 IoU）；用**质量分 = 面积 × 检测分 × 正脸程度**（5 点关键点估偏转）比较优劣；每个 track 只留**目前最好**的一张（更好才替换）。
- 抓拍的落盘时机是**track 结束（`max_age` 到）那一刻**，不是每帧、不是刚绑到脸——因为在人还在画面时最佳脸随时会被刷新；落盘时可（可选）算 512 维特征入库/查重（第 20、25 章）。全程没绑到脸就诚实留空。
- 人脸检测贵，工程上靠**隔 N 帧跑、只对没好脸的 track 跑、框够大才跑**降频，再往上就是第 29 章的**并发与部署**。

## 28.9 练习

1. **双绊线让进出判定更稳**：单条计数线在"擦线/横跳"时容易误判。仿照第 23 章的**双绊线**思路，在主流向上并排画**两条**线 L1、L2，规定必须**先穿 L1 再穿 L2**（顺序对上）才算一次"进"，反过来才算"出"。改写 `cross_direction` 与状态记录，想清楚每个 track 要多存什么状态（提示：记"已经穿过哪条线、朝哪个方向"），并说明它为什么能把徘徊者的横跳过滤掉。

2. **按小时统计人流**：给 `PeopleSystem` 增加"分时段计数"。每记一次进/出时带上**时间戳**，把 `GateCounter` 换成 `HashMap<u8 /*小时*/, (u64, u64)>`。跑完一段视频后打印一张"每小时进/出"表，并画出哪个时段最挤（可结合本书 dataviz 思路做个简单柱状）。

3. **同一张脸多次抓拍去重**：真实场景里同一个人可能出画又入画，被当成两个 track，落盘两张几乎一样的脸。在 `on_track_lost` 落盘前，对最佳脸 `crop` 算 `face_embedding`，与"已落盘特征库"逐一 `cosine` 比对，**相似度超过阈值（如 0.5）就判为同一人、不再重复存**（指路第 20 章人脸库、第 25 章）。写出这段查重逻辑，并讨论阈值设高/设低分别会带来什么问题。

4. **把计数线做成可配置 JSON**：现在计数线是硬编码的两个点。设计一份配置文件（如 `line.json`：`{"a":[0,540],"b":[1920,540],"in_dir":"down"}`），用 `serde` 反序列化到 `PeopleSystem`，让运维不改代码就能调整线的位置和"进"的方向。想一想 `in_dir` 字段怎么和 28.2 的 `side` 符号映射对应上——如何用一个配置项完成 28.3 表格里"方向校准"那一步。
