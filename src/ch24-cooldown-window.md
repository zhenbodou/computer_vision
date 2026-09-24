# 第 24 章 · 时间与去重过滤器：Cooldown、Window 与去重

> **本章导读**：第 22 章的空间过滤器解决了"事件发生在**哪里**、该不该算"，第 23 章的绊线告诉我们"有没有越界、往哪个方向越"。但它们都是**逐帧**工作的——模型每秒吐几十帧，跟踪偶尔还会抖，于是**同一件事会被触发几十次**。把这些原始触发直接丢给报警或计数系统，结果就是**报警刷屏、重复计数**。本章补上业务过滤的最后一块拼图：**时间维度的过滤**。我们用 **Cooldown（冷却 / 防抖）**、**Window（滑动时间窗）** 和 **去重（dedup）** 三件套，把"脏、乱、刷屏的原始触发"整理成"干净、不重复的业务事件"，并用一个 `EventFilter` 把它们串成一条流水线。读完你能亲手写出一个不刷屏、不重复计数的事件过滤器。

## 24.1 为什么要在时间维度上过滤

先看一个真实场景。摄像头 25 fps，一个人走到绊线上停了 2 秒才迈过去。这 2 秒 = **50 帧**，每一帧模型都检测到这个人、跟踪都给他一个稳定 ID、绊线判断"他在线附近反复横跳"。于是：

- 如果**每次触发就 `+1` 计数**：一个人过线，你数出了 30 多次。
- 如果**每次触发就报警**：2 秒内弹了 30 多条"有人越界"，值班员直接崩溃。

问题的根源是：**模型和跟踪是"连续信号"，而业务要的是"离散事件"。** 一个人越一次线，业务只想要**一条**记录，不管底层触发了多少次。

打两个生活比喻，你立刻就懂本章三件套分别在干什么：

| 生活场景 | 现象 | 对应技术 |
| --- | --- | --- |
| **门铃防连按** | 你戳门铃戳了 5 下，屋里只"叮咚"一次，之后按也不响，过几秒才恢复 | **Cooldown（冷却）**：报过一次，短时间内压住同类 |
| **手机通知合并** | 群里刷了 10 条消息，通知栏只显示"3 个新会话"，而不是响 10 次 | **Window（时间窗聚合）**：窗内累计、达阈值才提醒 |
| **已读回执** | 同一条消息不会重复标记"已读第 2 次" | **去重（dedup）**：同一个体同一件事只算一次 |

这三件套处在整条视频分析流水线的末端，把上游的"原始触发"收拾干净再交给下游：

```text
逐帧检测 ─▶ 跟踪(第21章) ─▶ 空间过滤(第22章) ─▶ 绊线(第23章)
                                                     │
                                          原始触发（几十次/秒，会抖动、会重复）
                                                     │
                                                     ▼
                                     ┌────────────────────────────┐
                                     │         EventFilter         │
                                     │  ① 去重  ② 窗口确认  ③ 冷却 │
                                     └────────────────────────────┘
                                                     │
                                          干净业务事件（不刷屏、不重数）
                                                     │
                                                     ▼
                                        报警 / 计数系统（第27章）
```

先把本章统一用到的**事件类型**定义出来，后面所有代码都复用它。它带着上游给的 `track_id`（第 21 章的跟踪 ID）、`class_id`（COCO 类别）和 `direction`（第 23 章的越线方向）：

```rust,ignore
/// 事件类型（来源见第 22、23 章）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventKind {
    LineCross,  // 绊线越界（第 23 章）
    RoiIntrude, // 区域入侵（第 22 章）
    Loiter,     // 徘徊
}

/// 一条"候选事件"：还没经过时间过滤，可能是抖动、可能重复
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub kind: EventKind, // 什么事
    pub track_id: u64,   // 谁触发的（第 21 章的跟踪 ID）
    pub class_id: usize, // COCO 类别（如 person=0）
    pub direction: i8,   // 方向：+1 进 / -1 出 / 0 无（第 23 章）
}
```

## 24.2 时间基准：用帧号还是墙钟？

三件套全都要回答"**现在是什么时候、距上次过了多久**"。这个"时间"从哪来，直接决定算得对不对。

Rust 标准库给了两把尺子：

- **`std::time::Instant`**：单调递增的墙钟时刻，`Instant::now()` 取当前时刻，两个 `Instant` 相减得到 `Duration`。它反映的是**真实流逝的物理时间**。
- **帧号 `frame_index`**：视频里的第几帧。配合帧率 `fps`，可换算成"视频内部时间"：`t_video = frame_index / fps`。

选哪把尺子，取决于你处理的是**实时流**还是**离线回放**：

| 场景 | 该用什么当"现在" | 为什么 |
| --- | --- | --- |
| **实时流**（RTSP 摄像头） | `Instant::now()` | 帧到达 = 真实时间流逝；掉帧、卡顿都真实体现在时间差里 |
| **离线回放**（分析一段 mp4） | `frame_index / fps` | 你可能 10 倍速处理、也可能因 CPU 慢而 0.5 倍速，`Instant::now()` 是"处理时间"不是"视频时间"，拿它做冷却会**全错** |

这是新手最容易踩的坑：**离线批量分析时用 `Instant::now()`**。你 5 分钟跑完 1 小时的录像，一个 3 秒冷却会把视频里相隔 30 秒的两次真实事件当成"同一秒"全压掉。回放**必须**用帧号换算的视频时间。换算很简单：

```rust,ignore
use std::time::Duration;

/// 把"帧号 + 帧率"换算成从视频开头起算的时间
fn frame_to_media_time(frame_index: u64, fps: f64) -> Duration {
    Duration::from_secs_f64(frame_index as f64 / fps) // 25 fps 下，第 75 帧 = 3.0 秒
}
```

小数字换一遍：`fps = 25` 时**每帧 40 ms**；一个 **3 秒**冷却 = **75 帧**。所以帧号和时间是可以互相翻译的。

本章后面的实现统一用 `Instant` 演示（贴近实时流）。**要用在回放上，只需把每个 `now: Instant` 换成 `media_time: Duration`——比较逻辑一模一样，都是"做减法、比大小"。** 工程上通常会把过滤器对时间类型做成泛型，或干脆统一传一个"从流开始起算的 `Duration`"，让实时和回放走同一套代码。

## 24.3 Cooldown：冷却与防抖

**Cooldown（冷却）** 的规则一句话：**某类事件触发一次后，在 `T` 时间内屏蔽同类事件。** 做法也一句话：记住"上次触发时刻 `last_fire`"，新事件来了先看 `now - last_fire < T`，是就丢弃。

```rust,ignore
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 冷却器：某个 key 触发后，在 cooldown 时间内屏蔽同 key 的事件。
/// key 用泛型，就能自由选择"按什么维度冷却"。
pub struct Cooldown<K> {
    cooldown: Duration,
    last_fire: HashMap<K, Instant>, // 每个 key 记一个"上次放行时刻"
}

impl<K: std::hash::Hash + Eq + Clone> Cooldown<K> {
    pub fn new(cooldown: Duration) -> Self {
        Self { cooldown, last_fire: HashMap::new() }
    }

    /// 返回 true = 放行（可以上报），false = 冷却中、丢弃
    pub fn allow(&mut self, key: K, now: Instant) -> bool {
        if let Some(&last) = self.last_fire.get(&key) {
            // saturating_duration_since 不会 panic：即使 now 早于 last 也返回 0
            if now.saturating_duration_since(last) < self.cooldown {
                return false; // 还在冷却窗内
            }
        }
        self.last_fire.insert(key, now); // 记下这次放行时刻
        true
    }
}
```

> 小提示：用 `now.saturating_duration_since(last)` 而不是 `now - last`。后者在 `now` 早于 `last` 时会 **panic**（合成时间戳、多线程乱序时可能发生），前者会安全地返回 0。

**冷却效果用小数字走一遍**（`cooldown = 2` 秒，同一个人反复在线上横跳）：

```text
t=0.0s  → 表里没记录          → 放行！last_fire = 0.0
t=0.3s  → 0.3 - 0.0 = 0.3 < 2 → 丢弃
t=1.5s  → 1.5 - 0.0 = 1.5 < 2 → 丢弃
t=2.1s  → 2.1 - 0.0 = 2.1 ≥ 2 → 放行！last_fire = 2.1
t=2.4s  → 2.4 - 2.1 = 0.3 < 2 → 丢弃
结论：几十次原始触发 → 只上报 2 次。
```

### 按什么维度冷却？

关键设计问题：`key` 选什么？这决定"谁压住谁"。泛型 `Cooldown<K>` 让你按需切换：

| 冷却维度 | `key` 类型 | 效果 | 适合场景 |
| --- | --- | --- | --- |
| **全局** | `EventKind` | 整个画面这类事 `T` 秒只报一次 | 单点、简单场景 |
| **按 track** | `(u64, EventKind)` | 每个目标各自冷却 | 多人同时越线，各报各的，互不压制 |
| **按区域** | `(u32, EventKind)` | 每条绊线 / 每个 ROI 各自冷却 | 多区域周界 |
| **区域 + 目标** | `(u32, u64, EventKind)` | 最细粒度 | 复杂场景 |

一个常见 bug：用**全局**冷却，结果 A、B 两人同时越线，A 报了、B 被 A 的冷却压掉漏报。多目标场景通常要**按 track 冷却**（`key = (track_id, kind)`）。

### 冷却时长怎么定？

| `T` 取值 | 后果 |
| --- | --- |
| **太短**（如 0.2s） | 抖动直接穿透，还是刷屏 |
| **太长**（如 60s） | 两次真实事件被合并，**漏报第二次** |
| **按业务定** | 报警场景常用 3~10s；纯计数场景一般不用冷却，而用下面的"去重" |

记住一句话：**Cooldown 适合"报警"**（同一件事可以再报，只是别太频繁）；**去重适合"计数"**（一个个体过一次线，永远只 `+1`）。两者解决的是不同业务诉求。

## 24.4 Window：滑动时间窗

Cooldown 是"报过就压住一段时间"，比较粗暴。很多业务需要更聪明的判断：**看一段时间窗里的整体情况再决定**。这就是 **Window（滑动时间窗）**。它有两类最常见的用法。

### 用法一：计数窗——"N 秒内 M 次才算数"

比如"**1 分钟内越界 ≥3 次才算入侵**"（偶尔蹭一下不报，反复试探才报）。做法：用一个**时间戳队列**记住窗内的每次命中，新事件进队尾，过期的从队头丢，队列长度就是"窗内命中数"。`VecDeque`（双端队列）天生适合这活。

```rust,ignore
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// 滑动计数窗：统计"最近 window 时间内"发生了多少次
pub struct CountWindow {
    window: Duration,
    hits: VecDeque<Instant>, // 窗内每次命中的时间戳，从旧到新
}

impl CountWindow {
    pub fn new(window: Duration) -> Self {
        Self { window, hits: VecDeque::new() }
    }

    /// 记录一次命中，丢弃过期项，返回当前窗内命中数
    pub fn push(&mut self, now: Instant) -> usize {
        self.hits.push_back(now); // 新事件进队尾
        self.evict(now);          // 队头的过期项清掉
        self.hits.len()
    }

    /// 丢弃窗口外（太旧）的事件
    fn evict(&mut self, now: Instant) {
        while let Some(&front) = self.hits.front() {
            if now.saturating_duration_since(front) > self.window {
                self.hits.pop_front(); // 队头过期，丢
            } else {
                break; // 队头没过期，后面更新的更不会过期（队列有序）
            }
        }
    }
}
```

**进窗 / 出窗**是滑动窗的灵魂：新事件永远从**队尾进**；每次操作都从**队头往外丢**过期项。因为时间戳单调递增、队列天然有序，一旦队头没过期就可以 `break`，不用扫全队列。

小数字走一遍（`window = 60s`，阈值 `= 3`）：

```text
t=10s 越界 → 窗内 [10]        count=1 <3  不报
t=25s 越界 → 窗内 [10,25]     count=2 <3  不报
t=40s 越界 → 窗内 [10,25,40]  count=3 ≥3  报警！
t=75s 越界 → 先丢过期：75-10=65>60，丢掉 10；25、40 保留
             窗内 [25,40,75]  count=3 ≥3  再次满足
```

### 用法二：确认窗 / N-of-M——抗单帧误检

模型偶尔会**单帧误检**（这一帧突然把柱子认成人）。如果一帧命中就报，噪声全放进来了。**N-of-M 确认窗**的规则：**最近 M 次判定里至少 N 次命中，才算"确认"。** 它吃掉孤立的抖动。这类窗常按**帧**算（每帧喂一个"命中 / 未命中"），用定长队列即可：

```rust,ignore
use std::collections::VecDeque;

/// N-of-M 确认窗：最近 M 帧里 ≥N 帧命中才算"确认"
pub struct ConfirmWindow {
    n: usize,
    m: usize,
    recent: VecDeque<bool>, // 最近 M 帧的命中情况
}

impl ConfirmWindow {
    pub fn new(n: usize, m: usize) -> Self {
        Self { n, m, recent: VecDeque::with_capacity(m) }
    }

    /// 送入本帧判定（命中 true / 未命中 false），返回是否"已确认"
    pub fn update(&mut self, hit: bool) -> bool {
        self.recent.push_back(hit);
        if self.recent.len() > self.m {
            self.recent.pop_front(); // 只保留最近 M 帧
        }
        let hits = self.recent.iter().filter(|&&b| b).count();
        hits >= self.n
    }
}
```

小数字（`3-of-5`：最近 5 帧 ≥3 帧命中才确认）：

```text
稳定目标：  ...1 0 1 1 0...  最近5帧 [1,0,1,1,0] → 3 命中 ≥3 → 确认 ✓
单帧误检：  ...0 0 1 0 0...  最近5帧 [0,0,1,0,0] → 1 命中 <3 → 噪声被吃掉 ✗
```

计数窗回答"够不够频繁"，确认窗回答"够不够稳定"，按业务选用或叠加。

## 24.5 去重：同一件事只算一次

计数场景最怕**重复计数**。核心手段是**去重（dedup）**：用一个键唯一标识"这件事"，记进 `HashSet`，见过就不再算。这个键的核心是 **`track_id` + 事件类型**——同一个跟踪目标的同一类事件，只算第一次。

```rust,ignore
use std::collections::HashSet;

/// 去重器：同一个 (track_id, 事件类型) 只算一次
#[derive(Default)]
pub struct Dedup {
    seen: HashSet<(u64, EventKind)>,
}

impl Dedup {
    /// 第一次见到返回 true（放行、计数），之后再见到返回 false
    pub fn first_time(&mut self, track_id: u64, kind: EventKind) -> bool {
        self.seen.insert((track_id, kind)) // insert 返回 true 表示"原来没有"
    }
}
```

`HashSet::insert` 的返回值直接就是我们要的语义：**插入成功（原来没有）= 第一次见 = 放行**。一行搞定。

### 进出计数：按方向去重

第 23 章的双绊线会给出**方向**（`+1` 进 / `-1` 出）。一个人可能先进后出，进要计一次、出也要计一次，但各自都只算一次。把**方向也放进键**即可：

```rust,ignore
use std::collections::HashSet;

/// 进出双向计数器：每个 track 的"进"和"出"各自只算一次
#[derive(Default)]
pub struct GateCounter {
    seen: HashSet<(u64, i8)>, // 键 = (track_id, direction)
    pub entered: u64,
    pub exited: u64,
}

impl GateCounter {
    /// 处理一次绊线事件（direction: +1 进 / -1 出）
    pub fn on_cross(&mut self, track_id: u64, direction: i8) {
        if self.seen.insert((track_id, direction)) {
            match direction {
                1 => self.entered += 1,  // 这个 track 第一次"进"
                -1 => self.exited += 1,  // 这个 track 第一次"出"
                _ => {}
            }
        }
    }
}
```

### 去重的软肋：跟踪 ID 跳变

去重的全部可靠性都押在 **`track_id` 稳定**上。可第 21 章讲过：目标被遮挡后重新出现，跟踪器往往给它一个**新 ID**；两个目标交叉时 ID 还可能**互换**。这会直接击穿去重：

- **ID 跳变**（同一个人换了新 ID）→ 去重表里查不到 → **重复计数**（一个人被数成两个）。
- **ID 互换**（A、B 交换 ID）→ 可能把该计的当成"已见过"而**漏计**。

所以去重的上限就是跟踪质量的上限（呼应第 21、23 章）。工程上通常会：让跟踪器输出更稳的 ID（如 ByteTrack）、结合外观特征续接 ID（关联问题见第 25 章）、或对"疑似同一人短时间重进"再叠一层冷却兜底。

## 24.6 三合一：EventFilter

现在把三件套串成一条流水线。任务约定的处理顺序是 **去重 → 窗口确认 → 冷却**，一个事件三关全过才放行：

```text
Event ─▶ ① 去重（这个 track 这类事报过没？）
             │过
             ▼
         ② 窗口确认（窗内累计够阈值没？）
             │过
             ▼
         ③ 冷却（同类事刚报过没？）
             │过
             ▼
          放行上报
```

用一个 `struct EventFilter` 把状态和逻辑收在一起，对外只暴露一个 `accept`：

```rust,ignore
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
// 复用 24.1 定义的 Event / EventKind

/// 过滤器配置
pub struct FilterConfig {
    pub once_per_track: bool, // 去重：每个 track 的同类事件只放行一次（计数场景开）
    pub window: Duration,     // 计数窗长度
    pub confirm: usize,       // 窗内累计达到几次才"确认"（=1 即不需要攒）
    pub cooldown: Duration,   // 确认后，同类事件冷却多久
}

/// 三合一事件过滤器：去重 → 窗口确认 → 冷却
pub struct EventFilter {
    cfg: FilterConfig,
    seen: HashSet<(u64, EventKind)>,                       // 去重表
    windows: HashMap<(u64, EventKind), VecDeque<Instant>>, // 每 key 一个计数窗
    last_fire: HashMap<EventKind, Instant>,                // 冷却表
    last_active: HashMap<u64, Instant>,                    // 每个 track 最近活跃时刻（清理用）
}

impl EventFilter {
    pub fn new(cfg: FilterConfig) -> Self {
        Self {
            cfg,
            seen: HashSet::new(),
            windows: HashMap::new(),
            last_fire: HashMap::new(),
            last_active: HashMap::new(),
        }
    }

    /// 事件从产生到上报的总闸门。返回 true = 应当上报。
    pub fn accept(&mut self, ev: Event, now: Instant) -> bool {
        self.last_active.insert(ev.track_id, now); // 记录活跃，供 gc 清理
        let key = (ev.track_id, ev.kind);

        // ① 去重：这个 track 的这类事件已经放行过？直接丢
        if self.cfg.once_per_track && self.seen.contains(&key) {
            return false;
        }

        // ② 窗口确认：入窗、丢弃过期，看累计是否达阈值
        let confirmed = {
            let win = self.windows.entry(key).or_default();
            win.push_back(now);
            while let Some(&front) = win.front() {
                if now.saturating_duration_since(front) > self.cfg.window {
                    win.pop_front();
                } else {
                    break;
                }
            }
            win.len() >= self.cfg.confirm
        };
        if !confirmed {
            return false; // 还没攒够，不确认
        }

        // ③ 冷却：同类事件刚报过就压住
        if let Some(&last) = self.last_fire.get(&ev.kind) {
            if now.saturating_duration_since(last) < self.cfg.cooldown {
                return false;
            }
        }

        // 三关都过：登记去重、刷新冷却、清空该 key 的窗口，放行
        self.seen.insert(key);
        self.last_fire.insert(ev.kind, now);
        if let Some(w) = self.windows.get_mut(&key) {
            w.clear();
        }
        true
    }

    /// 清理长时间不活跃的 track 状态，防止 HashMap 无限膨胀（见 24.7）
    pub fn gc(&mut self, now: Instant, ttl: Duration) {
        let expired: Vec<u64> = self
            .last_active
            .iter()
            .filter(|(_, &t)| now.saturating_duration_since(t) > ttl)
            .map(|(&id, _)| id)
            .collect();
        for id in expired {
            self.last_active.remove(&id);
            self.seen.retain(|(tid, _)| *tid != id);
            self.windows.retain(|(tid, _), _| *tid != id);
        }
    }
}
```

配置一变，同一套代码就能表达不同业务：

| 业务需求 | 配置 |
| --- | --- |
| **进出计数**（一人过一次 `+1`） | `once_per_track = true`、`confirm = 1`、`cooldown = 0` |
| **周界报警**（别刷屏，可复报） | `once_per_track = false`、`confirm = 1`、`cooldown = 5s` |
| **反复试探才算入侵** | `once_per_track = false`、`window = 60s`、`confirm = 3`、`cooldown = 60s` |

真正落地成一套完整的报警/推送/存证系统，是第 27、28 章的实战内容；本章只负责把事件"洗干净"。

## 24.7 工程坑：状态清理、多路隔离与时间一致性

这几件套都是**有状态**的，长时间运行的服务里，状态管理才是真正的坑。

**① 状态无限增长 = 内存泄漏。** `seen`、`windows`、`last_fire` 都以 `track_id` 为键，而跟踪 ID **只增不减**——跑一天下来几十万个死 ID 全堆在内存里。必须定期清理不活跃的 track，这就是上面 `gc` 的作用：记录每个 track 的 `last_active`，超过 `ttl`（比如 30 秒没再出现）就把它相关的所有状态删掉。工程上通常挂在跟踪器的"track 消失"回调上，或每隔几秒扫一次。

**② 多路视频必须状态隔离。** 一台服务器同时分析 16 路摄像头，**绝不能共用一个 `EventFilter`**——否则 1 号摄像头的 track 3 会和 5 号摄像头的 track 3 撞键，互相污染去重和冷却。两种正确做法：**每路一个 `EventFilter` 实例**（最清晰），或把摄像头号也拼进键 `(camera_id, track_id, kind)`。前者更推荐，天然隔离、也方便按路销毁。

**③ 时间戳来源必须一致。** 同一个过滤器里，比较的两个时间必须来自**同一把尺子**：要么全是 `Instant::now()`（实时流），要么全是帧号换算的视频时间（回放），**绝不能混用**。一旦混用，减出来的差值毫无意义，冷却和窗口会给出随机结果。多路、多线程环境里，也要确保时间戳在流水线上一路传下来，而不是每个环节各自 `Instant::now()`（各环节耗时不同会引入偏差）。

**④ 顺序也有讲究。** 本章按"去重 → 窗口 → 冷却"排列：先去重能让"一次性事件"最省——查到见过直接返回，连窗口都不进；不开去重（允许复报）时，窗口和冷却谁先谁后语义不同，要按业务想清楚。

## 24.8 小结

- 模型逐帧输出 + 跟踪抖动，会让**一件事触发几十次**；本章在**时间维度**上把原始触发洗成干净业务事件。
- **Cooldown（冷却）**：记 `last_fire`，`now - last_fire < T` 就丢。适合**报警**；按 `(track_id, kind)` 等维度选择"谁压住谁"；太短刷屏、太长漏报。
- **Window（时间窗）**：用 `VecDeque` 存时间戳，新事件进队尾、过期从队头丢。**计数窗**判"够不够频繁"（N 秒 M 次），**N-of-M 确认窗**判"够不够稳定"（抗单帧误检）。
- **去重（dedup）**：键 = **`track_id` + 事件类型**（进出计数再加 `direction`），`HashSet::insert` 一行搞定；可靠性受制于跟踪 ID 稳定性（第 21、23 章）。
- **`EventFilter`** 把三者串成 `去重 → 窗口确认 → 冷却` 流水线，靠 `FilterConfig` 一套代码适配计数 / 报警多种业务。
- 长跑服务三大纪律：**定期 `gc` 清理死 track 状态**、**多路视频状态隔离**、**时间戳来源统一**（实时用 `Instant`、回放用帧号换算）。

## 24.9 练习

1. **冷却维度实验**：用 24.3 的 `Cooldown<K>`，构造 A、B 两人（`track_id` 5 和 8）同时越线的事件序列，分别用 `key = EventKind`（全局）和 `key = (u64, EventKind)`（按 track）跑一遍，观察全局冷却如何把 B 漏掉，验证"多目标要按 track 冷却"。
2. **回放时间基准**：给 `CountWindow` 写一个测试，先用 `Instant::now()` 模拟"5 倍速处理 1 小时录像"（真实只花 12 分钟），再用 `frame_to_media_time` 换算的视频时间跑同样数据，对比两者的"1 分钟内 3 次"判定结果，解释为什么回放必须用帧号。
3. **N-of-M 抗噪**：给 `ConfirmWindow` 喂一段含单帧误检的序列（如 `0 0 1 0 0`）和一段稳定命中序列，验证 `3-of-5` 能吃掉孤立噪声；再想想 `confirm` 从 3 调到 1 会怎样。
4. **状态清理**：给 `EventFilter` 造 10 万个只出现一次的 `track_id`，调用 `accept` 后打印 `seen`/`windows` 的大小；再调用 `gc` 并观察内存回收。思考：如果不清理，服务跑一周会怎样？多路摄像头共用一个 `EventFilter` 又会出什么问题？
