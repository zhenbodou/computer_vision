# 事件规则层 共享数据模型（EVENTS_MODEL.md）

> 本文件不属于书稿。**第十八部分（事件规则层，ch83–93）所有章节必须使用同一套数据模型**，
> 否则各章各自定义 `Detection`/`BBox` 会互相对不上、无法交叉引用。凡本部分章节，
> 代码一律基于下面这套类型与约定；第 83 章负责把它介绍给读者，其余章节直接沿用并 `use` 它，
> 交叉引用写「见[第 83 章](ch83-filtering-basics.md)定义的 `Detection`/`Frame`」。

## 定位：事件层在整条流水线的什么位置

```text
摄像头/视频 → 解码成帧 → 检测器(第9部分) → [跟踪(第13部分)给稳定 id] → ★事件规则层(本部分)★ → 业务告警/联动
                                   Detection            Track                    Event
```

事件层**只消费检测/跟踪的结果**（`Detection`/`Track`），**不关心**检测器/跟踪器怎么实现。
所以本部分示例**不依赖任何深度学习 crate**，用代码合成/手写 `Vec<Frame>` 当输入即可自包含、可复现。
这正是学术书常略过、而工业落地天天在写的「把检测结果变成有用告警」的一层。

## 坐标与时间约定（全部统一）

- 图像坐标：**原点左上角**，x 向右、y 向下（和 `image` 库一致）。
- 包围盒用 **左上角 (x, y) + 宽高 (w, h)**，`f32` 像素。
- 时间戳统一用 **`u64` 毫秒**（字段名 `ts_ms`）。合成数据时自己造递增时间戳即可，**不要用系统时钟**（示例要可复现；写到需要真实时钟处用 `std::time::Instant`/`SystemTime` 并说明）。
- 目标「落地点/脚点」用**包围盒底边中点** `bottom_center()`——人站地面上，脚点比中心更适合判断「过线」「进区域」。

## canonical 类型（第 83 章原样给出，后续章节 `use` 之）

```rust
/// 轴对齐包围盒：左上角 (x, y) + 宽高 (w, h)，单位像素。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self { BBox { x, y, w, h } }

    /// 中心点
    pub fn center(&self) -> (f32, f32) { (self.x + self.w / 2.0, self.y + self.h / 2.0) }

    /// 底边中点（脚点）——过线/入区域判定常用这个而非中心
    pub fn bottom_center(&self) -> (f32, f32) { (self.x + self.w / 2.0, self.y + self.h) }

    pub fn area(&self) -> f32 { (self.w * self.h).max(0.0) }

    /// 交并比 IoU ∈ [0,1]，用于同帧去重 / 跨帧关联
    pub fn iou(&self, o: &BBox) -> f32 {
        let x1 = self.x.max(o.x);
        let y1 = self.y.max(o.y);
        let x2 = (self.x + self.w).min(o.x + o.w);
        let y2 = (self.y + self.h).min(o.y + o.h);
        let iw = (x2 - x1).max(0.0);
        let ih = (y2 - y1).max(0.0);
        let inter = iw * ih;
        let uni = self.area() + o.area() - inter;
        if uni <= 0.0 { 0.0 } else { inter / uni }
    }
}

/// 一个检测框：来自检测器（第 9 部分）。事件层只消费它。
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String, // 类别名，如 "person" / "car"
    pub score: f32,    // 置信度 [0,1]
}

/// 一帧的检测结果（带时间戳）。事件层的输入通常是 Vec<Frame> 或帧流。
#[derive(Clone, Debug)]
pub struct Frame {
    pub ts_ms: u64,
    pub detections: Vec<Detection>,
}

/// 跟踪后的目标：跟踪器（第 13 部分）给每个目标分配**稳定 id**。
/// 很多规则（停留时长、绊线方向、轨迹）都基于 track_id 才能跨帧连起来。
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u64,
    pub label: String,
    pub history: Vec<(u64, BBox)>, // (ts_ms, bbox) 按时间递增
}

/// 事件层的输出：一条业务事件/告警。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String,          // "roi" | "tripwire" | "intrusion" | "dwell" | ...
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
```

## 过滤/规则的统一心智模型

把事件层想成**一条可插拔的流水线**：检测结果流过一串「规则/过滤器」，
每一环要么**筛掉**一些检测（阈值、ROI），要么**在满足条件时吐出 `Event`**（绊线、停留）。
概念上是这样一个 trait（各章可按需简化，不强制都实现它，但心智模型统一）：

```rust
/// 一个规则/过滤环节：吃一帧（或一批 Track），产出若干事件（可能为空）。
/// 有状态的规则（时间窗、cooldown、绊线方向）自己持有 &mut self 里的状态。
pub trait Stage {
    fn process(&mut self, frame: &Frame) -> Vec<Event>;
}
```

## 依赖与运行约定（务必遵守，避免污染主工程）

- **默认只用 std**，示例就能 `cargo run` 跑通、可复现（合成 `Vec<Frame>` 当输入）。
- ch91（事件联动/推送）真实推送用 `ureq`（纯 Rust HTTP）或 webhook，但**可运行 demo 用一个 `trait Sink`＋打印实现**来模拟，正文再给出「换成 ureq 真发 HTTP」的片段并注明需 `cargo add ureq`。
- ch92（多路并发）用 **`std::thread` + `std::sync::mpsc`** 讲清「每路一线程、结果汇到一个通道」即可；`rayon`/`tokio` 作为进阶选项一句带过，别硬引重运行时。
- ch93（规则引擎）配置用 `serde` + `serde_json`（纯 Rust）；可运行 demo 可以用手写结构体，正文再展示 JSON 配置驱动。
- 需要新增 crate 时，加到主工程 `code/Cargo.toml` 前先确认是纯 Rust；`ureq`/`serde`/`serde_json`/`rayon` 均为纯 Rust，可用。
- 几何点在多边形内（ROI）用**射线法 point-in-polygon**，自己写十来行即可，别引库。

## 各章职责边界（避免内容重叠）

| 章 | 主题 | 核心产出 |
|----|------|----------|
| 83 | 过滤基础：为什么要后处理 + 数据模型 + 阈值过滤 + 类别/尺寸过滤 | 引入上面全部类型；置信度阈值、类别白名单、最小/最大框尺寸过滤 |
| 84 | ROI 过滤（感兴趣区域） | 多边形 ROI + 射线法点在多边形内；只保留脚点落在 ROI 里的检测 |
| 85 | 时间窗过滤（N 帧内命中 M 次才算数） | 滑动窗口/连续帧确认，抖动抑制（去偶发误检） |
| 86 | cooldown 去重（同一目标短时间只报一次） | 冷却时间、基于 track_id 或 IoU 的同帧 NMS 式去重 |
| 87 | 绊线（tripwire / 越线检测） | 线段 + 目标脚点轨迹跨线判定 + **方向**（A→B / B→A，用叉积定侧） |
| 88 | 闯入与停留（intrusion & dwell time） | 进入 ROI 触发闯入；在 ROI 内持续 ≥ T 秒触发停留（基于 track 历史算时长） |
| 89 | 轨迹与行为分析 | 由 Track.history 算速度/方向/停留/徘徊；简单行为判定（逗留、逆行） |
| 90 | 布防排班与防遮挡（arming & tamper） | 时段布防（排班表）、区域布防开关；镜头遮挡/移位/模糊检测（画面统计量突变） |
| 91 | 事件联动与推送 | Sink 抽象（打印/webhook/存库）、去抖后的事件如何送出、ureq 发 HTTP 片段 |
| 92 | 多路并发与工程架构 | 每路视频一线程、mpsc 汇聚；背压、丢帧策略；整体架构图 |
| 93 | 规则引擎 | 把 83–92 的规则做成**可配置**（JSON）的组合链，一个 `Pipeline` 串起多个 `Stage` |

> 交叉引用：83 是地基，其余章都 `use` 它的类型并写「见第 83 章」。88 依赖 87 的过线思想与 84 的 ROI；89 依赖 Track；93 把前面所有 Stage 组装起来。
