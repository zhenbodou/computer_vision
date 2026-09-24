# 全书统一接口契约（_INTERFACE_CONTRACT.md）

> 本文件不进入 mdBook 构建（只构建 `src/`），是**跨章节 API 的唯一事实来源**。
> 第 21～28 章的实战代码互相调用，必须共用同一套签名，否则读者照抄会编译不过。
> 任何改动这些类型的编辑，都要回来同步本文件，并检查所有引用章节。
>
> 配套：`_deprecated_old_chapters/_WRITING_GUIDE.md`（文风/结构/共享结构体 `BBox`/`Detection`/`Track`）。

---

## 1. 共享数据类型（定义章 → 全书复用）

| 类型 | 定义章 | 关键字段 / 方法 |
| --- | --- | --- |
| `BBox` | 第 16 章 | `x1,y1,x2,y2:f32`；`width/height/area/center/bottom_center/iou` |
| `Detection` | 第 16 章 | `bbox:BBox, score:f32, class_id:usize` |
| `Track` | 第 21 章 | `id:u64, bbox:BBox, class_id:usize, score:f32` |
| `Keypoint/Pose` | 第 17 章 | 关键点相关 |
| `Face` | 第 20 章 | `bbox:BBox, score:f32, landmarks:[(f32,f32);5]` |

坐标约定：**出了各自模块一律是原图像素坐标**（第 10 章逆 letterbox 之后）。

---

## 2. Detector（定义章：第 26 章 `model.rs`）

```rust
pub struct Detector { session: Session }   // + 可选内部字段

impl Detector {
    pub fn load(model_path: &str) -> Result<Self>;            // 唯一构造器（不是 new）
    fn  warmup(&mut self) -> Result<()>;                      // load 内部自动调用
    pub fn infer(&mut self, input: Array4<f32>) -> Result<Array3<f32>>;   // [1,3,640,640]→[1,84,8400]
    pub fn detect(&mut self, img: &RgbImage) -> Result<Vec<Detection>>;   // ★ 便捷方法：一帧原图→一批框
}
```

- **构造器是 `load`，不是 `new`。** 输入名 `"images"`、输出名 `"output0"`。
- `Session::run` 需要 `&mut self`；`Session` 是 `Send+Sync` 但**不能在同一实例上并发 `run`**。批量并行 = 每个 Rayon 线程各持一个 `Detector`（`for_each_init`）。
- **`detect` 是 27/28 章的统一入口**：内部串 `preprocess → infer → postprocess`，用通用默认阈值（conf 0.25 / NMS IoU 0.45），返回原图像素坐标的 `Vec<Detection>`。要自定义阈值走 `infer + postprocess`（如 `run_single`）。
- `detect` 返回 `Result` ⇒ **调用它的 `process_frame` 必须返回 `Result`**，主循环用 `if let Err(e) = ... { eprintln!(...); continue; }` 跳过坏帧、不崩整条流水线。

**必须遵守此约定的章节**：第 27 章（`Detector::load` + `self.detector.detect(frame)?`）、第 28 章（同）。

---

## 3. Tracker（定义章：第 21 章）

```rust
pub struct Tracker { /* tracks, next_id, iou_thresh, max_age, min_hits, just_removed */ }

impl Tracker {
    pub fn new() -> Self;                                     // 无参；内部默认 iou 0.3 / max_age 30 / min_hits 3
    pub fn update(&mut self, dets: &[Detection]) -> Vec<Track>;   // 框进、带 ID 的框出
    pub fn just_removed(&self) -> Vec<u64>;                  // ★ 本帧被删除(达 max_age)的 track id
}
```

- **构造器是无参 `new()`**（参数走内部默认；要暴露调参可另加 `with_params`，但 27 章统一用 `new()`）。
- `just_removed()` 返回**最近一次 `update` 中被清理**的 id（owned `Vec`，避免借用冲突：调用点常在同一 `self` 上再借 `&mut` 其他字段）。第 28 章靠它在 track 结束时提交人脸抓拍落盘。
- `just_removed` 在每次 `update` 开头重算：先收集 `time_since_update > max_age` 的 id，再 `retain`。

**必须遵守**：第 27 章（`Tracker::new()`）、第 28 章（`new()` + `just_removed()`）。

---

## 4. Cooldown（定义章：第 24 章）

```rust
pub struct Cooldown<K> { cooldown: Duration, last_fire: HashMap<K, Instant> }

impl<K: Hash + Eq + Clone> Cooldown<K> {
    pub fn new(cooldown: Duration) -> Self;                   // ★ 构造时传入冷却时长
    pub fn allow(&mut self, key: K, now: Instant) -> bool;    // ★ true=放行；内部用 saturating_duration_since
}
```

- **不是** `Cooldown::new()`（无参）+ `ready(key, now, cd)`。冷却时长在**构造时**传入，`allow` 只收 `(key, now)`。
- 维度由 `K` 决定：多目标场景按 `(track_id, kind)` 或 `track_id`，避免全局冷却把并发目标压掉。

**必须遵守**：第 27 章（`Cooldown::new(Duration::from_secs(cfg.cooldown_secs))` + `self.cooldown.allow(t.id, now)`）。

---

## 5. GateCounter（定义章：第 24 章）

```rust
#[derive(Default)]
pub struct GateCounter { seen: HashSet<(u64,i8)>, pub entered: u64, pub exited: u64 }

impl GateCounter {
    pub fn on_cross(&mut self, track_id: u64, direction: i8);  // +1 进 / -1 出；内部按 (id,dir) 去重
}
```

- 读数是字段 **`entered`（进）/ `exited`（出）**，不是 `down`/`up`。
- 记一次穿越用 **`on_cross(track_id, i8)`**，不是 `record(id, Dir)`。构造用 `GateCounter::default()`。

**必须遵守**：第 28 章（`on_cross` + `entered`/`exited`）。

---

## 6. 几何原语（定义章：第 23 章）

```rust
type Pt = (f32, f32);
fn side(a: Pt, b: Pt, p: Pt) -> f32;                          // 叉积符号：点在有向线哪一侧
fn segments_intersect(a: Pt, b: Pt, c: Pt, d: Pt) -> bool;    // ★ 鲁棒版：四次叉积、显式符号
fn foot_point(b: &BBox) -> Pt;                                // 底边中点（代表点）

pub enum CrossDir { AtoB, BtoA, None }                        // 几何层的方向枚举
pub struct Tripwire      { /* new(p1,p2); update(&[Track])->Vec<(u64,CrossDir)> */ }
pub struct DoubleTripwire{ /* new(a,b,max_gap); update(tracks,frame)->Vec<(u64,CrossDir)> */ }
```

- **线段相交统一用 `segments_intersect`（鲁棒显式符号版）**，不要用 `side(..)*side(..)<0` 的乘积形式（`segments_cross`），后者在退化/边界情形不稳。
- 其他章节需要相交判定时 **`use crate::{side, segments_intersect};`**，不再本地重定义。

**必须遵守**：第 27 章（删除本地 `side`/`segments_cross`，改 `use` 第 23 章）、第 28 章（已正确 `use crate::{side, segments_intersect}`）。

---

## 7. 方向约定（贯穿 23/24/28）

- **几何层**（第 23 章 `Tripwire`）产出 `CrossDir { AtoB, BtoA, None }`。
- **计数边界**（第 24 章 `GateCounter`）说 **`i8`：+1 = 进 / -1 = 出**。
- 需要把几何方向送进计数器时，**在边界处映射** `CrossDir::AtoB/BtoA → ±1`。
- 第 28 章自带的 `cross_direction` 直接返回 `Option<i8>`（+1 进 / -1 出），与 `GateCounter::on_cross` 无缝对接；复用第 23 章的 `side`/`segments_intersect` 做几何、不引入第二个方向枚举。
- ±1 与"进/出"的语义映射按图像 y 轴朝向**上线前校准一次**（第 28.3）。

---

## 8. process_frame 的错误传播约定

- `Detector::detect` / `infer` 返回 `Result` ⇒ 任何调用它们的 `process_frame` 返回 `anyhow::Result<...>`。
- 视频主循环**不 `?` 直接上抛**（会让一帧坏数据杀死 24/7 服务），而是：

```rust
for frame in frame_source {
    if let Err(e) = system.process_frame(&frame /*, now */) {
        eprintln!("处理帧失败: {:#}", e);
        continue;
    }
    // ... 正常读数/展示
}
```

**必须遵守**：第 27 章 `PerimeterAlarm::process_frame -> Result<Vec<u64>>`、第 28 章 `PeopleSystem::process_frame -> Result<()>`。
