# 第 93 章 设计一个可配置的规则引擎（过滤器管线）

> **本章导读**
> - 这是整个"事件规则层"的**集大成**。前面十章我们攒下了一整套规则：阈值、类别、尺寸（[第 83 章](ch83-filtering-basics.md)）、ROI（[第 84 章](ch84-roi-filter.md)）、时间窗（[第 85 章](ch85-temporal-window.md)）、cooldown 去重（[第 86 章](ch86-cooldown-dedup.md)）、绊线（[第 87 章](ch87-tripwire.md)）、闯入/停留（[第 88 章](ch88-intrusion-dwell.md)）……本章把它们统一成一种零件，拼成一台可组装的机器。
> - 用一个统一的 **`trait Stage`** 抽象所有规则，用一个 **`Pipeline`** 把多个 `Stage` 串成一条**可插拔的规则链**——检测流依次流过各环，产出 `Event`。
> - 用 **`serde` + `serde_json`** 让整条流水线**由 JSON 配置驱动**：先用手写结构体跑通，再展示同一条流水线**改一段 JSON 就换行为**。真实数字：同一批合成帧，调阈值、换 ROI，产出事件数随之而变——**一行代码都不用改、不用重新编译**。

## 场景：一个每周都要改规则的甲方

系统上线了，甲方的需求却像春天的野草：

> 周一："白天误报太多，把行人置信度阈值从 0.5 提到 0.7 吧。"
> 周三："布防区往左挪一点，把新开的侧门也圈进来。"
> 周五："周末人多，拥挤阈值从 3 人改成 5 人。"

如果每个规则、每个阈值都硬编码在 Rust 源码里，那你这一周就是：**改代码 → 重新编译 → 打包 → 停服务 → 部署 → 重启**，来回六趟。甲方那边还得停机——安防系统停机可是要写事故报告的。

工业界的标准解法是**数据驱动（data-driven）**：把"用哪些规则、每个规则什么参数"从代码里抽出来，放进一个**配置文件**（通常是 JSON/YAML）。程序启动时读配置、动态组装流水线。于是甲方那三条需求，全都变成"**改一下配置文件里的一个数字**"——不碰代码，不重新编译，甚至能做到不停机热加载。

这一章，我们就把前面十章的规则做成这样一台**可配置的规则引擎**。

## 原理：万物皆 `Stage`，串起来就是 `Pipeline`

回顾[第 83 章](ch83-filtering-basics.md)立下的心智模型：事件层是**一条可插拔的流水线**，检测结果流过一串"规则环"，每一环要么**筛掉一些检测**（阈值、ROI），要么**吐出事件**（闯入、拥挤）。现在我们把这个"环"用一个统一的 `trait` 钉死：

```rust
/// 一个规则环：处理一帧，可以就地筛掉帧里的一些检测（过滤类规则），
/// 并/或往 out 里吐出事件（告警类规则）。有状态的规则（cooldown）把状态放进 self。
pub trait Stage {
    fn name(&self) -> &str;
    fn process(&mut self, frame: &mut Frame, out: &mut Vec<Event>);
}
```

这里把第 83 章那个 `process(&Frame) -> Vec<Event>` **稍作扩展**：`frame` 改成 `&mut`，再加一个 `out` 收集事件。为什么？因为流水线里其实有**两类环**，这个签名要能同时容纳：

- **过滤环**（阈值、类别、ROI）：**就地修改 `frame`**（`retain` 掉不合格的检测），不吐事件。
- **事件环**（闯入、拥挤）：**读 `frame`**（此时已被上游过滤干净），满足条件就往 `out` 里 `push` 事件。

于是所有规则长成一个样，能塞进同一条链。比如置信度过滤（过滤环）和区域闯入（事件环）：

```rust
struct ScoreThreshold { min_score: f32 }
impl Stage for ScoreThreshold {
    fn name(&self) -> &str { "score_threshold" }
    fn process(&mut self, f: &mut Frame, _out: &mut Vec<Event>) {
        f.detections.retain(|d| d.score >= self.min_score); // 就地筛掉低分框
    }
}

struct PresenceAlarm { cooldown_ms: u64, message: String, last_fire: Option<u64> }
impl Stage for PresenceAlarm {
    fn name(&self) -> &str { "presence_alarm" }
    fn process(&mut self, f: &mut Frame, out: &mut Vec<Event>) {
        if f.detections.is_empty() { return; }        // 区内没人，不报
        let cool = matches!(self.last_fire, Some(t) if f.ts_ms - t < self.cooldown_ms);
        if !cool {                                      // 过了冷却期才报（去重，见第 86 章）
            self.last_fire = Some(f.ts_ms);
            out.push(Event { kind: "intrusion".into(), ts_ms: f.ts_ms, track_id: None,
                             message: self.message.clone() });
        }
    }
}
```

`Pipeline` 就是"一串 `Stage`"，每帧克隆一份、依次流过各环：

```rust
struct Pipeline { stages: Vec<Box<dyn Stage>> }
impl Pipeline {
    fn run(&mut self, frames: &[Frame]) -> Vec<Event> {
        let mut all = Vec::new();
        for fr in frames {
            let mut f = fr.clone();       // 过滤环会改帧，不能动原始输入
            let mut out = Vec::new();
            for s in &mut self.stages {    // 一帧依次流过每一环
                s.process(&mut f, &mut out);
            }
            all.extend(out);
        }
        all
    }
}
```

> ⚠️ **注意**：**环的顺序是有意义的，不能乱排。** 过滤环必须排在事件环**前面**——先把低分、错类、区外的脏检测清掉，事件环看到的才是"干净"的检测。如果把 `presence_alarm` 排在 `roi_filter` 前面，它就会对全画面（还没按区域筛）的检测报警，ROI 形同虚设。流水线是**有序**的，这一点在写配置时要格外小心。

## 动手写代码：先用手写结构体跑通

完整代码见 `code/examples/ch93_rule_engine.rs`（用到 `serde`/`serde_json`，已在主工程依赖里）：

```bash
cargo run --example ch93_rule_engine
```

我们合成 16 帧（2fps）的检测流，故意混入各种目标：一个 `car`（该被类别滤掉）、一个 0.40 分的低分框（该被阈值滤掉）、一个站在**左侧**区外的人 P4，以及右侧 ROI 里陆续出现的 P1/P2/P3（其中 P3 只有 0.70 分）。**原始检测框一共 52 个。**

先用**手写的 `PipelineConfig` 结构体**搭一条 baseline 流水线（阈值 0.5、只要 person、右侧 ROI、闯入冷却 3 秒、区内满 3 人算拥挤），跑一遍：

```text
── baseline（阈值0.5 / person / 右侧ROI） ──
   规则链：["score_threshold", "class_filter", "roi_filter", "presence_alarm", "crowd_alarm"]
   产出事件 3 条，按 kind：{"crowd": 1, "intrusion": 2}
     [intrusion] t=3.0s  有人进入布防区
     [crowd    ] t=5.0s  区内目标数达到 3 人
     [intrusion] t=6.0s  有人进入布防区
```

52 个原始框，经过五道环，收敛成 **3 条干净事件**：3 秒时有人进区（闯入）、5 秒时区内挤到 3 人（拥挤）、6 秒时冷却期过后再报一次闯入。这就是整个事件层要干的事——**把一堆框，变成几条有业务含义的告警**。

## 配置化：同一条流水线，改 JSON 就换行为

关键的一步来了。上面那个"手写结构体" `PipelineConfig`，其实**天生就能变成 JSON**——因为我们给它派生了 serde 的 `Serialize`/`Deserialize`。规则的种类用一个 **带标签的枚举** 表示：

```rust
/// 一条规则的配置规格。#[serde(tag = "type")] → JSON 里用 "type" 字段区分是哪种规则。
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StageSpec {
    ScoreThreshold { min_score: f32 },
    ClassFilter    { allow: Vec<String> },
    RoiFilter      { polygon: Vec<[f32; 2]> },
    PresenceAlarm  { cooldown_ms: u64, message: String },
    CrowdAlarm     { min_count: usize, cooldown_ms: u64 },
}

#[derive(Serialize, Deserialize)]
struct PipelineConfig { stages: Vec<StageSpec> }
```

注意这里有个漂亮的**分离**：`StageSpec` 是**无状态的"配置规格"**（能存进 JSON），而真正运行的 `Stage`（如带 `last_fire` 冷却状态的 `PresenceAlarm`）是**有状态的运行期对象**。一个 `build_stage(spec)` 函数负责把前者实例化成后者：

```rust
fn build_stage(spec: &StageSpec) -> Box<dyn Stage> {
    match spec {
        StageSpec::ScoreThreshold { min_score } => Box::new(ScoreThreshold { min_score: *min_score }),
        StageSpec::PresenceAlarm { cooldown_ms, message } =>
            Box::new(PresenceAlarm { cooldown_ms: *cooldown_ms, message: message.clone(), last_fire: None }),
        // ……其余规则同理
    }
}
```

把 baseline 配置用 `serde_json::to_string_pretty` 打印出来，就是这样一段人人能改的 JSON（片段）：

```json
{
  "stages": [
    { "type": "score_threshold", "min_score": 0.5 },
    { "type": "class_filter", "allow": ["person"] },
    { "type": "roi_filter", "polygon": [[200,150],[440,150],[440,330],[200,330]] },
    { "type": "presence_alarm", "cooldown_ms": 3000, "message": "有人进入布防区" },
    { "type": "crowd_alarm", "min_count": 3, "cooldown_ms": 10000 }
  ]
}
```

程序里用 `serde_json::from_str(json)` 把它读回来、`build_pipeline` 组装、`run` 跑。为了证明"JSON 驱动的和手写结构体是同一条流水线"，我们把 baseline 序列化再反序列化跑一遍：

```text
JSON 往返一致性检查：结构体产出 3 条，JSON 产出 3 条 → 一致 ✓
```

现在见证配置的威力——**下面两次改动，Rust 代码一行没动，只改了 JSON 字符串**：

**改动一：把置信度阈值 `0.5` 改成 `0.8`。**

```text
── 阈值抬到 0.8（P3=0.70 被滤掉，区内不足 3 人） ──
   产出事件 2 条，按 kind：{"intrusion": 2}
     [intrusion] t=3.0s  有人进入布防区
     [intrusion] t=6.0s  有人进入布防区
```

阈值一抬，0.70 分的 P3 被挡在门外，右侧区内最多只剩 2 个人——**拥挤告警直接消失了**，事件从 3 条降到 2 条。

**改动二：把 ROI 多边形从右侧 `[[200,150]…]` 换成左侧 `[[50,150]…]`。**

```text
── ROI 换到左侧（改由 P4 触发，右侧那几个人反而没了） ──
   产出事件 3 条，按 kind：{"intrusion": 3}
     [intrusion] t=2.0s  有人进入左侧布防区
     [intrusion] t=5.0s  有人进入左侧布防区
     [intrusion] t=8.0s  有人进入左侧布防区
```

区域一挪，原本被忽略的左侧行人 P4 成了主角，而右侧的 P1/P2/P3 全部出局——触发者彻底换了一批。

三次运行的总览，一目了然：

```text
  ① baseline（阈值0.5, 右ROI）  → 3 条：{"crowd": 1, "intrusion": 2}
  ② 阈值抬到 0.8               → 2 条：{"intrusion": 2}
  ③ ROI 换到左侧               → 3 条：{"intrusion": 3}
```

**同一批帧、同一套代码，只因配置不同，产出的事件就完全不同。** 甲方每周改的那些需求，到这里全都退化成"改一个 JSON 数字"——这就是可配置规则引擎的价值。

## 常见坑与提示

> 💡 **提示**：**加一种新规则 = 加一个 `impl Stage` + 一个枚举分支。** 想支持[第 87 章](ch87-tripwire.md)的绊线、[第 88 章](ch88-intrusion-dwell.md)的停留？只需：写一个 `TripwireStage` 实现 `Stage`、在 `StageSpec` 里加一个 `Tripwire { a, b, ... }` 分支、在 `build_stage` 里加一行映射。**流水线、配置解析、主程序统统不用动。** 这种"对扩展开放、对修改封闭"就是好架构的样子。（注意：绊线/停留需要跨帧的 `track_id`，所以它们的 `Stage` 得自己维护"上一帧脚点"这类状态，或让流水线传入 `Track` 而非 `Frame`——这是留给你的练习。）

> 🕳️ **坑**：**配置解析一定要校验，坏配置要"响亮地失败"。** `serde_json::from_str` 遇到 JSON 里写了个不认识的 `"type": "foobar"`、或 `min_score` 写成了字符串，会返回 `Err`——**千万别 `.unwrap()` 了事后当没事**。生产代码要把这个 `Err` 打成清晰的日志（"第 3 条规则配置非法：未知类型 foobar"）并拒绝启动，而不是带着半条流水线裸奔。宁可启动时就报错，也别上线后悄悄漏报。

> ⚠️ **注意**：**配置驱动 ≠ 配置万能，别把配置写成一门编程语言。** 规则引擎的甜区是"**已有规则的组合与调参**"。总有人想让配置支持 `if (score>0.5 && (在A区 || 在B区))` 这种任意逻辑表达式——一旦你开始在 JSON 里发明 if/else 和括号，就是在造一门蹩脚的编程语言，复杂度会失控。真需要那种灵活性时，正确做法是嵌入一个成熟的脚本引擎（如纯 Rust 的 `rhai`），而不是硬扩 JSON schema。**让配置管组合，让代码管逻辑。**

> 💡 **提示**：**热加载与每路独立配置。** 有了配置化，两个高级玩法水到渠成：① **热加载**——监听配置文件变化（或收到一个信号），重新 `from_str` + `build_pipeline`，把旧流水线原子替换掉，实现不停机改规则；② **每路一套配置**——[第 92 章](ch92-multistream.md)里每路视频线程各自持有一条按自己配置 `build` 出来的 `Pipeline`，于是"大门那路管绊线、仓库那路管停留"各行其是，代码却完全复用。

## 本章小结

- 把第 83–92 章的所有规则统一成一个 **`trait Stage`**（`process(&mut Frame, &mut Vec<Event>)`），过滤环就地改帧、事件环吐事件；用一个 **`Pipeline`** 把它们串成有序的规则链，检测流依次流过、产出事件。**环的顺序有意义：过滤在前，事件在后。**
- 用 **`serde` 带标签枚举** `StageSpec` 把"规则配置"和"运行期有状态 Stage"分离：配置能进 JSON，`build_stage` 负责实例化。`serde_json::from_str` 读配置 → `build_pipeline` 组装 → `run` 执行。
- 真实数字验证了配置的威力：同一批 52 个检测框，baseline 出 3 条事件；**只改 JSON**把阈值抬到 0.8，拥挤告警消失、降到 2 条；把 ROI 挪到左侧，触发者换了一批。**改行为不必改代码、不必重新编译**——这正是工业落地的核心。
- 好扩展（加规则 = 加一个 `impl` + 一个枚举分支）、坏配置要响亮失败、别把配置写成编程语言、配好了还能热加载与每路独立配置。

至此，整个**第十八部分·事件规则层**走完了：从[第 83 章](ch83-filtering-basics.md)认识"检测结果不能直接报警"、立起共享数据模型，到清洗、ROI、时间窗、去重、绊线、闯入停留、轨迹行为、布防防遮挡，再到[第 91 章](ch91-actions-push.md)的联动推送、[第 92 章](ch92-multistream.md)的多路并发，最后在这一章把它们**组装成一台可配置的引擎**。你已经掌握了工业视觉项目里最不起眼、却最决定成败的那一层——**把"看见"变成"有用的告警"**。

## 动手练习

1. **加一个尺寸过滤环**：仿照 `ScoreThreshold`，写一个 `SizeFilter { min_area, max_area }`（第 83 章的尺寸过滤），加进 `StageSpec` 和 `build_stage`，在 JSON 里插一条，验证碎框/巨框被滤掉。
2. **坏配置演练**：把 JSON 里某条规则的 `"type"` 改成 `"unknown"`，观察 `serde_json::from_str` 返回的 `Err` 长什么样，然后把主程序里的 `.unwrap()` 换成友好的错误提示（"第 N 条规则非法"）。
3. **接上分发**：把 `Pipeline` 产出的 `Vec<Event>` 喂给[第 91 章](ch91-actions-push.md)的 `FanOutSink`，让规则引擎的产出真正"发出去"，跑通"检测流 → 规则链 → 多路分发"的全链路。
4. **进阶·绊线环**：给流水线加一个 `TripwireStage`（第 87 章），让它自己维护"每个 track 上一帧的脚点"，把 `Pipeline::run` 的输入从 `Frame` 换成带 `track_id` 的结构，体会"有状态跨帧规则"如何融进这套 `Stage` 抽象。
