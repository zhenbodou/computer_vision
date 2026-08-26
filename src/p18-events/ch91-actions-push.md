# 第 91 章 事件联动与推送：抓拍、录像、Webhook/MQTT/Kafka

> **本章导读**
> - 前面八章我们把检测结果一路清洗、投票、去重、算成了干净的 `Event`（事件）。可**事件产生之后呢？** 光在内存里躺着没用——得**送出去**：发告警、存数据库、触发抓拍/录像、推到消息队列。这一章讲怎么把这"最后一公里"做得优雅又可靠。
> - 设计一个贯穿全书事件层的 **`trait Sink`（事件出口）** 抽象：无论下游是控制台、数据库、webhook 还是 Kafka，上层代码**只管把事件丢给 `Sink`**，不关心它到底送去哪。
> - 动手写出**打印 Sink、存内存/计数 Sink、多路 fan-out Sink（一个事件同时发多个下游）**，再套上**重试 + 死信队列**的可靠推送装饰器，并报真实分发数字。
> - 正文给出「换成 `ureq`（纯 Rust HTTP）真发 webhook」的代码片段，讲清 HTTP POST JSON、超时、重试、失败降级这些工业考量。demo 本身只用 std，免联网、可复现。

## 场景：告警响了，然后呢？

[第 90 章](ch90-arming-tamper.md)结束时，我们的事件层已经相当能打了：它能吐出"track#9 越线进入""保险柜前停留超时""镜头被遮挡"这样一条条精准的 `Event`。运维小哥很满意——直到他问了一句：

> "这些事件……我在哪能看到？手机能收到推送吗？能自动把那一刻的画面截下来存证吗？公司的大屏系统要对接，你给个接口呗？"

你愣住了。原来产生事件只是上半场，**把事件送到该去的地方**才是下半场。而"该去的地方"往往**不止一个**：

- 一条越线告警，可能同时要 **① 打到运维控制台**、**② 存进数据库留证**、**③ POST 给公司的告警平台（webhook）**、**④ 触发摄像头抓拍一张图**、**⑤ 推到 Kafka 让大数据组做统计**。
- 而且这些下游**可靠性天差地别**：打日志几乎不会失败；但 webhook 依赖网络，对方服务器可能超时、可能返回 500、可能正在重启。

如果你为每条推送写一坨 `if 要发webhook { ... } if 要存库 { ... }`，代码很快就烂成一团。这一章的核心，就是用一个小小的 **`trait`** 把这堆五花八门的出口收拾得服服帖帖。

> 📖 **术语**：**Sink（水槽 / 出口）**。数据流里，产生数据的一端叫 source（源），消费/吞掉数据的一端叫 **sink**。事件层的 sink 就是"事件的归宿"——把一条 `Event` 送到某个下游系统。webhook、数据库、消息队列、抓拍、录像，本质上都是**不同的 `Sink` 实现**。

## 原理：一个 trait 收编所有出口

回想[第 83 章](ch83-filtering-basics.md)那个"流水线一环 `Stage`"的心智模型——检测流过一串规则，最后吐出 `Event`。**`Sink` 就接在这条流水线的最末端**：

```text
… → 绊线/停留/遮挡等规则(ch87–90) → Event ─┬─→ [控制台 Sink]   打日志
                                           ├─→ [数据库 Sink]   存证 / 审计
                                           ├─→ [Webhook Sink]  POST JSON 给告警平台
                                           ├─→ [抓拍 Sink]     存一张当时的画面
                                           └─→ [Kafka Sink]    推给下游做统计
                                          （fan-out：一个事件，多个下游）
```

我们要的抽象只需回答两个问题：**"叫什么名字"**（出问题时好定位是哪个出口挂了）和**"把这条事件送出去，成了还是败了"**。于是 `trait Sink` 就三个方法：

```rust
/// 分发结果：Ok = 送达；Err(原因) = 失败（上层据此决定重试 / 降级）。
type SinkResult = Result<(), String>;

/// 事件出口：把一条去抖后的 Event 送到某个下游。
pub trait Sink {
    fn name(&self) -> &str;                       // 出口名字，用于日志与统计
    fn send(&mut self, ev: &Event) -> SinkResult; // 送一条事件，成败都要能看见
    fn report(&self) {}                           // 打印自己的内部统计（默认什么都不打）
}
```

**关键设计：`send` 返回的是 `Result` 而不是 `()`。** 一个只会"闷头发、不告诉你成没成"的出口是无法做可靠推送的——正因为能看见失败，我们后面才谈得上重试和降级。这是把玩具代码变成生产代码的分水岭。

> 💡 **提示**：`send` 取 `&mut self` 是有讲究的。像 webhook 这种出口需要在自己内部记"发了几次、失败几次"，甚至维护一个连接池；有状态，就得可变借用。无状态的打印 Sink 用不上 `&mut`，但为了统一接口，大家都长一个样。

## 动手写代码：从打印到可靠的多路分发

完整代码见 `code/examples/ch91_event_sink.rs`，**只用 std**，运行：

```bash
cargo run --example ch91_event_sink
```

我们不真跑检测器，直接**合成一批"去抖后的事件"**（模拟第 83–90 章各规则的产出：绊线进出、闯入、停留、遮挡……）当输入，一共 8 条。

### 出口一：打印 / 存内存计数

最朴素的两个 `Sink`。`PrintSink` 把事件打到控制台（永远成功）；`MemorySink` 把事件存进 `Vec` 并按 `kind` 计数——这就是"存库 / 审计"的极简版，真实项目里把 `push` 换成写数据库即可，接口不变：

```rust
/// ① 打印 Sink：打到控制台 / 日志。最简单，永远成功。
struct PrintSink;
impl Sink for PrintSink {
    fn name(&self) -> &str { "print" }
    fn send(&mut self, ev: &Event) -> SinkResult {
        println!("  [print] @{}ms {} {}", ev.ts_ms, ev.kind, ev.message);
        Ok(())
    }
}

/// ② 内存 / 计数 Sink：存起来并按 kind 计数（"存库 / 审计"的极简版）。
#[derive(Default)]
struct MemorySink {
    events: Vec<Event>,
    by_kind: HashMap<String, u32>,
}
impl Sink for MemorySink {
    fn name(&self) -> &str { "memory" }
    fn send(&mut self, ev: &Event) -> SinkResult {
        *self.by_kind.entry(ev.kind.clone()).or_insert(0) += 1;
        self.events.push(ev.clone());
        Ok(())
    }
    fn report(&self) { /* 打印 events.len() 与 by_kind，略 */ }
}
```

再写一个"统一分发入口"——它**不关心上游是哪条规则、下游是哪个出口**，只要是 `Sink` 就往里灌：

```rust
/// 把一批事件依次送进某个 Sink，返回 (送达数, 失败数)。
fn dispatch(events: &[Event], sink: &mut dyn Sink) -> (u32, u32) {
    let (mut ok, mut fail) = (0, 0);
    for ev in events {
        match sink.send(ev) {
            Ok(()) => ok += 1,
            Err(_) => fail += 1,
        }
    }
    (ok, fail)
}
```

存内存 Sink 跑完，`report()` 打印（数字真实可复现）：

```text
== B) 存内存 / 计数 Sink ==
分发条数：8
         └─ 存入 8 条，按 kind：dwell 2，intrusion 2，tamper 1，tripwire 3
```

### 出口二：fan-out 一个事件发多个下游

真实系统里一条告警要同时发好几处。`FanOutSink` 持有**一串下游** `Vec<Box<dyn Sink>>`，`send` 时挨个转发，**各记各的成败**——某个下游挂了不影响别的：

```rust
/// ④ Fan-out Sink：一个事件同时发往多个下游，各记各的成败。
/// 它自己也是一个 Sink——所以能被别的 Sink 再包一层，也能直接喂给 dispatch。
struct FanOutSink {
    sinks: Vec<Box<dyn Sink>>,
    ok: Vec<u32>,
    fail: Vec<u32>,
}
impl Sink for FanOutSink {
    fn name(&self) -> &str { "fanout" }
    fn send(&mut self, ev: &Event) -> SinkResult {
        let mut any_fail = false;
        for (i, s) in self.sinks.iter_mut().enumerate() {
            match s.send(ev) {
                Ok(())  => self.ok[i] += 1,
                Err(_)  => { self.fail[i] += 1; any_fail = true; }
            }
        }
        if any_fail { Err("部分下游失败".into()) } else { Ok(()) }
    }
    // report()：逐个下游打印 名字 + 送达/失败，再让下游自报统计
}
```

> 💡 **提示**：注意 `FanOutSink` **本身也实现了 `Sink`**。这是这套抽象最舒服的地方——出口可以**像积木一样嵌套**：fan-out 里塞一个 fan-out、或塞一个"重试装饰器"（下面就用到）。这种"整体也是部分"的结构叫**组合模式（composite pattern）**。

为了演示，第三个下游是个**模拟 webhook** 的 `WebhookSink`。真发 HTTP 会依赖网络、没法复现，所以我们用一个**确定性的"抖动网络"**来模拟它：每次发送有约 30% 的概率失败（超时或 5xx），用[第 90 章](ch90-arming-tamper.md)那个确定性 LCG 造可复现的成败序列。把 `[打印, 存内存, webhook]` 三个下游 fan-out，跑 8 条事件：

```text
== C) Fan-out 到 3 个下游（webhook 无重试）==
整体：5 条全部送达 fan-out，3 条至少一个下游失败
    下游 print    送达  8  失败  0
    下游 memory   送达  8  失败  0
    下游 webhook  送达  5  失败  3
         └─ 累计发送尝试 8 次，其中瞬时失败 3 次
=> 一个事件同时进了 3 个下游；webhook 有事件因瞬时故障丢了（没有重试兜底）
```

一个事件同时进了 3 个下游：打印和存库稳如老狗（8/8），**但 webhook 因为网络抖动，8 条里丢了 3 条**。丢的这 3 条告警，运维平台永远收不到了——这在安防场景是不可接受的。

### 出口三：重试 + 死信，让推送变可靠

网络抖动大多是**瞬时**的：这一下超时了，隔 200 毫秒再试一次多半就通了。所以我们写一个**装饰器** `RetrySink`：它包住任意一个"可能失败"的 `Sink`，失败就重试；重试若干次仍不行，就把事件塞进**死信队列（dead-letter queue）**兜底，**绝不悄悄丢弃**。

```rust
/// ⑤ 重试 + 失败降级装饰器：包住任意一个"可能失败"的 Sink（典型是 webhook）。
struct RetrySink {
    inner: Box<dyn Sink>,      // 被包住的真实出口
    max_retries: u32,
    retries_used: u32,
    dead_letter: Vec<Event>,   // 重试耗尽仍失败的事件，落这里等补发
}
impl Sink for RetrySink {
    fn name(&self) -> &str { "retry" }
    fn send(&mut self, ev: &Event) -> SinkResult {
        for attempt in 0..=self.max_retries {
            match self.inner.send(ev) {
                Ok(()) => { self.retries_used += attempt; return Ok(()); }
                Err(_) => { /* 真实项目此处指数退避 sleep 后再试；见下文 */ }
            }
        }
        self.dead_letter.push(ev.clone());        // 降级：进死信，不丢
        Err("重试耗尽，已进死信".into())
    }
}
```

把 webhook 用 `RetrySink`（最多重试 2 次）包起来，再 fan-out，同样 8 条事件：

```text
== D) 同样 3 个下游，但 webhook 外面套「重试(2 次) + 死信」==
整体：7 条 fan-out 全成功，1 条仍有下游失败（=重试耗尽进了死信）
    下游 print    送达  8  失败  0
    下游 memory   送达  8  失败  0
    下游 retry    送达  7  失败  1
         └─ 共发起重试 4 次，死信队列 1 条
         └─ 累计发送尝试 12 次，其中瞬时失败 5 次
=> 重试把大部分瞬时故障救了回来；实在救不回的进死信队列，绝不丢事件
```

对比看得很清楚：Demo C 里 webhook **丢了 3 条**；套上重试后，8 条里 **7 条成功送达**（发起了 4 次重试把瞬时故障救了回来），只有 1 条运气极差、连试 3 次都失败，被**稳稳接进死信队列**，等定时任务或人工补发——**一条都没丢**。这就是可靠推送的标准姿势：瞬时抖动靠重试救，真挂了靠死信兜底。

## 换成 ureq：真发一个 webhook

上面的 `WebhookSink` 是模拟的。真要发 HTTP，用纯 Rust 的 **`ureq`** 就好（它是同步阻塞式、无需 async 运行时，特别适合我们"每路一线程"的场景，见[第 92 章](ch92-multistream.md)）。先加依赖：

```bash
cargo add ureq --features json
```

> ⚠️ **注意**：下面按 **ureq 3.x**（本书写作时为 3.4）的 API 编写；ureq 2.x 用的是 `AgentBuilder::new().timeout(..)`，3.x 改成了 `Agent::config_builder()`。跨大版本 API 有变，**以你实际装到的版本文档为准**。

把 `WebhookSink::send` 里的模拟部分换成真请求：

```rust
use std::time::Duration;
use serde::Serialize;

#[derive(Serialize)]
struct Payload<'a> { kind: &'a str, ts_ms: u64, track_id: Option<u64>, message: &'a str }

fn send(&mut self, ev: &Event) -> SinkResult {
    // 建一个带「全局超时」的 agent（超时是可靠推送的命根子——绝不能无限等对方）
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(3)))
        .build()
        .into();

    let body = Payload { kind: &ev.kind, ts_ms: ev.ts_ms, track_id: ev.track_id, message: &ev.message };

    match agent.post(&self.url).send_json(&body) {
        Ok(_resp)                        => Ok(()),                         // 2xx：送达
        Err(ureq::Error::StatusCode(c))  => Err(format!("对方返回 {c}")),   // 4xx/5xx
        Err(e)                           => Err(format!("网络/超时错误：{e}")), // 传输层错误
    }
}
```

三点工业考量，正是上面 demo 里那套重试/降级要对付的：

- **超时（timeout）**：`timeout_global` 一定要设。不设的话，对方服务器"假死"会把你这条线程一直挂住——在多路并发里这是灾难（[第 92 章](ch92-multistream.md)细讲）。
- **区分两类失败**：`StatusCode` 是"对方收到了但拒绝了"（4xx 多半是你请求错了，**重试也没用**；5xx 是对方临时故障，**该重试**）；传输层错误（超时、连接被拒）几乎都是瞬时的，**该重试**。生产代码会据此决定"要不要重试"。
- **重试要退避**：demo 里为了可复现没 `sleep`。真实重试应**指数退避**（第 1 次等 200ms、第 2 次 400ms、第 3 次 800ms），避免对方刚缓过来又被你一拥而上打垮。

## 常见坑与提示

> 💡 **提示**：**抓拍、录像、MQTT、Kafka——都只是另一个 `Sink`。** 本章标题里那一串，本质上没有新东西：抓拍就是"收到事件时存一张当时的帧"的 `SnapshotSink`；录像是"把事件前后几秒的帧拼成一段"的 `ClipSink`；MQTT/Kafka 就是把 `send` 换成 `client.publish(topic, payload)` 的 `MqttSink`/`KafkaSink`（纯 Rust 有 `rumqttc`、`rdkafka` 等 crate）。抽象一旦立住，接新下游就是"再写一个 `impl Sink`"，上层代码一行不用改——这就是 `trait` 的价值。

> 🕳️ **坑**：**推送必须做幂等，否则重试会把告警发重。** 想想 Demo D：一条事件重试了好几次才成功。如果第 1 次其实**已经送到**对方、只是**返回的响应在网络上丢了**，你以为失败又重发一次——对方就收到**两条一模一样的告警**。工业做法：给每个 `Event` 带一个**唯一 id**（比如 `kind + track_id + ts_ms` 拼一个），下游按 id 去重（幂等消费）。这也是为什么[第 86 章](ch86-cooldown-dedup.md)的去重和本章的幂等要一起考虑。

> ⚠️ **注意**：**别在产生事件的线程里同步等 webhook。** `send_json` 会**阻塞**到对方响应或超时。如果你在检测/规则的主循环里直接调它，对方慢 3 秒，你的整条流水线就卡 3 秒、疯狂丢帧。正确姿势是把"分发"和"检测"**解耦**：事件先进一个队列，专门的分发线程去慢慢发。这正是[第 92 章](ch92-multistream.md)的并发架构要解决的问题。

> 💡 **提示**：**死信不是终点，是待办清单。** 进了死信队列的事件，要有人管：可以定时重放、可以落盘等人工处理、也可以攒够一批告警"XX 平台已失联 5 分钟"。死信队列的意义是**把"丢了"变成"晚点补"**——在安防、金融这类场景，这条命门绝不能省。

## 本章小结

- 事件产生只是上半场，**把事件可靠地送出去**才是下半场。用一个 `trait Sink`（`name` / `send` / `report`）把控制台、数据库、webhook、抓拍、录像、消息队列这些五花八门的出口统一收编，上层只管 `dispatch`，不关心下游是谁。
- `send` 返回 `Result` 是可靠推送的地基——**能看见失败**，才谈得上重试与降级。
- **fan-out** 让一个事件同时发多个下游，各记各的成败；`FanOutSink` 自己也是 `Sink`，可以像积木一样嵌套（组合模式）。
- 网络下游必然会抖：用 **`RetrySink` 装饰器**（重试 + 死信队列）把瞬时故障救回来、把真故障兜住。demo 里 8 条事件，无重试丢 3 条，加重试后 7 条送达、1 条稳稳进死信，一条不丢。
- 真发 HTTP 用纯 Rust 的 `ureq`：**必设超时**、**区分状态码错误与传输错误**、**重试要退避**、**推送要幂等**、**别在检测线程里同步阻塞发送**。

## 动手练习

1. **加一个抓拍 Sink**：写一个 `SnapshotSink`，收到事件时把 `ev.ts_ms` 记进一个"待抓拍清单"（真实项目里就是去缓存里捞那一刻的帧存盘）。把它加进 fan-out，验证它和别的下游并列工作。
2. **按 kind 路由**：写一个 `RouterSink`，内部持有一张 `HashMap<String, Box<dyn Sink>>`，按 `ev.kind` 把 `tamper` 发给运维、`dwell` 发给店长——不同事件走不同出口。
3. **幂等去重 Sink**：给 `Event` 造一个唯一 id（`format!("{}-{:?}-{}", kind, track_id, ts_ms)`），写一个 `DedupSink` 包在最外层，`send` 时先查 id 是否发过，发过就直接返回 `Ok` 不重复发。用它模拟"重试导致的重复"并验证被挡下。
4. **进阶·真发一次**：`cargo add ureq --features json`，找一个免费的 webhook 测试站点（如 webhook.site 给你的临时 URL），把 demo 里的 `WebhookSink::send` 换成上面的 `ureq` 版本，真发几条看看对方收到的 JSON。

---

下一章，我们要面对真正的工程规模问题：**一路摄像头一个线程，几十路怎么办？** 各路的检测/规则结果如何汇到一处统一分发？队列满了、处理不过来时，是丢最新的帧还是丢最旧的？我们会用 `std::thread` + `std::sync::mpsc` 搭出"多路→汇聚→分发"的经典架构，并把背压与丢帧策略讲透。👉 [第 92 章 多路流并发调度与工程架构](ch92-multistream.md)
