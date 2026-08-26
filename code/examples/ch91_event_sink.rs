//! 第 91 章 事件联动与推送：Sink 抽象 + fan-out 多路分发 + 重试降级
//! 运行：cargo run --example ch91_event_sink
//! 依赖：无（只用 std）——合成一批「去抖后的 Event」当输入，自包含、可复现。
//!       正文另给「换成 ureq 真发 webhook」的片段（需 cargo add ureq，本 demo 不引入）。
//!
//! Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md）。
//! 本章核心：把第 83–90 章各规则去抖后产出的 Event，经统一的 `trait Sink` 分发出去。

#![allow(dead_code)]

use std::collections::HashMap;

// ========================= 用到的共享数据模型（第 83 章） =========================
/// 事件层的输出：一条业务事件 / 告警（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String, // "tripwire" | "intrusion" | "dwell" | "tamper" | ...
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 分发结果：Ok = 送达；Err(原因) = 失败（上层据此决定重试 / 降级）。
/// 真实项目里失败原因用错误枚举更好，这里用 String 便于阅读。
type SinkResult = Result<(), String>;

/// 事件出口：把一条去抖后的 Event 送到某个下游（控制台 / 内存 / webhook / 消息队列…）。
/// - `send` 返回 Result：成败都要能看见，才谈得上重试与降级。
/// - `name` 用于日志与统计，出问题时知道「是哪个出口挂了」。
/// - `report` 打印该出口自己的内部统计（默认什么都不打），分发结束后做「体检报告」。
pub trait Sink {
    fn name(&self) -> &str;
    fn send(&mut self, ev: &Event) -> SinkResult;
    fn report(&self) {}
}

/// ① 打印 Sink：把事件打到控制台 / 日志。最简单，永远成功。
struct PrintSink;
impl Sink for PrintSink {
    fn name(&self) -> &str {
        "print"
    }
    fn send(&mut self, ev: &Event) -> SinkResult {
        let id = ev
            .track_id
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "  [print] @{:>5}ms {:<9} track={:<3} {}",
            ev.ts_ms, ev.kind, id, ev.message
        );
        Ok(())
    }
}

/// ② 内存 / 计数 Sink：把事件存起来并按 kind 计数——「存库 / 审计」的极简版。
/// 真实项目里换成写数据库、落文件皆可，接口 `send` 不变。
#[derive(Default)]
struct MemorySink {
    events: Vec<Event>,
    by_kind: HashMap<String, u32>,
}
impl Sink for MemorySink {
    fn name(&self) -> &str {
        "memory"
    }
    fn send(&mut self, ev: &Event) -> SinkResult {
        *self.by_kind.entry(ev.kind.clone()).or_insert(0) += 1;
        self.events.push(ev.clone());
        Ok(())
    }
    fn report(&self) {
        // 按 kind 名排序打印，保证输出可复现
        let mut kinds: Vec<_> = self.by_kind.iter().collect();
        kinds.sort_by(|a, b| a.0.cmp(b.0));
        let brief: Vec<String> = kinds.iter().map(|(k, v)| format!("{} {}", k, v)).collect();
        println!(
            "         └─ 存入 {} 条，按 kind：{}",
            self.events.len(),
            brief.join("，")
        );
    }
}

/// 确定性 LCG（同第 90 章思路）：给「抖动网络」造可复现的成败序列，免联网。
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 返回 [0,100) 的伪随机整数
    fn pct(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 100) as u32
    }
}

/// ③ 模拟 webhook Sink：真实项目里这里是一发 HTTP POST（见正文 ureq 片段）。
/// 为保持 demo 免联网、可复现，用一个「确定性抖动网络」模拟：每次发送尝试有
/// `fail_pct`% 的概率失败（超时 / 5xx）。这正是重试与降级要对付的东西。
struct WebhookSink {
    url: String,
    rng: Lcg,
    fail_pct: u32,
    attempts: u32, // 累计发送尝试次数（含被重试装饰器驱动的多次）
    failures: u32, // 其中失败的次数
}
impl WebhookSink {
    fn new(url: &str, fail_pct: u32, seed: u64) -> Self {
        WebhookSink {
            url: url.to_string(),
            rng: Lcg::new(seed),
            fail_pct,
            attempts: 0,
            failures: 0,
        }
    }
}
impl Sink for WebhookSink {
    fn name(&self) -> &str {
        "webhook"
    }
    fn send(&mut self, ev: &Event) -> SinkResult {
        self.attempts += 1;
        if self.rng.pct() < self.fail_pct {
            self.failures += 1;
            return Err(format!("POST {} 失败（超时/5xx）", self.url));
        }
        // 真实实现见正文 ureq 片段：把 ev 序列化成 JSON 发出去。这里只构造 payload 示意。
        let _payload = format!(
            "{{\"kind\":\"{}\",\"ts_ms\":{},\"track_id\":{},\"message\":\"{}\"}}",
            ev.kind,
            ev.ts_ms,
            ev.track_id.map(|i| i as i64).unwrap_or(-1),
            ev.message
        );
        Ok(())
    }
    fn report(&self) {
        println!(
            "         └─ 累计发送尝试 {} 次，其中瞬时失败 {} 次",
            self.attempts, self.failures
        );
    }
}

/// ④ Fan-out Sink：一个事件同时发往多个下游（控制台 + 存库 + webhook…）。
/// 某个下游失败不影响别的下游（各记各的成败），这正是工业分发的常态。
/// 它自己也是一个 Sink——所以能被别的 Sink 再包一层，也能直接喂给 dispatch。
struct FanOutSink {
    sinks: Vec<Box<dyn Sink>>,
    ok: Vec<u32>,   // 每个下游各自的送达计数
    fail: Vec<u32>, // 每个下游各自的失败计数
}
impl FanOutSink {
    fn new(sinks: Vec<Box<dyn Sink>>) -> Self {
        let n = sinks.len();
        FanOutSink {
            sinks,
            ok: vec![0; n],
            fail: vec![0; n],
        }
    }
}
impl Sink for FanOutSink {
    fn name(&self) -> &str {
        "fanout"
    }
    fn send(&mut self, ev: &Event) -> SinkResult {
        let mut any_fail = false;
        // sinks / ok / fail 是不相交的字段，可同时可变借用
        for (i, s) in self.sinks.iter_mut().enumerate() {
            match s.send(ev) {
                Ok(()) => self.ok[i] += 1,
                Err(_) => {
                    self.fail[i] += 1;
                    any_fail = true;
                }
            }
        }
        if any_fail {
            Err("部分下游失败".into())
        } else {
            Ok(())
        }
    }
    fn report(&self) {
        for (i, s) in self.sinks.iter().enumerate() {
            println!(
                "    下游 {:<8} 送达 {:>2}  失败 {:>2}",
                s.name(),
                self.ok[i],
                self.fail[i]
            );
            s.report(); // 每个下游再自报内部统计
        }
    }
}

/// ⑤ 重试 + 失败降级装饰器：包住任意一个「可能失败」的 Sink（典型是 webhook）。
/// - 失败就重试，最多 max_retries 次（共 max_retries+1 次尝试）；
/// - 仍失败则「降级」：把事件塞进死信队列（dead-letter），等人工 / 定时补发，绝不丢。
/// 这是可靠推送的标配：瞬时抖动靠重试救，真挂了靠死信兜底。它本身也是一个 Sink。
struct RetrySink {
    inner: Box<dyn Sink>,
    max_retries: u32,
    retries_used: u32,       // 累计重试次数（衡量瞬时故障有多严重）
    dead_letter: Vec<Event>, // 重试耗尽仍失败的事件，落到这里
}
impl RetrySink {
    fn new(inner: Box<dyn Sink>, max_retries: u32) -> Self {
        RetrySink {
            inner,
            max_retries,
            retries_used: 0,
            dead_letter: Vec::new(),
        }
    }
}
impl Sink for RetrySink {
    fn name(&self) -> &str {
        "retry"
    }
    fn send(&mut self, ev: &Event) -> SinkResult {
        let mut last_err = String::new();
        for attempt in 0..=self.max_retries {
            match self.inner.send(ev) {
                Ok(()) => {
                    self.retries_used += attempt; // attempt 次重试后才成功
                    return Ok(());
                }
                Err(e) => {
                    last_err = e;
                    // 真实项目此处应「指数退避」sleep(200ms→400ms→800ms) 再重试；
                    // demo 不 sleep，以免拖慢、且保持可复现。见正文。
                }
            }
        }
        // 重试耗尽：降级到死信队列，返回 Err 让上层知道「这条没能实时送达」。
        self.retries_used += self.max_retries;
        self.dead_letter.push(ev.clone());
        Err(format!(
            "重试 {} 次仍失败，已进死信：{}",
            self.max_retries, last_err
        ))
    }
    fn report(&self) {
        println!(
            "         └─ 共发起重试 {} 次，死信队列 {} 条",
            self.retries_used,
            self.dead_letter.len()
        );
        self.inner.report(); // 连带打印被包住的 webhook 的尝试 / 失败统计
    }
}

/// 把一批事件依次送进某个 Sink，返回 (送达数, 失败数)。
/// 这就是「事件产生后统一分发」的入口：上游是哪条规则不重要，出口实现 Sink 即可。
fn dispatch(events: &[Event], sink: &mut dyn Sink) -> (u32, u32) {
    let (mut ok, mut fail) = (0u32, 0u32);
    for ev in events {
        match sink.send(ev) {
            Ok(()) => ok += 1,
            Err(_) => fail += 1,
        }
    }
    (ok, fail)
}

/// 合成一批事件：模拟第 83–90 章各规则「去抖后」的产出（绊线进出、闯入、停留、遮挡…）。
/// 时间戳自己造递增值，不读系统时钟（可复现）。
fn make_events() -> Vec<Event> {
    vec![
        Event { kind: "tripwire".into(),  ts_ms: 1_000,  track_id: Some(1), message: "track 1 进 IN（门外→门内）".into() },
        Event { kind: "intrusion".into(), ts_ms: 1_200,  track_id: Some(7), message: "track#7 进入 ROI".into() },
        Event { kind: "tripwire".into(),  ts_ms: 1_600,  track_id: Some(2), message: "track 2 出 OUT（门内→门外）".into() },
        Event { kind: "dwell".into(),     ts_ms: 31_200, track_id: Some(7), message: "track#7 停留超时：32.5s（≥30s）".into() },
        Event { kind: "intrusion".into(), ts_ms: 35_000, track_id: Some(9), message: "track#9 进入 ROI".into() },
        Event { kind: "tamper".into(),    ts_ms: 40_000, track_id: None,    message: "设备异常(遮挡)".into() },
        Event { kind: "tripwire".into(),  ts_ms: 42_000, track_id: Some(9), message: "track 9 进 IN（门外→门内）".into() },
        Event { kind: "dwell".into(),     ts_ms: 65_000, track_id: Some(9), message: "track#9 停留超时：30.0s（≥30s）".into() },
    ]
}

fn main() {
    let events = make_events();
    println!("共有 {} 条去抖后的事件要分发。\n", events.len());

    // ---------- A) 单个 PrintSink ----------
    println!("== A) 单个打印 Sink ==");
    let mut print_sink = PrintSink;
    let (ok, _) = dispatch(&events, &mut print_sink);
    println!("分发条数：{}\n", ok);

    // ---------- B) 存内存 / 计数 Sink ----------
    println!("== B) 存内存 / 计数 Sink ==");
    let mut mem = MemorySink::default();
    let (ok, _) = dispatch(&events, &mut mem);
    println!("分发条数：{}", ok);
    mem.report();
    println!();

    // ---------- C) Fan-out 到 3 个下游（webhook 无重试兜底）----------
    println!("== C) Fan-out 到 3 个下游（webhook 无重试）==");
    let mut fan = FanOutSink::new(vec![
        Box::new(PrintSink),
        Box::new(MemorySink::default()),
        Box::new(WebhookSink::new("https://alarm.example.com/hook", 30, 76)),
    ]);
    // 注意：这里不再逐条打印（PrintSink 仍会打），重点看 fan-out 计数
    let (ok, fail) = dispatch(&events, &mut fan);
    println!("整体：{} 条全部送达 fan-out，{} 条至少一个下游失败", ok, fail);
    fan.report();
    println!("=> 一个事件同时进了 3 个下游；webhook 有事件因瞬时故障丢了（没有重试兜底）\n");

    // ---------- D) 给 webhook 套上「重试 + 死信」再 fan-out ----------
    println!("== D) 同样 3 个下游，但 webhook 外面套「重试(2 次) + 死信」==");
    let webhook = WebhookSink::new("https://alarm.example.com/hook", 30, 76);
    let reliable = RetrySink::new(Box::new(webhook), 2);
    let mut fan2 = FanOutSink::new(vec![
        Box::new(PrintSink),
        Box::new(MemorySink::default()),
        Box::new(reliable),
    ]);
    let (ok, fail) = dispatch(&events, &mut fan2);
    println!("整体：{} 条 fan-out 全成功，{} 条仍有下游失败（=重试耗尽进了死信）", ok, fail);
    fan2.report();
    println!("=> 重试把大部分瞬时故障救了回来；实在救不回的进死信队列，绝不丢事件");
}
