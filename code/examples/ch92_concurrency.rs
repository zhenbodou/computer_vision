//! 第 92 章 多路流并发调度与工程架构：每路一线程、mpsc 汇聚、背压与丢帧
//! 运行：cargo run --example ch92_concurrency
//! 依赖：无（只用 std：thread / mpsc / Arc / Mutex / Condvar / VecDeque）。
//!
//! Event 就是第 83 章给出的共享数据模型（见 _authoring/EVENTS_MODEL.md）。
//! 本章核心：把「每路视频一线程、各路结果汇到一个通道、统一分发」的工程架构落成可跑的代码，
//! 并把「通道满了怎么办」的三种策略（阻塞背压 / 丢最新 / 丢最旧）做成可复现的对照。
//!
//! 说明：Part 1、Part 2A 用真线程，但汇聚**总数**是确定的（不丢帧，全部送达）。
//!       Part 2B 的丢帧对照特意用**单线程确定性**演示，好让丢弃/处理条数可复现、
//!       讲清「留最旧还是留最新」这个策略差别；真并发下丢弃的**具体条数**会随调度微微浮动。

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::mpsc::{self, sync_channel, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ========================= 用到的共享数据模型（第 83 章） =========================
/// 事件层的输出：一条业务事件 / 告警（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Event {
    pub kind: String,
    pub ts_ms: u64,
    pub track_id: Option<u64>,
    pub message: String,
}
// ======================= 共享数据模型结束 =======================

/// 一路视频在某一帧处理完后，往汇聚通道里丢的一条消息。
/// 真实系统里这条消息背后是「解码→检测→跟踪→事件规则」跑完的结果；
/// 这里只保留调度需要的字段：哪一路、第几帧、时间戳、以及本帧是否命中了事件。
#[derive(Clone, Debug)]
struct Msg {
    stream_id: usize,
    frame_no: u64,
    ts_ms: u64,
    event: Option<Event>,
}

/// 模拟「一路视频处理完第 fno 帧」的产物：每 25 帧命中一次事件（可复现）。
fn process_one_frame(stream_id: usize, fno: u64) -> Msg {
    let ts_ms = 1000 + fno * 40; // 25 fps：每帧 +40ms（合成递增时间戳，不读系统时钟）
    let event = if fno % 25 == 24 {
        Some(Event {
            kind: "intrusion".to_string(),
            ts_ms,
            track_id: Some(stream_id as u64 * 1000 + fno),
            message: format!("流#{} 第{}帧触发闯入", stream_id, fno),
        })
    } else {
        None
    };
    Msg { stream_id, frame_no: fno, ts_ms, event }
}

// ============================================================================
// Part 1：每路一线程 + mpsc 单通道汇聚（真并发，汇聚总数确定）
// ============================================================================
fn part1_mpsc_fanin(n_streams: usize, frames_per: u64) {
    println!("== Part 1：每路一线程，mpsc 单通道汇聚 ==");

    // 一个通道，多个 Sender（每路 clone 一个），一个 Receiver（主线程）。
    let (tx, rx) = mpsc::channel::<Msg>();
    let mut handles = Vec::new();
    for sid in 0..n_streams {
        let tx = tx.clone(); // 每路自己的发送端
        handles.push(thread::spawn(move || {
            for fno in 0..frames_per {
                let msg = process_one_frame(sid, fno);
                // send 到共享通道；unbounded 通道不会阻塞、不会丢
                if tx.send(msg).is_err() {
                    break; // 接收端已挂，收摊
                }
            }
        }));
        // tx 在此 clone 的副本 move 进线程，线程结束时自动 drop
    }
    drop(tx); // 关键：主线程手里这把原始 Sender 必须 drop，否则 rx 永远等不到「全部结束」

    // 主线程：单通道汇聚，做聚合统计
    let mut per_stream = vec![0u32; n_streams];
    let mut events: Vec<Event> = Vec::new();
    for msg in rx {
        // rx 迭代到所有 Sender 都 drop 才结束
        per_stream[msg.stream_id] += 1;
        if let Some(e) = msg.event {
            events.push(e);
        }
    }
    for h in handles {
        h.join().unwrap();
    }

    let total: u32 = per_stream.iter().sum();
    println!("  {} 路，每路 {} 帧：", n_streams, frames_per);
    for (sid, &c) in per_stream.iter().enumerate() {
        println!("    流#{} 汇入 {} 帧", sid, c);
    }
    println!("  汇聚总帧数：{}（= {}×{}，不丢帧）", total, n_streams, frames_per);
    println!(
        "  汇聚到的事件数：{}（每路每 25 帧 1 次 → 每路 {} 次）",
        events.len(),
        frames_per / 25
    );
    println!("  => 这些事件接下来交给第 91 章的 Sink 统一分发（打印 / 存库 / webhook…）\n");
}

// ============================================================================
// Part 2A：有界阻塞队列（Arc<Mutex<VecDeque>> + Condvar）——满则「背压」，不丢帧
// ============================================================================
/// 经典的有界阻塞队列：队满时 push 阻塞（背压），队空时 pop 阻塞。
/// Condvar 是让线程「睡着等条件成立」的原语：不用忙等空转 CPU。
struct BoundedBlockingQueue<T> {
    inner: Mutex<VecDeque<T>>,
    not_full: Condvar,  // 「队列不满了」的通知
    not_empty: Condvar, // 「队列非空了」的通知
    cap: usize,
    peak: Mutex<usize>, // 记录历史峰值长度，用来证明「内存有界」
}

impl<T> BoundedBlockingQueue<T> {
    fn new(cap: usize) -> Self {
        BoundedBlockingQueue {
            inner: Mutex::new(VecDeque::new()),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
            cap,
            peak: Mutex::new(0),
        }
    }

    /// 入队：满了就等 not_full（生产者被「背压」住，摄像头端会感到卡——这正是代价）。
    fn push(&self, item: T) {
        let mut q = self.inner.lock().unwrap();
        while q.len() >= self.cap {
            q = self.not_full.wait(q).unwrap();
        }
        q.push_back(item);
        let len = q.len();
        drop(q);
        {
            let mut p = self.peak.lock().unwrap();
            if len > *p {
                *p = len;
            }
        }
        self.not_empty.notify_one();
    }

    /// 出队：空了就等 not_empty。
    fn pop(&self) -> T {
        let mut q = self.inner.lock().unwrap();
        while q.is_empty() {
            q = self.not_empty.wait(q).unwrap();
        }
        let item = q.pop_front().unwrap();
        self.not_full.notify_one();
        item
    }
}

fn part2a_backpressure(n_streams: usize, frames_per: u64, cap: usize) {
    println!("== Part 2A：有界阻塞队列（cap={}）——满则背压，不丢帧 ==", cap);
    let q: Arc<BoundedBlockingQueue<Msg>> = Arc::new(BoundedBlockingQueue::new(cap));

    let mut producers = Vec::new();
    for sid in 0..n_streams {
        let q = Arc::clone(&q);
        producers.push(thread::spawn(move || {
            for fno in 0..frames_per {
                q.push(process_one_frame(sid, fno)); // 满时在此阻塞（背压）
            }
        }));
    }

    // 消费者：总数已知（不丢帧），恰好取 n_streams*frames_per 条
    let total = n_streams as u64 * frames_per;
    let q_c = Arc::clone(&q);
    let consumer = thread::spawn(move || {
        let mut per_stream = std::collections::HashMap::new();
        for _ in 0..total {
            let msg = q_c.pop();
            *per_stream.entry(msg.stream_id).or_insert(0u32) += 1;
        }
        per_stream
    });

    for p in producers {
        p.join().unwrap();
    }
    let per_stream = consumer.join().unwrap();

    let got: u32 = per_stream.values().sum();
    let peak = *q.peak.lock().unwrap();
    println!("  汇聚总数：{}（应等于 {}×{}={}），丢弃：0", got, n_streams, frames_per, total);
    println!("  队列历史峰值长度：{} ≤ cap {}（内存有界，从不无限堆积）", peak, cap);
    println!("  => 代价：队满时生产者被阻塞等待（背压会「顶」到摄像头端）。适合能减速的源。\n");
}

// ============================================================================
// Part 2B：队满就丢——「丢最新」 vs 「丢最旧」（确定性对照，条数可复现）
// ============================================================================
fn part2b_drop_policies(n_frames: u64, cap: usize) {
    println!("== Part 2B：队满就丢——丢最新 vs 丢最旧（生产者猛推 {} 帧，缓冲 cap={}）==", n_frames, cap);
    let ts = |i: u64| 1000 + i * 40; // 帧号 → 时间戳

    // ---------- 策略一：丢最新（sync_channel(cap) + try_send，满了发不进就丢当前这帧）----------
    let (tx, rx) = sync_channel::<u64>(cap);
    let mut dropped_new = 0u64;
    for i in 0..n_frames {
        match tx.try_send(i) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => dropped_new += 1, // 缓冲已满 → 丢掉「最新」这帧
            Err(TrySendError::Disconnected(_)) => break,
        }
    }
    drop(tx);
    let survivors_new: Vec<u64> = rx.iter().collect(); // 缓冲里活下来的（最旧的 cap 帧）

    // ---------- 策略二：丢最旧（Arc<Mutex<VecDeque>> 环形缓冲：满了先弹出最旧再放新）----------
    // 真并发时它就是 Part 2A 那把队列去掉阻塞、改成 pop_front 腾位；这里单线程演示策略本身。
    let ring: Arc<Mutex<VecDeque<u64>>> = Arc::new(Mutex::new(VecDeque::with_capacity(cap)));
    let mut dropped_old = 0u64;
    for i in 0..n_frames {
        let mut q = ring.lock().unwrap();
        if q.len() == cap {
            q.pop_front(); // 丢掉「最旧」那帧，给最新的腾地方
            dropped_old += 1;
        }
        q.push_back(i);
    }
    let survivors_old: Vec<u64> = ring.lock().unwrap().iter().copied().collect(); // 最新的 cap 帧

    // ---------- 对照 ----------
    let brief = |v: &[u64]| {
        if v.is_empty() {
            "（空）".to_string()
        } else {
            format!(
                "帧#{}..#{}（ts {}..{}ms）",
                v[0],
                v[v.len() - 1],
                ts(v[0]),
                ts(v[v.len() - 1])
            )
        }
    };
    println!("  丢最新：处理 {} 帧，丢弃 {} 帧；活下来的是 {}", survivors_new.len(), dropped_new, brief(&survivors_new));
    println!("  丢最旧：处理 {} 帧，丢弃 {} 帧；活下来的是 {}", survivors_old.len(), dropped_old, brief(&survivors_old));
    println!("  => 丢弃条数相同，但「丢最新」留下的是**陈旧帧**，「丢最旧」永远处理**最新帧**（延迟低）。");
    println!("     安防/实时场景通常选「丢最旧」：宁可跳过旧画面，也要盯住此刻发生了什么。");
}

fn main() {
    // Part 1：4 路，每路 100 帧
    part1_mpsc_fanin(4, 100);

    // Part 2A：4 路，每路 50 帧，有界缓冲 16
    part2a_backpressure(4, 50, 16);

    // Part 2B：猛推 50 帧，缓冲 8
    part2b_drop_policies(50, 8);
}
