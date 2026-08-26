//! 第 75 章 RTSP / 流媒体接入与实时处理管线
//! 运行：cargo run --example ch75_rtsp_pipeline
//! 依赖：仅 std（thread + mpsc + Mutex/Condvar）+ image 合成帧；纯 Rust，不连真实摄像头。
//!
//! 实时管线 = 采集 → 处理。当「处理」慢于「到帧」，就必须在两种策略间取舍：
//!   A) 无界队列、不丢帧：全部处理，但队列越堆越长，端到端延迟持续累积（越看越「慢半拍」）。
//!   B) 有界队列 + 丢最旧帧：队列满就丢掉最旧的一帧，牺牲完整性换取「始终处理最新帧」的低延迟。
//! 本例用一个「采集线程」按固定节奏产帧（模拟拉流），一个「处理线程」慢速消费，
//! 分别跑 A、B 两种策略，打印 产出/处理/丢弃 帧数与端到端延迟对比。
//! 真实 RTSP 拉流骨架（retina，纯 Rust 收流）见文件末尾注释与正文——不纳入本可运行示例。

use image::{ImageBuffer, Rgb, RgbImage};
use std::collections::VecDeque;
use std::sync::mpsc::{sync_channel, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 48;
const N_FRAMES: usize = 40; // 采集总帧数
const CAPTURE_PERIOD: Duration = Duration::from_millis(20); // 采集节奏：每 20ms 一帧 ≈ 50 fps
const PROCESS_COST: Duration = Duration::from_millis(45); // 单帧处理耗时 ≈ 45ms ≈ 22 fps（追不上！）
const BOUND_CAP: usize = 4; // 策略 B 的有界队列容量

/// 一帧「带时间戳的画面」：采集时刻 + 画面 + 帧序号。时间戳用于事后算端到端延迟。
#[allow(dead_code)] // seq 供真实实现做丢帧诊断；img 在 heavy_process 里被读
struct Stamped {
    seq: usize,
    captured_at: Instant,
    img: RgbImage,
}

/// 合成一帧：移动的小方块。真实场景这里换成「从 RTSP 解码出的帧」。
fn synth_frame(seq: usize) -> RgbImage {
    let mut img: RgbImage = ImageBuffer::from_pixel(W, H, Rgb([30, 30, 30]));
    let x = (seq as u32 * 3) % (W - 8);
    for dy in 0..8 {
        for dx in 0..8 {
            img.put_pixel(x + dx, 20 + dy, Rgb([200, 80, 80]));
        }
    }
    img
}

/// 模拟一次「重处理」（如跑一个检测模型）：做点真实的灰度求和 + 固定耗时 sleep。
fn heavy_process(s: &Stamped) -> u64 {
    let mut acc = 0u64;
    for p in s.img.pixels() {
        acc += (p[0] as u64 + p[1] as u64 + p[2] as u64) / 3;
    }
    thread::sleep(PROCESS_COST); // 模拟模型推理耗时
    acc
}

/// 策略 A：无界队列（std::mpsc::channel），不丢帧。
/// 全部帧都会被处理，但当处理慢于到帧，队列 backlog 增长，端到端延迟持续累积。
fn run_unbounded() -> (usize, usize, Duration, usize) {
    let (tx, rx) = std::sync::mpsc::channel::<Stamped>();
    let producer = thread::spawn(move || {
        for seq in 0..N_FRAMES {
            let f = Stamped {
                seq,
                captured_at: Instant::now(),
                img: synth_frame(seq),
            };
            if tx.send(f).is_err() {
                break;
            }
            thread::sleep(CAPTURE_PERIOD);
        }
        // tx 在此 drop → rx 收到 None，处理线程退出
    });

    let mut processed = 0usize;
    let mut max_backlog = 0usize;
    let mut last_latency = Duration::ZERO;
    for s in rx.iter() {
        // 处理前先看看积压了多少（近似：粗估通道长度靠已收未处理，无法直接查 → 用延迟侧面反映）
        let latency = s.captured_at.elapsed(); // 端到端延迟 = 现在 - 采集时刻
        last_latency = latency;
        let _ = heavy_process(&s);
        processed += 1;
        // 用「延迟 / 处理耗时」粗略反推积压帧数
        max_backlog = max_backlog.max((latency.as_millis() / PROCESS_COST.as_millis().max(1)) as usize);
    }
    producer.join().ok();
    (N_FRAMES, processed, last_latency, max_backlog)
}

/// 策略 B：有界队列 + 丢最旧帧。用 Arc<(Mutex<VecDeque>, Condvar)> 手搓一个容量受限的缓冲：
/// 队列满时 pop_front（丢掉最旧的一帧）再 push_back（放入最新帧），保证处理端总拿到「最新的一批」。
fn run_bounded_drop_oldest() -> (usize, usize, usize, Duration) {
    // 队列 + 「生产是否结束」标志；Condvar 用于处理线程在空队列上等待
    let buf = Arc::new((Mutex::new((VecDeque::<Stamped>::new(), false)), Condvar::new()));
    let dropped = Arc::new(Mutex::new(0usize));

    let buf_p = Arc::clone(&buf);
    let dropped_p = Arc::clone(&dropped);
    let producer = thread::spawn(move || {
        let (lock, cvar) = &*buf_p;
        for seq in 0..N_FRAMES {
            let f = Stamped {
                seq,
                captured_at: Instant::now(),
                img: synth_frame(seq),
            };
            {
                let mut g = lock.lock().unwrap();
                if g.0.len() >= BOUND_CAP {
                    g.0.pop_front(); // 丢最旧帧
                    *dropped_p.lock().unwrap() += 1;
                }
                g.0.push_back(f);
                cvar.notify_one();
            }
            thread::sleep(CAPTURE_PERIOD);
        }
        // 置结束标志并唤醒处理线程
        let mut g = lock.lock().unwrap();
        g.1 = true;
        cvar.notify_one();
    });

    let (lock, cvar) = &*buf;
    let mut processed = 0usize;
    let mut max_latency = Duration::ZERO;
    loop {
        let s = {
            let mut g = lock.lock().unwrap();
            // 队列空且生产未结束 → 等待
            while g.0.is_empty() && !g.1 {
                g = cvar.wait(g).unwrap();
            }
            if g.0.is_empty() && g.1 {
                break; // 生产结束且取空 → 退出
            }
            g.0.pop_front().unwrap()
        };
        max_latency = max_latency.max(s.captured_at.elapsed());
        let _ = heavy_process(&s);
        processed += 1;
    }
    producer.join().ok();
    let dropped = *dropped.lock().unwrap();
    (N_FRAMES, processed, dropped, max_latency)
}

fn main() {
    println!(
        "采集 {N_FRAMES} 帧，采集节奏 {}ms/帧(≈{} fps)，单帧处理 {}ms(≈{} fps) → 处理追不上到帧",
        CAPTURE_PERIOD.as_millis(),
        1000 / CAPTURE_PERIOD.as_millis(),
        PROCESS_COST.as_millis(),
        1000 / PROCESS_COST.as_millis(),
    );

    println!("\n—— 策略 A：无界队列 / 不丢帧 ——");
    let t0 = Instant::now();
    let (produced_a, processed_a, last_lat_a, backlog_a) = run_unbounded();
    println!(
        "产出 {produced_a} 帧，处理 {processed_a} 帧，丢弃 0 帧；末帧端到端延迟 {}ms，估算峰值积压 ~{backlog_a} 帧",
        last_lat_a.as_millis()
    );
    println!("总耗时 {}ms（≈处理端速度决定，采集被拖着走）", t0.elapsed().as_millis());

    println!("\n—— 策略 B：有界队列(容量 {BOUND_CAP}) + 丢最旧帧 ——");
    let t1 = Instant::now();
    let (produced_b, processed_b, dropped_b, max_lat_b) = run_bounded_drop_oldest();
    println!(
        "产出 {produced_b} 帧，处理 {processed_b} 帧，丢弃 {dropped_b} 帧；最大端到端延迟 {}ms",
        max_lat_b.as_millis()
    );
    println!("总耗时 {}ms（≈采集时长，处理端始终追最新帧）", t1.elapsed().as_millis());
    // 不变式：产出 = 处理 + 丢弃
    assert_eq!(produced_b, processed_b + dropped_b, "帧账目必须平衡");

    println!("\n结论：A 一帧不丢但延迟随时间累积（末帧延迟 {}ms）；B 丢了 {dropped_b} 帧，换来延迟被压在 {}ms 内。",
        last_lat_a.as_millis(), max_lat_b.as_millis());
    println!("实时监控要「看最新」→ 选 B；离线转码/不能丢帧 → 选 A 并接受延迟或加速处理。");

    // 附：sync_channel 的「丢最新帧」变体（满则 try_send 失败，直接丢当前这一帧）——最省事，但丢的是新帧。
    demo_sync_channel_drop_newest();
}

/// 用标准库 sync_channel(cap) 做「丢最新帧」：通道满时 try_send 返回 Full，生产者直接丢掉当前帧。
/// 与策略 B「丢最旧」相对——实现更简单，但保留的是较旧的帧。此处只跑几帧演示计数。
fn demo_sync_channel_drop_newest() {
    let (tx, rx) = sync_channel::<usize>(2);
    let prod = thread::spawn(move || {
        let mut dropped = 0;
        for seq in 0..12 {
            match tx.try_send(seq) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => dropped += 1, // 通道满 → 丢当前（最新）帧
                Err(TrySendError::Disconnected(_)) => break,
            }
            thread::sleep(Duration::from_millis(5));
        }
        dropped
    });
    let consumer = thread::spawn(move || {
        let mut got = 0;
        for _ in rx.iter() {
            thread::sleep(Duration::from_millis(20)); // 消费慢
            got += 1;
        }
        got
    });
    let dropped = prod.join().unwrap();
    let got = consumer.join().unwrap();
    println!(
        "\n附：sync_channel(容量2) 丢最新帧变体 → 产出 12，处理 {got}，丢弃 {dropped}（产出=处理+丢弃 校验：{}）",
        got + dropped == 12
    );
}

// ============================================================================
// 「换成真实 RTSP 流」的骨架（retina 0.4.20，纯 Rust 收流；需 `cargo add retina tokio`，
//  且底层还要一个 H.264 解码器把 f.data() 的码流变成像素——见正文。不纳入本可运行示例）：
//
// use retina::client::{Session, SessionOptions, SetupOptions, PlayOptions};
// use retina::codec::CodecItem;
// use futures::StreamExt;
//
// #[tokio::main]
// async fn main() -> Result<(), anyhow::Error> {
//     let url = "rtsp://user:pass@192.168.1.10:554/stream1".parse()?;
//     let mut session = Session::describe(url, SessionOptions::default()).await?;
//     session.setup(0, SetupOptions::default()).await?;          // 选第 0 路（视频）
//     let mut demuxed = session.play(PlayOptions::default()).await?.demuxed()?;
//     while let Some(item) = demuxed.next().await {
//         if let CodecItem::VideoFrame(f) = item? {
//             // ⚠️ f.data() 是编码后的 H.264 NAL 单元，不是像素！
//             //    要得到 RgbImage，还需送进 H.264 解码器（openh264=C / ffmpeg 子进程）。
//             let _encoded: &[u8] = f.data();
//             let _ts = f.timestamp();               // 带 RTP 时间戳，用于对齐
//             // decode(_encoded) -> RgbImage -> 塞进上面的有界队列 -> 处理线程
//         }
//     }
//     Ok(())
// }
// ============================================================================
