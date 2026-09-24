# 第 29 章 · 工程化：并发、性能与部署

> **本章导读**：走到这里，第 26~28 章的三个实战程序都能出正确结果了——检测、跟踪、绊线、抓拍，一样不少。但"能出结果"和"能上线"之间还差一大截：单线程喂不满 GPU，GPU 大把时间在空等；每帧串行跑，总延迟卡在最慢的那一环上；模型没预热，头一帧慢得吓人；程序在你的开发机跑得欢，换到边缘盒子上直接起不来。本章就干一件事：把 demo 升级成能 7×24 小时稳定跑的产品。我们会讲流水线并发（本章重点）、数据并行、推理侧性能、零拷贝、可观测性与稳定性、跨平台部署，最后给一张上线检查清单。这是全书的收尾——把前面所有零件拧成一台能交付的机器。

## 29.1 为什么需要工程化：串行流水线的浪费

先看看"能跑但不够快"的程序长什么样。第 26 章那个端到端检测程序，一帧的处理是四步顺着来的：**解码 → 预处理 → 推理 → 后处理**。单线程串行跑，时间线是这样的：

```text
时间轴（毫秒）→
帧1: [解码 8][预处理 5][    推理 20    ][后处理 3]
                                                帧2: [解码 8][预处理 5][    推理 20    ]...

单帧总延迟 = 8 + 5 + 20 + 3 = 36 ms  →  最高 ~28 FPS
GPU 只在“推理 20ms”里干活，其余 16ms 全在空等 → GPU 利用率 ≈ 20/36 ≈ 55%
```

两个问题一眼看穿：

1. **总延迟 = 各环节相加。** 四步排队走，谁也不能提前开工，36ms 是硬加出来的。
2. **GPU 大量空闲。** 解码、预处理、后处理这 16ms 里，GPU（推理主力）在干等。你花大价钱买的显卡，一半时间在摸鱼。

症结在于**串行**：CPU 在解码时，GPU 闲着；GPU 在推理时，CPU 又闲着。两边轮流上班，从不同时开工。工程化的第一刀，就是把这条串行流水线改成**并发流水线**——让解码、推理这些环节像工厂流水线一样**同时运转**，各干各的，用传送带（channel）把上一环的产出递给下一环。

## 29.2 流水线并发：把四步拆成四条线（本章重点）

工厂流水线的精髓是：**工位不等人**。焊接工焊完一个零件立刻传给下一个工位，自己马上焊下一个，绝不站着等下游装配完。我们照搬这个思路：把解码、预处理、推理、后处理各自放进一个（或多个）**线程**，用**channel（通道）**当传送带把它们串起来。

```text
[解码线程] ──ch1──▶ [预处理线程] ──ch2──▶ [推理线程] ──ch3──▶ [后处理线程]
   CPU                 CPU/SIMD              GPU               CPU

四个工位同时开工后的时间线：
解码:  [F1][F2][F3][F4][F5]...        ← 一直在解码后面的帧
预处理:    [F1][F2][F3][F4]...
推理:         [   F1   ][   F2   ][   F3   ]  ← GPU 几乎不停歇！
后处理:                 [F1][F2][F3]...

吞吐 = 1 / 最慢环节 = 1 / 20ms ≈ 50 FPS（几乎翻倍）
```

关键转变：**吞吐不再等于各环节相加，而是取决于最慢的那一环**（这里是推理 20ms）。GPU 从"忙 20 闲 16"变成"基本一直在推理"。注意单帧延迟没变小（甚至因排队略增），但**单位时间处理的帧数翻了倍**——这正是吞吐与延迟的区别（第 14 章讲 batch 时提过）。

### 骨架代码：crossbeam channel + 线程

Rust 标准库有 `std::sync::mpsc`，但它只支持"多生产者单消费者"，且没有好用的有界/超时 API。工程上更常用 **`crossbeam::channel`**——支持多生产者多消费者、有界通道、`try_send`/`recv_timeout`，正好对上我们的需求。先定义流水线里流动的"载体"（复用全书统一类型）：

```rust,ignore
use crossbeam::channel::{bounded, Receiver, Sender};
use ndarray::{Array3, Array4};
use std::thread;
use anyhow::Result;

// 复用前几章：Detector（第14/16章）、Detection/BBox（全书统一类型）、LetterboxInfo（第10章）

/// 解码出的一帧原图
struct Decoded { id: u64, img: image::RgbImage }
/// 预处理后的输入张量（含 letterbox 还原信息）
struct Prepped { id: u64, input: Array4<f32>, info: LetterboxInfo }
/// 推理产出的原始输出 [1,84,8400]
struct Raw    { id: u64, out: Array3<f32>, info: LetterboxInfo }
// 上面三个载体对应前三条传送带；第四个工位（后处理）产出最终的 Vec<Detection>，
// 就地打框/上报——它是流水线终点，不再往下一条通道传，所以不必再定义一个载体类型。
```

主函数：用 **`bounded` 有界通道**连接各 stage，`thread::spawn` 起线程。

```rust,ignore
fn main() -> Result<()> {
    // 每段之间用容量很小的有界通道连接 —— 容量小是故意的，见下文“背压”
    let (tx_dec, rx_dec) = bounded::<Decoded>(4);
    let (tx_pre, rx_pre) = bounded::<Prepped>(4);
    let (tx_inf, rx_inf) = bounded::<Raw>(4);

    // ① 解码线程：从视频/RTSP 一帧帧读出来，塞进 ch1
    let h1 = thread::spawn(move || decode_stage("rtsp://cam/stream", tx_dec));
    // ② 预处理线程：ch1 取原图 → letterbox+归一化 → 塞进 ch2
    let h2 = thread::spawn(move || preprocess_stage(rx_dec, tx_pre));
    // ③ 推理线程：Session 在这里【只建一次】，ch2 取张量 → run → 塞进 ch3
    let h3 = thread::spawn(move || infer_stage(rx_pre, tx_inf));
    // ④ 后处理线程：ch3 取原始输出 → 阈值+NMS+还原坐标 → 打框/上报
    let h4 = thread::spawn(move || postprocess_stage(rx_inf));

    // 等所有线程收尾（? 把线程内的 anyhow::Error 冒泡上来）
    h1.join().unwrap()?; h2.join().unwrap()?;
    h3.join().unwrap()?; h4.join().unwrap()?;
    Ok(())
}
```

每个 stage 就是"从上游 `recv`、干活、往下游 `send`"的循环。以预处理为例：

```rust,ignore
fn preprocess_stage(rx: Receiver<Decoded>, tx: Sender<Prepped>) -> Result<()> {
    // for 循环遍历 Receiver：上游把 Sender 全部 drop 后，循环自然结束
    for Decoded { id, img } in rx {
        let (input, info) = preprocess(&img, 640);        // 复用第 10 章
        // send 会在下游通道满时【阻塞】——这就是背压（见下）
        if tx.send(Prepped { id, input, info }).is_err() {
            break; // 下游线程没了，收工
        }
    }
    Ok(()) // 循环结束 → tx 被 drop → 下游的 for 也随之结束（优雅收尾链）
}
```

推理 stage 是把 Session 建一次、循环里只 `run` 的地方——完全对应第 14 章"Session 只建一次"的铁律，只是搬进了独立线程。

### 背压：有界通道满了就顶住上游

为什么通道容量设成 `4` 而不是无限大？这是**背压（backpressure）**的核心。设想推理慢（20ms/帧）、解码快（8ms/帧）：解码疯狂产帧，推理消化不动，帧就堆在通道里。**如果通道无界，内存会一路涨到 OOM（内存耗尽）程序崩溃。**

有界通道的 `send` 在通道满时会**阻塞**，一路把"慢"传导回最上游——解码线程 `send` 被卡住，自然就慢下来，跟着推理的节奏走。整条流水线自动以最慢环节的速率运行，内存稳定在"通道容量 × 单帧大小"这个可控的上限内。这就是背压：**下游顶不住，就让上游等一等，用阻塞换来内存可控。**

```text
解码(快) ──▶ [■■■■] 满了！──▶ 推理(慢)
              ↑ send 阻塞，解码被迫等待 → 内存不再上涨
```

### 丢帧策略：实时流处理不过来，就主动丢旧帧

背压对**离线任务**是完美的——处理一个视频文件、跑一个图片文件夹，慢就慢点，**一帧都不能少**。但**实时流**（RTSP 摄像头、直播）不一样：摄像头以 25FPS 恒速推帧，你处理不过来时，背压会把解码卡住，导致**积压的是越来越旧的画面**，报警延迟越拖越大——报"3 秒前有人闯入"毫无意义。

实时场景的正解是**主动丢帧**：与其攒一堆过时的旧帧,不如丢掉旧的、永远处理最新的。用 **`try_send`** 代替 `send`——满了不阻塞，直接丢：

```rust,ignore
fn decode_stage_realtime(url: &str, tx: Sender<Decoded>) -> Result<()> {
    let mut id = 0u64;
    let mut dropped = 0u64;
    loop {
        let img = read_next_frame(url)?; // 从 RTSP 读一帧
        id += 1;
        match tx.try_send(Decoded { id, img }) {
            Ok(()) => {}
            // 通道满 = 下游忙不过来 → 丢掉这一帧，绝不阻塞解码
            Err(crossbeam::channel::TrySendError::Full(_)) => {
                dropped += 1;
                if dropped % 100 == 0 { tracing::warn!(dropped, "实时丢帧累计"); }
            }
            Err(crossbeam::channel::TrySendError::Disconnected(_)) => break,
        }
    }
    Ok(())
}
```

离线要全帧、实时可丢帧，这是视频分析特有的工程抉择，一句话总结：

| | 离线批处理（视频文件/图库） | 实时流（RTSP/直播） |
| --- | --- | --- |
| 帧的取舍 | **一帧不丢**，慢就慢 | 处理不过来就**丢旧帧**，保最新 |
| 通道满时 | `send` **阻塞**（背压顶住上游） | `try_send` **失败即丢** |
| 优化目标 | 结果完整、吞吐最大 | 延迟最低、画面最新鲜 |
| 典型指标 | 总处理时长 | 端到端延迟、丢帧率 |

## 29.3 数据并行：一批一起算，SIMD 加速缩放

流水线并发是把**不同的活**分给不同线程（任务并行）；数据并行是把**同一种活的一批数据**摊给多核同时算。两者常常叠加使用。

**`rayon`** 让数据并行简单到只改一个词：`iter()` 换成 `par_iter()`。比如离线要预处理一整个文件夹的图，或后处理时对一批候选框并行算 NMS：

```rust,ignore
use rayon::prelude::*;

// 一批图并行预处理：rayon 自动切分给线程池，吃满所有 CPU 核
let inputs: Vec<(Array4<f32>, LetterboxInfo)> =
    images.par_iter().map(|img| preprocess(img)).collect();

// 一批检测框并行做点独立计算（各框互不依赖，天然可并行）
detections.par_iter_mut().for_each(|d| d.score = calibrate(d.score));
```

预处理里最吃 CPU 的一步是**缩放**（letterbox 要把原图等比缩到 640）。`image` 自带的 `resize` 是通用实现，而 **`fast_image_resize`** 用 SIMD（单指令多数据，一条指令同时算多个像素）专门优化了缩放，同样质量下常快 **3~10 倍**：

```rust,ignore
use fast_image_resize::{images::Image, PixelType, Resizer, ResizeOptions};

// 把 src（原图）缩放到 dst（640×640）；实际 API 随版本略有差异，以文档为准
let src = Image::from_vec_u8(w, h, rgb_bytes, PixelType::U8x3)?;
let mut dst = Image::new(640, 640, PixelType::U8x3);
let mut resizer = Resizer::new();          // Resizer 可复用，别每帧新建
resizer.resize(&src, &mut dst, &ResizeOptions::new())?;  // SIMD 加速
```

对比一下：`image::imageops::resize` 胜在零依赖、够用；`fast_image_resize` 胜在**快**，实时高分辨率视频里这几毫秒/帧的差距很关键。同理，**CHW 排布转换（HWC→CHW）、`u8→f32` 归一化**这些逐像素操作，也都能用 SIMD 或 rayon 并行加速——第 10 章埋过伏笔，这里正式兑现。

## 29.4 推理侧性能：预热、EP、batch 与并发 run

推理往往是流水线里最慢、最烧钱的一环，值得单独榨性能。

**1. 预热（warmup）——必做。** 引擎第一次推理往往慢 2~10 倍：图优化、显存分配、计算核首次编译全挤在第一发里（第 14 章讲过）。上线前先拿假输入空跑几次把它"焐热"，别让第一个真实请求当小白鼠：

```rust,ignore
// 用真实形状的假输入预热，把冷启动开销吃掉
let dummy = Array4::<f32>::zeros((1, 3, 640, 640));
for _ in 0..3 {
    let _ = session.run(ort::inputs!["images" => TensorRef::from_array_view(&dummy)?])?;
}
tracing::info!("模型预热完成，可以接客了");
```

**2. 执行提供器（EP）选择。** 有 NVIDIA 显卡上 CUDA/TensorRT，苹果芯片上 CoreML，Intel 平台上 OpenVINO，都没有就 CPU 兜底——这是第 13、14 章的核心内容，`ort` 里按优先级注册即可：

```rust,ignore
use ort::session::{builder::GraphOptimizationLevel, Session};

let mut session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_execution_providers([
        #[cfg(feature = "tensorrt")] ort::ep::TensorRT::default().build(),
        #[cfg(feature = "cuda")]     ort::ep::CUDA::default().build(),
        #[cfg(feature = "coreml")]   ort::ep::CoreML::default().build(),
    ])? // 排前面的先试，不支持就自动回退，最后退到 CPU
    .commit_from_file("yolov8n.onnx")?;
```

> ⚠️ EP 注册失败默认**静默回退 CPU**（第 14 章的坑）。上线前务必确认"以为在用 GPU、其实在跑 CPU"没发生。

**3. batch 推理。** 多路摄像头凑齐 N 帧一次推，输入 `[N,3,640,640]`、输出 `[N,84,8400]`。GPU 像大灶台，炒一份和炒十份耗时差不多，凑满了单张成本更低——**吞吐↑，但要等凑齐、单帧延迟↑**。这是吞吐与延迟的经典权衡，细节见第 13、14 章。

**4. FP16/INT8 量化。** 把权重从 float32 降到 float16 甚至 int8，模型更小、算得更快，精度略降——用速度换精度的开关，怎么导出量化模型见第 13 章。

**5. Session 要复用，但不能在多个线程间直接并发 `run`。** 本书使用的 `ort 2.0.0-rc.13` 中，`Session::run()` 需要 `&mut self`。官方这样设计，是因为部分 EP 的分配器或统计组件不能安全地并发访问。最简单可靠的结构就是 29.2 的流水线：一个推理线程独占一个 Session，其他线程负责解码、预处理和后处理。确实要开多个推理线程时，每个线程各建并长期复用一个 Session，并评估额外的模型内存：

```rust,ignore
for _ in 0..4 {
    thread::spawn(move || {
        let mut session = build_session()?; // 每个工作线程创建一次
        while let Ok(input) = recv_input() {
            let _ = session.run(input)?;    // 在线程内反复复用
        }
        Ok::<(), anyhow::Error>(())
    });
}
```

若用 `Mutex<Session>` 包住单个 Session，代码虽然能共享，但锁会把 `run` 串行化；它适合统一所有权，不会增加推理并行度。优先先调好 ORT 的内部线程数或使用 batch，再考虑多 Session。

## 29.5 内存与零拷贝：别每帧都重新分配

高帧率下，"每帧都 new 一个大 buffer、算完就扔"是隐形的性能杀手——内存分配器被反复折腾，还制造 GC 式的抖动。三条省内存的思路（点到为止，深入见第 10 章）：

- **复用缓冲区。** 预处理的输出张量、`fast_image_resize` 的 `Resizer`、临时 `Vec`，都在循环外建一次、循环里反复写，别每帧 `Vec::new()`。有界通道里流动的 `Frame` 甚至可以做成对象池循环利用。
- **尽量零拷贝地衔接。** 解码输出直接喂给预处理，能借用就别 `clone`；取推理输出用视图（`try_extract_array`/`to_array_view` 拿到的是借用，不 `to_owned` 就不拷贝，第 14 章讲过）；ORT 的 IO Binding 还能省掉 CPU↔GPU 的来回搬运。
- **一趟做完。** `u8→f32` 归一化和 letterbox 填充别分两趟扫内存——在同一个循环里，读一个像素就地算好、写进目标位置，少一次几十 MB 的内存往返。

## 29.6 可观测性与稳定性：让程序"说话"，也让它"扛揍"

demo 崩了你重启一下就行；线上程序崩一次可能就是一起事故。要让它**看得见状态、扛得住异常**。

**结构化日志与分段耗时——用 `tracing`。** `println!` 上不了生产。`tracing` 能打带字段的结构化日志、能给每个 stage 计时、能按 span（跨度）串起一帧的完整轨迹：

```rust,ignore
use tracing::{info, warn, info_span};
use std::time::Instant;

// 给这一帧开一个 span，后面所有日志自动带上 frame_id
let span = info_span!("infer", frame_id = id);
let _guard = span.enter();

let t = Instant::now();
let out = session.run(/* ... */)?;
info!(elapsed_ms = t.elapsed().as_millis(), "推理完成"); // 结构化字段，好检索、好聚合
```

**统计端到端 FPS/延迟。** 记录每帧"进流水线"到"出结果"的时间戳，滑动窗口算平均延迟和 FPS，暴露出来（打日志或走 metrics）。没有这个数字，你根本不知道系统是不是在悄悄变慢。

**错误恢复——单点故障别拖垮整机。** 视频流里坏一帧太正常了（网络抖动、解码错误）。**一帧解码失败就跳过，绝不 `panic`**；`panic` 会让整个线程死掉、流水线断裂。用 `anyhow::Result` 兜住每一步，坏帧记个 `warn` 接着跑：

```rust,ignore
loop {
    match read_next_frame(url) {
        Ok(img) => { /* 正常送入流水线 */ }
        Err(e) => { warn!(error = %e, "解码失败，跳过此帧"); continue; } // 不 panic
    }
}
```

RTSP **断流重连**同理：读流报错不是终点，是"歇一下重连"的信号——退避重试（sleep 递增），连上了继续跑：

```rust,ignore
loop {
    match connect_rtsp(url) {
        Ok(stream) => run_until_error(stream),           // 正常跑，直到断开
        Err(e) => {
            warn!(error = %e, "RTSP 断开，5 秒后重连");
            std::thread::sleep(std::time::Duration::from_secs(5)); // 退避重连
        }
    }
}
```

**优雅退出（graceful shutdown）。** 收到停止信号（Ctrl-C / SIGTERM）别硬杀——那会丢掉在途的帧、留下没落盘的报警。正确姿势：置一个原子标志位，让**最上游停止产帧**，然后 **drop 掉 Sender**；下游的 `for rx` 循环取空后自然结束，一路把管道里的存货处理完（drain）再退出。crossbeam 通道天然支持这个"Sender 全 drop → Receiver 收尾"的链式收场。

## 29.7 部署与跨平台：从"我这能跑"到"哪都能跑"

**1. `cargo build --release`——必须！** debug 构建没开优化，推理和图像处理能慢 **10~50 倍**。别拿 debug 版本量性能、更别拿它上线。这是最容易忘、代价最大的一条：

```bash
cargo build --release          # 开满优化，产物在 target/release/
```

**2. 动态库依赖要随程序分发。** `ort` 依赖 `onnxruntime` 动态库（Linux `.so` / Windows `.dll` / macOS `.dylib`）。开发机上它自动下载好了，但**部署时目标机器没有**——程序一跑就报"找不到库"。两种解法：

- 把 `libonnxruntime.so` 和可执行文件放一起，运行前设 `LD_LIBRARY_PATH`（Linux）：

  ```bash
  LD_LIBRARY_PATH=./libs ./my_detector    # 告诉动态链接器去哪找库
  ```
- 或者编译时写死 **rpath**（把库的搜索路径烧进可执行文件），部署时不用再设环境变量，更省心。

**3. 交叉编译到 ARM（边缘盒子、Jetson、RK 芯片）。** 边缘设备大多是 ARM 架构，跟你的 x86 开发机不一样。用 `--target` 指定目标平台交叉编译（tract 纯 Rust，交叉编译尤其省心，见第 14 章）：

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
# 注意：ort 还需为 ARM 目标准备对应的 onnxruntime 库；tract 无此负担
```

**4. 容器化一句话。** 用 Docker 把可执行文件、动态库、模型、配置打进一个镜像，"镜像即环境"，彻底告别"我这能跑你那不行"。

**5. 模型与配置随包分发。** 模型文件（`.onnx`）、类别表、阈值/ROI 等配置（用 `serde` 读 JSON/YAML），都要跟程序一起交付，并做**版本管理**——模型换了、配置改了要能追溯。

开发机和边缘设备的环境差异，上线前对齐这张表：

| 维度 | 开发机 | 边缘设备 |
| --- | --- | --- |
| CPU 架构 | x86_64 | ARM（aarch64） |
| 算力 | 强，常有独显 | 弱，靠 CPU 或板载 NPU |
| 内存 | 充裕（16G+） | 紧张（1~4G，更要背压+丢帧） |
| 推理加速 | CUDA / CoreML | 厂商 NPU SDK 或 CPU |
| 构建方式 | 本机 `--release` | 交叉编译 `--target` |
| 动态库 | 自动下载 | 手动随包分发 |

## 29.8 上线检查清单（Checklist）

把 demo 推上生产前，逐项对勾——这张单子是本章所有要点的浓缩：

- [ ] **Release 编译**：`cargo build --release`，绝不拿 debug 版上线。
- [ ] **预热**：服务启动先空跑几次，避免首帧超慢。
- [ ] **Session 复用**：单推理线程只建一个；多推理线程则每线程各建一个并长期复用，别按帧重建。
- [ ] **有界通道防爆内存**：stage 间用 `bounded`，靠背压把内存钉在可控上限。
- [ ] **丢帧策略**：实时流用 `try_send` 满了丢旧帧；离线用 `send` 一帧不丢。
- [ ] **断流重连**：RTSP 断开能自动退避重连；单帧失败跳过不 `panic`。
- [ ] **日志与监控**：`tracing` 结构化日志 + 端到端 FPS/延迟统计，看得见状态。
- [ ] **资源上限**：内存、通道容量、线程数都设了明确上限，跑一夜不涨。
- [ ] **优雅退出**：收到信号能 drain 通道、处理完在途帧再停。
- [ ] **模型/配置版本管理**：模型、类别表、阈值/ROI 配置随包分发且可追溯。

## 29.9 承上启下：系统能跑起来了，然后呢？

到这里，第八部分画上句号：你手里的系统已经不是"能跑一次的 demo"，而是一个**并发、稳定、能部署上线**的视频分析服务——检测、跟踪、业务判定串成流水线，GPU 喂得饱，坏帧压不垮它，还能优雅退出、跨平台部署。

但"能稳定地跑"离"资深工程师能拍胸脯交付"还差一层。系统一旦上线，有三个问题会立刻迎面而来，而它们**在前八部分里我们一直绕着走**：

- **它到底准不准？** 你把框画得又快又稳，可这些框对不对、漏了多少、误报多少？没有量化的数字，"准"就只是一句感觉，换个新模型也说不清是变好还是偷偷变差。——这是**第 30 章：模型评估与度量**（mAP、PR 曲线、阈值决策）要回答的。
- **改一版会不会悄悄变差？** 你重构了后处理、换了个新模型、调了个参数——怎么保证没把坐标改偏、没让某个关键类偷偷掉点、没让性能劣化一倍？视觉的 bug 常常不崩不报，只是结果慢慢变糟。——这是**第 31 章：测试与性能剖析**（单测、golden 回归、基准、火焰图）要守住的。
- **上线之后怎么长期养着它？** 模型要在不停服的前提下升级、报警要一条不丢地送到上层平台、系统半夜变慢或模型悄悄变差要有人第一时间知道。——这是**第 32 章：生产化与运维**（热更新、监控闭环、可靠投递、数据漂移）要处理的，也正是第 27、28 章里那句"报警怎么可靠送出去，后面讲"欠下的账。

这三个问题——**准不准、会不会退化、怎么运维**——正是"会把模型跑通的人"和"资深机器视觉开发工程师"之间的最后一段距离。**第九部分**就专门补齐它们。翻过这一页，我们从"让系统跑起来"，走向"对系统负责到底"。

## 29.10 小结

- 串行流水线的总延迟是各环节相加、且 GPU 大量空闲；**流水线并发**把解码/预处理/推理/后处理拆成多线程用 channel 连接，吞吐取决于最慢环节而非相加，GPU 几乎不停歇。
- **背压**（有界通道满了阻塞上游）把内存钉在可控上限，防止 OOM；实时流处理不过来时用 `try_send` **主动丢旧帧**保新鲜——离线要全帧、实时可丢帧，是视频分析特有的抉择。
- **数据并行**（rayon 一批并行、fast_image_resize 的 SIMD 缩放）与流水线并发叠加；推理侧靠**预热、EP 选择、batch、量化和 Session 复用**榨性能；`ort 2.0.0-rc.13` 的同一个 Session 不能直接跨线程并发 `run`。
- 稳定性靠**可观测性**（tracing 结构化日志 + FPS/延迟统计）和**错误恢复**（坏帧跳过不 panic、RTSP 断流重连、信号触发优雅 drain 退出）。
- 部署第一铁律是 **`--release` 编译**；再管好动态库分发（`LD_LIBRARY_PATH`/rpath）、交叉编译到 ARM（`--target`）、容器化、模型与配置版本管理。上线前对着**检查清单**逐项打勾。
- 本章让系统**能稳定地跑**；但"准不准、会不会退化、怎么运维"三问还没答——**第九部分（第 30~32 章）**补齐模型评估、测试剖析与生产运维，才真正从"能跑"走到"资深可交付"。

## 29.11 练习

1. **改成三线程流水线**：把第 26 章的端到端检测程序，用 `crossbeam::channel::bounded` 拆成"解码预处理 / 推理 / 后处理"三个线程连接起来。跑同一个视频，对比改造前后的 FPS，验证吞吐提升。
2. **加实时丢帧**：给上题的解码环节换成 `try_send`，通道满时丢帧并累计 `dropped` 计数。把通道容量从 16 逐步调到 1，观察丢帧率和端到端延迟怎么变——体会"容量越小越新鲜、但丢得越多"。
3. **用 tracing 统计各 stage 耗时**：给每个 stage 包一个 `info_span!` 并打 `elapsed_ms`，跑一段视频后找出瓶颈在哪一环。若瓶颈是推理，比较两种方案：增大 batch；或开两个推理线程、每线程各持有一个 Session。记录吞吐、延迟和内存占用后再决定。
4. **给推理加 warmup 并对比首帧延迟**：在正式处理前用假输入空跑 3 次预热，记录"预热版"和"不预热版"的**第一帧**处理耗时，看看首帧慢了多少倍——这正是"上线要预热"的实证。
