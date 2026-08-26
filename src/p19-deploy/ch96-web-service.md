# 第 96 章 Axum 视觉服务：并发、背压与 SLO

> **本章导读**
> - 学完你能：把一个纯 Rust 视觉算法包成 HTTP 服务——别人 `POST` 一张图，你返回 JSON 结果（尺寸、平均灰度）或处理后的 PNG。
> - 掌握 axum 三件套：**路由（Router）、处理器（handler）、提取器（extractor）**，并用 `Arc<State>` 让模型**只加载一次、全局共享**。
> - 学会把 CPU 密集处理丢进 `spawn_blocking`，**别卡死 async 运行时**；用信号量做**背压**，过载时快速失败而不是撑爆内存。
> - 配齐生产要素：请求体上限、超时、健康检查、结构化日志、优雅关停——这正是 [第 2 章](../p01-start/ch02-why-rust.md) 所说 Rust“单二进制、高并发、部署省心”的主场。

## 场景引入：训练好的能力，怎么给别人用？

在 [第 94 章](ch94-onnx.md) 你已经能用 tract 加载 ONNX 模型做纯 Rust 推理，在前面各章也写了一堆图像处理算法。现在产品经理来了：“前端 App 和另一个 Java 系统都想用你这个能力，能不能给个接口，我上传一张图，你返回结果？”

这就是**服务化（serving）**：把算法从“本地跑的命令行程序”变成“网络上随叫随到的 HTTP 服务”。约定好协议（比如 `POST /process` 传图片字节、返回 JSON），谁都能调用，不管对方用什么语言。

为什么用 Rust + axum 来做这件事？回想 [第 2 章](../p01-start/ch02-why-rust.md)：视觉服务往往要**扛并发**（同时来几十路请求）、**省内存**（边缘盒子内存紧张）、**好部署**（一个二进制文件拷过去就跑）。Rust 没有 GIL，`tokio` 异步运行时能用少量线程扛住大量连接；编译出的单文件常常只有几十 MB——服务化恰恰是 Rust 的主场。`axum` 则是当前 Rust 生态最主流的 Web 框架，构建在 `tokio` + `tower` 之上，路由简洁、类型安全。

## 原理：axum 是怎么工作的

### 三件套：路由 → 提取器 → 处理器

axum 的心智模型非常直白，一个请求的旅程是这样的：

```text
                  ┌──────── 中间件层（tower Layer）────────┐
HTTP 请求  ──▶   │  体积上限(413) → 超时(408) → 访问日志   │  ──▶  路由匹配
                  └────────────────────────────────────────┘         │
                                                                      ▼
   HTTP 响应  ◀──  处理器 handler  ◀──  提取器 extractor（从请求里“抽”出需要的东西）
```

- **路由 `Router`**：把 URL 路径映射到处理器。`Router::new().route("/health", get(health)).route("/process", post(process))`。
- **处理器 handler**：一个 `async fn`，返回任何实现了 `IntoResponse` 的类型（如 `Json<T>`、`StatusCode`、`(StatusCode, Json<T>)`）。
- **提取器 extractor**：handler 的**参数**就是提取器，axum 按类型自动从请求里“抽料”。常用的有：`State<T>` 抽共享状态、`Bytes` 抽原始请求体字节、`Json<T>` 抽并反序列化 JSON body、`Path`/`Query` 抽路径/查询参数。

> 📖 **术语**：**提取器（extractor）**。axum 的招牌设计——你在函数签名里写下想要什么类型，框架就负责从 HTTP 请求里解析出来喂给你，解析失败自动返回合适的 4xx。声明式，不用手抠 header 和 body。

我们的视觉服务收的是**原始图片字节**（前端直接把文件二进制放进 body，最省事），所以用 `body: Bytes` 这个提取器即可。若要走网页表单的多文件上传，则用 `multipart`（需给 axum 打开 `multipart` feature），复杂一些，本章从简。

### 关键一：模型只加载一次，用 `Arc<State>` 共享

新手最容易犯的错，是在 handler 里每来一个请求就 `load_model()` 一次——模型动辄几十上百 MB，每次重新加载会让延迟爆炸、内存抖动。**正确做法：进程启动时加载一次，放进共享状态**。

axum 用 `.with_state(state)` 注入状态，handler 里用 `State(state): State<Arc<AppState>>` 取出。`Arc`（原子引用计数）让所有请求**只读共享**同一份数据，没有拷贝。tract 的 `model.run(&self, ...)` 恰好是只读的，天生适合塞进 `Arc` 共享：

```rust
struct AppState {
    model_version: String,        // 随响应返回，便于定位“哪个版本算的”（呼应 ch94）
    model: TypedRunnableModel<..>,// ch94 的 tract 模型，启动时加载一次
    limiter: Semaphore,           // 背压闸门，见下
}
```

### 关键二：别卡死运行时——`spawn_blocking`

`tokio` 是**异步**运行时，靠“少量线程 + 任务永不长时间阻塞”来扛高并发。可图像解码、逐像素遍历、模型前向推理，全是**CPU 密集的同步阻塞**操作。如果直接在 `async` handler 里跑，就会**霸占**一个 worker 线程几十毫秒甚至几秒，其他请求全被堵在门外——整个服务“假死”。

解法：把这段重活丢进 `tokio::task::spawn_blocking`，它有一个专门的阻塞线程池，不会拖累负责收发网络的 async 线程。

```rust
let stats = tokio::task::spawn_blocking(move || compute_stats(&body))
    .await
    .map_err(|_| AppError::Internal)??;   // 外层 JoinError→500，内层解码错误→400
```

### 关键三：背压——过载时快速失败，而不是 OOM

呼应 [第 92 章](../p18-events/ch92-multistream.md) 的多路并发。如果“来多少收多少”，突发流量一来，成千上万张图同时在内存里解码，进程瞬间 OOM 被杀。**背压（backpressure）** 就是给系统装一个闸门：同时处理的请求数有上界，超了就明确拒绝。

用一个**有界信号量** `Semaphore`（许可数 = 允许的最大并发处理量）即可：进 handler 先 `try_acquire()` 抢许可，抢不到立刻返回 `503 Service Unavailable`（或 `429 Too Many Requests`），绝不排队堆内存。用排队论的 Little 定律看，系统内在处理的请求数 \( L = \lambda W \)（到达率 × 单请求处理时长）；要让 \( L \) 有上界、内存可控，就必须给并发设闸门。

## 动手写代码：一个最小可跑的 axum 视觉服务

> ⚠️ **注意**：axum + tokio 是一棵较重的异步依赖树，**绝不能并入主 `code/Cargo.toml`**（会拖慢、污染主工程）。本章代码放在**独立工程** `code/service_demo/`，用空 `[workspace]` 声明自己是独立根，与主工程隔离（同 `wasm_demo/` 的手法）。

`code/service_demo/Cargo.toml`（版本为本机 2026-08 实测解析值，Rust 1.97.1）：

```toml
[workspace]                       # 声明独立 workspace 根，不被上层 code/ 吸入

[package]
name = "service_demo"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"                                              # 实测 0.8.9
tokio = { version = "1", features = [                     # 实测 1.53.1
    "rt-multi-thread", "macros", "net", "signal", "time",
] }
tower-http = { version = "0.7", features = ["trace", "timeout"] }  # 实测 0.7.0
serde = { version = "1", features = ["derive"] }          # 实测 1.0.229
serde_json = "1"                                          # 实测 1.0.151
image = "0.25"                                            # 实测 0.25.10（与主工程同版本）
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

核心 handler 与状态（完整源码见 `code/service_demo/src/main.rs`）：

```rust
/// 全局共享状态：启动时构造一次，Arc 让所有 handler 只读共享。
struct AppState {
    model_version: String,   // 随响应返回，便于排查
    limiter: Semaphore,      // 背压闸门：最多同时处理这么多请求
}

/// POST /process —— 收图片字节，返回宽高 + 平均灰度。
async fn process(
    State(state): State<Arc<AppState>>,
    body: Bytes,                         // 提取器：原始请求体字节
) -> Result<Json<ProcessResult>, AppError> {
    // 背压：抢不到许可就快速失败（503），不把活儿堆进内存
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?;
    if body.is_empty() {
        return Err(AppError::BadImage("请求体为空".into()));
    }
    let version = state.model_version.clone();
    // CPU 密集：解码 + 遍历像素，丢进阻塞线程池，别卡住 async 运行时
    let (width, height, avg_gray) =
        tokio::task::spawn_blocking(move || compute_stats(&body))
            .await
            .map_err(|_| AppError::Internal)??;
    Ok(Json(ProcessResult { width, height, avg_gray, model_version: version }))
}

/// 纯 Rust 视觉处理：解码 → 宽高 + 平均灰度（同步函数，交给 spawn_blocking 跑）
fn compute_stats(bytes: &[u8]) -> Result<(u32, u32, f64), AppError> {
    let img = image::load_from_memory(bytes).map_err(|e| AppError::BadImage(e.to_string()))?;
    let (w, h) = img.dimensions();          // 需要 GenericImageView trait
    let gray = img.to_luma8();
    let sum: u64 = gray.pixels().map(|p| p[0] as u64).sum();
    let avg = if w * h > 0 { sum as f64 / (w as u64 * h as u64) as f64 } else { 0.0 };
    Ok((w, h, avg))
}
```

统一的错误类型让“状态码”与“业务错误”对上号（错误处理的分层思想见 [第 112 章](../p21-mastery/ch112-engineering.md) 的 `thiserror`/`anyhow`）：

```rust
enum AppError { BadImage(String), Overloaded, Internal }

impl IntoResponse for AppError {          // 让 AppError 能直接当 handler 的错误返回
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::BadImage(m) => (StatusCode::BAD_REQUEST, m),           // 400
            AppError::Overloaded  => (StatusCode::SERVICE_UNAVAILABLE, "服务繁忙".into()), // 503
            AppError::Internal    => (StatusCode::INTERNAL_SERVER_ERROR, "内部错误".into()),// 500
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}
```

最后把路由、中间件、状态组装起来并启动（含体积上限、超时、日志、优雅关停）：

```rust
let state = Arc::new(AppState { model_version: "demo-v1".into(), limiter: Semaphore::new(64) });
let app = Router::new()
    .route("/health", get(health))
    .route("/process", post(process))
    .route("/grayscale", post(grayscale))            // 收图 → 返回灰度 PNG 字节
    .layer(DefaultBodyLimit::max(8 * 1024 * 1024))   // 请求体上限 8 MiB，超限自动 413
    .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(10)))
    .layer(TraceLayer::new_for_http())               // 结构化访问日志
    .with_state(state);
let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
axum::serve(listener, app)
    .with_graceful_shutdown(shutdown_signal())       // 收到 Ctrl-C 停止收新请求，等在途完成
    .await.unwrap();
```

### 跑起来 + curl 冒烟（以下为本机真实响应，非示意）

```bash
cd code/service_demo
cargo run                       # 监听 0.0.0.0:3000

# 另开一个终端：
curl -s http://127.0.0.1:3000/health
curl -s -X POST --data-binary @../out_ch74_frame.png http://127.0.0.1:3000/process
curl -s -X POST --data-binary @../out_ch74_frame.png http://127.0.0.1:3000/grayscale -o gray.png
```

实际返回（一张 160×120 的测试图）：

```text
GET  /health   → {"available_permits":64,"model_version":"demo-v1","status":"ok"}
POST /process  → {"width":160,"height":120,"avg_gray":70.69,"model_version":"demo-v1","elapsed_ms":5}
POST /grayscale→ gray.png（file 验证：PNG image data, 160 x 120, 8-bit grayscale）
```

错误路径也实测符合预期：空 body → `400`；非图片字节 → `400 {"error":"The image format could not be determined"}`；9 MiB 的超大 body → `413`。

## 常见坑与提示

> 🕳️ **坑：请求体大小限制。** axum 默认 body 上限约 **2 MB**，一张高清大图直接被拦。要用 `DefaultBodyLimit::max(n)` 调大，但**绝不能设成无限**——否则攻击者传个“解码炸弹”（几 KB 压缩、解开几个 GB）就能打爆内存。本例设 8 MiB，实测 9 MiB 请求返回 `413`。

> 🕳️ **坑：阻塞操作卡死 async 运行时。** 解码、逐像素、模型推理都是 CPU 密集同步操作，直接在 `async fn` 里跑会霸占 worker 线程、让服务“假死”。**一律用 `spawn_blocking`** 丢到阻塞线程池。这是异步服务里最隐蔽也最致命的坑。

> 🕳️ **坑：模型每请求重复加载。** 别在 handler 里 `load_model()`。启动时加载一次，放进 `Arc<AppState>`，`.with_state()` 注入，所有请求共享。

> ⚠️ **注意：超时/取消未必能停下已派发的活。** `TimeoutLayer` 和客户端断开能取消 async future，但**已经进入 `spawn_blocking` 的 CPU 任务无法被中断**，会一直跑完。要真正响应取消，得在算法内部埋检查点（定期查 flag）。所以超时更多是“别让客户端干等”，而非“立刻省下算力”。

> 💡 **提示：健康检查要分层。** 生产上 `/health` 至少区分三层：**存活**（进程没死）、**就绪**（模型/依赖加载完）、**真实小样本推理**（喂一张固定图跑通）。模型加载失败的实例必须让健康检查失败，好让负载均衡把它摘掉，别继续接流量。

> 💡 **提示：接口版本化 + 可观测。** 响应里带上 `model_version`、请求 id，日志记录 QPS、状态码分布、P50/P95/P99 延迟、队列/许可占用。定 **SLO**（如“P99 < 200ms、可用性 99.9%”）时，压测要**同时报告错误率**——靠疯狂丢请求换来的漂亮延迟是自欺欺人。

## 本章小结

- **服务化**就是把视觉算法包成 HTTP 接口；Rust + axum 的无 GIL 高并发、低内存、单二进制部署，让它成为服务化的主场（呼应 [ch02](../p01-start/ch02-why-rust.md)）。
- axum 三件套：**Router 路由、handler 处理器、extractor 提取器**；收原始图用 `Bytes`，返回用 `Json<T>` 或直接给 PNG 字节。
- 三条生死线：模型用 `Arc<State>` **只加载一次**共享；CPU 密集处理用 **`spawn_blocking`** 不卡运行时；用 **`Semaphore` 背压**过载快速失败（503）而非 OOM。
- 生产要素缺一不可：**体积上限（413）、超时、分层健康检查、结构化日志、优雅关停**，配上统一的错误→状态码映射。
- 配套 `code/service_demo/` 已 `cargo build` 通过，并实测 `curl` 冒烟：`/health`、`/process`、`/grayscale` 及 400/413 错误路径全部符合预期。

## 动手练习

1. **接一个真模型（衔接 ch94）**：把 [第 94 章](ch94-onnx.md) 的 tract 模型加载进 `AppState`，新增 `POST /classify`，在 `spawn_blocking` 里跑推理，返回 top-1 类别与置信度 JSON。注意模型只在 `main` 里加载一次。
2. **压测找饱和点**：用 `wrk` 或 `hey` 对 `/process` 加压，从低到高调并发，观察吞吐、P95 延迟与 503 比例的拐点；再把 `Semaphore` 许可数调大/调小，看背压闸门如何影响“过载表现”。
3. **返回可视化结果**：新增 `POST /edges`，用 [第 15 章](../p03-classic/ch15-edges.md) 的 Canny 边缘检测处理上传图，返回边缘图 PNG。
4. **加鉴权与限流**：写一个 tower 中间件校验 `Authorization` 头（无效返回 401），并按客户端 IP 做简单的每秒请求数限流，思考它和 `Semaphore` 背压的分工。

> 💡 **下一章预告 / 延伸**：本章是“单机最小服务”。真正端到端把模型部署成生产 Web 服务（配置、容器、指标、灰度发布）见 [第 107 章](../p20-projects/ch107-deploy-service.md)；把服务塞进资源受限的边缘盒子见 [第 111 章](../p21-mastery/ch111-embedded.md)；性能压榨（并行、SIMD）见 [第 108 章](../p21-mastery/ch108-performance.md)。
