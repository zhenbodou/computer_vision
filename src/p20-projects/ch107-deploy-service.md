# 第 107 章 项目：端到端把模型部署成 Web 服务

> **本章导读**
> - 学完你能：把一个 ONNX 模型用**纯 Rust**包成一个真能跑的 HTTP 推理服务——别人 `POST` 一张图，你返回 `{"label":"red","confidence":0.9859,...}` 这样的 JSON。这是第二十部分（实战项目）的**部署收官**。
> - 把两条已经各自跑通的链路**焊死成一条**：[第 94 章](../p19-deploy/ch94-onnx.md) 的「tract 加载 ONNX 推理」+ [第 96 章](../p19-deploy/ch96-web-service.md) 的「axum 服务骨架」→ 一条 `模型 → 纯 Rust Web 服务 → 真实请求 → JSON` 的最小可用闭环。
> - 学会一个关键的**可复现验证**手法：用 `tower` 的 `oneshot` 把请求**直接打进 `Router`**，不占端口、不联网，就能真发请求、断言响应——正文所有数字都是这样跑出来的。
> - 诚实认清**最小骨架 vs 生产系统**的差距：批处理、GPU、负载均衡、鉴权、监控、模型版本管理、限流……本章只打通「能用」的那一层。

## 场景引入：模型训练完了，怎么让全世界调用？

你在前面十九部分里造过、训过、跑过一堆模型。到了 [第 94 章](../p19-deploy/ch94-onnx.md)，你已经能用纯 Rust 的 tract 加载 `.onnx` 做推理；到了 [第 96 章](../p19-deploy/ch96-web-service.md)，你搭起了 axum 服务的骨架（路由、提取器、`Arc` 共享状态、`spawn_blocking`、背压）。但那两章是**分开**的：ch94 在命令行里 `println!` 出结果，ch96 的服务里放的还是「求平均灰度」的占位逻辑。

产品经理最后一公里的需求很朴素：「把你那个模型，做成一个接口，我上传图片、你回结果。」——这就是本章要交付的东西：**把真模型接进真服务**。我们要让服务在**启动时把模型加载一次**、放进共享状态，然后每来一个请求就走完整条推理流水线，返回结构化 JSON。全程纯 Rust，编译出一个单二进制，拷到哪台机器都能跑。

## 原理：一条推理请求在服务里的完整旅程

### 把 ch94 和 ch96 焊在一起

先看清楚一条 `POST /infer` 请求在服务里要走完的路：

```text
                              进程启动时：build .onnx → tract 加载 → Arc<模型>  (只做一次)
                                                    │  只读共享
HTTP 请求(图片) ─▶ 提取器 ─▶ base64/字节解码 ─▶ 预处理 ─▶ 张量 ─▶ tract.run ─▶ softmax/argmax ─▶ JSON
   (axum, ch96)              (客户端错→400)     (归一化)  ([1,3])   (ch94)      (取 top-1)      (serde)
                                       └──────── 这一整段 CPU 密集，丢进 spawn_blocking ────────┘
```

- **左半段是 ch96 的功劳**：`axum` 的提取器把请求体抽出来，`Arc<AppState>` 让模型只加载一次、所有请求只读共享，`spawn_blocking` 把 CPU 密集的解码+推理挪出 async 线程，`Semaphore` 做背压。
- **中间那段是 ch94 的功劳**：`tract_onnx::onnx().model_for_read(...).with_input_fact(...).into_optimized().into_runnable()` 把 `.onnx` 变成可反复调用的模型，`model.run(tvec!(...))` 出结果。
- 本章的活，就是**把两段的接缝焊好**：预处理产出的张量形状要和模型的输入契约对上，模型的输出 logits 要在 Rust 侧做 softmax/argmax 变成人能看懂的「类别 + 置信度」。

### 一条铁律：模型只算数，业务语义归服务

[第 94 章](../p19-deploy/ch94-onnx.md) 反复强调过：`.onnx` 里只有「怎么算」（算子图 + 权重），**没有**「算完怎么用」——类别索引到名字的映射表、图像该怎么预处理、要不要归一化，统统不在模型里。所以在我们的服务里：**预处理**（解码 → 三通道均值 → 归一化）和**后处理**（softmax → argmax → 查类别表 `["red","green","blue"]`）都得由服务自己实现。模型吐出的是一串 logits \\( z \\)，服务用 softmax 把它变成概率：

\\[ p_i = e^{z_i} / \sum_j e^{z_j} \\]

取概率最大的那一类作为预测，其概率就是「置信度」。

### 怎么验证「服务真能跑」而不占端口？

真起一个监听 `0.0.0.0:3000` 的服务、再开一个终端 `curl`，当然可以（ch96 就是这么冒烟的）；但这**没法自动化、没法进 CI、还可能撞端口**。更专业的做法是用 `tower` 的 `ServiceExt::oneshot`：`axum` 的 `Router` 本身就是一个 `tower::Service`，`oneshot` 能把一个构造好的 `Request` **直接喂进 `Router`**、拿回 `Response`，全程在同一个进程的内存里完成，**不绑端口、不走网卡**。这就是「可复现集成测试」的正解——本章正文里那些响应 JSON，全是这样真跑出来、逐字复制过来的。

## 动手：把 tract 模型包成 axum 服务

> ⚠️ **注意**：这棵依赖树（axum + tokio + tract）较重，**绝不能并入主 `code/Cargo.toml`**。本章代码放独立工程 `code/projects/ch107_deploy/`，用空 `[workspace]` 声明独立根（同 `service_demo/`、`ch94_tract_onnx/` 的隔离手法）。

`code/projects/ch107_deploy/Cargo.toml`（版本为本机 2026-08 `cargo add` 实测解析值，Rust 1.97.1）：

```toml
[workspace]                     # 声明独立 workspace 根，不被上层 code/ 吸入

[dependencies]
axum = "0.8"                    # 实测 0.8.9：路由/提取器/响应
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "signal"] }  # 1.53.1
tower = { version = "0.5", features = ["util"] }   # 0.5.3：提供 oneshot（可复现测试）
tract-onnx = "0.23.5"           # 纯 Rust ONNX 推理（ch94）
prost = "0.14"                  # 仅用于「在 Rust 里造 .onnx」（ch94 手法）
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] } # 只留需要的编解码器
serde = { version = "1", features = ["derive"] }   # 请求/响应结构
serde_json = "1"
base64 = "0.22"                 # 解码 base64 入图
anyhow = "1"                    # 内部管道错误
thiserror = "2"                 # HTTP 边界错误类型 → 状态码
```

**第一步：启动时把模型加载进共享状态。** 为了「免下载、即刻可跑、可复现」，我们沿用 ch94 的手法——在 Rust 里手搓一个极小的分类 `.onnx`（一个线性头 \\( z = x \cdot W \\)，\\( W = 6\,I \\) 放大差距让置信度更干脆），当作「按主色调分类 red/green/blue」的模型。**真实项目里 `.onnx` 由 PyTorch 导出**，加载那段 API 一模一样，只是模型更大、预处理更复杂。

```rust
type OnnxModel = Arc<TypedRunnableModel>;  // into_runnable() 返回 Arc<...>，run 签名是 fn run(self: &Arc<Self>)

/// 全局共享状态：进程启动时构造一次，Arc 让所有 handler 只读共享。
struct AppState {
    model: OnnxModel,       // 只 load 一次（tract 的 run 只读，天生适合共享）
    model_version: String,  // 随响应返回，便于定位「哪个版本算的」
    limiter: Semaphore,     // 背压闸门
}

fn load_model() -> anyhow::Result<OnnxModel> {
    let onnx = build_tiny_onnx_bytes();          // ch94 式：prost 造 .onnx 字节（不落盘）
    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(&onnx))?  // 从内存读，纯 Rust 解析 protobuf
        .with_input_fact(0, f32::fact([1, 3]).into())?      // 钉死输入 dtype+形状
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}
```

**第二步：请求/响应结构（serde）+ 完整推理流水线。**

```rust
#[derive(Deserialize)]
struct InferRequest { image_base64: String }

#[derive(Serialize)]
struct InferResponse {
    label: String, class_id: usize, confidence: f64,
    scores: Vec<f64>, model_version: String, elapsed_ms: u128,
}

/// 预处理 → 张量 → tract 推理 → softmax → argmax → 组装响应。
fn infer(model: &OnnxModel, bytes: &[u8], version: &str) -> Result<InferResponse, AppError> {
    let feat = preprocess(bytes)?;               // 解码失败 → BadRequest(400)
    let input = Tensor::from_shape(&[1, 3], &feat).context("构造输入张量失败")?;
    let out = model.run(tvec!(input.into())).context("tract 推理失败")?;
    let view = out[0].to_plain_array_view::<f32>().context("读取输出失败")?;  // 0.23 API
    let scores = softmax(view.as_slice().context("输出不连续")?);
    let (class_id, &top) = scores.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).context("空输出")?;
    Ok(InferResponse { label: CLASSES[class_id].into(), class_id,
        confidence: round4(top), scores: scores.iter().map(|&x| round4(x)).collect(),
        model_version: version.into(), elapsed_ms: /* ... */ 0 })
}
```

**第三步：错误分层——`thiserror` 定边界、`anyhow` 串管道。** 这正好落地 [第 112 章](../p21-mastery/ch112-engineering.md) 的分工：库/边界层用 `thiserror` 定义**明确的错误种类**并映射到状态码，内部管道用 `anyhow` 携带上下文往上冒泡。`#[from] anyhow::Error` 让任何内部错误 `?` 一下就变成 `500`：

```rust
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("请求有误：{0}")] BadRequest(String),   // 客户端问题 → 400
    #[error("服务繁忙")]       Overloaded,           // 背压满 → 503
    #[error(transparent)]      Internal(#[from] anyhow::Error), // 内部错误 → 500
}
impl IntoResponse for AppError { /* 匹配 → (StatusCode, Json({"error":..})) */ }
```

**第四步：handler + 路由。** `/infer` 收 base64 JSON、`/infer_raw` 收原始字节，两条都把 CPU 活丢进 `spawn_blocking`；另配 `GET /health` 健康检查和一个极简 HTML 上传页 `GET /`。

```rust
async fn infer_raw(State(state): State<Arc<AppState>>, body: Bytes)
    -> Result<Json<InferResponse>, AppError>
{
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?; // 背压
    if body.is_empty() { return Err(AppError::BadRequest("请求体为空".into())); }
    Ok(Json(run_blocking(&state, body.to_vec()).await?))  // 解码+推理丢进阻塞线程池
}

fn build_app() -> anyhow::Result<Router> {
    let state = Arc::new(AppState { model: load_model()?,
        model_version: "color-head-v1".into(), limiter: Semaphore::new(64) });
    Ok(Router::new()
        .route("/", get(index)).route("/health", get(health))
        .route("/infer", post(infer_json)).route("/infer_raw", post(infer_raw))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))   // 8 MiB，超限自动 413
        .with_state(state))
}
```

**第五步：可复现冒烟——`oneshot` 直打 `Router`。** 不占端口、不联网，合成几张纯色 PNG，发请求、断言、打印真实 请求→响应：

```rust
async fn call(app: &Router, method: &str, uri: &str, body: Body) -> (StatusCode, String) {
    use tower::ServiceExt;                        // 提供 oneshot
    let req = Request::builder().method(method).uri(uri)
        .header("content-type", "application/json").body(body).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();      // 请求直接进 Router
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}
```

### 跑起来（以下为本机 `cargo run --release` 的真实输出，逐字复制）

```bash
cd code/projects/ch107_deploy
cargo run --release          # 默认：进程内冒烟（oneshot），不占端口，打印真实响应后退出
# cargo run --release -- serve            # 真部署模式：绑定 0.0.0.0:3000（Ctrl-C 优雅关停）
# cargo test  --release                   # 同一套断言进 CI 防回归
```

```text
==== ch107 进程内冒烟（tower oneshot，不占端口/不联网）====
[PASS] GET /health   → 200 {"available_permits":64,"model_version":"color-head-v1","num_classes":3,"status":"ok"}
[PASS] POST /infer      红图·base64 JSON → 200 {"label":"red","class_id":0,"confidence":0.9859,"scores":[0.9859,0.007,0.007],"model_version":"color-head-v1","elapsed_ms":0}
[PASS] POST /infer      绿图·base64 JSON → 200 {"label":"green","class_id":1,"confidence":0.9747,"scores":[0.0112,0.9747,0.0141],...}
[PASS] POST /infer_raw  蓝图·原始字节    → 200 {"label":"blue","class_id":2,"confidence":0.9821,"scores":[0.0079,0.01,0.9821],...}
[PASS] POST /infer      非法 base64      → 400 {"error":"base64 解码失败：Invalid symbol 64, offset 0."}
[PASS] POST /infer      合法 base64 非图 → 400 {"error":"无法解码图像：The image format could not be determined"}
[PASS] POST /infer_raw  空 body          → 400 {"error":"请求体为空"}
==== 冒烟通过 7/7 ====
```

红图 `[230,20,20]` 三通道均值归一化后约 \\( [0.902, 0.078, 0.078] \\)，过 \\( z = 6x \\) 得 logits \\( [5.41, 0.47, 0.47] \\)，softmax 后 red 概率 \\( 0.9859 \\)——**模型判定「红色主导」，置信度 98.59%**，和真实输出逐字一致。整条 `模型 → 纯 Rust 服务 → 真实请求 → JSON` 的闭环，就这么在纯 Rust 里跑通了。

## 常见坑与提示

> 🕳️ **坑：tract「纯 Rust」也有个星号——但和 ort 完全不是一回事。** 用 `cargo tree -i cc` 一查会发现：`tract-linalg` 有个**构建期**依赖 `cc`。别慌——它是拿 `cc` 当**汇编器**来编译 tract 自带的**手写 SIMD 汇编内核**（`arm64/*.S`、`x86_64/avx512*/*.S`，仓库里 0 个 `.cpp`），为的是 CPU 上跑得快；**没有任何 `-sys` crate、不链接任何外部 C/C++ 运行库、不需要在目标机装 onnxruntime**。这和 [ch94](../p19-deploy/ch94-onnx.md) 里 `ort`「底层链接 C++ onnxruntime」是**两种性质**：tract 编译期需要一个能认目标架构指令的汇编器（交叉编译时汇编器不支持就回退到通用 Rust 内核），但**产物是自包含的单二进制**，部署时零外部依赖。

> 🕳️ **坑：模型每请求重复加载。** 新手最容易在 handler 里 `load_model()`——几十上百 MB 的模型每次重载，延迟爆炸、内存抖动。**启动时加载一次**放进 `Arc<AppState>`（`tract` 的 `run(self: &Arc<Self>, ...)` 只读，天生适合共享），`.with_state()` 注入，所有请求共享同一份。

> ⚠️ **注意：预处理必须和训练端逐项一致（ch94 铁律，线上翻车头号根因）。** 尺寸、RGB/BGR 通道序、`/255`、逐通道 mean/std、NCHW 布局——任何一项和训练时对不上，模型看到的就是「另一个分布」的数据，Python 侧测得好好的，换到 Rust 结果全错。本章模型只吃 3 个均值特征所以预处理极简，真实模型这里是最该写单元测试对齐的地方。

> 💡 **提示：用 `oneshot` 做集成测试，别真起端口。** `Router` 就是 `tower::Service`，`app.clone().oneshot(request)` 把请求直接打进内存里的路由栈，不绑端口、不联网、可进 CI、可断言状态码和响应体。本章 `cargo test` 里就是同一套 7 条断言，改坏任何一处都会红。

> ⚠️ **注意：`spawn_blocking` 里的活取消不了。** 超时中间件、客户端断开能取消 async future，但**已经进入 `spawn_blocking` 的 CPU 推理无法被中断**，会跑完为止（ch96 同款坑）。想真正响应取消，得在推理循环里埋检查点。

## 本章小结

- 本章是实战部分的**部署收官**：把 [ch94](../p19-deploy/ch94-onnx.md) 的 tract 推理和 [ch96](../p19-deploy/ch96-web-service.md) 的 axum 骨架**焊成一条**可运行的推理服务，打通 `模型 → 纯 Rust Web 服务 → 真实请求 → JSON` 的最小可用闭环。
- 服务的三根生死线（复用 ch96）：模型用 `Arc<AppState>` **只加载一次**共享、CPU 推理丢进 **`spawn_blocking`** 不卡运行时、**`Semaphore` 背压**过载快速失败。
- **模型只算数，业务语义归服务**：类别表、预处理、softmax/argmax 后处理都在 Rust 侧实现（ch94 契约铁律）；错误用 `thiserror`（边界）+ `anyhow`（管道）分层映射到 400/500/503。
- **可复现验证**用 `tower::oneshot` 把请求直打 `Router`——不占端口、不联网，本章 7/7 冒烟与所有响应数字均为本机真实输出、逐字一致。
- tract 是**自包含单二进制、零外部 C/C++ 运行库**（其 `cc` 只是构建期汇编 SIMD 内核，与 ort 链接 onnxruntime 性质不同）——这正是纯 Rust 部署到边缘的底气（[ch111](../p21-mastery/ch111-embedded.md)）。

## 动手练习

1. **换真模型**：把练手的手搓 `.onnx` 换成 PyTorch 导出的真分类模型（如 MNIST/ResNet），改 `preprocess`（resize 到训练尺寸 + 逐通道归一化 + NCHW）和 `with_input_fact` 的形状，替换 `CLASSES` 类别表——服务其余部分一行不用改。
2. **补背压压测**：给 `/infer` 加压（`hey`/`wrk`），把 `Semaphore::new(64)` 调大调小，观察 503 比例与 P95 延迟的拐点，验证背压闸门的作用。
3. **加模型版本管理**：让 `AppState` 同时持有两个版本的模型，`/infer?version=v2` 按查询参数选路；响应里的 `model_version` 已经为此埋好了伏笔。
4. **补一个 `/metrics`**：统计各状态码计数、推理耗时的 P50/P95，用 JSON 暴露（呼应 ch96 的可观测性）；思考它和真正的 Prometheus 指标差在哪。

> 💡 **诚实边界 / 延伸**：本章交付的是「能用」的最小骨架。真正的生产推理服务还需要：**批处理**（攒够一批一起推理提吞吐）、**GPU/加速**（纯 Rust CPU 跑大模型吃力，见 [ch95 量化](../p19-deploy/ch95-quantization.md) 与 [ch109 GPU](../p21-mastery/ch109-gpu.md)）、**负载均衡与多副本**、**鉴权与限流**、**监控告警**、**模型版本管理与灰度发布**、**优雅关停**（本章 `serve` 模式已含 Ctrl-C 优雅关停）。把模型塞进资源受限的边缘盒子见 [第 111 章](../p21-mastery/ch111-embedded.md)，性能压榨见 [第 108 章](../p21-mastery/ch108-performance.md)。至此，你已经能把一个纯 Rust 视觉模型**从训练一路交付到线上接口**——这就是全书实战部分的终点。
