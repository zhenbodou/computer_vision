// 第 107 章配套：端到端把模型部署成纯 Rust Web 推理服务（axum + tract，零 C/C++ 依赖）
//
// 本例把「ch94 的 tract ONNX 加载」和「ch96 的 axum 服务骨架」合流，跑通
//   模型  →  纯 Rust Web 服务  →  真实 HTTP 请求  →  JSON 结果
// 这条最小可用的部署闭环。全程纯 Rust、可复现、无需下载任何模型、无需联网。
//
// 端点：
//   GET  /            极简 HTML 上传/说明页（字符串）
//   GET  /health      健康检查：进程存活 + 模型就绪 + 类别数 + 剩余并发许可
//   POST /infer       收 JSON {"image_base64": "..."} → 预处理 → tract 推理 → JSON 结果
//   POST /infer_raw   收「原始图片字节」（body）      → 预处理 → tract 推理 → JSON 结果
//
// 演示的部署工程点：
//   1. 启动时把模型 load 一次放进 Arc<AppState>，所有请求只读共享（tract model.run(&self) 只读）。
//   2. CPU 密集的解码 + 推理放进 spawn_blocking，绝不阻塞 async 运行时（同 ch96）。
//   3. thiserror 定义 HTTP 边界错误类型 + anyhow 串起内部管道；两者一起映射到状态码。
//   4. serde 定义请求/响应结构；base64 与原始字节两种入图方式。
//   5. Semaphore 背压：过载快速失败 503，而不是把请求堆进内存 OOM。
//
// 运行：
//   cargo run --release            # 默认：进程内冒烟（tower oneshot 把请求直接打进 Router），
//                                  #        不占端口/不联网，打印真实 请求→响应，断言全过后退出
//   cargo run --release -- serve   # 真部署模式：绑定 0.0.0.0:3000 起服务（Ctrl-C 优雅关停）

use std::sync::Arc;
use std::time::Instant;

use anyhow::Context; // 给普通 Result 挂上下文，向上冒泡成 anyhow::Error
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;

use prost::Message; // 把 protobuf 结构体编码成字节（仅用于「在 Rust 里造 .onnx」）
use tract_onnx::pb; // tract 内置的 ONNX protobuf 类型（ModelProto 等）
use tract_onnx::prelude::*;

// tract 把 .onnx 优化成的「可执行模型」。into_runnable() 返回 Arc<...>（tract 内部即用 Arc
// 共享计划），run() 的签名是 fn run(self: &Arc<Self>, ...)，天生只读、Send + Sync，
// 因此可安心塞进我们自己的 Arc<AppState> 跨线程共享（每个请求在 spawn_blocking 里 model.run）。
type OnnxModel = Arc<TypedRunnableModel>;

/// 类别表——注意这是「业务语义」，不在 .onnx 里，必须由服务另行打包（ch94 的铁律）。
const CLASSES: [&str; 3] = ["red", "green", "blue"];

// ===================== Part A：在 Rust 里造一个极小的分类 .onnx =====================
// 与 ch94 同法：手搓 ModelProto → prost 序列化成 .onnx 字节。真实项目里 .onnx 由
// PyTorch 的 torch.onnx.export 导出，这里手搓只是为了「免下载、即刻可跑、完全可复现」。
//
// 模型本身是一个「按主色调分类」的线性分类器：
//   输入  x = [meanR, meanG, meanB]（图像三通道均值，已归一化到 [0,1]），形状 [1,3]
//   logits = x · W          W = 6·I（对角 6.0，放大差距让 softmax 置信度更干脆）
//   预测类别 = argmax(logits) ∈ {red, green, blue}
// 换成真实 ResNet/YOLO，只是模型更大、预处理更复杂，这套「加载→推理」的 API 一模一样。

fn f32_io(name: &str, shape: &[i64]) -> pb::ValueInfoProto {
    let dim = shape
        .iter()
        .map(|&d| pb::tensor_shape_proto::Dimension {
            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: pb::tensor_proto::DataType::Float as i32,
                shape: Some(pb::TensorShapeProto { dim }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn f32_init(name: &str, shape: &[i64], data: Vec<f32>) -> pb::TensorProto {
    pb::TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: data,
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], output: &str) -> pb::NodeProto {
    pb::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.to_string()],
        op_type: op.to_string(),
        ..Default::default()
    }
}

/// 造出 .onnx 的字节（不落盘，直接给 tract 从内存读）。
fn build_tiny_onnx_bytes() -> Vec<u8> {
    // W [3,3] = 6·I：对角线放大，让「主色调」在 softmax 后置信度更干脆。
    let s = 6.0f32;
    #[rustfmt::skip]
    let w = vec![
        s,   0.0, 0.0,
        0.0, s,   0.0,
        0.0, 0.0, s,
    ];
    let graph = pb::GraphProto {
        name: "color_head".to_string(),
        node: vec![node("MatMul", &["input", "W"], "logits")],
        initializer: vec![f32_init("W", &[3, 3], w)],
        input: vec![f32_io("input", &[1, 3])],
        output: vec![f32_io("logits", &[1, 3])],
        ..Default::default()
    };
    let model_proto = pb::ModelProto {
        ir_version: 7,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "rust_cv_book".to_string(),
        graph: Some(graph),
        ..Default::default()
    };
    let mut buf = Vec::new();
    model_proto.encode(&mut buf).expect("protobuf 编码不应失败");
    buf
}

/// 启动时构造模型：造 .onnx 字节 → tract 从内存加载 → 钉输入形状 → 优化 → 可执行。
fn load_model() -> anyhow::Result<OnnxModel> {
    let onnx = build_tiny_onnx_bytes();
    let model = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(&onnx))
        .context("tract 解析 .onnx 失败")?
        .with_input_fact(0, f32::fact([1, 3]).into())
        .context("设定输入 fact 失败")?
        .into_optimized()
        .context("图优化失败")?
        .into_runnable()
        .context("生成可执行模型失败")?;
    Ok(model)
}

// ===================== 共享状态 / 请求 / 响应 结构 =====================

/// 全局共享状态：进程启动时构造一次，Arc 让所有 handler 只读共享。
struct AppState {
    model: OnnxModel,     // ch94 的 tract 模型，只 load 一次
    model_version: String, // 随响应返回，便于定位「哪个版本算的」
    limiter: Semaphore,   // 背压闸门：最多同时处理这么多请求
}

/// POST /infer 的请求体：一张 base64 编码的图片。
#[derive(Deserialize)]
struct InferRequest {
    image_base64: String,
}

/// 推理响应（两个入图端点共用），自动序列化成 JSON。
#[derive(Serialize)]
struct InferResponse {
    label: String,       // top-1 类别名，如 "red"
    class_id: usize,     // top-1 类别下标
    confidence: f64,     // top-1 的 softmax 概率
    scores: Vec<f64>,    // 各类别 softmax 概率（按 CLASSES 顺序）
    model_version: String,
    elapsed_ms: u128,    // 服务端处理耗时
}

// ===================== 统一错误：thiserror 定边界 + anyhow 串管道 =====================

/// HTTP 边界错误类型。thiserror 负责「错误怎么显示」，IntoResponse 负责「映射到哪个状态码」。
#[derive(Debug, thiserror::Error)]
enum AppError {
    /// 客户端问题：图片解码失败 / base64 非法 / body 为空 → 400。
    #[error("请求有误：{0}")]
    BadRequest(String),
    /// 背压闸门已满 → 服务繁忙 503。
    #[error("服务繁忙")]
    Overloaded,
    /// 内部管道出错（张量构造、推理、线程 join…）→ 500。
    /// #[from] 让任何 anyhow::Error 都能用 `?` 直接冒泡成本变体。
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.clone()),
            AppError::Overloaded => (StatusCode::SERVICE_UNAVAILABLE, "服务繁忙，请稍后重试".into()),
            AppError::Internal(e) => {
                // 内部错误细节记日志，别泄露给客户端
                eprintln!("[500] 内部错误：{e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "内部错误".into())
            }
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}

// ===================== 预处理 + 推理（同步，交给 spawn_blocking 跑）=====================

/// 纯 Rust 预处理：解码 → RGB → 三通道均值 / 255 → 特征向量 [meanR, meanG, meanB]。
/// 真实分类器这里会 resize 到固定尺寸 + 逐通道 mean/std 标准化 + NCHW；本例的模型只吃
/// 3 个均值特征，故预处理也从简——但「预处理必须和训练端逐项一致」这条铁律不变（ch94）。
fn preprocess(bytes: &[u8]) -> Result<[f32; 3], AppError> {
    let img =
        image::load_from_memory(bytes).map_err(|e| AppError::BadRequest(format!("无法解码图像：{e}")))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8();
    let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
    for p in rgb.pixels() {
        sr += p[0] as u64;
        sg += p[1] as u64;
        sb += p[2] as u64;
    }
    let n = (w as u64 * h as u64).max(1) as f32;
    Ok([
        sr as f32 / n / 255.0,
        sg as f32 / n / 255.0,
        sb as f32 / n / 255.0,
    ])
}

/// 数值稳定的 softmax（减最大值防溢出）。
fn softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&x| ((x - m) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// 保留 4 位小数，让打印/JSON 里的数字稳定、可复现。
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// 完整推理：预处理 → 构造张量 → tract 推理 → softmax → argmax → 组装响应。
/// 解码/base64 错误 → BadRequest(400)；张量/推理错误经 anyhow → Internal(500)。
fn infer(model: &OnnxModel, bytes: &[u8], version: &str) -> Result<InferResponse, AppError> {
    let started = Instant::now();
    let feat = preprocess(bytes)?; // 400 路径
    // 下面这些是「服务端内部」错误：出错就是 500。用 anyhow 挂上下文，? 冒泡成 AppError::Internal。
    let input = Tensor::from_shape(&[1, 3], &feat).context("构造输入张量失败")?;
    let out = model.run(tvec!(input.into())).context("tract 推理失败")?;
    let view = out[0]
        .to_plain_array_view::<f32>()
        .context("读取输出张量失败")?; // tract 0.23：用 to_plain_array_view，不是旧的 to_array_view
    let logits = view.as_slice().context("输出不连续")?;

    let scores = softmax(logits);
    // argmax
    let (class_id, &top) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .context("空输出")?;

    Ok(InferResponse {
        label: CLASSES[class_id].to_string(),
        class_id,
        confidence: round4(top),
        scores: scores.iter().map(|&x| round4(x)).collect(),
        model_version: version.to_string(),
        elapsed_ms: started.elapsed().as_millis(),
    })
}

// ===================== handlers =====================

/// GET / —— 极简说明/上传页（字符串）。生产上你会放真正的前端，这里只做占位演示。
async fn index() -> Html<&'static str> {
    Html(
        r#"<!doctype html><meta charset="utf-8"><title>ch107 推理服务</title>
<h1>纯 Rust 推理服务（ch107）</h1>
<p>按主色调分类：red / green / blue。</p>
<form id="f"><input type="file" id="img" accept="image/*"><button>识别</button></form>
<pre id="out"></pre>
<script>
f.onsubmit = async (e) => {
  e.preventDefault();
  const b = await img.files[0].arrayBuffer();
  const r = await fetch('/infer_raw', {method:'POST', body:b});
  out.textContent = await r.text();
};
</script>"#,
    )
}

/// GET /health —— 健康检查：进程存活 + 模型就绪 + 类别数 + 剩余并发许可。
async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model_version": state.model_version,
        "num_classes": CLASSES.len(),
        "available_permits": state.limiter.available_permits(),
    }))
}

/// POST /infer —— 收 JSON {"image_base64": "..."}。
async fn infer_json(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, AppError> {
    // 背压：拿不到许可就快速 503。permit 借用 state.limiter，一直持有到本函数结束——
    // 也就是覆盖整个 CPU 处理阶段，真正把并发压在闸门内。
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?;
    // base64 解码属于「客户端给的数据不对」→ 400
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(req.image_base64.trim())
        .map_err(|e| AppError::BadRequest(format!("base64 解码失败：{e}")))?;
    Ok(Json(run_blocking(&state, bytes).await?))
}

/// POST /infer_raw —— 收原始图片字节（body）。Bytes 提取器要放在最后一个参数。
async fn infer_raw(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<InferResponse>, AppError> {
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?;
    if body.is_empty() {
        return Err(AppError::BadRequest("请求体为空".into()));
    }
    Ok(Json(run_blocking(&state, body.to_vec()).await?))
}

/// 把 CPU 密集的「解码 + 推理」丢进阻塞线程池，别卡住 async 运行时（ch96 铁律）。
/// 借用 state 保住调用方的 permit，同时 clone 出一个 Arc 供 'static 的阻塞闭包持有。
async fn run_blocking(state: &Arc<AppState>, bytes: Vec<u8>) -> Result<InferResponse, AppError> {
    let st = Arc::clone(state);
    let version = st.model_version.clone();
    let resp = tokio::task::spawn_blocking(move || infer(&st.model, &bytes, &version))
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("spawn_blocking join 失败：{e}")))??;
    //   ^ 外层 ?：JoinError → 500      ^ 内层 ?：infer 的 AppError（400 或 500）原样冒泡
    Ok(resp)
}

/// 组装路由 + 中间件 + 共享状态。冒烟测试与真实起服务都复用它。
fn build_app() -> anyhow::Result<Router> {
    let state = Arc::new(AppState {
        model: load_model()?,
        model_version: "color-head-v1".to_string(),
        limiter: Semaphore::new(64), // 背压：最多 64 路并发进入 CPU 阶段
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/infer", post(infer_json))
        .route("/infer_raw", post(infer_raw))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024)) // 请求体上限 8 MiB，超限自动 413
        .with_state(state);
    Ok(app)
}

// ===================== 进程内冒烟：tower oneshot（不占端口 / 不联网）=====================

/// 合成一张纯色 RGB 图并编码成 PNG 字节（自包含测试输入）。
fn solid_png(rgb: [u8; 3], side: u32) -> Vec<u8> {
    use image::{ImageFormat, RgbImage};
    let img = RgbImage::from_pixel(side, side, image::Rgb(rgb));
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut buf), ImageFormat::Png)
        .expect("PNG 编码不应失败");
    buf
}

/// 把一个请求用 tower 的 oneshot 直接打进 Router，取回 (状态码, body 文本)。
async fn call(app: &Router, method: &str, uri: &str, body: axum::body::Body) -> (StatusCode, String) {
    use tower::ServiceExt; // 提供 oneshot
    let req = axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body)
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// 冒烟测试：发若干请求、打印真实 请求→响应、断言、返回 (通过数, 总数)。
async fn smoke(app: &Router) -> (usize, usize) {
    let b64 = |png: &[u8]| base64::engine::general_purpose::STANDARD.encode(png);
    let red = solid_png([230, 20, 20], 16);
    let green = solid_png([20, 210, 30], 16);
    let blue = solid_png([15, 25, 220], 16);

    // 每条：(说明, 方法, uri, body, 期望状态码, 期望 body 里应包含的子串)
    let mut checks: Vec<(String, StatusCode, String, bool)> = Vec::new();
    let mut n_pass = 0usize;
    let total;

    // 用闭包收集会比较啰嗦，这里顺序跑，逐条打印
    macro_rules! run {
        ($desc:expr, $method:expr, $uri:expr, $body:expr, $want_status:expr, $want_sub:expr) => {{
            let (st, txt) = call(app, $method, $uri, $body).await;
            let ok = st == $want_status && txt.contains($want_sub);
            if ok { n_pass += 1; }
            println!("[{}] {} {}", if ok {"PASS"} else {"FAIL"}, $method, $uri);
            println!("      说明 : {}", $desc);
            println!("      响应 : {} {}", st.as_u16(), txt);
            checks.push(($desc.to_string(), st, txt, ok));
        }};
    }

    println!("==== ch107 进程内冒烟（tower oneshot，不占端口/不联网）====");
    run!("健康检查", "GET", "/health", axum::body::Body::empty(),
         StatusCode::OK, "\"status\":\"ok\"");
    run!("红图 · base64 JSON 入图", "POST", "/infer",
         axum::body::Body::from(json!({"image_base64": b64(&red)}).to_string()),
         StatusCode::OK, "\"label\":\"red\"");
    run!("绿图 · base64 JSON 入图", "POST", "/infer",
         axum::body::Body::from(json!({"image_base64": b64(&green)}).to_string()),
         StatusCode::OK, "\"label\":\"green\"");
    run!("蓝图 · 原始字节入图", "POST", "/infer_raw",
         axum::body::Body::from(blue.clone()),
         StatusCode::OK, "\"label\":\"blue\"");
    run!("非法 base64 → 400", "POST", "/infer",
         axum::body::Body::from(json!({"image_base64": "@@@not-base64@@@"}).to_string()),
         StatusCode::BAD_REQUEST, "base64");
    run!("合法 base64 但不是图片 → 400", "POST", "/infer",
         axum::body::Body::from(json!({"image_base64": b64(b"hello, not an image")}).to_string()),
         StatusCode::BAD_REQUEST, "无法解码图像");
    run!("空 body → 400", "POST", "/infer_raw", axum::body::Body::empty(),
         StatusCode::BAD_REQUEST, "请求体为空");

    total = checks.len();
    println!("==== 冒烟通过 {}/{} ====", n_pass, total);
    (n_pass, total)
}

// ===================== main：默认跑冒烟；`-- serve` 真起服务 =====================

async fn serve(app: Router, addr: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("服务已启动: http://{addr}  (Ctrl-C 优雅关停)");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            println!("\n收到关停信号，停止接收新连接，等待在途请求完成…");
        })
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = build_app()?;
    if std::env::args().nth(1).as_deref() == Some("serve") {
        let addr = std::env::args().nth(2).unwrap_or_else(|| "0.0.0.0:3000".to_string());
        serve(app, &addr).await
    } else {
        let (pass, total) = smoke(&app).await;
        if pass != total {
            anyhow::bail!("冒烟未全过：{pass}/{total}");
        }
        Ok(())
    }
}

// cargo test 也复用同一套冒烟断言，防回归。
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn smoke_all_pass() {
        let app = build_app().expect("build_app");
        let (pass, total) = smoke(&app).await;
        assert_eq!(pass, total, "冒烟未全过");
    }
}
