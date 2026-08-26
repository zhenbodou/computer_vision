// 第 96 章配套：最小可跑的 axum 视觉服务
//
// 提供三个接口：
//   GET  /health     健康检查，返回 {"status":"ok", ...}
//   POST /process    收原始图片字节 -> 解码 -> 返回宽高 + 平均灰度 (JSON)
//   POST /grayscale  收原始图片字节 -> 转灰度 -> 返回处理后的 PNG 字节
//
// 演示的“生产要点”：
//   1. 模型/配置只加载一次，用 Arc<AppState> 在所有请求间共享（这里放版本号；
//      真实场景把 ch94 的 tract 模型放进来，见下方 run_model 注释）。
//   2. CPU 密集的解码/遍历放进 spawn_blocking，绝不阻塞 async 运行时。
//   3. 有界并发许可（Semaphore）做背压：过载时快速返回 503，而不是把请求
//      堆进内存直到 OOM（呼应第 92 章多路并发）。
//   4. 请求体大小上限（DefaultBodyLimit）：超限自动 413，防大图/解码炸弹。
//   5. 超时中间件 + 优雅关停（ctrl_c）。
//   6. 统一错误处理：解码失败 -> 400，过载 -> 503，内部错误 -> 500。
//
// 运行：cargo run    （默认监听 0.0.0.0:3000）

use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use image::{DynamicImage, GenericImageView, ImageFormat};
use serde::Serialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// 全局共享状态：进程启动时构造一次，用 Arc 让所有 handler 只读共享。
struct AppState {
    /// 模型/配置版本号，随每个响应返回，方便定位“哪个版本算出来的”（呼应 ch94）。
    model_version: String,
    /// 背压闸门：最多允许 `permits` 个请求同时进入 CPU 处理阶段。
    /// 拿不到许可就立刻 503，避免过载压垮进程。
    limiter: Semaphore,
    // 真实推理服务在这里放模型，例如（前向引用 ch94 的 tract）：
    //   model: tract_onnx::prelude::TypedRunnableModel<...>,
    // tract 的 model.run(&self, ...) 是只读的，天然适合放进 Arc 共享。
}

/// /process 的返回结构，自动序列化成 JSON。
#[derive(Serialize)]
struct ProcessResult {
    width: u32,
    height: u32,
    /// 平均灰度：0（全黑）~ 255（全白）。
    avg_gray: f64,
    model_version: String,
    /// 服务端处理耗时（毫秒），方便观测 P50/P95。
    elapsed_ms: u128,
}

/// 统一的应用错误，映射到不同 HTTP 状态码。
enum AppError {
    /// 图片解码失败 / 请求体为空 -> 客户端问题，400。
    BadImage(String),
    /// 背压闸门已满 -> 服务繁忙，503（可按需改 429）。
    Overloaded,
    /// spawn_blocking 崩了、编码失败等 -> 500。
    Internal,
}

// 让 AppError 能直接作为 handler 的错误返回值：axum 会调用 into_response。
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::BadImage(m) => (StatusCode::BAD_REQUEST, m),
            AppError::Overloaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                "服务繁忙，请稍后重试".to_string(),
            ),
            AppError::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "内部错误".to_string(),
            ),
        };
        // 错误也返回 JSON，前端好解析
        (status, Json(json!({ "error": msg }))).into_response()
    }
}

/// GET /health —— 健康检查。
/// 这里返回“进程存活 + 配置就绪”；真实服务还应区分“依赖就绪”和“跑一次小样本推理”，
/// 避免模型加载失败的实例仍然接流量（见本章正文）。
async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model_version": state.model_version,
        // 剩余可用并发许可，可作为简单的负载观测
        "available_permits": state.limiter.available_permits(),
    }))
}

/// POST /process —— 收图片字节，返回宽高 + 平均灰度。
async fn process(
    State(state): State<Arc<AppState>>,
    body: Bytes, // 原始请求体字节（Bytes 是 Arc 支撑的，clone 很便宜）
) -> Result<Json<ProcessResult>, AppError> {
    // 背压：拿不到许可就快速失败，不把活儿堆进内存
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?;

    if body.is_empty() {
        return Err(AppError::BadImage("请求体为空".to_string()));
    }

    let started = Instant::now();
    let version = state.model_version.clone();

    // CPU 密集：解码 + 遍历每个像素求和。放到阻塞线程池，别卡住 async 运行时。
    let (width, height, avg_gray) = tokio::task::spawn_blocking(move || compute_stats(&body))
        .await
        .map_err(|_| AppError::Internal)??; // 外层 JoinError->500，内层解码错误->400

    Ok(Json(ProcessResult {
        width,
        height,
        avg_gray,
        model_version: version,
        elapsed_ms: started.elapsed().as_millis(),
    }))
}

/// POST /grayscale —— 收图片字节，返回灰度化后的 PNG 字节。
async fn grayscale(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Response, AppError> {
    let _permit = state.limiter.try_acquire().map_err(|_| AppError::Overloaded)?;
    if body.is_empty() {
        return Err(AppError::BadImage("请求体为空".to_string()));
    }

    let png = tokio::task::spawn_blocking(move || to_gray_png(&body))
        .await
        .map_err(|_| AppError::Internal)??;

    // 手动设置 Content-Type，返回二进制 PNG
    Ok(([(header::CONTENT_TYPE, "image/png")], png).into_response())
}

/// 纯 Rust 视觉处理：解码 -> 宽高 + 平均灰度。放在同步函数里，交给 spawn_blocking 执行。
fn compute_stats(bytes: &[u8]) -> Result<(u32, u32, f64), AppError> {
    let img =
        image::load_from_memory(bytes).map_err(|e| AppError::BadImage(e.to_string()))?;
    let (w, h) = img.dimensions(); // 需要 GenericImageView trait
    let gray = img.to_luma8(); // 转 8 位灰度
    let sum: u64 = gray.pixels().map(|p| p[0] as u64).sum();
    let n = (w as u64) * (h as u64);
    let avg = if n > 0 { sum as f64 / n as f64 } else { 0.0 };
    Ok((w, h, avg))
}

/// 解码 -> 转灰度 -> 编码成 PNG 字节。
fn to_gray_png(bytes: &[u8]) -> Result<Vec<u8>, AppError> {
    let img =
        image::load_from_memory(bytes).map_err(|e| AppError::BadImage(e.to_string()))?;
    let gray = img.to_luma8();
    let mut buf = Vec::new();
    DynamicImage::ImageLuma8(gray)
        .write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .map_err(|_| AppError::Internal)?; // 编码失败属于服务端问题
    Ok(buf)
}

// 真实服务里，tract 推理大致这样接进 handler（前向引用 ch94）：
//   fn run_model(state: &AppState, input: Tensor) -> TractResult<Tensor> {
//       let out = state.model.run(tvec!(input.into()))?;   // model 只加载一次、只读共享
//       Ok(out[0].clone().into_tensor())
//   }
// 在 handler 里同样用 spawn_blocking 包起来跑，避免阻塞运行时。

/// 等待 Ctrl-C，触发优雅关停。
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("无法监听 Ctrl-C 信号");
    tracing::info!("收到关停信号，停止接收新连接，等待在途请求完成…");
}

#[tokio::main]
async fn main() {
    // 日志：默认 info；可用 RUST_LOG 环境变量覆盖
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,tower_http=info")),
        )
        .init();

    // 状态只构造一次；Semaphore 许可数 = 允许的最大并发处理量（按 CPU 核数 / 压测调）
    let state = Arc::new(AppState {
        model_version: "demo-v1".to_string(),
        limiter: Semaphore::new(64),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/process", post(process))
        .route("/grayscale", post(grayscale))
        // 请求体上限 8 MiB：超限 axum 自动返回 413（防大图/解码炸弹）
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        // 单请求超时 10s：超时返回 408 Request Timeout
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(10),
        ))
        // 结构化访问日志
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:3000";
    let listener = TcpListener::bind(addr).await.expect("端口绑定失败");
    tracing::info!("服务已启动: http://{addr}  (Ctrl-C 优雅关停)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("服务运行出错");
}
