# 附录 A：常用 crate 速查表

> **怎么用这份速查**
> - 这是一张按**用途分类**的 crate 地图：每个 crate 标了**实测版本、是否纯 Rust、干什么用、书里哪章讲、一句话用法或最该记的坑**。
> - 选依赖前先来这里扫一眼：**纯 Rust 打勾**的可以放心用；**打叉的（只有一个 `ort`）** 会拖进外部 C/C++，本书只在"务实取舍"处提名，不进主工程。
> - 所有版本号都是**本机实测 `cargo add` 解析值**（2026-08，Rust 1.97.x，macOS/darwin）；标 ✅ 的在本机 `cargo run`/`cargo build` 或 `cargo check` 真编过，标 📄 的是按 docs.rs 对应版本页查证。
> - 遇到「加了这个 crate 就编不过」的具体报错，配套看 [附录 D：常见编译错误与排查](d-troubleshooting.md)。

**全书一条铁律先摆这儿**：只用**纯 Rust**，**不碰 OpenCV / `opencv` crate**。纯 Rust 换来的是——**单文件静态部署、交叉编译省心、没有 GC、内存安全、没有 Python/C++ 运行时拖累**。代价是某些能力纯 Rust 生态确实还没补齐（通用视频解码、成熟通用 OCR、量化工具链、CUDA 级 GPU 性能）——这些**第八节「诚实边界」**一条条列清楚，并给务实替代。**知道边界在哪，本身就是专家能力的一部分。**

## 一、图像与经典视觉基石（主工程 `code/` 默认依赖，全部纯 Rust）

这五个是全书前半程（第 1–31 章、特征与经典算法）的地基，都在主工程 [`code/Cargo.toml`](../p01-start/ch03-setup.md) 里，`use` 即可。

| crate | 实测版本 | 纯 Rust | 干什么 | 主要章节 | 一句话 |
|---|---|:---:|---|---|---|
| `image` | 0.25.10 | ✅ | 读写图片、像素访问、颜色转换、缩放裁剪旋转 | [7](../p02-image-basics/ch07-load-save.md)–[12](../p02-image-basics/ch12-histogram.md) | `image::open(p)?` → `DynamicImage`；`RgbImage`/`GrayImage`/`Luma`/`Rgb` |
| `imageproc` | 0.27.0 | ✅ | 滤波、边缘、形态学、轮廓、霍夫、模板匹配、角点、绘制、几何 warp | [13](../p03-classic/ch13-convolution.md)–[22](../p03-classic/ch22-frequency.md)、[23](../p04-features/ch23-corners.md)–[26](../p04-features/ch26-homography-stitching.md) | 建在 image 之上；`filter`/`edges`/`morphology`/`contours`/`hough`/`geometric_transformations` |
| `ndarray` | 0.17.2 | ✅ | 多维数组 / 张量批量运算 | [29](../p05-numeric/ch29-ndarray.md)、[31](../p05-numeric/ch31-image-as-tensor.md) | `Array2::from_shape_vec`、`.dot()`、`.mean()`；大块数据首选 |
| `nalgebra` | 0.35.0 | ✅ | 线性代数、小矩阵、几何变换、SVD、特征分解 | [30](../p05-numeric/ch30-nalgebra.md)、[64](../p14-3d/ch64-calibration.md)–[68](../p14-3d/ch68-sfm-slam.md) | `Matrix3`/`Vector3`、`.svd(true,true)`、`.try_inverse()` |
| `rustfft` | 6.4.1 | ✅ | 快速傅里叶变换（频域滤波） | [22](../p03-classic/ch22-frequency.md) | `FftPlanner::new().plan_fft_forward(n)`；处理 `Complex<f32>` |

> 💡 **image 与 imageproc 的版本要咬死**：imageproc 0.27 是针对 image 0.25 编的，混用不同大版本会因 `Pixel`/`GenericImage` trait 对不上而编译失败。本书全程 `image 0.25.10 + imageproc 0.27.0`。
> 🕳️ **`imageproc::gradients` 的 Sobel 返回真导数的 8 倍**（3×3 核系数绝对值和=8）。凡把它和"原始像素差/帧差"放进同一个方程（光流、亚像素边缘）**先 /8 统一量纲**。

## 二、工程化：并行、错误处理、序列化、基准测试（均纯 Rust，已入主工程）

| crate | 实测版本 | 纯 Rust | 干什么 | 主要章节 | 一句话 |
|---|---|:---:|---|---|---|
| `rayon` | 1.12.0 | ✅ | 数据并行（多核） | [108](../p21-mastery/ch108-performance.md) | `.par_iter()`/`.par_chunks()` 一改就多核；实测 7–101× |
| `thiserror` | 2.0.20 | ✅ | 库层自定义错误枚举 | [112](../p21-mastery/ch112-engineering.md) | `#[derive(Error)]` + `#[error("...")]` |
| `anyhow` | 1.0.104 | ✅ | 应用层错误聚合 | [112](../p21-mastery/ch112-engineering.md) | `anyhow::Result<T>`、`?` 一路上抛、`.context(...)` |
| `serde` | 1.0.229 | ✅ | 序列化框架（derive） | [93](../p18-events/ch93-rule-engine.md)、[113](../p21-mastery/ch113-datasets-mlops.md) | `#[derive(Serialize, Deserialize)]` |
| `serde_json` | 1.0.151 | ✅ | JSON 读写（规则配置、COCO 标注） | [93](../p18-events/ch93-rule-engine.md)、[113](../p21-mastery/ch113-datasets-mlops.md) | `serde_json::from_str`/`to_string_pretty` |
| `criterion` | 0.8.2 | ✅ | 统计学基准测试（dev-dep） | [108](../p21-mastery/ch108-performance.md) | `cargo bench`；配 `[[bench]] harness=false` |

> 🕳️ **加 `[[bench]]`/`[[example]]` 到 `Cargo.toml`，对应源文件必须同时存在**，否则 `cargo metadata` 及任何 cargo 命令都因 manifest 解析失败而全线报错。

## 三、深度学习与模型推理（第 7–16、19 部分）

**只有 `ort` 不是纯 Rust**（它链接外部 C++ onnxruntime）。其余三个都纯 Rust；`candle` 是全书深度学习章的主力。

| crate | 实测版本 | 纯 Rust | 定位 | 主要章节 | 验证 |
|---|---|:---:|---|---|:---:|
| `candle-core` / `candle-nn` | 0.11.0 | ✅ | HF 出品，safetensors 原生，手搓/训练/微调/推理，CPU/CUDA/Metal | [39](../p07-dl-basics/ch39-dl-ecosystem.md)、[42](../p08-classification/ch42-candle-classify.md)–[44](../p08-classification/ch44-finegrained.md)、[46](../p09-detection/ch46-yolo-inference.md)–[73](../p15-generative/ch73-clip-vlm.md)、[99](../p20-projects/ch99-ocr.md)–[102](../p20-projects/ch102-defect-system.md) | ✅ `cargo run` |
| `burn` | 0.21.0 | ✅ | 泛型后端 + Autodiff，类 PyTorch 的训练框架，一套代码切后端 | [40](../p07-dl-basics/ch40-burn-train.md) | 📄 + ✅ ch40 |
| `tract-onnx` | 0.23.5 | ✅ * | Sonos 出品，纯 Rust ONNX 推理引擎，交叉编译友好 | [94](../p19-deploy/ch94-onnx.md)、[107](../p20-projects/ch107-deploy-service.md) | ✅ `cargo run` |
| `ort` | 2.0.0-rc.13 | ❌ | ONNX Runtime 的 Rust 绑定，**底层链 C++ onnxruntime**；算子最全、性能最强、支持 GPU | [94](../p19-deploy/ch94-onnx.md) | 📄 未编译 |

> **`*` tract 的纯度星号（诚实交代，全书据此措辞）**：`tract-linalg` 有一个**构建期** `cc` 依赖，但它只当**汇编器**用，编译 tract **自带的 SIMD 汇编内核**（`.S` 文件，仓库里 **0 个 `.cpp`**）——**没有 `-sys` crate、不链接任何外部 C/C++ 运行库、产物是自包含单二进制**。这与 `ort` 链接外部 C++ onnxruntime 是**性质完全不同**的两回事：tract 依然满足"纯 Rust / 可交叉编译 / 自包含"的诉求，**依然是本书跑现成 ONNX 的首选**。交叉编译到某平台时，若目标汇编器不认那套指令，tract 会**回退到通用纯 Rust 内核**（能跑、稍慢）——这反过来印证它不依赖外部 C 库。
>
> 💡 **一句话选型**：只推理 + 要纯 Rust → **tract**；只推理 + 要最强性能/最全算子（可接受 C++）→ **ort**；加载 safetensors / 手搓·微调（LLM、Transformer）→ **candle**；从零训练 + 要多后端 PyTorch 式体验 → **burn**。

**candle GPU/加速** 走 feature：`cuda`/`metal`/`accelerate`/`mkl`（默认 CPU）。**candle 有两个高频坑**（详见附录 D）：CPU 后端**随机不可复现**（`set_seed` 报错），要自造确定性初始化；`binary_cross_entropy_with_logit` **数值不稳定**且 target 须 f32，多标签训练自写稳定版 BCE。

## 四、传统机器学习：linfa 全家桶（第 6 部分）

| crate | 实测版本 | 纯 Rust | 干什么 | 主要章节 |
|---|---|:---:|---|---|
| `linfa` + `linfa-svm`/`linfa-trees`/`linfa-clustering`/`linfa-logistic`/`linfa-bayes`/`linfa-reduction`/… | 0.8.1 | ✅ | 经典 ML：SVM、决策树、KMeans、逻辑回归、朴素贝叶斯、PCA | [33](../p06-ml/ch33-linfa.md)–[35](../p06-ml/ch35-digit-recognition.md) |

> 🕳️ **头号版本坑：linfa 0.8.1 内部锁死 `ndarray 0.16.1`，而本书其余章节用 `ndarray 0.17.2`。** 同一个 `Cargo.toml` 里两者并存会拉入**两个 ndarray**，你的 `Array2`（0.17）和 linfa 要的 `Array2`（0.16）是**不同类型**，报 `expected ArrayBase… found ArrayBase…`。**对策**：linfa 示例做成**独立工程**，把 ndarray 也钉到 `0.16`。基础 API 在 0.16/0.17 基本一致，读者无感。范式统一为 `Model::params(...).fit(&dataset)?` → `.predict(...)`。

## 五、读码与 OCR（第 17 部分、第 98–99 章）

| crate | 实测版本 | 纯 Rust | 能力 | 与 image 0.25.10 | 主要章节 |
|---|---|:---:|---|:---:|---|
| `qrcode` | 0.14.1 | ✅ | QR **生成** | ✅ 兼容 | [81](../p17-industrial/ch81-code-reading.md)、[98](../p20-projects/ch98-qrcode.md) |
| `rqrr` | 0.10.1 | ✅ | QR 定位 + **解码** | ✅ 兼容 | [81](../p17-industrial/ch81-code-reading.md)、[98](../p20-projects/ch98-qrcode.md) |
| `rxing` | 0.9.2 | ✅ | 1D/QR/DataMatrix **解码**（ZXing 移植） | ❌ **锁死 image =0.25.8** → 独立工程 | [81](../p17-industrial/ch81-code-reading.md) |
| `barcoders` | 2.0.0 | ✅ | 一维码**仅生成** | ✅ | [81](../p17-industrial/ch81-code-reading.md) |
| `datamatrix` | 0.3.3 | ✅ | DataMatrix 编+解（**只吃已二值化模块矩阵，不吃相机原图**） | — | [81](../p17-industrial/ch81-code-reading.md) |
| `ocrs` + `rten` | 0.12.2 / 0.25.0 | ✅ | OCR（rten = 纯 Rust ONNX 运行时，需配 ONNX 模型） | — | [81](../p17-industrial/ch81-code-reading.md)、[99](../p20-projects/ch99-ocr.md) |

> 🎯 **`qrcode` + `rqrr` 是只读 QR 的黄金组合**，端到端可复现（生成→旋转→缩放→加噪→读回，实测闭环）。
> 🕳️ **`rqrr` 的 `MetaData.ecc_level` 是格式信息原始 2 bit**（M=0b00 / L=0b01 / H=0b10 / Q=0b11），**不是** L<M<Q<H 的序号，别直接当序号查表。

## 六、视频与流媒体（第 16 部分）

| crate | 实测版本 | 纯 Rust | 能力 | 主要章节 |
|---|---|:---:|---|---|
| `image`（gif feature，默认开） | 0.25.10 | ✅ | 多帧 **GIF/APNG** 读写（纯 Rust 唯一顺畅的"视频"形态） | [74](../p16-video/ch74-video-io.md) |
| `retina` | 0.4.20 | ✅ | RTSP/RTP 客户端（异步，需 tokio）——**收到的是编码后 H.264 NAL，不是像素** | [75](../p16-video/ch75-rtsp-streaming.md) |

> ⚠️ **诚实边界**：通用视频解码（H.264/H.265/mp4）**没有成熟纯 Rust 方案**（`image::open` 不认 mp4；`symphonia` 纯 Rust 但只解音频）。务实路线：① `ffmpeg` **子进程**抽帧成 PNG/裸 RGB 喂给 Rust（最通用，你的 Rust 代码仍纯，只是依赖系统 ffmpeg）；② `ffmpeg-next` FFI（进程内解码，非纯 Rust）。实时管线用 `std` 的 `mpsc`/`sync_channel` 就够（丢帧策略见第 [75](../p16-video/ch75-rtsp-streaming.md)、[92](../p18-events/ch92-multistream.md) 章）。

## 七、服务化、GPU、WebAssembly、跨框架（第 18–21 部分）

| crate | 实测版本 | 纯 Rust | 干什么 | 主要章节 |
|---|---|:---:|---|---|
| `axum` | 0.8.9 | ✅ | Web 框架（把模型做成 HTTP API） | [96](../p19-deploy/ch96-web-service.md)、[107](../p20-projects/ch107-deploy-service.md) |
| `tokio` | 1.53.1 | ✅ | 异步运行时（`spawn_blocking` 跑 CPU 密集推理） | [96](../p19-deploy/ch96-web-service.md)、[107](../p20-projects/ch107-deploy-service.md) |
| `tower` / `tower-http` | 0.5.3 / 0.7.0 | ✅ | 中间件（超时、trace）、`oneshot` 进程内测试 | [96](../p19-deploy/ch96-web-service.md)、[107](../p20-projects/ch107-deploy-service.md) |
| `tracing` + `tracing-subscriber` | 0.1 / 0.3 | ✅ | 结构化日志/观测 | [96](../p19-deploy/ch96-web-service.md) |
| `wgpu` + `pollster` + `bytemuck` | 30.0.1 / 1.0.1 / 1.25.2 | ✅ | 跨平台 GPU 计算（WGSL compute，可上 wasm） | [109](../p21-mastery/ch109-gpu.md) |
| `wasm-bindgen` | 0.2.127 | ✅ | 编到 WebAssembly，在浏览器跑视觉算法 | [110](../p21-mastery/ch110-wasm.md) |
| `prost` | 0.14 | ✅ | protobuf；配 tract 在纯 Rust 里**造 `.onnx`** | [94](../p19-deploy/ch94-onnx.md) |

> 🕳️ 独立服务/GPU/wasm 工程的 `Cargo.toml` **必须写一个空 `[workspace]`**，否则会被上层 `code/` 的工作区吸入导致依赖串味。
> 🕳️ `prost` 版本**必须与 tract 内部一致（0.14.x）**，否则 `Message::encode` 类型对不上。
> ⚠️ **GPU/CUDA 边界**：纯 Rust 通用 GPU = `wgpu` 写 WGSL；训练上 GPU = `burn-wgpu` 一行换后端。**CUDA 路线（`cust`/candle 的 cuda feature）非纯 Rust、绑 NVIDIA、依赖 C 运行库**；纯 Rust 目前没有 cuBLAS/cuDNN 的对等物，极致性能仍属 CUDA。

## 八、纯度总账与"诚实边界"（本书的立场）

**纯度分四级**，全书据此选型：

| 级别 | 含义 | 代表 |
|---|---|---|
| ✅ 纯 Rust | 无任何外部 C/C++ 运行库，自包含单二进制 | image、imageproc、ndarray、nalgebra、candle、burn、rqrr、qrcode、axum、wgpu… |
| ✅ * 纯 Rust（构建期汇编器） | 仅**构建期**调 `cc` 当汇编器编自带 asm，无外部运行库 | **tract-onnx** |
| ⚠️ 纯 Rust但有版本枷锁 | 本身纯 Rust，但锁死了某依赖的旧版本 → 需独立工程隔离 | linfa（锁 ndarray 0.16）、rxing（锁 image 0.25.8） |
| ❌ 非纯 Rust | 链接外部 C/C++ 运行库 | **ort**（C++ onnxruntime） |

**纯 Rust 目前还够不着的能力（→ 务实替代）**——这不是缺点清单，是让你少走弯路的地图：

| 能力 | 纯 Rust 现状 | 务实替代 |
|---|---|---|
| 通用视频解码 H.264/H.265/mp4 | ❌ 无成熟方案 | `ffmpeg` 子进程抽帧（你的 Rust 仍纯）/ `ffmpeg-next` FFI |
| 成熟通用 OCR | ❌ 无（Tesseract=C++、PaddleOCR=Py/C++） | 受限字符集可纯 Rust 自己做（连通域+小 CNN，见第 [99](../p20-projects/ch99-ocr.md) 章）；通用文字 → CRNN/PaddleOCR **导出 ONNX** + tract（代码纯 Rust，依赖外部模型） |
| 模型量化工具链（PTQ/QAT） | ❌ 无（都在 Python：PyTorch/ORT/TensorRT） | 训练侧量化好 → 导出量化 ONNX/GGUF → Rust（tract/candle）加载推理 |
| 成熟 MLOps（实验追踪/编排/血缘） | ❌ 无对标（MLflow/W&B/DVC/Airflow 都在 Python） | Rust 做高性能数据流水线/预处理/serving，接入 Python 中枢 |
| CUDA 级 GPU 性能（cuBLAS/cuDNN） | ⚠️ 有 wgpu 但非极致 | 极致性能走 CUDA（非纯 Rust）；纯 Rust 用 wgpu/burn-wgpu 够教学与多数落地 |
| 大模型级 SAM / Stable Diffusion | ⚠️ candle 官方 example 需下载数 GB 权重、CPU 极慢 | 只讲机制，或上带权重的官方 example / GPU |

> **一句话总纲**：**纯 Rust 在"经典视觉 + 中小模型推理 + 高性能数据流水线 + 单文件部署 + 交叉编译"这条主线上，今天就完全够用、而且很香。** 真正够不着的是"通用视频解码、成熟通用 OCR、量化/MLOps 工具链、CUDA 级性能"这几块——本书从不为了"纯"而硬造，遇到就诚实说明并给替代。

## 九、三条最常踩的版本冲突（编译前先记住）

1. **linfa ↔ ndarray**：linfa 0.8.1 锁 `ndarray 0.16.1`，与主工程 `0.17.2` 冲突 → linfa 示例独立工程 + ndarray 钉 0.16。
2. **rxing ↔ image**：rxing 0.9.2 锁 `image =0.25.8`，与主工程 `0.25.10` 冲突 → 读码高级场景独立工程隔离。
3. **prost ↔ tract**：造 `.onnx` 时 `prost` 必须与 tract 内部同为 `0.14.x`。

外加两条"结构性"约定：**独立工程（candle/burn/tract/axum/wgpu/wasm 等重依赖）一律带空 `[workspace]`**；**`[[bench]]`/`[[example]]` 条目与源文件必须成对存在**。

---

选依赖时的黄金问题永远是那句：**它是纯 Rust 吗？会不会拖进 C/C++ 运行库？** 打勾的放心用，打叉的（就 `ort` 一个）想清楚再用。具体报错怎么修，翻 [附录 D](d-troubleshooting.md)。
