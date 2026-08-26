# 附录 D：常见编译错误与排查

> **怎么用这份排查手册**
> - 全书上百个配套示例，作者在本机一个个踩过的坑都收在这里，按**症状 → 原因 → 改法**排列。编译报错时，把关键字（报错里的类型名、trait 名、宏名）在本页 `Ctrl+F` 一下，多半能直接命中。
> - 分两大类：**① 编译期报错**（编不过，编译器会拦你，相对好办）；**② 能编译却跑错**（编译器放行、程序不 panic，但结果是错的——这类最危险，独占第九节）。
> - 版本相关的坑先翻 [附录 A](a-crates.md) 的"版本冲突速查"；本页只讲"报了什么错、为什么、怎么改"。
> - 一条心法：**Rust 的编译器是你最严格也最忠诚的同事**。它拦下的，几乎都是真问题；它拦不下的（第九节那些），才需要你自己长记性。

## 一、依赖与工程配置类（最先排除）

**现象：加了 `[[bench]]` 或 `[[example]]` 后，`cargo run`/`cargo test`/`cargo metadata` 全部报错，提示找不到 target 源文件。**
- **原因**：`Cargo.toml` 里声明了 `[[bench]] name = "xxx"`（或 `[[example]]`），但 `benches/xxx.rs`（或 `examples/xxx.rs`）**还没建**。manifest 解析在**任何** cargo 命令前发生，一处引用悬空，全线瘫痪。
- **改法**：**声明和源文件必须成对**。先建空文件（哪怕只有 `fn main(){}`）再写 `Cargo.toml` 条目，或反过来。本书主工程的 `[[bench]] name="ch108_bench"` 对应的 `benches/ch108_bench.rs` 始终在位。

**报错：`expected struct ArrayBase<...>, found struct ArrayBase<...>`（两个看起来一样的 ndarray 类型对不上）。**
- **原因**：`Cargo.tree` 里存在**两个 ndarray 大版本**。最常见是 linfa 0.8.1 锁死 `ndarray 0.16.1`，而你的工程用 `0.17.2`——你的 `Array2<f64>`（0.17）喂不进 linfa 要的 `Array2<f64>`（0.16），它俩是**不同 crate 实例里的不同类型**。
- **改法**：把 linfa 示例拆成**独立工程**，其中 ndarray 也钉到 `0.16`（`ndarray = "0.16"`）。同理 rxing 0.9.2 锁 `image =0.25.8`，与主工程 `image 0.25.10` 冲突，也用独立工程隔离。见 [第 33 章](../p06-ml/ch33-linfa.md)、[附录 A 第九节](a-crates.md)。

**现象：独立小工程 `cargo build` 时莫名去编上层目录一堆无关依赖，或版本被"拽偏"。**
- **原因**：这个小工程落在了上层 `code/` 的 **workspace** 覆盖范围内，被当成工作区成员，依赖被统一解析（"串味"）。
- **改法**：独立工程的 `Cargo.toml` 里加一个**空 `[workspace]`** 表，声明"我自成一区、不归任何上层工作区管"。本书所有 `dl_labs/*`、`projects/*`、`service_demo`、`gpu_demo`、`wasm_demo` 都这么做。

**报错：造 `.onnx` 时 `prost` 的 `Message::encode` 类型对不上 / trait 不满足。**
- **原因**：你直接 `cargo add prost` 装了最新版，但 tract 0.23.5 内部用的是 `prost 0.14.x`，protobuf 消息类型来自 tract 重导出的那份，版本不一致就对不上。
- **改法**：`prost` 显式钉到与 tract 一致的 `0.14`。用 `tract_onnx::pb`（tract **公开**了 protobuf 定义）来构造 `.onnx`。见 [第 94 章](../p19-deploy/ch94-onnx.md)。

## 二、nalgebra 与线性代数类

**现象：用 SVD 解齐次方程 \\( A\mathbf{x}=\mathbf{0} \\)（如单应、基础矩阵 DLT），最小情形（方程数刚好等于未知数、\\( m<n \\)）解出来完全不对。**
- **原因**：**瘦 SVD（thin SVD）的陷阱**。当 \\( A \\) 是 \\( m\times n \\) 且 \\( m<n \\) 时，nalgebra 的 `svd.v_t` **不含**你要的那个零空间向量——最小奇异值对应的右奇异向量被"瘦掉"了，你取到的"末行"是错的。ch106 全景拼接就栽在这：四角误差飙到 513px。
- **改法**：**改解特征分解**。构造 \\( A^\top A \\)（\\( n\times n \\)，一定满秩维度），取其**最小特征值对应的特征向量**：
  ```rust
  let ata = a.transpose() * &a;              // n×n 对称半正定
  let se = ata.symmetric_eigen();            // 特征分解
  // ⚠️ symmetric_eigen 的特征值【未排序】，必须自己找 argmin
  let (mut k, mut best) = (0, f64::INFINITY);
  for i in 0..se.eigenvalues.len() {
      if se.eigenvalues[i] < best { best = se.eigenvalues[i]; k = i; }
  }
  let sol = se.eigenvectors.column(k);        // 最小特征值对应的解
  ```
  改完 ch106 从 513px 收敛到 1.965px。**注意**：`svd.v_t` 末行取解**只在超定（\\( m>n \\)，方程多于未知数）时才对**——三维重建里点数充足的 DLT 属于此列，是安全的。见 [第 26 章](../p04-features/ch26-homography-stitching.md)、[第 64 章](../p14-3d/ch64-calibration.md)。

**坑：`symmetric_eigen()` 的特征值默认不排序**——别假设 `eigenvalues[0]` 是最小/最大的，永远自己 `argmin`/`argmax`（见上）。

**现象：Umeyama/Procrustes 求相似变换（人脸对齐），偶尔把脸"镜像翻转"了。**
- **原因**：从协方差矩阵 SVD 恢复旋转 \\( R=UV^\top \\) 时，可能得到 \\( \det(R)=-1 \\)（含镜像的正交矩阵），这不是纯旋转。
- **改法**：**镜像校正**——当 \\( \det(UV^\top)<0 \\)，在中间插一个 \\( \mathrm{diag}(1,\dots,1,-1) \\)：
  ```rust
  let mut s = Matrix2::identity();
  if (u.determinant() * v_t.determinant()) < 0.0 { s[(1,1)] = -1.0; }
  let r = u * s * v_t;   // 保证是纯旋转
  ```
  见 [第 54 章](../p11-keypoints/ch54-face-landmarks.md)。

**坑：`imageproc` 的 `warp_into` / `warp` 传了单应却 warp 反了方向。** `warp_into` **内部会对你给的投影求逆**——它按"对输出每个像素、反查源图"工作。所以要传**源 → 目标**的单应 \\( H \\)，函数内部自动用 \\( H^{-1} \\) 反查。方向错就是把 \\( H \\) 和 \\( H^{-1} \\) 传反了。见 [第 26 章](../p04-features/ch26-homography-stitching.md)、[第 106 章](../p20-projects/ch106-panorama.md)。

## 三、candle 训练与推理类（深度学习章高频）

**报错/现象：CPU 后端调 `set_seed` 报错，或每次运行结果都不一样、无法复现。**
- **原因**：candle 的 CPU 后端对全局 `set_seed` 支持不完整，参数随机初始化不受它控制。
- **改法**：**自造确定性初始化**——用固定种子的 LCG 生成初始权重，`Var::from_tensor` 建可训练参数。全书 DL 章都这么保证"数字可复现"。见 [第 42 章](../p08-classification/ch42-candle-classify.md)。

**报错：`binary_cross_entropy_with_logit` 传入 target 时类型/形状报错，或训练中途出现 NaN、loss 不降。**
- **原因**：两个坑叠加。① target 必须是 **f32**（你若用 u32 标签直接传会报错）;② 该函数**数值不稳定**——logit 绝对值一大，`exp` 溢出成 NaN。
- **改法**：target 先转 f32；BCE **自己写数值稳定版**（等价变形，避免 `exp` 溢出）：
  ```text
  loss = max(x, 0) - x * y + log(1 + exp(-|x|))
  ```
  多标签分类（ch44）、GAN 判别器（ch70）都靠这个稳住。见 [第 44 章](../p08-classification/ch44-finegrained.md)、[第 70 章](../p15-generative/ch70-gan.md)。

**报错：`cross_entropy` / `nll_loss` 的 index 张量类型不符，或 `argmax`/`gather`/`index_select` 报 dtype 错。**
- **原因**：candle 里**索引/标签类张量必须是 `u32`**（不是 i64、不是 f32）。`arange` 也要写 `Tensor::arange(0u32, n, dev)`。
- **改法**：标签建成 `u32`；`gather`/`index_select` 的 index 张量转 `u32`。

**现象：想做逐像素分类（分割），或想给交叉熵加类别权重，发现 candle 没有现成 API。**
- **原因**：candle 的 `cross_entropy` **不支持 class weight**，也不直接吃 `[N,C,H,W]` 的分割 logits。
- **改法**：① 逐像素 CE：`permute` 成 `[N,H,W,C]` → **`.contiguous()`** → `reshape([N*H*W, C]`，标签同样拉平，再套普通 CE。② 加权 CE **自己写**：`log_softmax` 后 `gather` 出真实类的 log 概率，乘上每类权重求平均。ch102 缺陷分割用 `w=4` 压少数类，把误报从 10.5% 压到 **0**。**注意**：`permute` 后一定要 `.contiguous()`，否则后续 `reshape` 因内存非连续而报错。见 [第 49 章](../p10-segmentation/ch49-semantic-seg.md)、[第 102 章](../p20-projects/ch102-defect-system.md)。

**报错：找不到 `ConvTranspose2d`，或上采样层反向传播报错/梯度为空。**
- **原因**：candle **没有** `ConvTranspose2d`；且它的 `upsample_nearest2d` **反向仅支持整数倍**，`bilinear` 上采样**根本没有反向**。
- **改法**：上采样用 `upsample_nearest2d`（整数倍）**+ 一个普通 `Conv2d`** 组合代替转置卷积（U-Net 解码器就这么搭）。**训练里别用 bilinear 上采样**（无梯度）。连带结论：需要 bilinear 反向的 **RoIAlign 在 candle 里训练不可行**——把 RoI 缩放放到 Rust 图像侧预处理，只让网络吃固定尺寸。见 [第 49 章](../p10-segmentation/ch49-semantic-seg.md)、[第 50 章](../p10-segmentation/ch50-instance-seg.md)。

**现象：ArcFace / 带 margin 的人脸损失训练时 loss 变 NaN。**
- **原因**：算 \\( \sin\theta=\sqrt{1-\cos^2\theta} \\) 时，`cos` 因浮点误差略大于 1，`1-\cos^2` 成微小负数，`sqrt` 负数 → NaN。
- **改法**：开方前 **`relu`**（把负的夹到 0）：`(1.0 - cos2).relu().sqrt()`。见 [第 57 章](../p12-face/ch57-face-recognition.md)。

**现象：用 1×1 卷积做的"头"去回归绝对坐标（关键点像素位置、检测框中心），怎么训都收敛不到。**
- **原因**：**卷积天生平移等变**，1×1 卷积头在每个空间位置共享权重、"看不见自己在哪"，**无法输出绝对坐标**。
- **改法**：两条路——① **网格中心 + 偏移**（在每个格点回归相对偏移，YOLO 式解码）;② 换 **全连接（FC）头**（FC 能看到全局位置，ch54 人脸关键点用 FC 头把 MSE 从 0.28 干到 0.00007）。见 [第 46 章](../p09-detection/ch46-yolo-inference.md)、[第 54 章](../p11-keypoints/ch54-face-landmarks.md)。

**报错：`.t()`（转置）或 `permute` 之后的张量做 `reshape`/`matmul` 报"非连续"。**
- **原因**：转置/置换只改 stride 不搬数据，结果**内存非连续**，很多算子要求连续布局。
- **改法**：紧跟一个 **`.contiguous()`**。CLIP 的对称 InfoNCE 里 `logits.t().contiguous()` 就是为此。见 [第 73 章](../p15-generative/ch73-clip-vlm.md)。

## 四、tract 推理类

**报错：`to_array_view` 找不到 / 类型不对，取不出推理结果。**
- **原因**：API 名记错了。tract 取输出用的是 **`to_plain_array_view`**（不是 `to_array_view`）。
- **改法**：`result[0].to_plain_array_view::<f32>()?`。从内存字节加载模型用 `model_for_read(&mut Cursor::new(bytes))`（省落盘）。`into_runnable()` 返回的是 `Arc<TypedRunnableModel<...>>`（有零泛型别名，直接存进 `AppState` 共享）。见 [第 94 章](../p19-deploy/ch94-onnx.md)、[第 107 章](../p20-projects/ch107-deploy-service.md)。

**疑问：tract 构建时看到它调用了 `cc`，是不是"其实不是纯 Rust"？**
- **答**：不是。`tract-linalg` 的 `cc` 只当**汇编器**编译 tract **自带的 `.S` SIMD 内核**（仓库里 0 个 `.cpp`），**不链接任何外部 C/C++ 运行库**，产物是自包含单二进制——与 `ort`（链接外部 C++ onnxruntime）性质完全不同。交叉编译到冷门平台时 tract 会**回退纯 Rust 通用内核**（能跑、稍慢），反证它不依赖外部 C 库。全书据此把 tract 标为"✅ *纯 Rust"。详见 [附录 A 第三节](a-crates.md)。

## 五、imageproc 与图像处理类

**现象：把 `imageproc` 的 Sobel 梯度和"帧差/像素差"放进同一个方程（光流、亚像素定位），结果量纲差了 8 倍。**
- **原因**：`imageproc::gradients` 的 **Sobel 返回真导数的 8 倍**（3×3 核系数绝对值和 = 8）。
- **改法**：与其它一阶差分混用前**先 `/8`** 统一量纲。见 [第 15 章](../p03-classic/ch15-edges.md)、[第 59 章](../p13-tracking/ch59-optical-flow.md)。

**坑：`draw_line_segment_mut` 只接收 `f32` 坐标、且没有箭头**——画方向/向量得自己补箭头两笔线段。这不是报错，是"画出来不如预期"。见 [第 87 章](../p18-events/ch87-tripwire.md)。

## 六、axum / tokio 服务类

**报错：`axum::Server::bind(...)` 找不到 / 已废弃。**
- **原因**：axum 0.7+ 移除了旧的 `Server` API。
- **改法**：用 `let listener = tokio::net::TcpListener::bind(addr).await?;` + `axum::serve(listener, app).await?`。见 [第 96 章](../p19-deploy/ch96-web-service.md)。

**报错：在 `spawn_blocking` 里跑推理，`?` 处理错误时报类型不对 / 编不过。**
- **原因**：`spawn_blocking` 返回 `Result<T, JoinError>`，而你的闭包内部又返回一层 `Result` → **双层 Result**。
- **改法**：**双 `??`**——`let out = tokio::task::spawn_blocking(move || infer(...)).await??;`（第一个 `?` 解 `JoinError`，第二个解业务错误）。见 [第 96 章](../p19-deploy/ch96-web-service.md)、[第 107 章](../p20-projects/ch107-deploy-service.md)。

**报错：并发限流用 `Semaphore`，permit 与"把 state move 进 `spawn_blocking`"发生借用冲突。**
- **原因**：permit 借用了 `AppState`，而 `spawn_blocking` 要求闭包 `'static`、把 state **move** 进去，二者打架。
- **改法**：`try_acquire_owned()` 拿**owned permit**（`Arc<Semaphore>`），把 permit 一起 move 进闭包；拿不到 permit 直接返回 **503**（背压）。请求体大小用 `DefaultBodyLimit` 限。见 [第 96 章](../p19-deploy/ch96-web-service.md)、[第 107 章](../p20-projects/ch107-deploy-service.md)。

**想在测试里免起端口跑服务？** `Router` 本身就是个 `Service`：`app.clone().oneshot(request).await?` 进程内打一发，CI 复用同一套。**注意** `oneshot` 需要 tower 的 **`util`** feature。ch107 冒烟测试 7/7 就靠它。

## 七、WebAssembly 与交叉编译类

**报错（macOS 上交叉编译）：`ld: unknown options` 之类的链接器报错。**
- **原因**：macOS 自带的链接器不认 GNU 风格参数。
- **改法**：用 **`cross`**（容器化交叉编译）或切 **musl** 目标，别硬用系统 ld。见 [第 110 章](../p21-mastery/ch110-wasm.md)。

**现象：把 image + imageproc 编到 wasm，产物暴涨、依赖树到 149 个 crate。**
- **原因**：这套图像栈拉入大量传递依赖，wasm 下体积敏感。
- **改法**：wasm 里只留**真正用到**的算法，能自己手写的小算子就别拉整个 imageproc；按需裁 feature。另外 **wasm 里没有线程、没有文件系统、`file://` 受限**——涉及这些的代码要有 web 侧替代。见 [第 110 章](../p21-mastery/ch110-wasm.md)。

## 八、能编译却跑错——最危险的静默陷阱 ⚠️

这一节没有编译器帮你。**程序编得过、跑不崩、还给你一个"看起来挺合理"的数字——但它是错的。** 这类 bug 才是区分新手和老手的地方，逐条记牢。

| 现象（不报错，但结果错） | 根因 | 改法 |
|---|---|---|
| 自造随机数（LCG）生成的类别/坐标分布诡异、周期性重复、某一类一个都没生成 | 取 LCG **低位**：`& 1` 掷布尔、对 **2 的幂**取模（`% 4` 等于只取最烂的低 2 位）——第 \\(k\\) 低位周期仅 \\(2^k\\)，直接退化 | 一律走**高位**：`unit() = (state >> 11) as f64 / ...`，`below(n) = (unit()*n) as usize`，`boolean() = unit() < 0.5`——永远安全。（对**非 2 的幂**的 `n` 做 `% n` 只是质量平平、不会退化，但别养成习惯。）见 [第 58 章](../p12-face/ch58-liveness-attributes.md) |
| 级联系统（检测→对齐→识别、分割→分类）单看每级都好，**端到端却很差**（ch101 从 0% 起步、ch99 整串 85%→20.5%） | **级联域偏移**：下游模型在"理想输入"上训练，上线却吃上游的**真实输出**（框紧贴/偏移、长宽比失真），分布对不上 | 下游**必须在上游真实产出上训练**；对齐两段式的**几何归一化方式要一致**（居中保长宽比 vs 紧贴拉伸）。ch101 这么改后 0%→40.7%→83.3%。见 [第 101 章](../p20-projects/ch101-face-attendance.md)、[第 99 章](../p20-projects/ch99-ocr.md) |
| RANSAC 内点率漂亮（79%）、重投影误差极小（0.4px），但和真值一比差了几百像素 | **自洽的错模型**：自相似纹理（BRIEF 描述子在重复花纹上误匹配）让一堆错匹配彼此"自圆其说" | **内点漂亮 ≠ 模型对**。必须**独立真值复核**；选中等尺度、非重复纹理的匹配区。见 [第 106 章](../p20-projects/ch106-panorama.md) |
| 检测/识别只看 top-1 就下结论，实际召回很差 | 只报 top-1 掩盖了漏检/误配；以图搜图按位置乱配还能"top-1 对" | 报 **precision@K / mAP / 混淆矩阵**，不是单一 top-1。ch100 原始像素 precision@5 仅 45.3%（学习嵌入 98.6%）。见 [第 100 章](../p20-projects/ch100-image-search.md)、[第 48 章](../p09-detection/ch48-metrics-annotation.md) |
| 计时/测速/停留时长类事件，丢帧后时间全部偏移（ch97 920ms→880ms） | 用"帧序号 × 名义帧间隔"当时间戳，丢帧后累计漂移 | **用采集端真实时间戳**，不要用帧号推时间。见 [第 97 章](../p20-projects/ch97-realtime-pipeline.md) |
| 绊线/越线判断：脚点正好压线时漏判，或延长线上的点被误判越线 | ① 脚点压线 → 叉积 `d1·d2` 浮点退化到 0 → 端点退化漏判；② 把绊线当**无限直线**而非**线段** | ① 避开边界/放宽等号/加迟滞；② 线段相交要"**两组叉积严格异号**"（既判点在线两侧，也判投影落在线段内）。见 [第 87 章](../p18-events/ch87-tripwire.md)、[第 103 章](../p20-projects/ch103-perimeter-intrusion.md)、[第 104 章](../p20-projects/ch104-people-counting.md) |
| 关节角/姿态在肢体接近伸直时数值乱跳、计数误触发 | 三点求夹角在近共线时**病态**（分母趋 0，微小抖动放大成大角度变化） | **时间平滑** + **滞回状态机**（双阈值）。ch105 单阈值 0% → 滞回 97.5%。见 [第 105 章](../p20-projects/ch105-pose-counter.md) |
| 受限字符集 OCR，某一类被系统性认错（ch99 混淆几乎全压 0→O，21 次） | 该类的**唯一判别特征恰好落在成像退化会吃掉的地方**（0 与 O 只差内部一个暗点，轻微模糊即抹平） | 设计数据/预处理时，**专防"唯一区别落在会被退化吃掉之处"的类**；1/I、5/S 判别特征跨多像素反而糊不掉。见 [第 99 章](../p20-projects/ch99-ocr.md) |
| 字符粘连时连通域切分整串正确率崩到 1% | 字距为负 → 相邻字被**连通域并成一块**，这是纯连通域方案的**天花板** | 认清边界：粘连场景需投影切分/滑窗/序列模型（CRNN），连通域切不动。见 [第 99 章](../p20-projects/ch99-ocr.md) |
| `rqrr` 解码后按 `ecc_level` 序号查表，纠错等级全错 | `MetaData.ecc_level` 是格式信息**原始 2 bit**（M=0b00/L=0b01/H=0b10/Q=0b11），**不是** L<M<Q<H 的序号 | 按 2-bit 原值映射，别当序号。见 [第 98 章](../p20-projects/ch98-qrcode.md) |
| 训练/验证分数虚高，上线暴跌 | **数据划分泄漏**：同一视频相邻帧、同一 image_id 的框被随机拆进训练和验证 | 按**主体/视频/设备/时间/image_id 分组**划分。ch47 随机划分 mAP 虚高 24 个点；ch113 按 image_id 划分 0 泄漏 vs 随机 5 泄漏。见 [第 47 章](../p09-detection/ch47-train-detector.md)、[第 113 章](../p21-mastery/ch113-datasets-mlops.md) |
| 生成模型（扩散）只覆盖了目标分布的一部分（环形只生成一半） | **谱偏置**：网络偏爱低频，高频结构学得慢/学不全 | 加长训练/改网络容量/换噪声调度；用覆盖度指标而非只看单张好看。见 [第 71 章](../p15-generative/ch71-diffusion.md) |
| GAN 训练"稳"了但只会生成一两种样本 | **模式坍缩**：判别器/生成器博弈失衡 | 条件 GAN、稳定化技巧；用**召回/覆盖度**监控（ch70 朴素崩到 1/6 模式，条件 GAN 6/6）。见 [第 70 章](../p15-generative/ch70-gan.md) |

> **这一节的元教训**：**编译通过只是第一关**。合成数据要**故意造得能让差距显形**（划分泄漏、域偏移、退化尺度），指标要用**能暴露问题的那个**（precision@K、混淆矩阵、覆盖度，而非单一 top-1），级联系统要**逐级对齐几何与分布**。这套"怀疑精神"，比任何单个 API 都值钱——它正是 [附录 G](g-expert-standard.md) 说的"失败可解释、部署可核对"。

---

排查的顺序永远是：**先信编译器**（一二三四五六七节，它给的报错几乎都是真问题），**再信真值**（第八节，编译器管不了的，用独立复核和会暴露问题的指标去抓）。两关都过了，才敢说"这段代码对了"。配套的 crate 版本与纯度问题，回 [附录 A](a-crates.md)。
