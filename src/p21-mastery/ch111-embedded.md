# 第 111 章 边缘与嵌入式部署：资源、热设计与故障恢复

> **本章导读**
> - 想清楚一件事：视觉程序很少跑在豪华服务器上，更多是塞进**摄像头盒子、树莓派、Jetson、工控机**——资源受限、还得 7×24 稳定运行。
> - 认识 Rust 在边缘的四张王牌：**无 GC、内存占用小、单个静态二进制、交叉编译强**。这正是它在这里赢的原因。
> - 学会**交叉编译**（含实测的链接坑与 `cross` 解法）、**面向边缘的体积瘦身配置**，以及用纯 Rust 的 `tract` 在边缘加载 ONNX 做推理的思路。
> - **诚实**划清边界：`no_std` 到底意味着什么，`image`/`imageproc` 为什么在真正的裸机单片机上跑不了。

## 场景引入：把桌面上的检测程序塞进巴掌大的盒子

你在开发机上用 Rust 写好了一个入侵检测程序，跑得又快又稳。现在要把它部署到工地围栏上的**边缘盒子**里——一块基于 ARM64 芯片、512 MB 内存、无风扇、密封防尘的小板子，装在铁皮箱里日晒雨淋。

麻烦立刻来了：这块板子的 CPU 架构（`aarch64`）和你的开发机不一样；它内存小得可怜，跑不动臃肿的运行时；它没有散热风扇，CPU 一热就降频；它可能几个月没人管，断电、断网、磁盘写满都得自己扛过去。

这套约束，恰恰是 Rust 的主场。回顾 [第 2 章](../p01-start/ch02-why-rust.md)：**没有 GC 就没有停顿**（实时视频不掉帧），**编译产物是单个静态二进制**（拷过去就能跑，不用在盒子上装一堆依赖），**交叉编译是一等支持**（在你的开发机上就能编出 ARM64 的可执行文件）。带 GC 的语言在这里会被内存和停顿拖累，Python 那套解释器 + 依赖更是没地方装。

## 原理：为什么边缘是 Rust 的主场

先用一张表把优势说透（都从"边缘"这个视角看）：

| 维度 | 带 GC / 解释器的语言 | Rust |
|------|----------------------|------|
| 运行时停顿 | GC 的 stop-the-world 会让实时视频掉帧 | 无 GC，延迟可预测 |
| 内存占用 | 运行时 + GC 堆，基线就偏高 | 小而可控，适合几百 MB 的盒子 |
| 交付方式 | 要带解释器 / 虚拟机 / 一堆动态库 | **单个（可静态链接的）二进制**，拷过去就跑 |
| 交叉编译 | 折腾，常要在目标机上重装环境 | 一等支持，一条命令编出异构架构产物 |

### 交叉编译：在 A 机器上编出 B 机器的程序

**交叉编译（cross compilation）** 就是在一种架构的机器（host，你的开发机）上，编出能在另一种架构机器（target，边缘盒子）上运行的程序。Rust 用 **target 三元组** 描述目标平台，比如树莓派 64 位系统是 `aarch64-unknown-linux-gnu`。

装目标平台的标准库很简单：

```bash
rustup target add aarch64-unknown-linux-gnu
```

但你满心欢喜地直接编译，往往当场翻车。本机实测（macOS 开发机编 ARM64 Linux）：

```bash
cargo build --release --target aarch64-unknown-linux-gnu
```

```text
error: linking with `cc` failed: exit status: 1
  ld: unknown options: --as-needed -Bstatic -Bdynamic --gc-sections ...
```

> 📖 **术语**：这个错误的根源是**链接器（linker）**。`rustc` 能顺利把 Rust 代码编成 ARM64 机器码，但最后一步"链接"要用到**目标平台的链接器和 C 运行库（libc）**。你开发机上的链接器（这里是 Apple 的 `ld`）根本不认识 GNU 风格的链接参数，于是报"unknown options"。缺的不是 Rust，是**目标平台的工具链和系统库**。

解决办法有两条。最省心的是 **`cross`**——它用 Docker 容器封装好了各目标平台的交叉工具链和 libc，你几乎零配置：

```bash
cargo install cross                 # 需要本机装好 Docker / Podman
cross build --release --target aarch64-unknown-linux-gnu
```

另一条是手动装交叉工具链（如 `aarch64-linux-gnu-gcc`），再在 `.cargo/config.toml` 里指定链接器。能用 `cross` 就优先用 `cross`，省下大把折腾时间。

> 💡 **提示**：想要一个**完全静态、不挑目标机 glibc 版本**的二进制，用 musl 目标：`aarch64-unknown-linux-musl`。它把 libc 也静态链进去，编出来的二进制拷到任何 ARM64 Linux 上都能跑，彻底告别"目标机 glibc 太老"的链接/运行报错。这正是 Rust "单文件部署"在边缘最爽的兑现。

## 边缘推理选型：tract 还是 ort

边缘设备上跑深度模型，纯 Rust 生态里 **`tract`（Sonos 出品）是交叉编译最省心的选择**。原因很直接：它是**纯 Rust、零 C/C++ 依赖**的 ONNX 推理引擎——自己用 Rust 解析 `.onnx` 并在 Rust 图上推理。这意味着上面那套 `cross build` 能把它和你的程序一起干净地编成一个 ARM64 二进制，**没有额外的 C++ 运行库要交叉编译**。

对比之下，`ort`（ONNX Runtime 的 Rust 绑定）**底层链接 C++ 的 onnxruntime**：它推理性能和算子覆盖最强，但在边缘要么得为目标平台找到 / 交叉编译对应的 C++ 运行库，要么依赖它下载预编译库——在冷门 ARM 板子上常常没有现成的，折腾成本陡增。选型逻辑详见 [第 39 章](../p07-dl-basics/ch39-dl-ecosystem.md)，ONNX 推理的完整用法见 [第 94 章](../p19-deploy/ch94-onnx.md)。

> 💡 **提示**：一句话决策——**边缘 + 要纯 Rust 交叉编译省心 → tract；能接受 C++ 依赖、要极致性能 / 全算子 / GPU → ort。** 大多数边缘视觉盒子跑的是中小模型，`tract` 足够。

### 模型瘦身：让模型装得进盒子

边缘设备内存和算力都紧张，动辄几百 MB 的浮点模型往往塞不下、也跑不动。**量化（quantization）** 是最主流的瘦身手段：把 32 位浮点权重压成 8 位整数（INT8），模型体积和内存占用直接降到约 1/4，整数运算在很多边缘芯片上还更快。代价是轻微精度损失，通常可接受。量化的原理、工具与精度权衡，本书 [第 95 章](../p19-deploy/ch95-quantization.md) 专门讲，这里只提醒你：**边缘部署，先想清楚模型能不能瘦下来**。

## 动手：交叉编译 + 边缘瘦身配置 + tract 推理骨架

**① 交叉编译命令**（前面已给，汇总）：

```bash
rustup target add aarch64-unknown-linux-gnu   # 装目标平台标准库
cargo install cross                           # 装 cross（需 Docker）
cross build --release --target aarch64-unknown-linux-gnu
# 产物：target/aarch64-unknown-linux-gnu/release/<你的程序>，直接拷到盒子上跑
```

**② 面向边缘的 `[profile.release]` 瘦身配置**（放进你程序的 `Cargo.toml`）：

```toml
[profile.release]
opt-level = "z"     # 体积优先（追求速度可用 3；边缘常缺存储，'z' 更稳妥）
lto = true          # 链接期优化，跨 crate 消冗余，显著减小体积
codegen-units = 1   # 只用一个代码生成单元，优化更充分（编译更慢，换更小更快的产物）
strip = true        # 去掉符号信息，二进制更小
panic = "abort"     # panic 直接中止，不生成栈展开表，再省一截体积
```

这套配置能把二进制显著压小——对存储只有几 GB、还要留给日志和模型的边缘盒子很关键。

**③ tract 在边缘加载 ONNX 推理的思路骨架**（tract-onnx 0.23.5，纯 Rust；完整代码见 [第 94 章](../p19-deploy/ch94-onnx.md)，此处只给结构）：

```rust
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    // 1) 加载 .onnx，声明输入的 dtype 与形状，做图优化，得到可执行模型
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?
        .into_optimized()?
        .into_runnable()?;

    // 2) 准备输入张量（实际中由摄像头帧预处理而来：缩放、归一化、HWC->CHW）
    let input: Tensor = Tensor::zero::<f32>(&[1, 3, 224, 224])?;

    // 3) 推理，取回输出（0.23 里用 to_plain_array_view，不是旧版的 to_array_view）
    let result = model.run(tvec!(input.into()))?;
    let output = result[0].to_plain_array_view::<f32>()?;
    println!("输出形状 = {:?}", output.shape());
    Ok(())
}
```

关键在于：这段代码连同它的依赖，**能被 `cross build` 一起干净地交叉编译成 ARM64 二进制**——因为 `tract` 全程纯 Rust，没有任何 C/C++ 库拖后腿。把这个二进制和 `model.onnx` 拷到盒子上，就能跑推理了。

## no_std 的诚实边界

你可能听过 `no_std`——让 Rust 程序脱离标准库运行。这里必须**诚实划清一条界线**，否则容易踩空。

> ⚠️ **注意**：本章说的"边缘设备"（树莓派、Jetson、工控机、大多数摄像头盒子）都**跑着 Linux、有完整的 `std`**。在它们上面，`image`、`imageproc`、`tract` 都能正常用，你**根本不需要 `no_std`**——交叉编译好、拷过去就行。

真正需要 `no_std` 的是**裸机单片机（MCU）**，比如 Cortex-M 系列——它们没有操作系统、内存以 KB 计、通常连堆分配都要自己配。那是另一个世界：

> 🕳️ **坑**：`image`、`imageproc`、`ndarray`（默认）、`tract` **都依赖 `std`**，在 `no_std` 的 MCU 上**编不过**。想在单片机上做视觉，你要么手写极简的逐像素处理（在 `[u8]` 缓冲上直接算），要么用嵌入式专用生态（如 `embedded-graphics` 做显示）。别指望把本书的视觉栈原样搬到单片机上——那不是"配置一下"能解决的，是整条依赖链的问题。一句话：**有 Linux 的边缘盒子，Rust 视觉栈畅通无阻；裸机 MCU 视觉，是需要另起炉灶的专门领域。**

## 常见坑与提示

> 🕳️ **坑：glibc 版本不匹配**。用 `aarch64-unknown-linux-gnu` 编的二进制，拷到 glibc 更老的目标机上会报 `GLIBC_x.xx not found`。对策：用 `cross`（它用较老的基础镜像保证兼容），或干脆上 **musl 静态链接**（`aarch64-unknown-linux-musl`）彻底绕开 glibc 依赖。

> ⚠️ **注意：内存与浮点不足**。边缘盒子内存小，大模型或大分辨率一上来就 OOM（内存耗尽）被系统杀掉。先算好预算：模型权重 + 推理中间张量 + 图像缓冲要装得下。部分低端 MCU 甚至没有硬件浮点单元（FPU），浮点运算奇慢——这也是**量化到整数**在边缘受欢迎的原因之一。

> 🕳️ **坑：实时性不达标**。30 FPS 意味着每帧预算只有 \\( 1000 / 30 \approx 33 \\) 毫秒，从采集、预处理、推理到后处理全得塞进去。别只看开发机上的速度，要在**目标设备**上实测端到端延迟；跟不上就降分辨率、抽帧、或换更小的模型。

> ⚠️ **注意：功耗与散热被严重低估**。密封无风扇的盒子里，CPU/NPU 满载几分钟就会**热降频（thermal throttling）**——你在实验室测的峰值性能，装进铁皮箱晒太阳后可能腰斩。做容量规划要按**持续散热后的稳态吞吐**来算，而不是开机头几秒的峰值。散热、功耗预算是边缘部署绕不开的物理约束。

## 本章小结

- 边缘/嵌入式（树莓派、Jetson、工控机、摄像头盒子）资源受限、要长期稳定运行，正是 Rust 的主场：**无 GC、内存小、单静态二进制、交叉编译强**。
- 交叉编译先 `rustup target add`，直接 `cargo build` 常在**链接期**失败（缺目标链接器 + libc）；用 **`cross`**（Docker 封装工具链）最省心，追求跨机兼容可上 **musl 静态链接**。
- 边缘推理选型：**`tract` 纯 Rust、零 C++ 依赖、交叉编译最省心**（首选）；`ort` 性能强但链接 C++ 库、边缘更折腾。模型太大就靠**量化**（[第 95 章](../p19-deploy/ch95-quantization.md)）瘦身。
- 用 `[profile.release]` 的 `opt-level='z'` / `lto` / `codegen-units=1` / `strip` / `panic='abort'` 把二进制压小。
- **诚实边界**：有 Linux 的边缘盒子有 `std`，视觉栈畅通、不需要 `no_std`；`image`/`imageproc`/`tract` 都依赖 `std`，真正的裸机 MCU 视觉是另起炉灶的专门领域。

## 动手练习

1. **编一个 ARM64 二进制**：`rustup target add aarch64-unknown-linux-gnu`，然后对一个最小的 Rust 程序尝试 `cargo build --target ...`。观察是否在链接期报错，再用 `cross build` 对比。
2. **量瘦身效果**：把一个用到 `image` 的小程序，分别用默认 release 和本章的 `[profile.release]` 瘦身配置各编一次，比较二进制体积。
3. **画部署预算表**：给一个"512 MB 内存、无风扇 ARM64 盒子、要跑 15 FPS 检测"的场景，列出内存预算（模型 + 张量 + 图像缓冲）、每帧时间预算、以及散热降频后的性能余量。
4. **想一想**：同样一个 ONNX 模型，用纯 Rust 的 `tract` 交叉编译 vs 用链接 C++ 的 `ort`，在一块冷门 ARM 板子上，各会遇到什么不同的麻烦？为什么说 `tract` 在这里"省心"？

> 下一章我们收尾"通往精通"这条路，聊聊把这一切串成可维护工程的**最佳实践**：项目结构、错误处理、测试与 CI/CD。👉 [第 112 章 工程化最佳实践](ch112-engineering.md)
