# 第 109 章 GPU 加速：Metal、CUDA 与 wgpu 的性能模型

> **本章导读**
> - 学完你能：判断一个视觉任务**到底值不值得上 GPU**，并用**纯 Rust** 的 `wgpu` 写一个真正跑在显卡上的 **compute shader**（WGSL），把结果读回来和 CPU 对拍。
> - 建立 GPU 的**性能心智模型**：并行吞吐高 ≠ 延迟低；决定收益的是**算术强度**和**数据搬运**，而不是"核心多"。
> - 诚实盘点纯 Rust 的 GPU 生态：`wgpu`、`burn-wgpu`、以及非纯 Rust 但性能最强的 CUDA/`candle` 路线，各自的位置和短板。
> - 本章的 GPU 示例**在本机 Apple M1 Pro（Metal 后端）实测通过**，正文数字都是真实运行结果；但它**需要一个可用的 GPU 适配器**——headless / 无卡的 CI 环境可能拿不到，示例会优雅跳过而非崩溃。

## 场景：一张 4K 图，CPU 有点喘

[第 108 章](ch108-performance.md) 我们把 CPU 榨得很干：`rayon` 多核并行、SIMD、缓存友好的内存布局。但有些活儿，CPU 天生就吃力——比如对一张 4K 图（约 800 万像素）做一个稍重的逐像素滤波，或者把一个批次的图片喂进深度网络。这些计算有个共同点：**每个像素（或每个元素）的处理彼此独立**，谁也不等谁。

这正是 GPU 的主场。一颗现代 GPU 有**几千个小核心**，同一条指令能同时作用在成千上万个数据上。卷积、逐像素运算、矩阵乘法——这些视觉计算的骨干操作，**天然就是大规模并行的**。

但先别急着把所有东西都塞进 GPU。GPU 更像一辆**大巴**：一次能拉很多人（吞吐高），可你要先把人凑齐、等它发车、到站再把人放下来（延迟高、有固定开销）。只送一个人？打车（CPU）更快。这一章的核心，就是教你看清"什么时候该坐大巴"。

## 原理：GPU 快在哪，又慢在哪

### 直觉：几千个小核心，但要先"搬货"

CPU 有几个非常强的核心，擅长复杂的串行逻辑和低延迟响应；GPU 有海量弱核心，擅长"同一件事重复做几百万遍"。把灰度化想象成给一百万个像素每个都算一次加权和——CPU 得一个个（或十几个并行地）算过去，GPU 却能几乎"一把梭"。

可 GPU 有自己的内存，和 CPU（主机，host）的内存是分开的（device 内存）。要让 GPU 干活，得先把数据从 host **搬到** device，算完再把结果**搬回** host。一次 GPU 往返，时间大致是：

\\[ T_{\text{端到端}} \approx T_{\text{H2D 上传}} + T_{\text{kernel 计算}} + T_{\text{D2H 读回}} + T_{\text{启动与同步}} \\]

只有当 \\( T_{\text{kernel}} \\) 足够大、大到能盖过前后那几项搬运和启动开销，上 GPU 才划算。衡量这一点的关键词叫**算术强度（arithmetic intensity）**：每搬运一个字节的数据，能摊上多少次计算。灰度化的算术强度极低（搬 4 字节、只算一次加权和），深度卷积的算术强度很高（搬一次、算成百上千次）——**算术强度越高，GPU 的收益越大**。

### 什么时候上 GPU：一张判断表

| 场景 | 建议 | 理由 |
|------|------|------|
| 小图、单张、要**低延迟**（如实时单帧响应） | **CPU**（rayon/SIMD，见 [第 108 章](ch108-performance.md)） | 搬运和启动开销盖过收益 |
| 轻算子（灰度、阈值、加减） | 多半 **CPU**（独显上尤其） | 算术强度太低，搬运主导 |
| **大批量**图片、**大分辨率**、**重滤波** | **GPU** | 算术强度够高，吞吐碾压 |
| **深度模型**训练 / 推理 | **GPU** | 矩阵乘密集，正是 GPU 的主场 |
| 能把"预处理→推理→后处理"**串在 GPU 上**不中途读回 | **GPU** | 省掉来回搬运，收益最大 |

> 💡 **提示**：最贵的往往不是计算，而是**搬运**。工程上一条重要原则是：**数据一旦上了 GPU，就尽量让它待在 GPU 上**，把预处理、推理、后处理串起来跑完再一次性读回，别每个算子后面都往回搬一趟。

## 纯 Rust 的 GPU 生态（诚实盘点）

本书坚持纯 Rust。好消息是：**纯 Rust 真的能跑 GPU 通用计算，而且跨平台**。下面这张表把选项摊开讲清楚，包括哪些不纯、短板在哪。

| 方案 | 纯 Rust？ | 平台 | 定位与短板 |
|------|-----------|------|-----------|
| **`wgpu`** | ✅ | Metal / Vulkan / DX12 / GL / WebGPU | 写 **WGSL** compute shader 做通用并行计算，跨平台、可上浏览器（[第 110 章](ch110-wasm.md)）。**纯 Rust 通用 GPU 计算的首选**。短板：要自己写 kernel、管 buffer，比 CUDA 生态糙一些 |
| **`burn-wgpu`** | ✅ | 同 wgpu | burn 的 wgpu 后端。训练/推理**换个后端类型就上 GPU**，不用碰 shader（见下） |
| **`candle` + CUDA/Metal** | ⚠️ CUDA 非纯 | NVIDIA(CUDA) / Apple(Metal) | 性能强、生态偏 LLM；Metal 特性在 Apple 上纯度尚可，CUDA 路线绑 NVIDIA、依赖 C 运行库 |
| **`cust` / CUDA** | ❌ | 仅 NVIDIA | Rust 绑 CUDA，性能天花板最高，但**非纯 Rust、锁死 NVIDIA**，交叉编译/部署重 |
| **CPU 并行**（rayon/SIMD） | ✅ | 任意 CPU | 不是 GPU，但小任务/低延迟场景常常更优（[第 108 章](ch108-performance.md)） |

> 📖 **术语**：**WGSL（WebGPU Shading Language）**。给 GPU 写"每个核心都跑一遍"的小程序（kernel）用的语言，语法有点像 Rust。`wgpu` 把它编译到各平台的原生着色器（Metal 的 MSL、Vulkan 的 SPIR-V 等）。写 WGSL 的心智和写 CUDA C 不同：没有指针满天飞，靠**绑定（binding）**声明输入输出 buffer，靠 `@workgroup_size` 和内置的 `global_invocation_id` 定位"我是第几个核心、该处理哪个数据"。

### 最省事的纯 Rust 上 GPU：burn-wgpu 一行换后端

如果你的目标是**训练或推理一个网络**，根本不用自己写 shader。还记得 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 那个用 burn 训练的 CNN 吗？它的后端是这么定义的：

```rust
use burn::backend::{Autodiff, NdArray};
type B = Autodiff<NdArray>;   // 纯 Rust CPU 后端
```

想让**同一份网络代码、同一套训练循环**跑到 GPU 上，理论上只要把后端换成 wgpu：

```rust
use burn::backend::{Autodiff, Wgpu};
type B = Autodiff<Wgpu>;      // 换成 wgpu 后端 → 跑 Metal/Vulkan/DX12
```

网络结构、`forward`、优化器、`loss.backward()` **一个字都不用改**——这就是 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 反复强调的"Backend 泛型"设计带来的红利。（`Cargo.toml` 里相应把 feature 从 `ndarray` 换/加成 `wgpu`；burn 版本演进快，精确 feature 拼写以你所用版本的官方文档为准。）这是"纯 Rust 上 GPU"最省心的一条路。

## 动手：用 wgpu 写一个 compute shader

不过，为了真正理解 GPU 是怎么干活的，我们还是要亲手写一次原始的 compute shader。任务很简单：**把一张 RGBA 图在 GPU 上灰度化**，再读回来和 CPU 结果对拍。

> ⚠️ **注意**：本示例**需要一个可用的 GPU 适配器**（Metal / Vulkan / DX12 / GL）。它作为独立小工程放在 `code/gpu_demo/`（**不进主 `code/` 构建**，因为 wgpu 依赖较重、且平台相关）。在 headless / 无显卡 / 部分 CI 环境里可能拿不到适配器——代码里对这种情况**优雅退出并打印提示**，不会 panic。依赖版本（本机实测）：`wgpu = "30.0.1"`、`pollster = "1.0.1"`、`bytemuck = "1.25.2"`。

### 第一步：WGSL kernel——每个核心处理一个像素

```wgsl
@group(0) @binding(0) var<storage, read>       inp: array<u32>;   // 输入：打包的 RGBA
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;  // 输出

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) { return; }   // 尾部越界保护（总核心数是 64 的倍数）
    let px = inp[i];
    let r = f32(px & 0xffu);
    let g = f32((px >> 8u) & 0xffu);
    let b = f32((px >> 16u) & 0xffu);
    let y = u32(0.299 * r + 0.587 * g + 0.114 * b);  // 经典亮度加权
    outp[i] = y | (y << 8u) | (y << 16u) | (0xffu << 24u);
}
```

关键点：`@workgroup_size(64)` 说"每 64 个核心编一组"，`global_invocation_id.x` 就是"我是全局第几个核心"——我们让第 `i` 个核心处理第 `i` 个像素。因为总核心数会向上取整到 64 的倍数，**必须**用 `if (i >= arrayLength(...))` 把尾部多出来的核心挡掉。

### 第二步：Rust 侧的五步骤骨架

Rust 侧的流程是固定套路：**拿适配器 → 建 device/queue → 建 buffer → 建 pipeline+bind group → dispatch → 读回**。核心骨架如下（完整可运行版本见 `code/gpu_demo/src/main.rs`）：

```rust
// 1) 拿适配器：headless/无卡会 Err —— 优雅退出，不 panic
let instance = wgpu::Instance::default();
let adapter = match instance.request_adapter(&Default::default()).await {
    Ok(a) => a,
    Err(e) => { eprintln!("拿不到 GPU 适配器：{e:?}（CI/无卡环境属正常，跳过）"); return; }
};
let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

// 2) 三个 buffer：输入、输出（可拷贝）、读回（可 map 回 CPU）
let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("in"), contents: bytemuck::cast_slice(&input),
    usage: wgpu::BufferUsages::STORAGE });
// out_buf: STORAGE | COPY_SRC ; read_buf: MAP_READ | COPY_DST（略，见完整工程）

// 3) 把 WGSL 编成 compute pipeline（layout=None 让 wgpu 自动推断绑定布局）
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: None, source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: None, layout: None, module: &shader,
    entry_point: Some("main"), compilation_options: Default::default(), cache: None });

// 4) dispatch：向上取整覆盖所有像素；每 64 像素一个 workgroup
let mut enc = device.create_command_encoder(&Default::default());
{
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(((n + 63) / 64) as u32, 1, 1);
}
enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes);  // D2H 拷贝
queue.submit(Some(enc.finish()));

// 5) map + poll(Wait) 阻塞到 GPU 完成，再读回
let slice = read_buf.slice(..);
slice.map_async(wgpu::MapMode::Read, |_| {});
device.poll(wgpu::PollType::wait_indefinitely()).ok();
let gpu: &[u32] = bytemuck::cast_slice(&slice.get_mapped_range().unwrap());
```

> 🕳️ **坑（wgpu 30 的 API 变动）**：`request_adapter` 现在返回 `Result`（不是老资料里的 `Option`）；`device.poll(...)` 的参数从 `Maintain::Wait` 变成了 `PollType::wait_indefinitely()`；`get_mapped_range()` 返回 `Result` 要 `unwrap()`；`entry_point` 是 `Option<&str>`。wgpu 版本演进快，跨版本这些签名常变——**认准你锁定的版本、对照当时的文档**。

### 跑起来：真实数字（Apple M1 Pro，Metal）

为了看清"算术强度决定收益"，`gpu_demo` 里还放了一个**重算子**：每像素做 300 次三角迭代（模拟重预处理），和轻的灰度算子对比。在本机（Apple M1 Pro，统一内存）实测：

```text
适配器：Apple M1 Pro （后端 Metal，类型 IntegratedGpu）
图像：1920x1080 = 2073600 像素

[轻·灰度]
  GPU 冷启动首次 = 3.89ms （含 shader 编译/驱动预热）
  GPU 预热后     = 0.36ms   CPU（向量化）= 0.79ms   加速 = 2.20x
  GPU↔CPU 最大差 = 1（f32 舍入，正常）

[重·每像素 300 次迭代]
  GPU 冷启动首次 = 13.70ms
  GPU 预热后     = 11.61ms   CPU = 10270.52ms   加速 = 884.47x
  GPU↔CPU 最大差 = 16（300 次混沌迭代放大了两端数学库的微小差异）
```

这四组数字，每一个都在讲一条道理，值得逐条读懂：

- **冷启动 vs 预热**：灰度算子第一次跑要 `3.89ms`（含 shader 编译、pipeline 建立、驱动预热），预热后只要 `0.36ms`——差了十倍。**这就是为什么 benchmark 必须先预热、再多次取样**（呼应 [第 108 章](ch108-performance.md)）。拿冷启动那一次的时间下结论，是新手最常见的误判。
- **轻算子 GPU 竟然赢了？** 预热后灰度 GPU（`0.36ms`）比向量化 CPU（`0.79ms`）快约 `2.2x`。但请特别注意：**本机是统一内存（unified memory）**，CPU 和 GPU 共享同一块物理内存，所谓 host↔device"拷贝"几乎免费。**换成独立显卡（discrete GPU，走 PCIe）就完全是另一回事**：灰度这种轻算子往往会**输给 CPU**——因为一次 H2D + 一次 D2H 要各搬约 8MB 数据过 PCIe，这点搬运时间足以淹没那点微不足道的计算。这就是**收支平衡点（break-even）**：算子越轻、数据越小，越不值得上 GPU。（我手头没有独显不能实测，但这是众所周知的经验法则。）
- **重算子 GPU 碾压**：每像素 300 次迭代，CPU 要 `10.3` 秒，GPU 只要 `11.6` 毫秒——快了近 **900 倍**。算术强度一高，搬运开销瞬间变得微不足道，GPU 的几千核心全面发力。**这才是 GPU 真正该干的活。**
- **精度要对拍**：轻算子最大误差是 `1`（f32 加权和的舍入差），可以接受；重算子却飙到 `16`——300 次 `sin/cos` 的混沌迭代，把 GPU 数学库和 CPU 数学库之间**每一步的微小差异层层放大**了。这提醒你：**GPU 的结果永远要和一个 CPU 参考实现对拍，且要用容差而非精确相等**；误差可能随计算深度累积。

> 💡 **提示**：`gpu_demo` 用 `if want { ... }` 只在需要时才把数据拷成 `Vec` 对拍，计时循环里则跳过拷贝——**别把不属于被测路径的开销算进计时**（同样呼应 [第 108 章](ch108-performance.md) 的基准纪律）。

## 常见坑与提示

> 🕳️ **坑**：**小任务上 GPU，纯亏。** 单张小图做个轻算子，搬运 + 启动开销远大于计算收益。判断标准是算术强度，不是"GPU 核心多"。拿不准就两条路都测一遍、按端到端时间说话。

> ⚠️ **注意**：**benchmark 必须预热。** 首次 dispatch 包含 shader 编译、pipeline 构建、驱动初始化（本例 `3.89ms` vs 预热后 `0.36ms`）。测 GPU 一定要先空跑几次预热，再多次取样报告（best-of-N 或中位数 + 方差）。

> 🕳️ **坑**：**f32/f16 精度差异，结果务必和 CPU 对拍。** GPU 的浮点实现、算子顺序、`sin/cos` 等超越函数的近似，都可能和 CPU 略有出入。浅算子差 1 个灰阶无所谓，深流水线（或用了 f16/bf16）误差会累积。**永远保留一份 CPU 标量参考实现，用容差比对**，别指望逐位相等。

> 🕳️ **坑**：**适配器不一定拿得到。** headless 服务器、部分容器、无显卡的 CI，`request_adapter` 会失败。生产代码要**优雅降级到 CPU 路径**，而不是直接 panic。本章示例正是这么处理的。

> ⚠️ **注意**：**WGSL 不是 CUDA。** 没有裸指针和 `malloc`，靠 `@group/@binding` 声明 buffer、靠 `workgroup_size` 和 `global_invocation_id` 定位数据。从 CUDA 迁移过来的读者要转换心智；好处是同一份 WGSL 能跑遍 Metal/Vulkan/DX12，甚至浏览器（[第 110 章](ch110-wasm.md)）。

## 专家补强：GPU 算子的验收

> 本节呼应 [附录 G](../appendix/g-expert-standard.md) 的证据标准——GPU 代码"能跑"只是起点，**能量化收益、能对拍精度、能优雅降级**才算掌握。

真正把一个 GPU 算子交付到生产，至少要做到：

- **带一个 CPU 标量参考实现**，用**随机尺寸、随机 stride、边界与退化输入**做差分测试（不能只测一张规整图）。
- **分段计时**：分别测 H2D 上传、kernel 计算、D2H 读回和端到端，报告 **break-even 图像尺寸**、P95、峰值显存和数值误差——而不是笼统一句"快了 X 倍"。
- **重叠传输与计算**：用双缓冲（double buffering）让"上传下一帧"和"计算当前帧"在时间线上重叠，并**证明它们确实重叠了**（看时间线/计数器，而非只看总时间）。
- **优雅降级**：适配器不可用、算子不支持时，回退到 CPU 路径，别让整条服务崩掉。

kernel 层面的调优要看 occupancy、寄存器/共享内存用量、访存是否合并（coalesced）、分支发散（divergence）——但记住**occupancy 高不等于一定快**，最终以时间线和硬件计数器为准，别凭感觉。

## 本章小结

- GPU 的优势是**高并行吞吐**，不是低延迟；一次往返的成本是 **H2D + kernel + D2H + 启动同步**。决定值不值得上 GPU 的是**算术强度**，不是核心数。
- 纯 Rust 的 GPU 路线：通用计算用 **`wgpu`**（写 WGSL compute shader，跨平台）；训练/推理用 **`burn-wgpu`**——**一行换后端**即可让 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 的网络跑上 GPU。CUDA/`candle` 性能最强但（CUDA）非纯 Rust、绑 NVIDIA。
- 本机实测（M1 Pro / Metal / 统一内存）：轻的灰度算子预热后 `0.36ms`、比 CPU 快 `2.2x`（**但独显上轻算子通常会输**）；重算子快近 **900x**。冷启动 `3.89ms` 说明**必须预热再测**；最大误差 `1`→`16` 说明**必须和 CPU 对拍、用容差**。
- 工程铁律：**数据尽量留在 GPU 上**、**小任务别上 GPU**、**结果一定对拍**、**适配器不可用要优雅降级**。

## 动手练习

1. **跑通并读数**：进入 `code/gpu_demo/` 跑 `cargo run --release`，记录你机器的适配器名、冷/暖时间和加速比。如果拿不到适配器（headless），观察它是否优雅退出——这本身就是一次"降级路径"验证。
2. **找 break-even**：把图像尺寸从 `256x256` 一路加到 `4096x4096`，画出"轻算子 GPU 端到端 vs CPU"的两条曲线，找出你机器上灰度算子的收支平衡点（独显和统一内存的拐点会很不一样）。
3. **加个阈值 kernel**：仿照灰度 WGSL 写一个二值化 kernel（`y > 阈值 ? 255 : 0`），同样和 CPU（[第 16 章](../p03-classic/ch16-threshold.md) 的阈值）对拍，确认最大误差为 0。
4. **进阶——换 burn 后端**：把 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 的训练工程后端从 `NdArray` 换成 `Wgpu`（改类型 + 改 feature），比较相同轮数下的训练时间和最终准确率，体会"一行换后端"的威力与代价。

下一章，我们把纯 Rust 的另一张王牌亮出来：**编译到 WebAssembly，让视觉算法直接在浏览器里跑**。👉 [第 110 章 编译到 WebAssembly，在浏览器跑视觉算法](ch110-wasm.md)
