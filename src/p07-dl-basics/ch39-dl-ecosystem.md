# 第 39 章 Rust 深度学习生态全景（candle/burn/tch/tract/ort）

> **本章导读**
> - 一次性认全 Rust 深度学习圈最主流的五个库：**candle、burn、tch、tract、ort**——分别谁出品、干什么、纯不纯 Rust。
> - 抓住两根最关键的标尺：**纯 Rust vs 依赖 C++**、**训练 vs 推理**。选型焦虑，一大半来自没先分清这两件事。
> - 拿到一张"典型场景 → 选哪个"的决策地图，并明确本书立场：**能纯 Rust 就纯 Rust，需要极致性能时才务实地上 C++ 后端，并讲清代价**。
> - 本章几乎没有代码（只有几行依赖和一段张量小实验），重在建立"生态大局观"。

## 场景：模型有了，用哪个库把它跑起来？

假设你手里有一个训练好的模型——可能是从 Hugging Face 下载的图像分类网络，也可能是同事导出的一个 `.onnx` 文件。现在你想用 Rust 把它跑起来，于是打开 crates.io 搜 "deep learning"，结果一排名字糊你一脸：`candle`、`burn`、`tch`、`tract`、`ort`、`dfdx`、`candle-onnx`……每个都说自己"高性能""易用"，到底该 `cargo add` 哪一个？

这正是新手最容易卡住的地方。好消息是：**这些库并不是在做同一件事的竞品，而是各自占着不同的生态位。** 只要先想清楚两个问题，选择范围立刻就从"五选一的焦虑"收敛成"照着地图走"。这两个问题就是本章的两根标尺。

## 先立两根标尺

**标尺一：纯 Rust，还是依赖 C++？**

[第 2 章](../p01-start/ch02-why-rust.md) 讲过，本书坚持"纯 Rust"是有实打实好处的：一条 `cargo build` 就能编译，不用在机器上折腾 C++ 编译器和动态库；产物是一个独立可执行文件，往服务器、边缘设备甚至浏览器（WebAssembly，见 [第 110 章](../p21-mastery/ch110-wasm.md)）上搬都省心；交叉编译（比如在 Mac 上编译出树莓派能跑的程序）几乎无痛。

而一旦某个库"底层是 C++"（典型如 `tch` 依赖 LibTorch、`ort` 依赖 onnxruntime），你就得额外准备好那套 C++ 运行库，交叉编译和静态链接也会变麻烦。换来的通常是**更强的性能、更全的算子**。所以纯不纯 Rust，本质是一道 **可移植性 vs 性能/生态** 的取舍题。

**标尺二：你要训练，还是只推理？**

- **训练（training）**：从零学一个模型，或在已有模型上继续学。需要**自动微分（autodiff）**、优化器、数据加载这一整套（下一章 [第 40 章](ch40-burn-train.md) 就会用 burn 真练一个）。
- **推理（inference）**：模型已经练好，你只负责"喂进去一张图、拿出来结果"。不需要反向传播，追求的是加载快、跑得快、依赖少。

想清楚这两点，五个库的定位就一目了然了。

## 五个主角逐个认识

### candle 0.11.0：Hugging Face 的纯 Rust 张量库

`candle` 由 **Hugging Face** 出品，是一个**纯 Rust** 的张量计算库，风格上很像"轻量版的 PyTorch 内核"。它原生支持 **safetensors** 这种安全、快速的权重格式（这正是 HF 上模型的标配），CPU 能跑，开了对应 feature 还能上 CUDA / Metal。它的主场是 **加载现成权重做推理、手搓 Transformer、跑大语言模型（LLM）与微调**。本书 [第 42 章](../p08-classification/ch42-candle-classify.md) 用它加载预训练模型做图像分类，就是典型用法。

> 📖 **术语**：**safetensors**。一种存模型权重的文件格式，本质是"一堆命名张量 + 一份描述形状/类型的头部"。相比 Python 老用的 `pickle`，它加载快、还不会在加载时执行任意代码（更安全），所以 HF 生态几乎都用它。

### burn 0.21.0：纯 Rust 的"PyTorch"

`burn` 是一个**纯 Rust**、面向**训练**的深度学习框架，体验上最接近 PyTorch：你定义网络结构、用自动微分求梯度、用优化器更新参数。它最有特色的设计是 **可插拔后端（backend）**——同一套模型代码，底层后端可以从"纯 Rust CPU（`ndarray`）"一键换成"跨平台 GPU（`wgpu`）"甚至复用 candle，训练时再套一层 **Autodiff** 装饰器就自动获得反向传播能力。想在 Rust 里"从零训练"，burn 是当前最"框架化"的选择，[第 40 章](ch40-burn-train.md) 会带你用它练一个识别手写数字的 CNN。

### tch：LibTorch 的 Rust 绑定（强大，但拖着 C++）

`tch` 是 **PyTorch 官方 C++ 库 LibTorch 的 Rust 绑定**。因为直接调用 LibTorch，它**功能最全、和 PyTorch 生态无缝**——PyTorch 里有的算子、能加载的 `.pt` 模型，它基本都吃得下。代价只有一句话，但很关键：**它不是纯 Rust，用之前你得先在机器上装好几百 MB 的 LibTorch，并配好库路径。** 对本书"一条 `cargo build` 搞定"的目标来说，这是明显的摩擦点，所以本书**不把 tch 作为主线**，只在这里让你认得它：当你已经重度绑定 PyTorch 生态、非用它不可时，知道 Rust 侧有这么个绑定即可。

> ⚠️ **注意**：本书没有把 tch 的版本固定下来实测（它要拉 LibTorch，安装成本高）。真要用，请自己 `cargo add tch` 并对照它当时依赖的 LibTorch 版本——两者版本必须匹配，否则链接会失败。

### tract 0.23.5：Sonos 的纯 Rust ONNX 推理引擎

`tract` 由做语音产品的 **Sonos** 开源，是一个**纯 Rust、零 C/C++ 依赖**的**推理**引擎，能直接加载 **ONNX**（和 NNEF）模型。它自己用 Rust 解析 `.onnx` 文件、在 Rust 构建的计算图上跑推理，因此**交叉编译、静态链接、上边缘设备都极其友好**——这正是"要纯 Rust 跑一个现成 ONNX 模型"时的首选。它的算子覆盖很广，但并非 100%，个别很新的模型/算子可能还没支持，遇到再考虑下面的 ort。

> 📖 **术语**：**ONNX（Open Neural Network Exchange）**。一种**跨框架的模型交换格式**。你在 PyTorch / TensorFlow 里训练的模型，都能导出成一个 `.onnx` 文件，再由 tract、ort 等各种推理引擎加载。可以把它理解成深度学习界的"PDF"——大家都能打开的通用格式。它是打通"训练框架"和"部署引擎"的桥梁，本书 [第 94 章](../p19-deploy/ch94-onnx.md) 会专门讲怎么用它做跨框架部署。

### ort 2.0.0-rc.13：ONNX Runtime 绑定（最快最全，代价是 C++）

`ort` 是微软 **ONNX Runtime 的 Rust 绑定**。它同样跑 ONNX 模型，但**底层链接的是 C++ 写的 onnxruntime**——所以它**不是纯 Rust**。换来的是这一票里**最强的推理性能、最全的算子覆盖，以及对 GPU / 各种执行后端（Execution Provider）的支持**。当你要跑很大的模型、或把性能榨到极致、又能接受"多带一个 C++ 运行库"时，ort 就是那个务实的选择。

> 🕳️ **坑**：**ort 2.x 还在 RC（候选发布）阶段，API 仍在变。** 它和 1.x 的 API 差异极大（1.x 用 `Environment` + `Value::from_array` 那一套），而且 2.x 各个 rc 之间，连 `inputs!` 宏要不要加 `?`、`try_extract_*` 方法叫什么名字都还在调整。真正写代码时，请**认准你锁定的具体版本号**（本书成书时是 `2.0.0-rc.13`），并对照当时的 docs.rs 复核签名，别照抄网上的旧例子。

## 一张表看懂版本与纯度

下面这张速览表把关键信息浓缩到一起（版本号均为本书成书时 `cargo` 实测解析值）：

| crate | 版本 | 纯 Rust？ | 定位 | 出品方 |
|---|---|---|---|---|
| `candle` | **0.11.0** | ✅ 纯 Rust | 张量库 / 加载 safetensors / LLM 与手搓模型、推理 | Hugging Face |
| `burn` | **0.21.0** | ✅ 纯 Rust | 多后端 + 自动微分的**训练**框架（类 PyTorch） | Tracel AI |
| `tract` | **0.23.5** | ✅ 纯 Rust（无 C/C++） | **ONNX 推理**引擎，交叉编译友好 | Sonos |
| `ort` | **2.0.0-rc.13** | ❌ 底层 C++ onnxruntime | ONNX 推理，性能与算子**最强** | ONNX Runtime |
| `tch` | 未固定（需装 LibTorch） | ❌ 底层 C++ LibTorch | PyTorch 绑定，功能最全 | LibTorch |

> 💡 **提示**：**别被版本号"劝退"。** 你可能会嘀咕："candle 才 0.11、burn 才 0.21，是不是还不成熟、不能用？" 在 Rust 生态里，`0.x` 只是说明它们还没承诺"API 永不破坏性变更"，**不代表不好用或不稳定**——这几个库都已经在真实项目里跑大模型、上生产了。真正要留心的反倒是 ort 那种 `rc`（候选发布）版本，因为它明确处在"还在定型"的阶段。

## 选型：一张决策地图

先给结论表（可直接当查询手册）：

| 需求场景 | 推荐 | 纯 Rust？ | 一句话理由 |
|---|---|---|---|
| 跑现成 ONNX 模型，要**纯 Rust、无 C++**、方便交叉编译/上边缘 | **tract** | ✅ | 直接加载 `.onnx` 推理，零外部运行库 |
| 跑**大模型 / 追求极致性能 / 要算子全 / 要 GPU** | **ort** | ❌ (C++) | 性能与算子覆盖最强，代价是链接 C++ |
| **加载 safetensors、手搓/微调模型（LLM、Transformer）** | **candle** | ✅ | HF 生态、safetensors 原生，CPU/CUDA/Metal |
| **从零训练，要自动微分 + 多后端、想要 PyTorch 式体验** | **burn** | ✅ | 泛型后端 + Autodiff，训练为主 |
| 已重度依赖 PyTorch 生态、非 LibTorch 不可 | tch | ❌ (C++) | 和 PyTorch 无缝，但要装 LibTorch |

把它画成一条"顺着问下来"的决策路径，更好记：

```text
你要干嘛？
├─ 只是把现成模型跑起来（推理）
│   ├─ 手上是 .onnx 文件
│   │   ├─ 要纯 Rust / 交叉编译 / 上边缘、浏览器 ── 选 tract ✅
│   │   └─ 要最强性能 / 最全算子 / GPU（可接受 C++）── 选 ort
│   └─ 手上是 safetensors 权重（HF 模型 / LLM）── 选 candle ✅
└─ 要训练模型（需要自动微分）
    ├─ 想要纯 Rust、PyTorch 式体验、能换后端 ── 选 burn ✅
    └─ 已深度绑定 PyTorch 生态 ── 用 tch（需装 LibTorch）
```

**本书的一贯立场**（打了 ✅ 的路径）：**优先纯 Rust**——推理用 **tract**、加载权重与手搓模型用 **candle**、训练用 **burn**。只有当任务确实需要极致性能、或纯 Rust 生态暂时缺算子时，才**务实地**搬出 **ort**，并且一定向你讲清"我们为此放弃了纯 Rust 纯度、多引入了一个 C++ 依赖"这笔账。这不是教条，而是把选择权和代价都摊开给你看。

## 上手感受：依赖行 + 一段 candle 张量实验

选型讲完，最后给点手感。加进 `Cargo.toml` 无非就是几行（按需选用）：

```toml
# 纯 Rust 三件套
candle-core = "0.11.0"
candle-nn   = "0.11.0"
burn        = { version = "0.21.0", features = ["ndarray"] } # 训练再加 "autodiff"
tract-onnx  = "0.23.5"
# 需要极致性能时（非纯 Rust，底层 C++）
ort         = "2.0.0-rc.13"
```

再跑一段 candle 的最小张量运算，感受一下"纯 Rust 也能很顺手"。这段代码在本机 `cargo run` 实测通过：

```rust
// candle 0.11.0（candle-core）—— 最小张量运算
use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu; // 纯 CPU，无需任何 GPU/C++ 依赖

    // 一个 2x3 张量（从数据 + 形状构造）和一个 3x2 张量（从嵌套数组字面量）
    let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device)?;
    let b = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;

    let c = a.matmul(&b)?;                 // (2,3) x (3,2) = (2,2)
    let cv: Vec<Vec<f32>> = c.to_vec2()?;  // 取回数据到 Rust 的 Vec
    println!("a @ b = {cv:?}");
    Ok(())
}
```

运行输出（真实可复现）：

```text
a @ b = [[22.0, 28.0], [49.0, 64.0]]
```

矩阵乘法的第一个元素 `22 = 1×1 + 2×3 + 3×5`，和你手算的一致——这说明 candle 的张量、形状、矩阵乘都按预期工作了。整个过程没有装任何 C++ 库、没有下任何运行时，`cargo run` 一把梭。这就是"纯 Rust"最舒服的地方。

## 常见坑与提示

> 🕳️ **坑**：**"能加载 ONNX" ≠ "支持你这个 ONNX"。** tract 算子覆盖广但非 100%，遇到很新的模型报"unsupported op"很正常。对策：先想办法把模型导出成更"标准"的算子集，或降低 ONNX opset 版本；实在不行再换 ort（它算子最全）。

> ⚠️ **注意**：**candle 对 ONNX 的支持很有限。** candle 的主场是 safetensors，社区虽有 `candle-onnx` 子 crate，但覆盖范围有限、用前务必自行验证。记住这个分工：**跑 ONNX 找 tract/ort，加载 safetensors 找 candle**，别拿 candle 硬啃 ONNX。

> 💡 **提示**：**ONNX 是打通训练与部署的桥梁，值得早点记住。** 很多真实流程是"PyTorch 训练 → 导出 ONNX → Rust 侧用 tract/ort 部署"。哪怕你现在还没到部署那一步，也先把这条链路记在心里，[第 94 章](../p19-deploy/ch94-onnx.md) 会把它讲透。

> 🕳️ **坑**：**首次编译 candle/burn 会比较久。** candle-core 0.11 会连带拉入 `safetensors`、`gemm`、`tokenizers` 等一堆依赖，第一次 `cargo build` 可能要几十秒到一两分钟，别以为卡死了——这是在编译整条依赖链，之后就有缓存了。

## 本章小结

- Rust 深度学习生态不是"五选一的竞品"，而是各占生态位。看清两根标尺就不迷路：**纯 Rust vs 依赖 C++**、**训练 vs 推理**。
- 五个主角：**candle**（HF、纯 Rust、safetensors、LLM 与手搓/推理）、**burn**（纯 Rust、多后端 + 自动微分的训练框架）、**tch**（LibTorch 绑定，功能全但要装 C++）、**tract**（Sonos、纯 Rust、ONNX 推理、交叉编译友好）、**ort**（ONNX Runtime 绑定，性能与算子最强但底层 C++）。
- 本书立场：**优先纯 Rust**——tract 推理、candle 加载、burn 训练；需要极致性能时才务实用 ort，并讲清放弃纯度、多带 C++ 依赖的代价。
- 版本号 `0.x` 不等于不能用；真正要小心的是 ort 那种 `rc` 版本的 API 变动。**ONNX** 是打通训练框架与部署引擎的通用桥梁。

## 动手练习

1. **对号入座**：给下面几个任务各选一个 crate 并说明理由——(a) 在树莓派上跑一个现成的 `.onnx` 图像分类模型；(b) 在 Mac 上从零训练一个小 CNN；(c) 加载一个 HF 上的 safetensors 大模型做推理；(d) 一个已有大量 PyTorch 代码、要求性能拉满的服务端项目。
2. **跑通张量实验**：把本章的 candle 例子敲进一个新工程（`cargo new` 后加上那两行依赖），`cargo run` 复现 `[[22.0, 28.0], [49.0, 64.0]]`。再试着把 `matmul` 换成逐元素相加 `(&a + &a)?`，观察形状和结果。
3. **查一手资料**：打开 tract 和 ort 的 docs.rs 页面，各找出"如何加载一个 `.onnx` 文件"的入口函数名，对比它们的第一步有什么不同（提示：一个在纯 Rust 里解析，一个把文件交给 C++ 运行时）。
4. **想一想**：如果你的产品要编译成 WebAssembly 在浏览器里跑推理，上面五个里哪些是现实可行的？哪些基本没戏？为什么？（提示：回到"纯 Rust vs C++"这根标尺。）

下一章，我们不再"纸上谈兵"——用纯 Rust 的 burn，**从零训练一个能识别手写数字的 CNN**。👉 [第 40 章](ch40-burn-train.md)
