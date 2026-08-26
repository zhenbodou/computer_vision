# 第 94 章 ONNX 跨框架推理：契约、验证与兼容性

> **本章导读**
> - 学完你能：用**纯 Rust 的 tract** 加载一个 `.onnx` 模型，设定输入形状，跑前向推理，取回输出——而且我们会**亲手在 Rust 里造一个极小的 `.onnx`**，把"加载 → 推理"整条链路真正跑通（本机 `cargo run` 实测，输出真实可复现）。
> - 看懂一个 `.onnx` 文件里到底装了什么：**算子图 + 权重 + 输入输出契约 + opset 版本**。
> - 记住一条铁律：**ONNX 只搬运"计算图"，不搬运业务语义**——类别表、预处理、NMS、许可证必须另行打包。
> - 学会把"能出数"升级为"可信的跨框架部署"：黄金样本对齐、逐层定位漂移、tract 与 ort 的选型。

## 场景：训练用 PyTorch，部署要纯 Rust，怎么接上？

假设算法同事在 GPU 服务器上用 PyTorch 训练好了一个分类模型，现在要把它塞进一台 ARM 边缘盒子里 7×24 运行（[第 111 章](../p21-mastery/ch111-embedded.md)）。你显然不想在这台巴掌大的盒子上装一整套 Python + PyTorch 运行时——太重、太慢、还难交叉编译。

**ONNX 就是那座桥。** 同事只需一句 `torch.onnx.export(...)`，就能把模型导出成一个 `model.onnx` 文件；你在 Rust 侧用 tract 加载它、跑推理。训练框架、编程语言、部署硬件**可以完全不一样**——这就是"训练与部署解耦"。[第 39 章](../p07-dl-basics/ch39-dl-ecosystem.md) 把 ONNX 比作"深度学习界的 PDF"：谁都能打开的通用格式。本章我们把这座桥**真正走一遍**。

## 原理：`.onnx` 里装了什么，tract 怎么把它跑起来

### 一个 `.onnx` 就是一张"算子图 + 权重"

别被二进制文件唬住。拆开一个 `.onnx`，里面其实就四样东西：

```text
model.onnx
├── graph（计算图）           一串节点 node，每个是一个算子 op：
│     input ──MatMul──▶ mm ──Add──▶ output    （Conv / Relu / MatMul / Add …）
│                                              节点用"输入名→输出名"连成有向无环图
├── initializer（初始化张量）  训练学到的权重/偏置，按名字存：W、B、conv1.weight …
├── input / output（输入输出契约）  每个的名字、dtype、形状——这就是"契约"
└── opset_import（算子集版本）  规定每个算子的确切语义，比如 opset 13
```

- **计算图**告诉引擎"先算什么、再算什么"；**初始化张量**是模型的"记忆"（参数）；**契约**规定"喂什么形状的数据进来、吐什么形状出去"；**opset** 则钉死每个算子在这个版本下的精确行为。
- 关键认知：**ONNX 描述的是"怎么算"，不包含"算完之后业务上怎么用"**。类别索引到中文名的映射表、图像该怎么预处理、检测框要不要做 NMS、模型的开源许可证——这些**都不在 `.onnx` 里**，必须你自己另行打包。这正是本章标题"契约"二字的分量。

### tract 加载推理的完整流程

tract（0.23.5，Sonos 出品，**纯 Rust\*、不链接任何外部 C/C++ 运行库**；\* 构建期有个 `cc` 只当**汇编器**编译 tract **自带的 SIMD 汇编内核**——仓库里 0 个 `.cpp`、无 `-sys` crate、产物是自包含单二进制，与 `ort` 链接外部 C++ onnxruntime 性质完全不同，详见 [附录 A](../appendix/a-crates.md)）加载并运行一个 `.onnx`，是固定的六步：

```text
onnx()  →  model_for_path(".onnx")  →  with_input_fact(0, dtype+shape)
        →  into_optimized()  →  into_runnable()  →  run(tvec!(输入))  →  取输出
```

其中最容易被忽略的是 **`with_input_fact`**。ONNX 允许**动态维度**（batch 常写成 `-1` 或一个符号名 `N`），可 tract 要先知道输入的**确切 dtype 和形状**，才能做图优化、预分配内存。所以你必须用 `with_input_fact(0, f32::fact([1, 3, 224, 224]).into())` 把第 0 个输入"钉死"。`into_optimized` 做常量折叠、算子融合等优化，`into_runnable` 得到一个可反复调用的可执行模型。

> 📖 **术语**：**fact（事实）**。tract 里用来描述一个张量"长什么样"的元信息——dtype（如 `f32`）加各维尺寸。你给输入设定 fact，等于把契约里那条"输入规格"告诉引擎。

### 预处理必须和训练端逐项一致

`.onnx` 通常**不含预处理**（除非导出时显式并进图里）。这意味着：**推理时喂进去的张量，必须和训练时喂给网络的张量，在每一个维度上逐项对齐**——尺寸、RGB/BGR 通道序、归一化（`/255` 或按通道 mean/std）、NCHW 布局。[第 42 章](../p08-classification/ch42-candle-classify.md) 和 [第 31 章](../p05-numeric/ch31-image-as-tensor.md) 反复强调过这条铁律；到了跨框架部署，它是**"离线好、线上崩"的头号根因**——你在 Python 里测得好好的模型，换到 Rust 侧结果全错，十有八九是预处理某一项对不上。

### tract 还是 ort？

[第 39 章](../p07-dl-basics/ch39-dl-ecosystem.md) 给过完整选型表，这里一句话回顾：**要纯 Rust、交叉编译、上边缘 → 选 tract**；**要极致性能、算子最全、GPU（可接受多带一个 C++ 运行库）→ 选 ort**（微软 ONNX Runtime 的 Rust 绑定，底层是 C++）。但无论选谁，从"可信部署"的角度都要补一条纪律：**拿同一批黄金输入，比对两个 runtime 的原始输出**（本章末"专家补强"细讲）。

## 动手：用 tract 加载 ONNX 并推理（含亲手造一个 `.onnx`）

> ⚠️ **注意（本章的诚实边界）**：真实的 `.onnx` 由 **PyTorch 的 `torch.onnx.export` 导出**，纯 Rust 生态里几乎没有"生产 ONNX"的工具。为了让本章**免下载、即刻可跑、且完全可复现**，我们用 tract 内置的 protobuf 类型在 Rust 里**手搓一个极小的 `.onnx`**（一个线性层 \\( y = x \cdot W + b \\)），再像加载真实模型一样把它读进来推理。**请记住：手搓 `.onnx` 只是为了自包含；加载 + 推理那段 API，跑这个玩具和跑真实 ResNet / YOLO 一模一样**，唯一区别是模型从哪来、形状多大、预处理怎么配。完整工程见 `code/dl_labs/ch94_tract_onnx/`，`cd` 进去 `cargo run --release` 即可复现。

`Cargo.toml`（tract 纯 Rust，`prost` 只在"手搓 `.onnx`"时用来序列化 protobuf；真实项目里 `.onnx` 是导出来的，不需要 `prost`）：

```toml
[dependencies]
tract-onnx = "0.23.5"   # Sonos 纯 Rust ONNX 推理引擎
prost = "0.14"          # 仅本例造 .onnx 时用来编码 protobuf
```

**第一步（Part A）：造一张"算子图 + 权重"并存成 `.onnx`。** 下面这段正是把前面讲的四样东西（节点、初始化张量、输入输出契约、opset）一一填进去——读它就是在读"一个 `.onnx` 的骨架"。构造 `ValueInfoProto` / `TensorProto` / `NodeProto` 的三个辅助小函数是样板代码，完整版在配套工程里：

```rust
use prost::Message;         // 把 protobuf 结构体编码成字节
use tract_onnx::pb;         // tract 内置的 ONNX protobuf 类型
use tract_onnx::prelude::*;

// 计算图：input[1,4] ──MatMul──▶ mm ──Add──▶ output[1,3]
let graph = pb::GraphProto {
    name: "tiny_linear".to_string(),
    node: vec![
        node("MatMul", &["input", "W"], "mm"),  // mm = input · W
        node("Add",    &["mm", "B"],  "output"), // output = mm + B（广播）
    ],
    initializer: vec![                            // 权重就是训练学到的参数
        f32_init("W", &[4, 3], w),                // W [4,3]
        f32_init("B", &[3], b),                   // 偏置 B [3]
    ],
    input:  vec![f32_io("input", &[1, 4])],       // 输入契约
    output: vec![f32_io("output", &[1, 3])],      // 输出契约
    ..Default::default()
};
let model_proto = pb::ModelProto {
    ir_version: 7,
    opset_import: vec![pb::OperatorSetIdProto { domain: String::new(), version: 13 }],
    graph: Some(graph),
    ..Default::default()
};
let mut buf = Vec::new();
model_proto.encode(&mut buf).unwrap();            // protobuf 序列化
std::fs::write("tiny_linear.onnx", &buf)?;        // 得到一个真正的 .onnx 文件
```

**第二步（Part B）：像加载任何真实模型一样，用 tract 跑推理。** 这段才是本章的主角，对任何 `.onnx` 都通用：

```rust
let model = tract_onnx::onnx()
    .model_for_path("tiny_linear.onnx")?             // 读 .onnx（纯 Rust 解析 protobuf）
    .with_input_fact(0, f32::fact([1, 4]).into())?   // 钉死第 0 个输入的 dtype+形状
    .into_optimized()?                               // 图优化
    .into_runnable()?;                               // 得到可执行模型

let input = Tensor::from_shape(&[1, 4], &[1.0f32, 2.0, 3.0, 4.0])?;
let result = model.run(tvec!(input.into()))?;        // 推理，输入用 tvec!(...) 包起来

// ⚠️ tract 0.23 取输出用 to_plain_array_view（旧版的 to_array_view 已移除）
let view = result[0].to_plain_array_view::<f32>()?;
println!("输出形状 = {:?}", view.shape());
println!("输出值   = {:?}", view.as_slice().unwrap());
```

`cargo run --release` 的**真实输出**：

```text
[A] 已写出 tiny_linear.onnx （218 字节的 protobuf）
[B] 输出形状 = [1, 3]
[B] 输出值   = [5.1, 6.2, 7.3]
```

手算校验一下：输入 `x = [1, 2, 3, 4]`，权重 `W` 让 `x·W = [1+4, 2+4, 3+4] = [5, 6, 7]`，再加偏置 `b = [0.1, 0.2, 0.3]`，正好得到 `[5.1, 6.2, 7.3]`。**一个真正的 `.onnx` 被纯 Rust 造出来、又被纯 Rust 加载跑通了**——全程没装任何 C++ 库、没下任何模型。

那真实模型从哪来？训练侧一行导出即可（这是 Python，跑在训练机上，与我们 Rust 侧无关）：

```python
# 训练侧（PyTorch）：把训练好的 model 导出成 model.onnx
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=13)
```

导出后，强烈建议先用 **Netron**（一个可视化 `.onnx` 的免费工具）打开看一眼：输入输出叫什么名、形状 dtype 是什么、有没有你没料到的算子。**别凭教程猜下标**（[第 46 章](../p09-detection/ch46-yolo-inference.md) 反复强调过这点）。

## 常见坑与提示

> 🕳️ **坑**：**输入 shape / dtype 不符，直接报错或结果全错。** 契约里写的是 `f32 [1,3,224,224]`，你喂了 `u8` 或 `[3,224,224]`（少了 batch 维），tract 要么在 `with_input_fact` / `run` 处报形状错，要么给你一堆垃圾数。喂之前先对齐 dtype、加够 batch 维。

> ⚠️ **注意**：**预处理不一致 → "离线好、线上崩"。** 尺寸、RGB/BGR、`/255`、每通道 mean/std、NCHW——任何一项和训练端不一致，模型看到的就是"另一个分布"的数据。Python 侧测得好好的，换到 Rust 结果莫名其妙差，先查这里。

> 🕳️ **坑**：**动态维度必须钉死。** ONNX 里 batch 常是 `-1` 或符号维，tract 在优化前必须知道确切形状，否则报错。用 `with_input_fact` 把每个输入的 dtype+shape 显式设定。

> 🕳️ **坑**：**算子不支持（`unsupported op`）。** tract 算子覆盖很广但非 100%，遇到很新的模型可能报某个算子不支持。对策：训练侧**降低 opset 版本**、或改用更"标准"的算子集重新导出；实在不行再上算子最全的 **ort**。

> ⚠️ **注意**：**ort 不是纯 Rust。** 它底层链接 C++ 的 onnxruntime，边缘/交叉编译更折腾；而且任何**落到 CPU 的算子回退（fallback）**都可能造成隐蔽的数据往返拖慢推理——要让它在日志里可见，别默默吃掉性能。

## 本章小结

- 一个 `.onnx` = **算子图 + 权重（initializer）+ 输入输出契约 + opset 版本**；它只描述"怎么算"，**不含类别表、预处理、NMS、许可证**——这些要另行打包。
- tract（纯 Rust）加载推理的固定流程：`onnx()` → `model_for_path` → **`with_input_fact`（钉死输入形状/类型）** → `into_optimized` → `into_runnable` → `run(tvec!(...))` → **`to_plain_array_view`**（0.23 API）。
- 我们**在 Rust 里手搓了一个极小 `.onnx` 并真跑通**，输出 `[5.1, 6.2, 7.3]` 可复现；这段加载+推理 API 与跑真实大模型完全一致。
- **预处理必须和训练端逐项一致**，这是跨框架部署最常见的翻车点。选型：纯 Rust/边缘 → tract；极致性能/算子全/GPU → ort（代价是 C++ 依赖）。

## 动手练习

1. **跑通并改形状**：把配套工程 `cargo run --release` 复现 `[5.1, 6.2, 7.3]`。再把 `W` 改成一个 `4×2` 矩阵、输出契约改成 `[1,2]`，看看输出怎么变（记得同步改 `with_input_fact` 和手算）。
2. **故意喂错**：把输入张量从 `[1,4]` 改成 `[1,3]` 或把 `f32` 换成别的 dtype，观察 tract 在哪一步、报什么错——亲身体会"契约不符"的表现。
3. **加一个算子**：在 `Add` 之后再接一个 `Relu` 节点（`node("Relu", &["output"], "y")`，并把图的输出名改成 `y`），把某个输出压成负数前后对比，验证 ReLU 生效。
4. **想一想**：如果同事导出的模型 batch 维是动态的 `N`，而你一次要推理 8 张图，`with_input_fact` 该写成什么？如果每次图片尺寸还不固定，又该怎么办？（提示：要么每种形状建一个 runnable，要么看模型是否支持动态尺寸。）

## 专家补强：从"能出数"到"可信的跨框架部署"

打印出一个输出只是起点。真正上线的跨框架推理，要把它当成一份**可验证的契约**来交付：

- **契约清单（manifest）**：一次部署至少打包模型文件及其 **SHA-256**、opset、动态轴、输入名/布局/dtype、预处理参数、输出语义、**类别表**、训练代码版本、许可证，以及一组**黄金输入输出**。加载时校验这些哈希，**不匹配就拒绝启动**——否则你可能用错版本的类别表配上对的权重，得到"数值正常、语义全错"的结果（这正是 [第 46 章](../p09-detection/ch46-yolo-inference.md) 的制品清单思路）。
- **黄金样本双 runtime 对齐**：固定至少 20 个**正常 + 边界**样本，在训练框架（PyTorch）和部署 runtime（tract / ort）分别跑，**先比原始输出、再比最终业务结果**。误差用**最大绝对/相对误差、余弦相似度、预测一致率**来描述，而不是拍脑袋说"差不多"。发现漂移时，**逐层二分对比中间张量**，能迅速定位到第一个开始漂移的节点——通常就是那个算子在两个 runtime 下语义有细微差异。
- **兼容性与回退可见**：测试错误形状、错误 dtype、缺失输出、不支持算子这些异常路径；任何 CPU fallback 都要在日志和 profile 中可见。

这几条的完整验收标准见 [附录 G](../appendix/g-expert-standard.md)。

下一章，我们让模型"瘦身提速"——**量化、剪枝与压缩**，为边缘和实时部署做准备。👉 [第 95 章](ch95-quantization.md)
