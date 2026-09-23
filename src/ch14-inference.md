# 第 14 章 · 模型推理：用 Rust 把模型跑起来

> **本章导读**：第 10 章我们把图片处理成了 `[1,3,640,640]` 的输入张量，第 11~13 章我们弄懂了模型文件长什么样、怎么读它的输入输出、怎么转换格式。万事俱备——本章就干一件事：**把张量喂给模型，让它算一遍，拿回输出张量**。我们用两套 Rust 方案来做：纯 Rust 的 `tract-onnx`（无外部依赖、编译省心）和微软 ONNX Runtime 的绑定 `ort`（性能标杆、能上 GPU/NPU）。读完你能写出一个可编译、可运行的推理程序，稳稳吐出 `float32[1,84,8400]` 这一堆数字。至于这堆数字怎么变成"猫在(100,50)、92%"的检测框——那是第 16 章后处理的活，本章只负责跑出它。

## 14.1 推理引擎到底在干什么

先把位置摆正。第 1 章那张"一张图片的奇幻漂流"流水线图里，有一格叫 **【推理】**：预处理把图变成张量之后，就轮到它上场；它吐出的不是标好框的图片，而是"一堆抽象的数字"。第 11 章我们又把这一步叫做**前向传播**（forward，输入数据顺着网络一层层往前算，最后从末端流出结果）。本章要做的，就是用代码把这一格、这一次前向传播真正跑起来。

问题是：模型文件（`.onnx`）本身**不能直接运行**。它只是一份"菜谱 + 食材清单"——记录了"先卷积、再激活、然后……"的计算图（graph）和一大堆权重数字，但它自己不会动。你需要一个程序把菜谱翻译成机器指令、按顺序执行，这个程序就是**推理引擎**（inference engine）。

打个比方：模型文件是一张 MP3，推理引擎是播放器。你双击 MP3 它不会自己响，是播放器在背后解码、混音、驱动声卡。推理引擎也一样，帮你默默做了四件事：

```text
   模型文件(.onnx)              输入张量 [1,3,640,640]
        │                              │
        ▼                              ▼
  ┌──────────────────────────────────────────────┐
  │                  推理引擎                       │
  │   ① 加载：解析计算图、把权重读进内存             │
  │   ② 优化：算子融合、常量折叠（能提前算的先算）    │
  │   ③ 规划：给每个中间张量安排好内存，能复用就复用   │
  │   ④ 执行：把每个算子派到 CPU/GPU/NPU 上算一遍     │
  └──────────────────────────────────────────────┘
                       │
                       ▼
              输出张量 [1,84,8400]
```

- **① 加载**：把 onnx 文件解析开，在内存里重建出计算图和权重。
- **② 图优化**：常量折叠（能提前算掉的先算掉）、算子融合（把 `Conv+ReLU` 合并成一个计算核，省一次内存往返）、布局调整。同一张图，优化后快 20%~50% 是常态。
- **③ 内存规划**：所有中间张量存哪、什么时候可以覆盖复用，提前排好。一张 4K 特征图动辄几十 MB，省下来不是小数。
- **④ 后端调度**：决定每个算子在哪算——CPU 上用 SIMD 指令，有 GPU/NPU 就调专用核。

所以结论很明确：**除非是教学玩具，否则别自己手写前向传播**。引擎比你写的快得多、稳得多，你只要学会怎么"喂"和怎么"取"。这两件事，就是本章的全部。

## 14.2 两大 Rust 方案：tract 还是 ort

Rust 世界里跑 ONNX，主流是两条路。它们的"喂→跑→取"三步几乎一模一样，区别在于底子和性能天花板：

| 维度 | `tract-onnx`（纯 Rust） | `ort`（ONNX Runtime 绑定） |
| --- | --- | --- |
| 本质 | 从零用 Rust 实现的推理引擎 | 微软 ONNX Runtime（C++）的 Rust 封装 |
| 外部依赖 | **无**，纯 Rust 静态编译 | 需要 `onnxruntime` 动态库（可自动下载） |
| 交叉编译 | **省心**，`cargo build --target` 基本直出 | 要为目标平台准备对应的库文件 |
| 硬件加速 | 主要吃 CPU | CPU + **GPU/NPU**，靠执行提供器（见下） |
| 算子覆盖 | 常见模型够用，冷门算子可能缺 | **最全**，工业界事实标准 |
| 极限性能 | 好，但比不过 ORT 的深度优化 | **标杆级** |
| 适合场景 | 嵌入式、边缘盒子、Rust 洁癖、快速跑通 | x86 服务器、要榨性能、要上 GPU |

`ort` 的杀手锏叫**执行提供器**（Execution Provider，简称 **EP**）——你可以把它理解成"这次推理交给哪块硬件、用哪套加速库来算"的插件开关。ONNX Runtime 支持的 EP 很全：

| EP | 面向硬件 / 平台 |
| --- | --- |
| CPU | 人人都有，兜底默认 |
| CUDA / TensorRT | NVIDIA 显卡（TensorRT 是其上更极致的加速） |
| CoreML | 苹果 M 系芯片 / iPhone 的神经引擎 |
| OpenVINO | Intel 的 CPU / 核显 / VPU |
| DirectML | Windows 上的通用 GPU 加速 |

**怎么选？** 一句话：

- 刚上手、要跑通、或者要交叉编译到嵌入式板子 → 先用 **tract**（14.3）。
- 生产环境要性能、机器上有 NVIDIA 显卡或苹果芯片 → 上 **ort**（14.4），配对应 EP。

好在两者接口高度相似，学会一个另一个照猫画虎。下面各写一个完整可编译的例子，模型统一用第 13 章导出的 `yolov8n.onnx`，输入统一用第 10 章的 `preprocess()`。

> **复用第 10 章的接口约定**：本章所有例子都调用第 10 章造好的预处理函数，签名是
> `pub fn preprocess(img: &RgbImage) -> (Array4<f32>, LetterboxInfo)`。
> 它吃一张 `image::RgbImage`，吐出两样东西：`ndarray::Array4<f32>`（形状正好 `[1,3,640,640]`、已 letterbox、已归一化到 `0~1`）和一个 `LetterboxInfo`（记着缩放比例和填充量，留给第 16 章把框还原回原图用）。本章用不到 `LetterboxInfo`，先用 `_` 接住。

## 14.3 tract：纯 Rust 跑通第一个模型

新建项目，拉依赖：

```bash
cargo new ch14_tract && cd ch14_tract
cargo add tract-onnx ndarray image
```

完整程序如下，四步走：加载 → 喂张量 → `run()` → 取输出。逐行有注释：

```rust
use image::RgbImage;
use ndarray::Array4;
use tract_onnx::prelude::*;

// 第 10 章造好的预处理，签名照搬过来即可：
// pub fn preprocess(img: &RgbImage) -> (Array4<f32>, LetterboxInfo);

fn main() -> TractResult<()> {
    // ① 加载模型：解析 onnx → 固定输入形状 → 图优化 → 编译成可执行计划
    let model = tract_onnx::onnx()
        .model_for_path("yolov8n.onnx")?
        // 把 0 号输入的“事实(fact)”钉死成 [1,3,640,640]。
        // 形状一旦确定，图优化和内存规划才能做到极致（动态 shape 会拖慢甚至报错）。
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?   // 图优化（14.1 的第 ② 件事）
        .into_runnable()?;   // 编译成可执行计划

    // ② 预处理：复用第 10 章的 preprocess，拿到标准 ndarray 的 Array4<f32>
    let img: RgbImage = image::open("test.jpg")?.to_rgb8();
    let (input, _info): (Array4<f32>, _) = preprocess(&img); // 形状 [1,3,640,640]

    // ③ 把 ndarray 里的数据搬进 tract 的张量。
    //    张量的本质就是“形状 + 一串连续的数字”，from_shape 正是照这个模型造的：
    let tensor = Tensor::from_shape(&[1, 3, 640, 640], input.as_slice().unwrap())?;

    // ④ 真跑：喂进“一个张量的列表”，拿回“一个张量的列表”
    let outputs = model.run(tvec!(tensor.into()))?;

    // ⑤ 取第一个（也是唯一的）输出，转成 ndarray 视图看形状
    let out = outputs[0].to_array_view::<f32>()?;
    println!("输出形状: {:?}", out.shape());               // [1, 84, 8400]
    println!("前 5 个数: {:?}", &out.as_slice().unwrap()[..5]);
    Ok(())
}
```

四步对应四个 API，一次记牢：

```text
model_for_path  → 读文件、重建计算图
with_input_fact → 固定输入形状（可选但强烈建议）
into_optimized  → 图优化
into_runnable   → 编译成可执行计划
run             → 真跑，喂 tvec![张量...]，拿回 outputs
```

**关于第 ③ 步那句 `Tensor::from_shape` 的桥接**，值得单独点一句：`preprocess()` 返回的是 crates.io 上标准 `ndarray` 的 `Array4`，而 tract 自带一份 `tract_ndarray`（同源，但版本号可能和你项目里的 `ndarray` 对不上）。两者类型"同名不同宗"时，直接 `input.into()` 会编译不过。最稳的桥就是借出底层数据切片（`input.as_slice()`）、按形状重建一个 tract 张量——绕开版本问题，永远能编过。**这也是使用 tract 时几乎唯一的槽点。**

如果第 16 章的后处理想要一个标准 `ndarray` 的 owned 数组，同样一句话搬回来：

```rust
// 把输出的 [1,84,8400] 变成标准 ndarray::Array3<f32>，交给第 16 章
let out3 = ndarray::Array3::from_shape_vec(
    (1, 84, 8400),
    out.as_slice().unwrap().to_vec(),
).unwrap();
println!("转成 ndarray: {:?}", out3.dim());   // (1, 84, 8400)
```

## 14.4 ort：接上工业级的 ONNX Runtime

`ort` 把微软的 ONNX Runtime 包成了 Rust API。用法和 tract 几乎一一对应，多出来的主要是**注册 EP** 和**要带一个动态库**。

依赖（写作时最新是 `2.0.0-rc.13`；`ort` 的 2.x 各 rc 之间 API 有过调整，工程上建议**锁定版本**）：

```toml
[dependencies]
ort = "=2.0.0-rc.13"   # 默认会自动下载官方预编译的 onnxruntime 动态库
ndarray = "0.16"
image = "0.25"
# 想用 NVIDIA 显卡，就打开对应 feature（还需机器上装好 CUDA/cuDNN）：
# ort = { version = "=2.0.0-rc.13", features = ["cuda"] }
```

> **注意**：`ort` 运行时依赖 `onnxruntime` 动态库（Linux 上是 `libonnxruntime.so`，macOS 是 `.dylib`，Windows 是 `onnxruntime.dll`）。默认特性下 `ort` 会自动帮你下载好；离线/受限环境要手动准备并让程序找得到它，否则运行时会报"找不到库"。这正是它比 tract"重"的地方。

完整程序：

```rust
use image::RgbImage;
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

// 同样复用第 10 章的 preprocess：
// pub fn preprocess(img: &RgbImage) -> (Array4<f32>, LetterboxInfo);

fn main() -> ort::Result<()> {
    // ① 建会话(Session)：图优化拉满 + 设线程数 + 注册 EP + 加载模型
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        // EP 按优先级排列：前面的硬件/库不可用，就自动往后退，最后退回 CPU。
        // 用 #[cfg] 按 feature 开关，没开的 feature 这行会被整行编译掉，因此永远能编过。
        .with_execution_providers([
            #[cfg(feature = "cuda")]
            ort::ep::CUDA::default().build(),
            #[cfg(feature = "coreml")]
            ort::ep::CoreML::default().build(),
        ])?
        .commit_from_file("yolov8n.onnx")?;

    // ② 预处理，拿到标准 ndarray 的 [1,3,640,640]（ort 直接吃标准 ndarray，不用像 tract 那样桥接）
    let img: RgbImage = image::open("test.jpg").unwrap().to_rgb8();
    let (input, _info): (Array4<f32>, _) = preprocess(&img);

    // ③ 构造输入并 run。"images" 是模型的输入名（第 12 章读元数据得到）。
    //    TensorRef::from_array_view 是“借用”数据、零拷贝，最省内存。
    let outputs = session.run(ort::inputs![
        "images" => TensorRef::from_array_view(&input)?
    ])?;

    // ④ 按输出名 "output0" 取回结果，try_extract_array 得到一个 ndarray 视图
    let out = outputs["output0"].try_extract_array::<f32>()?;
    println!("输出形状: {:?}", out.shape());               // [1, 84, 8400]
    println!("前 5 个数: {:?}", &out.as_slice().unwrap()[..5]);
    Ok(())
}
```

对照着看，和 tract 是同一套心智模型：

| 步骤 | tract | ort |
| --- | --- | --- |
| 加载 | `onnx().model_for_path(...)` | `Session::builder()?.commit_from_file(...)` |
| 固定形状 | `.with_input_fact(...)` | 一般无需（喂什么形状用什么，动态维度自适应） |
| 喂输入 | `tvec!(tensor.into())`（按顺序） | `inputs!["images" => ...]`（**按名字**） |
| 跑 | `model.run(...)` | `session.run(...)` |
| 取输出 | `outputs[0].to_array_view()` | `outputs["output0"].try_extract_array()` |

两个要点：一是 `inputs!` 和 `outputs[...]` 都**按名字**索引，名字（`"images"`、`"output0"`）来自第 12 章读到的模型元数据，**打错就报"找不到输入/输出"**；二是 `ort` 用的就是 crates.io 标准 `ndarray`，和 `image` 等 crate 互转顺滑，不用操心版本错开。

## 14.5 拿到输出之后：先别急着"看懂"它

两个例子都打印了同一行：`输出形状: [1, 84, 8400]`。这是本章的终点，也是很多新手最懵的地方——**输出就是一大堆浮点数字，不是框、不是类别、不是概率**。它的形状 `[1, 84, 8400]` 摊开来是：1 张图、8400 个候选、每个候选 84 个数（4 个框坐标 + 80 个 COCO 类别分数）。

跑一次，打印出来大概长这样（**具体数值随图片而变**，别对着抄）：

```text
输出形状: [1, 84, 8400]
前 5 个数: [5.83, 12.44, 20.71, 33.90, 48.15]
```

看着毫无意义，对吧？把这 `84×8400` 的数字矩阵，翻译成"有 3 个物体：猫(92%)在(100,50,180,140)、狗(88%)……"，要做的事包括：挑出分数够高的候选、把框坐标从 `640×640` 空间用 `LetterboxInfo` 还原回原图像素、再用 NMS（非极大值抑制）去掉重叠的重复框。**这些全是第 16 章后处理的内容，本章一概不碰。**

为什么要划这条线？因为"跑出数字"和"读懂数字"是两件独立的工程：换个引擎（tract↔ort）、换块硬件（CPU↔GPU），跑出来的 `[1,84,8400]` 应该几乎一模一样（误差 `<1e-4`）；而怎么解码这堆数字，只跟模型结构有关，跟用什么引擎跑无关。本章把"跑"这件事彻底吃透，你就拥有了一个稳定的"数字水龙头"——拧开它，第 16 章接着处理水就行。

所以本章的验收标准很朴素：**能打印出正确的输出形状 `[1,84,8400]` 和几个不为 NaN 的数值**，就算通关。

## 14.6 Batch 推理：一次喂 N 张

前面都是一次一张（batch=1）。如果你手上攒了一叠图（离线跑一个文件夹、或者多路摄像头凑齐了一批帧），可以**一次喂 N 张**：输入变成 `[N,3,640,640]`，输出相应变成 `[N,84,8400]`——多出来的那一维就是"这是批里第几张"。

代码几乎不用改：tract 把 `with_input_fact` 的形状改成 `[N,3,640,640]`（前提是模型导出时留了这个 batch 维，见第 13 章）；ort 直接喂一个 `[N,3,640,640]` 的数组即可。真正要想清楚的是**为什么要 batch**——这是一道吞吐（throughput）和延迟（latency）的权衡题：

| | batch=1（来一张跑一张） | batch=N（攒够 N 张一起跑） |
| --- | --- | --- |
| 单帧延迟 | **低**，来了立刻出结果 | 高，要等凑够 N 张、且一起算完才出 |
| 整体吞吐 | 一般 | **高**，硬件（尤其 GPU）被喂饱，单张摊薄的开销更小 |
| 典型场景 | 实时摄像头、报警，要求"快出结果" | 离线批处理一堆图/视频，追求"单位时间处理得多" |

一句话记法：**要快（延迟）用小 batch，要多（吞吐）用大 batch。** GPU 尤其吃这一套——它像个大食堂灶台，一次炒一份和一次炒十份耗时差不多，所以喂满了才划算；CPU 上 batch 带来的收益则温和得多。

要留意：能不能用 N>1，取决于模型导出时输入的 batch 维是**固定**的（导成 `1` 就只能喂 1 张）还是**动态**的（导成 `-1`/`N` 才能变）。这属于模型转换的范畴，第 13 章已经讲过怎么导出动态或固定 batch 的模型，这里直接用它的结论即可。

## 14.7 性能与工程实践

把 demo 变成能上线的推理程序，下面几条是"不做就慢/就崩"的硬常识。

**1. 会话(Session)只创建一次，反复复用。** 加载模型要读文件、建图、优化，动辄几百毫秒到几秒、还吃几百 MB 内存。**千万别放进循环**：

```rust
// ❌ 每帧都重新加载模型：每次几秒 + 内存反复申请释放，程序又慢又抖
for frame in frames { let model = load_model(); model.run(...); }

// ✅ 加载一次，循环里只调用 run
let model = load_model();
for frame in frames { model.run(...); }
```

**2. 一定要预热（warmup）。** 引擎**第一次推理往往慢 2~10 倍**——内存首次分配、计算核首次编译、CPU 缓存冷启动、GPU 上下文初始化都挤在这一发里。所以：正式服务前先空跑几次把它"焐热"；做性能测试(benchmark)时更要先预热再计时，否则数字毫无意义。

```rust
use std::time::Instant;

// 预热：拿真实尺寸的输入空跑几次
for _ in 0..10 { let _ = model.run(tvec!(tensor.clone().into()))?; }

// 正式计时：跑 100 次取平均
let t = Instant::now();
for _ in 0..100 { let _ = model.run(tvec!(tensor.clone().into()))?; }
println!("平均单次推理: {:?}", t.elapsed() / 100);
```

**3. 线程数按场景设。** ORT 默认会吃满 CPU 核心：单路摄像头、追求低延迟时，`with_intra_threads` 设小一点（甚至 1）反而稳；离线批处理追求吞吐，就放开让它多线程并行。

**4. 少拷贝、复用内存。** 每帧都新建输入缓冲、`to_owned()` 一份输出，都是白白的内存搬运。工程上常用的招：预处理的输出缓冲固定复用（第 10 章）；取输出优先用零拷贝**视图**（tract 的 `to_array_view`、ort 的 `try_extract_array` 拿到的都是借用，不 `to_owned` 就不拷贝）；ort 还提供 **IO Binding**，把输入/输出缓冲直接绑到设备内存上，省掉 CPU↔GPU 之间的来回搬运——高并发服务里这几微秒累积起来很可观。

**5. 把三段接成流水线。** 真实程序是"预处理 → 推理 → 后处理"连轴转。单张顺着跑没问题，但要吃满多核/多路视频时，通常把三段拆成流水线并发跑（一个线程解码预处理、一个线程推理、一个线程后处理，用队列衔接），让 GPU 不空等。**多路并发、批调度、部署打包这些工程话题，第 29 章会专门展开**，本章先把"单条链路正确、复用得当"打牢。

## 14.8 CPU / GPU / NPU 怎么选，EP 怎么配

硬件与 EP 的搭配，给一张实用速查表：

| 环境 | 推荐引擎 + EP | 说明 |
| --- | --- | --- |
| 开发机 / 无独显 | tract，或 ort + CPU | 跑通、调试、轻量部署，tract 最省事 |
| x86 服务器 + NVIDIA 显卡 | ort + CUDA（再进一步 TensorRT） | 吞吐最高，TensorRT 需额外转换/编译 |
| 苹果 M 系 / iPhone | ort + CoreML | 用上苹果神经引擎，功耗表现好 |
| Intel CPU / 核显 | ort + OpenVINO | Intel 平台上的性价比之选 |
| Windows 通用 GPU | ort + DirectML | 不限显卡品牌 |
| 嵌入式 / 边缘盒子 | tract（CPU），或厂商 NPU SDK | tract 交叉编译友好；榨 NPU 性能得用厂商 SDK |

配置上记住 EP 的两条脾气（14.4 的代码已体现）：

- **顺序即优先级**：`with_execution_providers([...])` 里排在前面的先试，不支持某个算子就回退到后一个，最后兜底到 CPU。
- **默认静默回退**：某个 EP 没注册成功（比如机器上没装 CUDA），`ort` 默认**不报错、悄悄退回 CPU**。这很友好，但也容易让你"以为在用 GPU、其实在跑 CPU"。想确认，就在注册时让它出错即停，或打印实际生效的 EP——上线前务必核对一次，别把这当成理所当然。

## 14.9 常见坑

推理这一步最容易踩的雷，集中在"输入对不对、库在不在、第一次为什么这么慢"：

| 现象 / 报错 | 病因 | 对策 |
| --- | --- | --- |
| `expected shape [...] got [...]` | 输入形状和模型要求不符 | 打印双方 shape 对比；确认是 `[1,3,640,640]` |
| `expected f32 got u8` 之类 | 输入 dtype 喂错 | 检查是否归一化成了 `f32`（INT8 量化模型才喂 `u8`） |
| 跑起来但输出全是 `NaN` | 预处理错了（常见：**忘了归一化**、除零、通道顺序反了） | 把送进模型的输入张量存成图肉眼看一眼 |
| tract 上又慢又占内存，或形状报错 | **动态 shape 没固定** | 用 `with_input_fact` 把输入钉成 `[1,3,640,640]` |
| 第一次推理特别慢，之后正常 | 冷启动（这是正常现象，不是 bug） | **预热**几次再计时/服务（14.7） |
| `no input named 'xxx'` / `output` | `inputs!`/`outputs[...]` 名字打错 | 以第 12 章读到的真实输入/输出名为准 |
| 运行时"找不到 onnxruntime 库" | ort 的动态库缺失或路径不对 | 让程序找得到 `.so/.dylib/.dll`（14.4 的注意） |
| 模型路径找不到 | 相对路径 vs 工作目录 | 用绝对路径，或确认在项目根目录运行 |
| 内存持续上涨 | Session 在循环里被反复创建 | 挪出循环，只加载一次（14.7 第 1 条） |

排障三板斧屡试不爽：**打印输入 shape、打印输入 dtype、把输入张量还原成图看一眼**。九成推理问题，都是"喂进去的东西不对"。

## 14.10 小结

- 推理引擎替你做了四件事：加载模型、图优化、内存规划、后端调度；模型文件本身不会动，别自己手写前向传播。这一步对应第 1 章流水线的【推理】、第 11 章的前向传播。
- 两大 Rust 方案：**tract** 纯 Rust、无外部依赖、交叉编译省心，适合 CPU/嵌入式；**ort** 是 ONNX Runtime 绑定，性能强、能靠 **EP** 上 GPU/NPU，代价是要带动态库。
- 两者都是"加载一次 → 喂张量 → `run()` → 取输出"，tract 按顺序喂、ort 按名字喂；输出统一是一堆数字 `[1,84,8400]`。
- 本章只负责**跑出这堆数字**并打印其形状；把数字解码成检测框（阈值、还原坐标、NMS）是第 16 章的事。
- 工程铁律：Session 只建一次并复用、**先预热再计时**、线程数按延迟/吞吐场景设、能零拷贝就零拷贝；batch 是吞吐与延迟的权衡（细节见第 13 章）。

## 14.11 练习

1. 用 14.3 的 tract 例子跑通 `yolov8n.onnx`，确认打印出 `[1, 84, 8400]`。再故意把 `with_input_fact` 那行删掉，观察行为有什么变化（可能变慢或直接报形状错）。
2. 用 14.4 的 ort 例子跑同一个模型和同一张图，把两个引擎输出的"前 5 个数"打印出来对比——它们应该几乎逐位相同（误差 `<1e-4`）。如果差很多，多半是预处理没对齐。
3. 给推理加上"预热 10 次 + 计时 100 次"，报告你机器上单张 YOLOv8n 的平均耗时；再把预热去掉、只跑 1 次计时，看看第一次比平均值慢多少倍。
4. 故意把第 10 章 `preprocess()` 里的归一化（除以 255）删掉再推理，打印输出的前几个值，体会"忘归一化"这个经典坑会把输出带成什么样。
