# 第 12 章 · 读懂模型：输入输出、shape 与元数据

> **本章导读**：上一章我们从原理上认识了神经网络和模型文件（第 11 章）。现在你手里多了一个 `yolov8n.onnx`——它到底"吃"什么、"吐"什么？这是任何推理工程的**第一步**，也是最容易被跳过、然后在预处理阶段悄悄翻车的一步。本章教你用 **Netron、Python、Rust** 三条路，把一个陌生模型的输入/输出规格（name、shape、dtype、布局）和元数据查得清清楚楚，并把 YOLOv8 的输出 `[1, 84, 8400]` 逐维解剖明白。真正把模型跑起来是第 14 章的事，本章只做一件事：**搞清楚它的"接口"长什么样。**

**学习目标：**

- 说清"搞懂输入输出"为什么必须排在写预处理代码之前
- 读懂输入四要素：**name / shape / dtype / 布局**，并知道归一化方式为什么模型文件里查不到
- 分清**固定维**和**动态维（dynamic axes）**，理解 shape 里 `-1` / `'batch'` 的含义
- 会用 Netron、Python（`onnx` / `onnxruntime`）、Rust（`ort` / `tract`）查规格
- 逐维解剖 YOLOv8 的输出张量 `[1, 84, 8400]`

## 12.1 为什么第一件事是"搞清输入输出"

想象你接手一个新项目，同事甩给你一个 `best.onnx` 文件说："这是训练好的检测模型，你把它接到我们的视频流上。"你打开一看——一个几十兆的二进制文件，啥也看不出来。

这时候有两种人。新手会立刻打开编辑器写 `resize(640, 640)`、`img / 255.0`，跑起来一堆报错或者输出全是乱框，然后卡两天。老手会先问四个问题：

```text
① 它要几个输入？每个输入叫什么名字？
② 每个输入是什么形状（shape）、什么数据类型（dtype）？
③ 像素要怎么摆（布局）、怎么缩放（归一化）？
④ 它输出几个张量？每个的形状、每一维什么含义？
```

**这就像拿到一个没有说明书的零件：你得先量清楚它的接口尺寸和进出线，才能把它焊到你的电路板上。** 模型也是一个"零件"，它的接口就是输入张量和输出张量。接口没搞清楚，后面全是猜。

为什么这么关键？因为**预处理代码的每一行都由输入规格决定**（预处理详见第 10 章）：

| 你要写的预处理步骤 | 由输入规格的哪一项决定 |
| --- | --- |
| 缩放到多大 | shape 里的 H、W（如 640×640） |
| 通道顺序摆成 RGB 还是 BGR | 训练时的约定（**模型文件里查不到！**） |
| 像素排成 NCHW 还是 NHWC | shape 的维度顺序 |
| 除以 255 还是减均值除方差 | 归一化约定（**模型文件里也查不到！**） |
| 张量用 f32 还是 f16 | dtype |

表格里两个"查不到"就是本章要反复敲黑板的大坑，12.2 会展开。先记住结论：**shape 和 dtype 能从模型文件里读出来；通道顺序和归一化方式通常读不出来，得查文档 + 实测。**

## 12.2 输入规格：name / shape / dtype / 布局 / 归一化

一个输入张量，你要搞清楚五件事。以我们贯穿全书的 YOLOv8 为例：

```text
输入张量 "images"
   name  = "images"              ← 名字：喂数据时要按名字对号入座
   shape = [1, 3, 640, 640]      ← 形状：四个数字各代表什么见下
   dtype = float32               ← 数据类型：每个数字占几字节、什么格式
   布局  = NCHW                   ← 内存怎么排（由 shape 维度顺序体现）
   归一化 = RGB, 像素 /255 到 0~1  ← ★ 模型文件里没有！查文档得来
```

**① name（名字）。** 推理时你要把数据"喂"给某个输入，靠的就是名字。单输入模型你可能感觉不到它的存在，但多输入模型（12.9 会讲）必须按名字对号入座，喂错了就报错或出乱码。

**② shape（形状）。** `[1, 3, 640, 640]` 这四个数字，配合布局 NCHW 来读：

```text
[  1,     3,      640,     640  ]
   │      │        │        └─ W  宽（像素）
   │      │        └─────────── H  高（像素）
   │      └──────────────────── C  通道数（RGB = 3）
   └─────────────────────────── N  batch，一次推理几张图
```

**NCHW** 是 **N**um（批量）、**C**hannel（通道）、**H**eight、**W**idth 的缩写，读作"N-C-H-W"。它规定了内存里像素的排列顺序：**先把整个 R 通道的所有像素铺完，再铺整个 G 通道，最后 B 通道**（这叫"平面式/planar"存储）。这和一张普通 JPG 解码后 `R,G,B,R,G,B...` 逐像素交错（"打包式/packed"，即 HWC）的摆法完全不同——把两者搞混，是预处理最经典的翻车点（内存布局详见第 4、10 章）。

**③ dtype（数据类型）。** `float32`（简写 f32）是最常见的：每个数字占 4 字节。也可能遇到 `float16`（半精度，省一半内存，第 13 章量化会讲）、`int8`（量化模型）、`uint8`（少数模型把归一化做进了图里，直接吃原始像素）。**dtype 必须严丝合缝**：模型要 f32，你喂 f64 或 u8，轻则报错重则结果全错。

**④ 布局（layout）。** 就是上面 NCHW 说的那件事——它其实不是一个独立字段，而是**通过 shape 的维度顺序体现**的。看到 `[1, 3, 640, 640]` 就知道是 NCHW；看到 `[1, 640, 640, 3]` 那就是 **NHWC**（通道在最后，TensorFlow 系模型常见）。这一栏没有单独的元数据，全靠你会看 shape。

**⑤ 归一化方式（normalization）——本章第一个大坑。** 模型文件（ONNX）记录的是**计算图**（有哪些算子、张量什么形状什么类型），但**通常不记录"喂进来之前该怎么预处理"**。也就是说：

> 像素要不要除以 255？要不要减去 ImageNet 均值 `[0.485, 0.456, 0.406]` 再除标准差？通道是 RGB 还是 BGR？letterbox 用什么灰度值填充（YOLO 用 114）？——**这些约定几乎都不在模型文件里。**

它们藏在训练代码、模型主页（model card）、README 里。你必须去查。YOLOv8 的官方约定是：**RGB 顺序、letterbox 到 640×640、像素值 `/255` 归一化到 `0~1`、不做均值方差**（预处理全流程见第 10 章）。

> **例外要留意**：有些导出工具会把归一化"焊进"计算图（比如在输入后加一个 `Div 255` 节点），这时模型直接吃 `0~255` 的原始像素，你反而不能再除一遍。判断方法就是本章教的——用 Netron 打开看看输入后面第一个算子是不是除法。还有些负责任的模型会把归一化参数写进**元数据**（12.5 节），那是最省心的情况。

## 12.3 动态维度：固定 vs 动态（dynamic axes）

再看一眼 shape，你可能会在实际模型里看到这样的东西：

```text
固定 shape :  [1,        3, 640, 640]
动态 shape :  [-1,       3, 640, 640]      ← 有些工具用 -1
动态 shape :  ['batch',  3, 640, 640]      ← 有些工具用符号名（更友好）
```

那个 `-1` 或 `'batch'` 叫**动态维（dynamic axis）**：这一维的大小在导出模型时**故意没有写死**，留到推理时再指定。

**生活比喻**：固定 shape 像一个只能停 1 辆车的车位，动态 batch 像一个"随到随停、来几辆停几辆"的停车场。最常见的动态维就是 **batch 维（N）**——训练时可能一次算 32 张图，部署时一次算 1 张，做成动态就不用为每种批量各导一个模型。

有的模型连 **H/W 也是动态**的（比如 `[1, 3, 'height', 'width']`），常见于分割、超分这类允许任意尺寸输入的模型。固定和动态的区别与影响：

| | 固定维（如 `640`） | 动态维（`-1` / `'batch'`） |
| --- | --- | --- |
| 含义 | 大小写死，只能喂这个尺寸 | 推理时由你指定具体值 |
| 好处 | 引擎能提前优化、分配内存，通常更快 | 灵活，一个模型应付多种批量/尺寸 |
| 代价 | 不灵活，换尺寸要重新导出 | 引擎难以极致优化；代码里**必须给出具体值** |
| 你要做的 | 直接按这个数准备数据 | 准备张量时把 `-1` 换成真实值（如 batch=1） |

**关键坑**：动态维在代码里**不能当数字用**。你不能拿 `-1` 去 `vec![0.0; 1 * 3 * 640 * 640]` 里算内存大小——你得先决定"这次喂 1 张"，把 batch 定为 1。看到 shape 里有 `-1` 或字符串，第一反应就是"这一维要我自己定"。（怎么把动态 batch 固定下来、或按需改 batch，是第 13 章模型转换的话题。）

## 12.4 输出规格：解剖 YOLOv8 的 `[1, 84, 8400]`

输入搞定，输出同样有 name、shape、dtype 三要素，方法完全一样。难点在于——**输出每一维到底什么含义，光看 shape 是看不出来的**，必须查文档 + 实测。这正是我们主模型的经典例子。

YOLOv8 检测模型只有一个输出，一般叫 `output0`，形状 `float32[1, 84, 8400]`。逐维拆开：

```text
形状 [1, 84, 8400]
      │   │    └─ 8400 : 候选框的数量（一次预测这么多个"可能的目标"）
      │   └────── 84   : 每个候选框的 84 个数字 = 4 个框坐标 + 80 个类别分数
      └────────── 1    : batch，一次一张图
```

**8400 从哪来？** YOLOv8 把 640×640 的图在三个尺度上打网格，每个格子出一个候选：

```text
80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400 个候选
```

**84 怎么分？** COCO 是 80 类（第 1 章约定：`person=0`、`car=2`、`cat=15`、`dog=16`…），所以：

```text
84 = 4  (框：中心 cx, cy 和 宽高 w, h)
   + 80 (80 个类别各自的分数)
```

画成表格看更直观（数值虚构）：

```text
              候选0    候选1    ...  候选8399
   cx     [  320.5,   12.0,  ...  ]   ← 4 行是框
   cy     [  210.2,   33.1,  ...  ]
   w      [   85.0,    9.2,  ...  ]
   h      [  120.4,    8.8,  ...  ]
   类别0  [   0.02,   0.61,  ...  ]   ← 后面 80 行是各类别分数
   类别1  [   0.00,   0.03,  ...  ]
   ...    （共 80 行）
   类别79 [   0.11,   0.00,  ...  ]
```

**光看 `[1, 84, 8400]` 你根本猜不到上面这套含义**——这就是本章要传达的核心认知：**输出的维度含义是"查"和"试"出来的，不是"看形状"看出来的。** 具体要查/实测清楚这几点：

- 84 里**前 4 个是框还是后 4 个**？框是 `cx,cy,w,h`（中心+宽高）还是 `x1,y1,x2,y2`（左上+右下）？YOLOv8 是前 4 个、且是中心宽高。
- 类别分数是**原始 logit 还是已经过了 sigmoid**（0~1）？这决定你要不要自己再激活一次。
- 坐标是**相对 640 画布**的，还是已经还原到原图？（YOLOv8 是相对 640 画布，得自己逆 letterbox 还原——第 16 章讲。）
- 甚至维度顺序也有坑：YOLOv8 是 `[1, 84, 8400]`，而老的 YOLOv5 是 `[1, 8400, 85]`（多一维 objectness，且 84/85 和 8400 位置调了个个儿）。**拿错了整套索引全乱。**

把这 8400×84 个数字变成"3 只猫在这些坐标"，需要置信度过滤、坐标还原、NMS 去重——那是**第 16 章**检测后处理的完整戏码。本章只要求你能把输出的**规格和每维含义**说清楚。

## 12.5 元数据（metadata_props）：模型自带的小抄

前面说归一化、类别名这些"模型文件里查不到"，其实有个**可能省心的例外**：ONNX 允许在 `metadata_props` 里存一批**自定义键值对**（都是字符串），负责任的导出流程会把有用信息塞进去。

比如 Ultralytics 导出的 YOLOv8，`metadata_props` 里常见这些键：

```text
metadata_props（键 = 值，值都是字符串）
  ─────────────────────────────────────────────
  author  = "Ultralytics"
  task    = "detect"                    ← 任务类型：检测/分割/分类/姿态
  stride  = "32"                        ← 网格最大步长
  names   = "{0: 'person', 1: 'bicycle', 2: 'car', ...}"  ← ★ 80 类的类别名！
  imgsz   = "[640, 640]"                ← 期望输入尺寸
```

其中 `names` 尤其宝贵：它直接告诉你**类别号对应的名字**，省得你去别处查 COCO 类别表。`task` 告诉你这是检测还是分割，`stride`/`imgsz` 帮你确认预处理尺寸。

**但别指望它一定存在。** 很多模型（尤其是 PyTorch 直接 `torch.onnx.export` 出来的）`metadata_props` 是空的。所以正确的姿态是：**先看有没有元数据，有就当小抄用；没有就老老实实查文档。** 12.7、12.8 的代码都会顺手把元数据打出来。

## 12.6 可视化神器：Netron

查规格最快、最直观的方式，不是写代码，而是一个免费工具——**Netron**（[netron.app](https://netron.app)）。它是模型界的"透视眼"。

用法简单到离谱：

```text
打开 netron.app（网页版，或下装桌面版）
   → 把 yolov8n.onnx 拖进去
   → 整张计算图就画出来了
```

拖进去你能立刻看到：

- **每个输入/输出节点**：点一下就显示 `name`、`shape`、`dtype`——本章前几节要查的东西一目了然。
- **整张计算图**：从输入到输出，每个算子（Conv、Add、Sigmoid…）怎么连的。想确认"归一化是不是焊进图里了"，就看输入后第一个节点是不是 `Div`（除以 255）。
- **模型属性**：点最上方的模型节点，右侧面板会列出 `metadata_props`（12.5 那些小抄）、ONNX opset 版本、生产者信息。

**工程建议**：拿到任何陌生模型，第一件事就是拖进 Netron 扫一眼。三十秒就能回答"几个输入几个输出、什么形状、归一化在不在图里"这些问题，比翻文档、写脚本快得多。写代码查规格（下面两节）是为了**自动化**和**在程序里拿到这些值**，人肉确认还是 Netron 最省事。

## 12.7 用工具查看（一）：Python 当命令行工具

第 1 章说过，Python 在模型这一侧我们就当"命令行工具"用——你只管跑脚本，不需要会写 Python。查规格有两个常用库，装一下：

```bash
pip install onnx onnxruntime
```

**方式 A：用 `onnx` 直接读计算图。** 这是"静态"读法，直接解析文件里的图定义：

```python
# inspect_onnx.py —— 用 onnx 库打印输入/输出/元数据
import onnx

model = onnx.load("yolov8n.onnx")

def dim_to_str(d):
    # 每一维要么是具体数字(dim_value)，要么是符号名(dim_param，即动态维)
    if d.HasField("dim_value"):
        return str(d.dim_value)
    if d.HasField("dim_param"):
        return d.dim_param          # 动态维，如 'batch'
    return "?"

def show(io_list, title):
    print(f"== {title} ==")
    for t in io_list:
        # elem_type 是个枚举数字，用 .Name() 转成 'FLOAT' 这样的可读名
        dtype = onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
        dims = [dim_to_str(d) for d in t.type.tensor_type.shape.dim]
        print(f"  name={t.name:10} dtype={dtype:8} shape={dims}")

show(model.graph.input, "INPUTS")
show(model.graph.output, "OUTPUTS")

print("== METADATA ==")
for prop in model.metadata_props:      # metadata_props 可能为空
    print(f"  {prop.key} = {prop.value[:70]}")
```

跑 `python inspect_onnx.py`，典型输出：

```text
== INPUTS ==
  name=images     dtype=FLOAT    shape=['1', '3', '640', '640']
== OUTPUTS ==
  name=output0    dtype=FLOAT    shape=['1', '84', '8400']
== METADATA ==
  task = detect
  names = {0: 'person', 1: 'bicycle', 2: 'car', ...}
```

> **坑**：某些老版本导出的模型，`model.graph.input` 里会**混进权重张量**（把几百个卷积核也当成"输入"列出来）。要只看"真正运行时要喂的输入"，请用下面的 `onnxruntime`，它只列真实输入。

**方式 B：用 `onnxruntime` 读会话（推荐）。** 这是"运行时"读法，和你将来真正推理时看到的完全一致，而且干净：

```python
# inspect_ort.py —— 用 onnxruntime 打印输入/输出/元数据
import onnxruntime as ort

sess = ort.InferenceSession("yolov8n.onnx",
                            providers=["CPUExecutionProvider"])

print("== INPUTS ==")
for i in sess.get_inputs():
    # i.shape 里动态维会直接是字符串，如 'batch'
    print(f"  name={i.name}  type={i.type}  shape={i.shape}")

print("== OUTPUTS ==")
for o in sess.get_outputs():
    print(f"  name={o.name}  type={o.type}  shape={o.shape}")

meta = sess.get_modelmeta()
print("metadata =", meta.custom_metadata_map)   # 就是 metadata_props
```

输出：

```text
== INPUTS ==
  name=images  type=tensor(float)  shape=[1, 3, 640, 640]
== OUTPUTS ==
  name=output0  type=tensor(float)  shape=[1, 84, 8400]
metadata = {'task': 'detect', 'names': "{0: 'person', ...}", ...}
```

`type=tensor(float)` 就是 f32；如果是动态 batch，`shape` 会显示成 `['batch', 3, 640, 640]`。

## 12.8 用工具查看（二）：Rust

Python 适合手头快速探查，但我们是要在 **Rust** 工程里把模型跑起来的，最终还是得用 Rust 拿到这些规格。两个推理引擎都能查（见规范第 5 节的 crate 表）。

**方式 A：用 `ort`（ONNX Runtime 的 Rust 绑定）。** 建个项目：

```bash
cargo new inspect_model && cd inspect_model
cargo add ort
```

```rust
// src/main.rs —— 用 ort 打印每个输入/输出的 name、dtype、shape
use ort::session::Session;
use ort::value::ValueType;

fn main() -> ort::Result<()> {
    // 加载模型，建立一个推理会话（session）
    let session = Session::builder()?
        .commit_from_file("yolov8n.onnx")?;

    println!("== 输入 ==");
    for input in &session.inputs {
        print_io(&input.name, &input.input_type);
    }

    println!("== 输出 ==");
    for output in &session.outputs {
        print_io(&output.name, &output.output_type);
    }

    // 顺手把元数据打出来（可能为空）
    let meta = session.metadata()?;
    if let Ok(name) = meta.name() {
        println!("模型名: {name}");
    }
    Ok(())
}

// 把一个输入/输出的类型信息拆开打印
fn print_io(name: &str, ty: &ValueType) {
    match ty {
        // 张量类型：dimensions 里 -1 表示动态维（如 batch）
        ValueType::Tensor { ty, dimensions, .. } => {
            println!("  {name:10}  dtype={ty:?}  shape={dimensions:?}");
        }
        other => println!("  {name:10}  (非张量) {other:?}"),
    }
}
```

`cargo run` 输出（`ty` 是 `TensorElementType`，Debug 打印成 `Float32`）：

```text
== 输入 ==
  images      dtype=Float32  shape=[1, 3, 640, 640]
== 输出 ==
  output0     dtype=Float32  shape=[1, 84, 8400]
```

动态 batch 会打印成 `shape=[-1, 3, 640, 640]`——记住 12.3 说的，代码里得把这个 `-1` 自己定成 1。

> **说明**：`ort` 默认会自动下载对应平台的 ONNX Runtime 库，`cargo run` 一般开箱即用。`ort` 处于 2.x 演进期，个别字段名（如 `input_type`/`inputs`）可能随版本微调，若编译报错，`cargo doc --open` 查一下当前版本的 `Session`/`ValueType` 即可，思路不变。真正跑推理是第 14 章。

**方式 B（可选）：用 `tract` 查 fact。** `tract` 是纯 Rust 的引擎，它把"某个张量的形状+类型"叫做 **fact（事实）**，用 `input_fact()` / `output_fact()` 按序号查询：

```bash
cargo add tract-onnx
```

```rust
// 用 tract 查询输入/输出的 fact（形状 + dtype）
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    // 加载为 InferenceModel（还没固定 shape 的原始图）
    let model = tract_onnx::onnx().model_for_path("yolov8n.onnx")?;

    // input_fact / output_fact 按序号(0,1,2...)查询
    println!("输入 0 的 fact: {:?}", model.input_fact(0)?);
    println!("输出 0 的 fact: {:?}", model.output_fact(0)?);
    Ok(())
}
```

输出的 fact 长这样，形状和 dtype 都在里面（`F32` 即 float32）：

```text
输入 0 的 fact: 1,3,640,640,F32
输出 0 的 fact: 1,84,8400,F32
```

如果模型是动态 batch，fact 里会显示成符号维（如 `batch,3,640,640,F32`）。两个引擎给出的信息一致，选哪个看你后续用哪个跑推理。

## 12.9 实战坑清单

把本章容易翻车的地方集中列一遍，照着排查：

| 坑 | 症状 | 排查 / 处理 |
| --- | --- | --- |
| shape 里有 `-1` 或符号维 | 分配内存时用 `-1` 算出负数/巨大值 | 那是动态维，自己定成具体值（batch 通常取 1） |
| dtype 不匹配 | 引擎报类型错误，或结果全是噪声 | 模型要 f32 就务必喂 f32，别喂 f64/u8（除非模型就是吃 u8） |
| 归一化 / 通道顺序猜错 | 能跑通但框全乱、分数极低 | 查文档：YOLOv8 是 RGB + `/255`；用 Netron 看归一化在不在图里 |
| 布局 NHWC 当成 NCHW | 输出乱七八糟 | 看 shape：`[1,3,640,640]` 是 NCHW，`[1,640,640,3]` 是 NHWC |
| 多输入 / 多输出 | 喂数据/取结果错位 | 一律**按 name 对号入座**，别靠位置猜；用本章脚本先把每个 name 列清楚 |
| 输出维度含义猜错 | 后处理解析出乱框 | 别只看形状，查文档 + 打印真实数值实测（YOLOv8 vs v5 的 84/85 差异就是教训） |

**多输入/多输出**再多说一句：本章的例子都是"单进单出"，但现实中有的模型是多输入（比如同时吃图像和一个尺寸参数），有的是多输出（比如分割模型同时吐检测框和掩码原型）。这时 12.7、12.8 的代码天然就能应付——因为我们是**遍历** `inputs`/`outputs` 列表、**按 name 打印**的，几个都照列不误。工程上取数据也永远按 name，绝不按位置硬猜。

## 12.10 小结

- 拿到任何模型，工程第一步永远是**搞清它吃什么、吐什么**，而不是急着写预处理——接口没量清楚，后面全是猜。
- 输入要查五件事：**name、shape、dtype、布局（NCHW/NHWC）、归一化方式**。前三项能从模型文件读出；**归一化和通道顺序通常读不出来，必须查文档 + 实测**，这是最大的坑。
- shape 里的 `-1` / `'batch'` 是**动态维**：推理时由你指定具体值，代码里不能当数字直接用。
- YOLOv8 输出 `[1, 84, 8400]`：`84 = 4 框 + 80 类`，`8400 = 三尺度网格候选数`。**每一维的含义靠查文档 + 实测，不是看形状看出来的**；变成检测框是第 16 章的事。
- 有些 ONNX 在 `metadata_props` 里存了类别名、任务类型、stride 等"小抄"，有就用、没有别指望。
- 查规格的三条路：**Netron**（拖进去就看，最快）、**Python**（`onnx` / `onnxruntime` 当命令行工具）、**Rust**（`ort` 遍历 `inputs`/`outputs`、`tract` 查 `input_fact`/`output_fact`，在工程里拿到这些值）。

## 12.11 练习

1. **动手查规格**：下载一个 `yolov8n.onnx`（`pip install ultralytics && yolo export model=yolov8n.pt format=onnx`），分别用 Netron、12.7 的 Python 脚本、12.8 的 Rust 程序查它的输入输出，确认三者给出的 name / shape / dtype 完全一致。
2. **找归一化证据**：把上面的模型拖进 Netron，观察输入 `images` 后面第一个算子是什么。它是不是一个"除以 255"的节点？据此判断：喂数据前你还需不需要自己做 `/255`？
3. **导出动态 batch 对比**：用 `yolo export ... dynamic=True` 再导一个动态版本，用 Python 脚本对比两个模型的输入 shape，找出固定版和动态版在 shape 显示上的区别，并说明这对你写"分配输入张量内存"的代码有什么影响。
4. **读懂陌生输出**：找一个**分类**模型（如 ResNet 的 ONNX），用本章工具查它的输出 shape。它大概是 `[1, 1000]`——对照本章方法说清楚：这两维分别是什么含义？为什么它比 YOLOv8 的输出好懂得多？（分类后处理见第 15 章。）
