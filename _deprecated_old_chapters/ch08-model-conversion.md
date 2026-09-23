# 第 8 章 · 模型转换：格式、Batch 与跨平台适配

> **本章导读**：训练模型的人用 PyTorch，跑模型的设备却要 ONNX、TFLite、RKNN……模型格式的不统一是部署工程师日常的主线剧情。本章教你这套"翻译"工作怎么做：以 ONNX 为中转站，把模型从训练框架翻成部署格式，看懂模型的输入输出（四要素），处理动态 batch 与静态 batch 的转换，最后盘点各平台的适配要点。学完本章，给你任何一个模型文件，你都能说出"它吃什么、吐什么、怎么跑到目标设备上"。

**学习目标**：
- 理解为什么需要模型转换，以及 ONNX 为什么是业界中转站
- 会用 Netron 和代码（Python / Rust）查看模型的输入名、输出名、shape、dtype
- 掌握动态 shape / 动态 batch 的概念，会做固定 batch 转换
- 记住各部署平台（ONNXRuntime/TensorRT/TFLite/NCNN/RKNN/OpenVINO/CoreML）的定位和 Rust 可用性
- 独立完成 YOLOv8n 从 PyTorch 到 ONNX 的完整转换流程

## 8.1 为什么模型需要转换

第 7.5 节说过：模型文件 = 权重 + 计算图。问题在于，**每家存这两样的"方言"都不一样**：

```text
PyTorch 的 .pt    ← 训练者的母语，但里面混着 Python 代码，设备跑不了
TensorFlow 的 .pb ← 谷歌系方言
        ↓ 谁也不服谁
部署平台各有各的口味：
  英伟达 GPU 要 TensorRT 引擎
  瑞芯微 NPU 要 RKNN
  安卓手机爱 TFLite
  腾讯 NCNN 主打手机 CPU
```

类比出国旅游：中国是国标插座，欧洲是欧标插座——**你的电器（训练产物）和酒店插座（部署平台）不配套，就需要一个转换插头**。模型转换器（converter）就是这个转换插头。

而转换插头自己也得有个标准件，否则 N 个训练框架 × M 个部署平台要造 N×M 个转换器。这个标准件就是 ONNX。

## 8.2 ONNX：业界的普通话

**ONNX（Open Neural Network Exchange）** 是微软和 Meta 发起的开放格式，设计目标只有一个：**让任何框架的模型都能翻译成它，再让任何平台都能从它翻译出去**。

它的技术形态（不必深究，有印象即可）：

- 文件本质是 **protobuf 序列化**的"计算图 + 权重"：graph 里列出所有节点（每个节点是 `Conv`/`Relu`/`MatMul` 这样的算子），initializer 里存权重张量
- **算子集版本（opset）**：ONNX 的"成语词典版本号"。opset 18 收录的算子比 opset 11 多，推理引擎说"我只支持到 opset 17"，你拿 opset 18 的模型给它就可能报"不认识某算子"。转换时选目标平台支持的 opset 是常规操作

ONNX 成为事实标准的原因：所有主流框架都支持导出（PyTorch 官方内置）、所有主要推理引擎都支持加载、还有神器 Netron 能可视化。我们后文的默认工作流：

```text
PyTorch (.pt) ──export──→ model.onnx ──convert──→ TFLite / RKNN / NCNN / TensorRT ...
                              │
                              └─→ 我们的 Rust 程序（tract / ort 直接吃）
```

## 8.3 训练框架 → ONNX 实战

以 YOLOv8n 为例（本书贯穿模型）。转换工具就是 ultralytics 包自带的命令：

```bash
pip install ultralytics onnx onnxsim
yolo export model=yolov8n.pt format=onnx
# 产出：yolov8n.onnx
```

这背后通用的是 PyTorch 的 `torch.onnx.export`，它的三个关键参数值得理解（看代码即可，不需要会写）：

```python
torch.onnx.export(
    model,               # 要导出的模型
    example_input,       # 一个样例输入（导出时跑一次，追踪计算图）
    "model.onnx",
    opset_version=17,    # 目标算子集版本（按部署平台支持程度选）
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes={       # 哪些维度是"活"的
        "images": {0: "batch"},        # 第 0 维叫 batch，可以变
        "output0": {0: "batch"},
    },
)
```

- `opset_version`：和第 8.2 节的词典版本对应，宁低勿高（兼容性）
- `dynamic_axes`：声明哪些维度是动态的（下详）
- `input_names/output_names`：给输入输出起名字——推理时按名字喂数据就靠它（第 9 章）

其他框架一句话：TensorFlow/Keras 用 `tf2onnx` 或 TFLite 自带转换器；大部分情况你拿到手的已经是 onnx，不需要自己导。

> 本节出现的 Python 命令，全部只当"命令行工具"敲，不要求理解 Python 语法——就像你不需要会造冰箱才能使用冰箱。

## 8.4 看懂模型：Netron

**Netron 是模型文件的"X 光机"**（github.com/lutzroeder/netron，可下载客户端也有网页版）。拿到任何模型，第一步拖进 Netron：

```bash
pip install netron
netron yolov8n.onnx   # 浏览器自动打开
```

你会看到：

```text
输入节点（左侧）： images: float32[1, 3, 640, 640]      ← 四要素来了三个
                 属性：名字 / 数据类型 / 形状
输出节点（右侧）： output0: float32[1, 84, 8400]       ← 第 10 章的主战场
中间（中间）： 密密麻麻几百个算子节点（Conv → SiLU → Concat …）
权重（灰色）： 每个 Conv 节点后面跟着 w/b 小方块
```

这张图告诉你的四要素——**输入名、输入 shape、输入 dtype、输出名、输出 shape、输出 dtype**（好吧是六要素）——是写推理代码的全部已知条件。第 9、10、11 章的代码都建立在"Netron 看来的这行字"之上。

## 8.5 模型"体检"工具链

转换出来的 onnx 先别急着用，三道体检：

```bash
pip install onnx onnxsim

# ① 合法性检查：文件结构是否完整、算子是否符合规范
python3 -c "import onnx; m = onnx.load('yolov8n.onnx'); onnx.checker.check_model(m); print('OK')"

# ② 简化：常量折叠（算一遍能提前算掉的节点直接算掉）、去除死节点
onnxsim yolov8n.onnx yolov8n_sim.onnx

# ③ 形状推理：把所有中间张量的形状算出来标注上（Netron 里显示不全时）
python3 -c "import onnx; m = onnx.shape_inference.infer_shapes(onnx.load('yolov8n_sim.onnx')); onnx.save(m, 'yolov8n_sim.onnx')"
```

onnxsim 尤其重要：训练框架导出的图经常有冗余（比如训练专用的节点忘了删、常量表达式没折叠），简化后的模型更小、加载更快、兼容性更好。**工作流上把它当必经步骤**。

## 8.6 用代码拿到输入输出（四要素的 N 种拿法）

Netron 适合人用眼睛看，但程序里要自动获取——两种代码方式都要会。

**方法一：Python onnx 库**（转换期用）：

```python
import onnx
m = onnx.load("yolov8n_sim.onnx")
g = m.graph
print("== 输入 ==")
for i in g.input:
    print(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
print("== 输出 ==")
for o in g.output:
    print(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
```

输出示例：

```text
== 输入 ==
images ['batch', 3, 640, 640]
== 输出 ==
output0 ['batch', 84, 8400]
```

`'batch'` 是符号名（动态维度），数字是静态维度——一眼看出哪些维可变。

**方法二：Rust + tract**（部署程序里用，本书主推）：

```rust
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx().model_for_path("yolov8n_sim.onnx")?;

    println!("== 输入 ==");
    for (i, fact) in model.input_outlet_facts()?.iter().enumerate() {
        println!("输入{}: {:?}", i, fact);
    }
    println!("== 输出 ==");
    for (i, fact) in model.output_outlet_facts()?.iter().enumerate() {
        println!("输出{}: {:?}", i, fact);
    }
    Ok(())
}
```

`TypedFact` 的 Debug 输出会包含形状（含符号维）和数据类型。

**方法三：Rust + ort**（用 ONNX Runtime 时）：

```rust
use ort::Session;

fn main() -> anyhow::Result<()> {
    let session = Session::builder()?.with_model_from_file("yolov8n_sim.onnx")?;
    for i in &session.inputs {
        println!("输入: 名={} 类型={:?} 形状={:?}", i.name, i.input_type, i.shape);
    }
    for o in &session.outputs {
        println!("输出: 名={} 类型={:?} 形状={:?}", o.name, o.output_type, o.shape);
    }
    Ok(())
}
```

三个方法按场景选：转换调试期用方法一，Rust 程序启动自检用方法二/三。**好习惯：推理程序启动时打印四要素，日志里留一份，排障时省得翻文件**。

## 8.7 动态 Shape 与 Batch 转换

### 静态 vs 动态

从 shape 看，模型分两种：

```text
静态：float32[1, 3, 640, 640]     ← 尺寸焊死，只能喂 640×640、一次一张
动态：float32[batch, 3, h, w]    ← 维度是符号，batch/宽高都能变
```

各有优劣：

| | 静态 shape | 动态 shape |
| --- | --- | --- |
| 灵活性 | 只能固定尺寸 | 尺寸/数量随便变 |
| 引擎优化 | 能做编译期优化（内存预分配、算子融合激进） | 优化保守，每次推理检查形状 |
| 嵌入式兼容 | **很多 NPU 编译器只认静态** | 常常不支持 |
| 典型场景 | 固定摄像头的工业检测 | 服务端处理各种尺寸请求 |

### 动态 batch 怎么来

导出时声明（8.3 节的 `dynamic_axes={0: "batch"}`）。想改成固定 batch=1？思路：

1. 用 onnx 库读图，把输入节点的 shape 改成 `[1, 3, 640, 640]`
2. 图里所有和 batch 相关的 `Reshape`/`Concat` 节点的常量参数同步改掉
3. 重新做形状推理，检查每个节点没破

工具上可以直接用 ONNX 官方的低阶 API 手写（几十行 Python），也有现成的 `polygraphy` / `onnx-graphsurgeon`（英伟达出品）：

```bash
pip install onnx-graphsurgeon
# 示例： surg 出图、改 shape、清理、保存（以官方文档的 reshape 示例为准）
```

**什么时候改固定 batch**：目标平台只支持静态（RKNN 早期版本、部分 TensorRT 工作流）、或你要榨性能（静态 shape 下 TensorRT 能做更狠的优化）。什么时候保留动态：服务端、批量大小不确定的离线处理。

### batch 转换的工程含义

"batch 转换"在工业语境下常指两件事，别混：

1. **动态 batch → 固定 batch**（上面讲的，改 shape）
2. **多 batch 推理的预处理/后处理配套**：你的 Rust 代码要能一次处理 `[4, 3, 640, 640]` 的输入和 `[4, 84, 8400]` 的输出——ndarray 上就是多一层循环，第 11.6 节会演示。

## 8.8 转各平台：一张大表

这是本章的"地图"。不要求记住细节，但要能在给定硬件时秒选路线：

| 平台 | 厂商/出身 | 目标硬件 | Rust 可用性 | 转换命令形态 |
| --- | --- | --- | --- | --- |
| ONNXRuntime | 微软 | CPU/GPU 通吃 | **极好**（ort crate） | 免转，直接吃 onnx |
| tract | 开源（纯 Rust） | CPU | **极好**（tract-onnx） | 免转，直接吃 onnx |
| TensorRT | 英伟达 | NVIDIA GPU | 一般（FFI/包装） | `trtexec --onnx=model.onnx --saveEngine=model.engine` |
| OpenVINO | 英特尔 | Intel CPU/核显 | 好（官方绑定） | `mo --input_model model.onnx` |
| TFLite | 谷歌 | 安卓/嵌入式 MCU | 一般（tflite-rs） | Python `tf.lite.TFLiteConverter` |
| NCNN | 腾讯 | 手机 CPU/GPU | 一般（C++ 为主） | `pnnx model.onnx` 或 `onnx2ncnn` |
| RKNN | 瑞芯微 | RK3588 等 NPU | 一般（官方 C API + FFI） | Python `rknn-toolkit2` 的 `rknn.build()` |
| CoreML | 苹果 | iOS/macOS | 弱（除非做苹果 App） | `coremltools.convert` |

经验之谈：

- **x86 工控机/服务器**：ONNXRuntime（ort）或 OpenVINO（Intel），ONNX 直接喂，最省事
- **NVIDIA Jetson**：TensorRT，性能是 ONNXRuntime 的 1.5~3 倍，值得折腾
- **瑞芯微/晶晨等 ARM 盒子**：厂商 NPU SDK（RKNN 等），ONNX 先转专用格式
- **纯 Rust 洁癖/交叉编译**：tract，零 C 依赖， musl 静态编译一把梭

## 8.9 跨平台适配的常见坑

1. **算子不支持**：某平台没有你的图里某个算子（常见于新 opset 的新算子或冷门算子）。报错形如 `Unsupported operator: xxx`。对策：升级/降级 opset 重导、用 onnxsim 简化、把该算子拆成基础算子等价实现、或换推理引擎。
2. **输入 layout 不一致**：ONNX 习惯 NCHW，TFLite 爱 NHWC。模型内部转置节点能吸收差异，但手写预处理时一定以 Netron 看到的为准（第 5.6 节）。
3. **数据类型**：很多嵌入式路径只收 FP16 甚至 INT8（第 8.10），FP32 模型要加量化步骤。
4. **opset 版本**：目标引擎支持到 17 就别导 18；太老的引擎反过来嫌你版本新。卡在中间最难受——这也是为什么 onnx 文件里会注明 `opset_import`。

## 8.10 量化：FP32/FP16/INT8 快速认知

类比：记账精确到分（FP32）、精确到毛（FP16）、约到十块（INT8）。模型权重的存储和计算精度越低：

- 体积：FP32 → FP16 减半 → INT8 再减半
- 速度：专用硬件上 INT8 算力经常是 FP32 的 2~4 倍
- 精度：通常掉一点点（1% 以内），INT8 掉得多些但常可接受

为什么敢量化？神经网络"抗造"：它学的是统计规律，个别权重的微小变化被亿级参数的冗余吸收掉了（类比：把一首歌的音量调低 1%，没人听得出来）。

两种量化路线（知道名字即可）：

- **PTQ（训练后量化）**：拿训好的模型直接量化，几行代码，掉点可接受时用——部署端最常用
- **QAT（量化感知训练）**：训练时就模拟量化误差，精度保持最好，流程重——精度敏感的量产项目用

## 8.11 实战：YOLOv8n 完整转换流程

把本章串起来走一遍：

```bash
# ① 准备环境
pip install ultralytics onnx onnxsim netron

# ② 导出
yolo export model=yolov8n.pt format=onnx
# → yolov8n.onnx

# ③ 体检 + 简化
python3 -c "import onnx; onnx.checker.check_model(onnx.load('yolov8n.onnx')); print('合法')"
onnxsim yolov8n.onnx yolov8n_sim.onnx

# ④ 打印四要素
python3 - <<'EOF'
import onnx
g = onnx.load("yolov8n_sim.onnx").graph
def dims(t): return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
print("输入:", [(i.name, dims(i)) for i in g.input])
print("输出:", [(o.name, dims(o)) for o in g.output])
EOF
# 预期输出：输入 images ['batch',3,640,640]，输出 output0 ['batch',84,8400]

# ⑤ Netron 肉眼确认（可选但强烈推荐）
netron yolov8n_sim.onnx
```

然后 Rust 侧再确认一遍（8.6 的代码跑通），**两相印证**，就可以放心进入第 9 章了。

## 8.12 小结

- 模型转换的本质：训练框架方言 → ONNX 普通话 → 部署平台方言。
- 四要素（输入输出名/shape/dtype）是写推理代码的全部已知条件：Netron 看人用、onnx 库/tract/ort 看程序用。
- 动态 shape 灵活、静态 shape 快且兼容嵌入式；batch 转换=改 shape + 改配套 Reshape 常量。
- 平台选择：x86 用 ONNXRuntime/OpenVINO，NVIDIA 用 TensorRT，ARM NPU 用厂商 SDK。
- 量化是部署的性能杠杆：PTQ 够用、QAT 保精度。

## 8.13 练习

1. 下载一个非 YOLO 的 onnx 模型（比如 onnx/models 仓库里的 squeezenet），用 Netron 找到它的四要素，再用 Rust tract 打印一遍对比。
2. 故意用旧版 opset 导出 YOLOv8n（查 ultralytics 文档找参数），观察 onnxsim 的日志输出有什么变化。
3. 写一个小工具：遍历任意 onnx 文件，统计算子种类出现次数（提示：遍历 `graph.node` 的 `op_type` 字段，用 Python 最方便）。
4. 思考并回答：为什么嵌入式 NPU 普遍偏爱静态 shape？（提示：从编译期内存规划角度想）
