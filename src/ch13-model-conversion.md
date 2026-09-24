# 第 13 章 · 模型转换：格式、Batch 与跨平台适配

> **本章导读**：第 12 章我们学会了打开一个模型文件、读出它的输入输出形状、数据类型和元数据——相当于把一张图纸看懂了。可现实里你拿到的图纸，常常"不是这个工地能用的版本"：训练组扔给你一个 PyTorch 的 `.pt`，可你要部署的是一块瑞芯微 NPU、一台 Intel 工控机、或者一部 iPhone，它们根本不认这个格式。本章只干一件事——**把模型从"训练时的样子"翻译成"目标硬件吃得下的样子"**。我们会讲清楚：为什么必须转换、ONNX 为什么是绕不开的中转枢纽、怎么从 PyTorch 导出 ONNX、**batch 维怎么选（固定还是动态）**、怎么把 ONNX 送上各家芯片（TFLite/NCNN/RKNN/TensorRT/CoreML）、怎么用量化给模型瘦身，以及最关键的——**转换完怎么验证它没被翻译错**。本章命令与脚本以 Python/bash 为主（把它们当"命令行工具"用），Rust 侧给一个"加载转换后 ONNX 能跑通"的冒烟测试。

**学习目标**：

- 说清"训练格式 ≠ 部署格式"，以及为什么不同芯片/引擎要吃不同格式
- 建立**格式全景图**：`.pt`/SavedModel → **ONNX（枢纽）** → 各家端侧格式
- 会用 `torch.onnx.export` 导出 ONNX，讲得清 `opset_version`、`input_names`、`dynamic_axes`
- 分清**固定 batch** 与**动态 batch**，知道各自的取舍和怎么改已有 ONNX 的 batch 维
- 认识跨平台转换工具链，知道"**算子不被目标平台支持**"是头号大坑
- 理解**量化**（FP32→FP16→INT8）的动机、PTQ 与 QAT 的区别、为什么 INT8 要校准数据集
- 会写"转换前后数值比对"来验证转换正确，并用 Rust 的 `ort` 加载验证能跑通

## 13.1 为什么要转换：训练格式 ≠ 部署格式

先想清楚一件事：训练组交给你的模型文件，天生就不是为你的现场准备的。

打个比方——训练格式像**厨师的私人手写菜谱本**：写满了草稿、注释、试验记录，还夹着"上次盐放多了"的便签，只有原来那间厨房（PyTorch + Python）看得懂、用得动。而部署格式像**贴在出餐口的标准工序卡**：精简、固定、去掉了所有多余信息，任何一个灶台（推理引擎、各家芯片）照着就能做。你要上线，需要的是工序卡，不是菜谱本。

具体来说，直接拿 PyTorch 的 **训练格式**（`.pt` / `.pth`）去部署，会撞上四堵墙：

1. **和 Python/框架强绑定**：`.pt` 通常要靠 PyTorch 运行时才能加载，而完整的 PyTorch + Python 环境动辄上 GB。一块嵌入式板子既装不下，也不该为跑一个模型背这么重的包袱。
2. **要脱离 Python**：真实产线上的推理服务多用 C++/Rust 写（就像本书），追求启动快、内存省、无 GIL。它们需要一个**自包含**的模型文件，而不是一段要 `import torch` 才能跑的代码。
3. **目标硬件只认自家格式**：NVIDIA 的 GPU 吃 TensorRT engine，瑞芯微 NPU 吃 `.rknn`，苹果芯片吃 CoreML，Intel 吃 OpenVINO IR。这就像不同国家的插座——你带的插头不对，电器再好也插不上，得先过一个转换头。
4. **转换顺手做优化**：转换器在翻译的同时，会做算子融合、常量折叠、为特定芯片重排计算——**部署格式往往比原始格式跑得更快**，这是白捡的收益。

一句话收束：**训练格式是给"改模型的人"用的，部署格式是给"跑模型的机器"用的，二者目标不同，所以要转换。** 接下来的问题是——从哪个格式转到哪个格式，中间要不要经过谁？这就要看格式全景图了。

## 13.2 格式全景与转换路线图

模型格式很多，但真正要记住的关系只有一句：**几乎所有转换，都先汇聚到 ONNX，再从 ONNX 发散到各家端侧格式。** ONNX 就是那个中转枢纽。

```text
                                  ┌─► TFLite ───────► 手机 / 嵌入式 (Android/MCU)
                                  ├─► NCNN / MNN ───► 手机 / 端侧 (腾讯 / 阿里)
  .pt / .pth (PyTorch) ─┐         ├─► RKNN ─────────► 瑞芯微 NPU (RK3588 / RK3566…)
  SavedModel (TF) ──────┼─► ONNX ─┼─► OpenVINO IR ──► Intel CPU / 核显 / VPU
  .pb / .h5 ────────────┘ (枢纽)  ├─► TensorRT ─────► NVIDIA GPU / Jetson
                                  └─► CoreML ───────► 苹果 A/M 芯片 (iPhone / Mac)
```

把"源格式 → 目标格式 → 工具 → 典型硬件/引擎"列成一张大表，随查随用：

| 源格式 | 目标格式 | 常用转换工具 | 典型硬件 / 引擎 |
| --- | --- | --- | --- |
| PyTorch `.pt` / `.pth` | ONNX | `torch.onnx.export` | 跨平台中转枢纽 |
| TF SavedModel / `.pb` | ONNX | `tf2onnx` | 跨平台中转枢纽 |
| ONNX | TFLite | `onnx2tf` | 安卓 / 嵌入式（Google） |
| ONNX | NCNN（`.param`+`.bin`） | `onnx2ncnn` | 手机端（腾讯） |
| ONNX | MNN | `MNNConvert` | 手机端（阿里） |
| ONNX | **RKNN** | `rknn-toolkit2` | 瑞芯微 NPU |
| ONNX | OpenVINO IR（`.xml`+`.bin`） | `ovc` / `mo` | Intel CPU / 核显 |
| ONNX | TensorRT engine | `trtexec` | NVIDIA GPU / Jetson |
| ONNX | CoreML（`.mlpackage`） | `coremltools` | 苹果芯片 |

> **注意方向别搞反**：`tf2onnx` 是 TensorFlow **→** ONNX（源头在 TF 时用它先汇入枢纽）；`onnx2tf` 是 ONNX **→** TF/TFLite（从枢纽发散到安卓时用它）。名字长得像，方向正相反，转错方向会直接报错。

看这张图有个直觉要建立：如果没有 ONNX 这个枢纽，`N` 种源格式对 `M` 种目标格式，就要写 `N×M` 个两两之间的转换器；有了枢纽，只需 `N` 个"汇入"+ `M` 个"发散"，共 `N+M` 个。这正是下一节要讲的、ONNX 存在的意义。

## 13.3 ONNX：部署界的"中转枢纽"

第 11 章我们把 ONNX 叫作"部署界的普通话"，这一节把这个比喻讲透。**ONNX（Open Neural Network Exchange，开放神经网络交换格式）** 是一个开放标准：它规定了"一个神经网络的计算图和权重（第 11 章讲过，模型文件里就这两样）该怎么写下来"，任何框架都能按这个标准**导出**，任何引擎都能按这个标准**导入**。

再打个比方——它像航空的**中转枢纽机场**。你不会指望"每个小城市都对飞每个小城市"，那航线会爆炸；现实是大家都先飞到枢纽（北京/上海），再转到目的地。ONNX 就是模型界的枢纽：PyTorch、TensorFlow 都先转成 ONNX，各家芯片工具再从 ONNX 出发。于是本书才敢说"主用 ONNX"——它是覆盖面最广、生态最全的那个中间格式。

这里要引入一个**每次转换都会碰到、也最容易踩坑**的概念：**opset（算子集版本，operator set version）**。

ONNX 用一套标准**算子（operator）** 来描述计算——`Conv`（卷积）、`Relu`、`Add`、`MatMul`（矩阵乘）……opset 就是这套算子的**版本号**，可以理解成"ONNX 的词汇表第几版"。版本越高，收录的算子越多、越新，一些新网络结构才表达得出来。

```text
opset 越低 ←─────────────────────────────────→ opset 越高
词汇少、老引擎都认            词汇多、能表达新算子，但老引擎可能不认
   (opset 9/11)                        (opset 17/19…)
```

选 opset 是个平衡活儿，也是新手常栽的地方：
- **选太高**：导出很顺，但下游那个老版本的 NPU 工具/推理引擎不认识这么新的算子，转换直接报错。
- **选太低**：老引擎都认，但你网络里的新算子表达不出来，导出时就报"不支持"。

经验做法：**先查清目标引擎支持到哪个 opset，再挑一个它认得、又够表达你模型的版本**。工程上 `opset 12`、`13`、`17` 是常见的安全档位。记住这个词，下面导出时马上就要用到它。

## 13.4 从 PyTorch 导出 ONNX

第一步转换——把 `.pt` 汇入枢纽。PyTorch 自带 `torch.onnx.export`，把它当一个命令行小工具用即可。下面是一段可跑的脚本骨架（以本书主模型 YOLOv8 的输入 `1×3×640×640` 为例）：

```python
# export_onnx.py —— 把 PyTorch 模型导出为 ONNX
import torch

# 1) 载入训练好的模型，并切到推理模式（关键！）
model = torch.load("yolov8n.pt")   # 真实项目里通常是构建网络再 load_state_dict
model.eval()                        # eval() 会关掉 Dropout、固定 BatchNorm，否则导出的图不对

# 2) 造一个"假输入"，形状要和真实输入完全一致：1 张、RGB、640×640
dummy = torch.randn(1, 3, 640, 640)

# 3) 导出
torch.onnx.export(
    model, dummy, "yolov8n.onnx",
    opset_version=12,                 # 选一个目标引擎认得、又够用的 opset（见 13.3）
    input_names=["images"],           # 给输入起个名字，Rust/推理时按名字喂数据
    output_names=["output0"],         # 给输出起名字，按名字取结果
    dynamic_axes=None,                # 这里先固定 batch，动态 batch 下一节讲
)
print("导出完成：yolov8n.onnx")
```

三个关键参数，务必讲清楚（用一张表钉死）：

| 参数 | 作用 | 不设会怎样 |
| --- | --- | --- |
| `opset_version` | 指定用 ONNX 的哪一版算子集导出（见 13.3） | 用默认版本，可能和目标引擎对不上 |
| `input_names` / `output_names` | 给输入/输出张量起**人类可读的名字** | 会得到 `input.1`、`188` 这种自动编号，喂数据/取结果时全靠猜 |
| `dynamic_axes` | 声明哪些维度是"可变的"（下一节主角） | 所有维度都被**写死**成 `dummy` 的形状 |

`input_names`/`output_names` 尤其重要：第 14 章用 Rust 推理时，是**按名字**把张量喂进去、按名字取出来的（本章末尾的冒烟测试就会用到 `"images"` 和 `"output0"`）。给个好名字，后面所有代码都清爽。

> **一个高频坑**：忘了 `model.eval()`。训练模式下 BatchNorm 和 Dropout 的行为和推理时不同，导出的计算图会把训练态的行为固化进去，结果就是"ONNX 跑出来和 PyTorch 对不上"。导出前先 `eval()`，是雷打不动的第一步。
>
> 顺便一提：像 Ultralytics 这类成熟项目已经封装好了导出命令，一行 `yolo export model=yolov8n.pt format=onnx opset=12` 就等价于上面这段脚本。但你得知道它背后在做什么，出问题才改得动。

## 13.5 Batch 转换：固定 batch 还是动态 batch

这是本章第一个重点。**batch（批大小）** 就是张量 `[N, C, H, W]` 里那个 `N`——一次喂几张图（第 11 章讲过）。导出时，这个 `N` 有两种处理方式，直接影响性能和适配。

**固定 batch**：把 `N` 写死成一个具体数字（通常是 1）。上一节 `dynamic_axes=None` 导出的就是它，ONNX 里输入形状就是硬编码的 `[1, 3, 640, 640]`——只能一次喂 1 张，喂 2 张直接报形状错。

**动态 batch**：把 `N` 这一维声明成"可变"，ONNX 里记作 `[-1, 3, 640, 640]`（`-1` 或一个符号名表示"这维运行时才定"）。喂 1 张、8 张、16 张都行。做法是设 `dynamic_axes`：

```python
torch.onnx.export(
    model, dummy, "yolov8n_dynamic.onnx",
    opset_version=12,
    input_names=["images"],
    output_names=["output0"],
    # 键是张量名，值是 {维度下标: 维度名}。这里把输入和输出的第 0 维都设成动态
    dynamic_axes={
        "images":  {0: "batch"},
        "output0": {0: "batch"},   # 输出 [1,84,8400] → [batch,84,8400]，别忘了一起改
    },
)
```

注意：**输入设了动态，输出对应的那一维也要一起设**，否则输出的 batch 还是被写死，喂多张时会对不上。

两者怎么选？看这张对比表：

| 维度 | 固定 batch（如 =1） | 动态 batch（`-1`） |
| --- | --- | --- |
| 速度 | **通常更快**：引擎编译期就知道确切尺寸，能把内存和流水线排到最优 | 略慢：要么按最大尺寸预留、要么运行时反复适配 |
| 显存/内存 | **更省**：按确定尺寸精确分配 | 偏多：常需按可能的最大 batch 预留 |
| NPU 友好度 | **高**：很多 NPU（含 RKNN）干脆只支持固定 `batch=1` | 低：不少端侧加速器不支持动态维 |
| 灵活性 | 差：换 batch 要重新导出 | 好：一个模型应付各种 batch |

结论很实用：**端侧/NPU 部署，优先固定 `batch=1`**——更快、更省、兼容性最好；只有在服务器上做吞吐优化、需要把多张图凑一批喂 GPU 时，动态 batch 才真正划算。

### 改已有 ONNX 的 batch 维

有时你手上只有一个别人导出好的 ONNX（拿不到原始 `.pt`），batch 却不合适，得直接改这个 ONNX。两种办法：

方法一，用 `onnx` 包**直接编辑计算图**的输入维度：

```python
# fix_batch.py —— 把 ONNX 输入的第 0 维从固定 1 改成动态
import onnx

m = onnx.load("yolov8n.onnx")
dim0 = m.graph.input[0].type.tensor_type.shape.dim[0]
dim0.dim_param = "batch"          # 用符号名表示动态；若要固定成 1 则设 dim0.dim_value = 1
onnx.save(m, "yolov8n_fixed.onnx")
```

方法二，用下一节会讲的 `onnxsim`，一条命令顺手把输入形状覆盖成固定值：

```bash
# 把输入 images 固定成 1×3×640×640（同时化简）
onnxsim in.onnx out.onnx --overwrite-input-shape images:1,3,640,640
```

> **坑：光改输入维度往往不够。** 计算图中间若有依赖 batch 的 `Reshape`、`Resize` 等算子（里面可能藏着写死的常量），只改输入头、不改这些常量，跑起来就会形状崩溃。所以**能从源头（`.pt`）重新导出，就别在成品 ONNX 上硬改**；实在只能改，改完一定要用 13.9 的方法验证一遍。

## 13.6 跨平台适配：把 ONNX 送上各家芯片

第二个重点。模型汇入 ONNX 枢纽后，就该发散到目标硬件了。逐个过一遍常见目的地，都很简短——重点是记住工具名和"它吃什么、吐什么"。

**TFLite（安卓/嵌入式）**：用 `onnx2tf`。它把 ONNX 转成 TensorFlow Lite 的 `.tflite`，可直接跑在安卓和很多 MCU 上。

```bash
onnx2tf -i yolov8n.onnx        # 输出一个文件夹，含 float32/float16/int8 多个 .tflite
```

**NCNN（手机端，腾讯）**：用 `onnx2ncnn`，产出 `.param`（结构）+ `.bin`（权重）两个文件。**强烈建议先用 onnxsim 化简**（见 13.8），否则常因碎算子转换失败。

```bash
onnx2ncnn yolov8n-sim.onnx yolov8n.param yolov8n.bin
```

**RKNN（瑞芯微 NPU）**：用官方 `rknn-toolkit2`，四步走——载入、配置、构建、导出。这是端侧部署的高频场景，给出脚本骨架：

```python
# to_rknn.py —— ONNX 转 RKNN（瑞芯微 NPU）
from rknn.api import RKNN

rknn = RKNN(verbose=True)
# 1) 配置：均值/标准差要和你的预处理对齐；目标平台要写对型号
#    这里 std=255 表示"喂 0~255 的 uint8，由 NPU 内部帮你归一化到 0~1"
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform="rk3588")
# 2) 载入 ONNX
rknn.load_onnx(model="yolov8n.onnx")
# 3) 构建；do_quantization=True 走 INT8 量化，必须给校准集（见 13.7）
rknn.build(do_quantization=True, dataset="calib_list.txt")
# 4) 导出 .rknn，拷到板子上就能被 NPU 加载
rknn.export_rknn("yolov8n.rknn")
rknn.release()
```

**TensorRT（NVIDIA GPU / Jetson）**：用自带的 `trtexec`，把 ONNX 编译成一个针对本机 GPU 优化的 `engine`。

```bash
# 基础：ONNX → engine
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine
# 加 FP16 提速（见 13.7）
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
# 动态 batch：必须给出 最小/最优/最大 三档形状
trtexec --onnx=yolov8n_dynamic.onnx --saveEngine=out.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:16x3x640x640
```

**CoreML（苹果芯片）**：用 `coremltools`。这条线略特殊——**新版 coremltools 推荐直接从 PyTorch 转**（对一个 traced 模型调 `ct.convert`），ONNX→CoreML 的老通道已经弃用。所以苹果这一路常常"绕过枢纽"，直接从 `.pt` 出（Ultralytics 的 `yolo export format=coreml` 也是这么干的）。

**OpenVINO（Intel）**：用 `ovc yolov8n.onnx`（或老工具 `mo`），产出 IR 格式的 `.xml`+`.bin`，跑在 Intel CPU/核显上。

讲完这些工具，必须重重敲一下**跨平台头号大坑**：

> **算子不被目标平台支持。** 每个目标平台（尤其 NPU）能实现的算子是**子集**——ONNX 有的算子，它不一定有。就像方言翻译：源语言里有个词，目标方言里没有对应的说法，翻不过去。表现有两种：转换时直接报 `unsupported op: XXX`；或者更阴险——**转换"成功"了，但推理结果整个错乱**。对策：① 换个 opset 试试（见 13.3）；② 先用 onnxsim 把怪算子化简掉（见 13.8）；③ 改网络结构，用等价的常见算子替换那个冷门算子；④ 让不支持的算子回退到 CPU 跑（NPU + CPU 混合推理）。无论如何——**转完一定要按 13.9 验证**，别信"没报错就是对了"。

## 13.7 量化：给模型瘦身

**量化（quantization）** 是转换时最常用的"瘦身+提速"手段，也是端侧部署几乎绕不开的一步。

它做的事：把权重和计算从高精度的浮点数，换成低精度的表示。常见三档：

```text
FP32 (32 位浮点, 4 字节/个) ──► FP16 (16 位浮点, 2 字节) ──► INT8 (8 位整数, 1 字节)
      基准精度                     体积减半, 几乎无损           体积 1/4, 要校准、会掉精度
```

**为什么要量化？** 两个动机：
1. **省**：FP32 换成 INT8，模型体积和内存带宽直接降到 1/4——对内存紧张的边缘设备是刚需。
2. **快**：很多 NPU 只有**定点（integer/fixed-point）** 算力，或者定点运算比浮点快好几倍。用整数算，正好喂到它们的强项上。

打个比方：FP32 像一张 4K 高清照片，INT8 像它的缩略图——小得多、加载快，细节略有损失但"看个大概"完全够用。又或者：记账时你不会写"3.14159926 元"，而是用"分"这个整数单位记成 314 分——**用整数格子去近似连续的小数，就是量化的本质**。

### 定点近似，手算一遍

量化怎么把浮点塞进整数格子？以把一层数值映射到 `uint8`（0~255，共 256 格）为例。设这层数值范围是 `[min, max] = [-1.0, 3.0]`：

```text
每格步长  scale = (max - min) / 255 = (3.0 - (-1.0)) / 255 = 4.0 / 255 ≈ 0.0157

量化   把 x = 1.0 塞进格子：q = round((1.0 - (-1.0)) / scale) = round(127.4) = 127
反量化 再还原：x' = -1.0 + 127 × 0.0157 ≈ 0.994
```

还原回来是 `0.994`，和原值 `1.0` 差了约 `0.006`——**这点误差就是量化"掉精度"的来源**。单个数字看着无所谓，但一层层累积起来，就可能让检测框偏、置信度变，所以量化后必须评估（见本节末）。

### PTQ 还是 QAT

INT8 量化有两条路线，用一张表分清：

| | **PTQ（训练后量化）** | **QAT（量化感知训练）** |
| --- | --- | --- |
| 全称 | Post-Training Quantization | Quantization-Aware Training |
| 怎么做 | 拿训练好的模型 + 一小批**校准数据**，直接量化 | 训练时就模拟量化误差，边训边适应 |
| 要重训吗 | 不用，几分钟搞定 | 要，得回到训练流程 |
| 精度 | 一般，个别模型掉得多 | 更高，接近原始精度 |
| 谁用 | **部署工程师（你）最常用** | 精度掉太多时，请训练组出手 |

作为部署工程师，你 90% 的场景用 **PTQ**：简单、快、不碰训练。只有当 PTQ 后精度掉到不能接受，才需要训练组做 QAT。

### 校准数据集：INT8 的必需品

PTQ 走 INT8 时，你会被要求提供一个**校准数据集（calibration dataset）**——这正是上面 RKNN 脚本里 `dataset="calib_list.txt"` 的东西。

为什么需要它？回看手算：要把浮点塞进整数格子，得先知道**每层数值的范围 `[min, max]`**，才能定出 `scale`。这个范围没法凭空猜，得拿一批**有代表性的真实图**喂进去跑一遍，统计出来。比喻成**校准一杆秤**：先放几个已知重量的砝码摸清刻度，之后称东西才准。

用法很朴素：准备几百张覆盖典型场景的真实图片（不用标注），列成一个清单文件喂给量化工具即可。

> **坑：校准集不代表实际场景，量化后精度会崩。** 如果你的现场是夜间红外，却拿白天彩色图去校准，统计出的数值范围完全对不上，量化后模型可能"整个瞎掉"。校准集要**贴近真实部署分布**。
>
> **铁律：量化后必须重新评估精度。** 在验证集上重跑一遍，对比量化前后的 mAP/准确率掉了多少，掉幅能接受才上线。量化是拿精度换速度和体积的**交易**，你得知道自己付出了多少。

## 13.8 精简与算子问题

转换失败或结果不对，八成栽在"算子"上。这一节讲两把常用的"钳子"。

**onnx-simplifier（`onnxsim`）**：ONNX 常常带一堆冗余——训练框架导出时会留下大量碎算子、可提前算好的常量、多余的 `Reshape`。`onnxsim` 做**常量折叠**（能提前算的先算掉）、去冗余、把碎算子合并成规整的一块。很多下游转换器（`onnx2ncnn`、`rknn-toolkit2` 等）都**强烈建议先 simplify**，否则容易转换失败。

```bash
pip install onnxsim
onnxsim yolov8n.onnx yolov8n-sim.onnx     # 输入 → 化简后输出
```

化简既能提升转换成功率，也能让模型跑得更快——是转换流程里几乎"免费"的一步，养成习惯先过一遍。

**opset 不匹配**：13.3 讲过，这里再点一次——它是转换报错的常见根因。目标引擎报"不认识某算子"时，先想想是不是 opset 选高了/选低了，换个版本重新导出往往就通了。

**自定义算子 / 不支持的算子**：训练侧若用了某框架特有的、或研究性质的冷门算子，ONNX 里没有标准对应，导出时会变成一个"自定义算子"——ONNX 勉强收下，下游转换器却更不认识。这本质上是 13.6 那个"方言翻不过去"的坑在源头的版本。对策一致：换 opset、用 onnxsim 化简、或改网络结构避开它。**转换出问题时，第一件事永远是定位到底是哪个算子在捣乱。**

## 13.9 验证转换正确性

最后，也是最容易被跳过、却最不该跳过的一步。**转换只是"翻译"，翻译完必须回译校对**——同一份输入，分别喂给转换前后的两个模型，比对输出数值对不对得上。

原则：**同输入 → 两模型 → 比最大绝对误差**。

```python
# verify.py —— 比对 PyTorch 原模型 与 导出的 ONNX 的输出
import numpy as np
import torch
import onnxruntime as ort

# 1) 同一份随机输入（真实项目里最好再补几张真实图）
x = np.random.rand(1, 3, 640, 640).astype(np.float32)

# 2) 原模型（PyTorch）跑一遍
model.eval()
with torch.no_grad():
    ref = model(torch.from_numpy(x)).numpy()

# 3) 转换后的 ONNX 跑一遍（按导出时起的输入名 "images" 喂）
sess = ort.InferenceSession("yolov8n.onnx")
out = sess.run(None, {"images": x})[0]

# 4) 比对：最大绝对误差应当很小
max_err = np.abs(ref - out).max()
print(f"最大绝对误差 = {max_err:.6f}")
assert max_err < 1e-3, "误差过大，转换可能出错了！"
```

误差阈值怎么定，看你做了什么：

| 转换类型 | 期望的最大绝对误差 | 说明 |
| --- | --- | --- |
| FP32 → FP32（纯格式转换） | `< 1e-5`，近乎一致 | 只是换壳，数值应几乎不变 |
| 经过 onnxsim / FP16 | `< 1e-3` 量级 | 有精度损失，放宽一些 |
| INT8 量化 | **别只看逐元素误差** | 用任务指标（mAP/准确率）比，见 13.7 |

> **两个验证坑**：① 别只用随机输入——随机数可能碰不到真实数据才会触发的分支，务必**补几张真实图**再比一遍。② 别忘了**预处理要一致**（第 10 章的 letterbox、归一化），输入都没对齐，比出来的误差没有意义。

### Rust 侧：加载转换后 ONNX 的冒烟测试

转换、验证都在 Python 里做完了，最后在 Rust 里确认一下——**这个 ONNX 能被我们的推理引擎加载、能前向、输出形状对**。用第 14 章介绍的 `ort`（ONNX Runtime 的 Rust 绑定）写一个最小冒烟测试：

```rust,ignore
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

fn main() -> ort::Result<()> {
    // 1) 加载转换/导出好的 onnx
    let mut session = Session::builder()?.commit_from_file("yolov8n.onnx")?;

    // 2) 造一个假输入：全 0 的 [1,3,640,640]，只为验证"能跑通"，不关心结果对不对
    let x = Array4::<f32>::zeros((1, 3, 640, 640));

    // 3) 按导出时起的输入名 "images" 喂进去（怎么查输入名？第 12 章教过）
    let outputs = session.run(ort::inputs![
        "images" => TensorRef::from_array_view(&x)?
    ])?;

    // 4) 取出名为 "output0" 的输出，打印形状，确认是我们期待的 [1, 84, 8400]
    let output = outputs["output0"].try_extract_array::<f32>()?;
    println!("跑通了！输出形状 = {:?}", output.shape()); // 期望 [1, 84, 8400]
    Ok(())
}
```

跑通并打印出 `[1, 84, 8400]`，就说明这个转换后的模型**能被 Rust 加载、能前向、形状符合预期**——转换链路到此闭环。

> `ort` 的 API 随版本略有出入（方法名以你 `cargo add ort` 拉到的版本为准）。这里只做一次"能加载、能前向、形状对"的冒烟测试；**完整、可长期维护的推理代码，以及 CPU/GPU/NPU 性能优化，是第 14 章的主题**。

**再进一步：跨平台 / 跨 EP 的一致性。** 上面比的是"转换前 vs 转换后"；系统上线后还有一个更隐蔽的问题——**同一个 ONNX，在不同机器、不同执行提供器（CPU / CUDA / CoreML，第 14 章）上跑，输出还对得上吗？** 答案是：**应当在容差内一致，但几乎不可能逐位相同**。因为不同硬件的浮点累加顺序、是否走 FP16、算子的具体实现都有差异（13.7 的量化、第 14 章的 EP 只会放大它）。由此有两条实践准则：

- **别指望逐位复现**：换台机器、换个 EP，用 `==` 死磕必然翻车——一律比**最大绝对误差**、留容差（同上表的 `1e-3` 量级）。
- **在部署目标平台上固定基线**：拿一组真实图在**上线要用的那套硬件/EP**上跑出输出、存成基线，这正是第 31 章 **golden 测试**要比对的"标准答案"。也正因为存在这种平台差异，golden 测试才**必须带容差**、不能用严格相等。

一句话：**转换验证保证"换了格式结果没变"，跨平台一致性保证"换了机器结果没跑偏"——两道关都用"同输入、比容差"这同一招守住。**

## 13.10 小结

- **训练格式 ≠ 部署格式**：`.pt` 和 Python/框架强绑定、笨重，而目标硬件只认自家格式——所以要转换，转换还能顺手做优化。
- **ONNX 是中转枢纽**：几乎所有转换都"先汇入 ONNX，再发散到端侧格式"，把 `N×M` 个转换器省成 `N+M` 个；`opset` 是它的"词汇表版本"，选错是报错常见根因。
- 用 `torch.onnx.export` 导出 ONNX，记牢 `model.eval()`、`opset_version`、`input_names/output_names`、`dynamic_axes` 四件事。
- **Batch**：端侧/NPU 优先**固定 `batch=1`**（更快、更省、兼容性好），服务器做吞吐才用**动态 batch**；改已有 ONNX 的 batch 维能改但有坑，最好从源头重导。
- **跨平台**：TFLite/NCNN/RKNN/TensorRT/CoreML 各有工具链；头号大坑是"**算子不被目标平台支持**"，可能报错也可能悄悄算错。
- **量化**：FP32→FP16→INT8 换体积和速度；INT8 走 PTQ 要**校准数据集**、且**必须重新评估精度**。
- **验证是必做项**：同输入喂转换前后两模型，比最大绝对误差（FP32 应 `<1e-5`，量化改用 mAP）；最后用 Rust 的 `ort` 加载做一次冒烟测试。

## 13.11 练习

1. **动态 vs 固定**：拿一个 `yolov8n.pt`（或任意 PyTorch 模型），分别导出"固定 batch=1"和"动态 batch"两个 ONNX。用第 12 章的方法读出两者的输入形状，确认一个是 `[1,3,640,640]`、另一个是 `[batch,3,640,640]`。再想想：如果目标是一块只支持 `batch=1` 的 RKNN NPU，你该选哪个？
2. **手算量化**：某一层数值范围是 `[min, max] = [-2.0, 6.0]`，量化到 `uint8`（256 格）。先算出 `scale`，再把 `x = 0.0` 和 `x = 3.5` 分别量化成整数、再反量化回来，算出各自的绝对误差。（提示：套 13.7 的两个公式。）
3. **验证转换**：用 13.9 的 `verify.py` 思路，比对"原始 ONNX"和"onnxsim 化简后的 ONNX"两者输出的最大绝对误差（都用 `onnxruntime` 跑即可，不必装 PyTorch）。误差应当在 `1e-3` 量级以内。想一想：如果误差大到 `0.5`，最可能是哪一步出了问题？
4. **查算子支持**：挑一个目标平台（如 TensorRT 或某型号 RKNN），上网找它的"支持算子列表"，再对照你的 ONNX 里用到的算子（可用 `onnx` 包遍历 `graph.node` 打印每个算子的 `op_type`），列出可能不被支持的算子。这一步就是 13.6 那个头号大坑的"提前排雷"。
