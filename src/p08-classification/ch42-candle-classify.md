# 第 42 章 用 Candle 加载预训练模型做图像分类

> **本章导读**
> - 学完你能：用 **candle**（0.11.0）搭一个小 CNN，把权重**存成 `.safetensors`、再加载回来**，然后跑通一整条**分类推理流水线**——预处理 → 前向 → softmax → argmax → 类别。
> - 你会亲手实现"加载预训练权重"这套机制：它和从 Hugging Face 下载一个 ResNet 的 `.safetensors` 来用，**是同一套 API、同一个套路**。
> - 我们用**小模型 + 合成小图**，让你**现在、在纯 CPU 上、不下载任何大模型**就能跑通全流程。代码在本机 `cargo run --release` 实测，正文数字均为真实输出。
> - 本章是全书第一次"用深度学习框架做一件真实任务"，我们把每一步都讲透。

## 场景：相册是怎么把照片分成"猫 / 狗 / 美食"的

[第 1 章](../p01-start/ch01-what-is-cv.md) 开篇就提过手机相册自动分类的例子。它背后是一个**图像分类器**：给一张图，吐出一个**类别标签**。今天的分类器几乎都是深度神经网络，而且很少有人从零训练——大家都直接用**别人在海量图片上训练好的模型**。

> 📖 **术语**：**预训练模型（pretrained model）**。别人已经在超大数据集（如 **ImageNet**，上百万张图、1000 个类别）上训练好、并把学到的**权重**公开出来的网络。你把权重下载下来加载进同样结构的网络，就能直接用它识图——相当于白捡了一个"识图大脑"。这就是 [第 41 章](../p07-dl-basics/ch41-training-tricks.md) 讲的迁移学习的地基。

那这些权重存在哪、怎么加载进 Rust？答案是 **safetensors** 文件（[第 39 章](../p07-dl-basics/ch39-dl-ecosystem.md) 介绍过），而 candle 原生就吃这种格式。

> ⚠️ **注意（本章的诚实边界）**：真正的 ResNet / MobileNet / ViT 是几十上百 MB 的权重，要联网从 Hugging Face 下载。为了让你**此刻就能跑通、且完全可复现**，本章改用一个**我们自己训练的小 CNN + 代码合成的小图**（延续 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 的三类形状：竖条 / 横条 / 方块）。**请记住：唯一的区别是"模型多大、数据从哪来"；"存权重 → 加载权重 → 预处理 → 前向 → 出类别"这条链路和跑真实大模型一模一样。** 学会了这个小例子，你就学会了加载任何 safetensors 模型。

## 原理：一条分类推理流水线

**直觉**：分类推理就像"把一张图翻译成一个词"。图片先被**收拾成模型认得的规范输入**，喂进网络算出每个类别的**分数**，分数经 softmax 变成**概率**，最后取概率最大的那个当答案。

```text
一张图片
   │  ① 预处理（务必和训练时一致！）
   ▼  resize→28x28 · /255 归一化 · HWC 转 CHW · 加 batch 维 · 按通道标准化
 [1, 3, 28, 28] 张量
   │  ② 前向 forward
   ▼
 logits [1, 3]     例：[2.1, -3.0, 0.4]  ← 原始分数，未归一、可正可负
   │  ③ softmax
   ▼
 概率 [0.83, 0.01, 0.16]   ← 全为正、加起来等于 1
   │  ④ argmax（取最大那项）
   ▼
 类别索引 0  →  查标签表  →  "竖条"
```

**第 ① 步预处理**就是 [第 31 章](../p05-numeric/ch31-image-as-tensor.md) 那一整套：把 `RgbImage` 变成 `f32` 张量，`/255` 归一化，**HWC 换成 CHW**，加上 **batch 维**凑成 `[N, C, H, W]`，再**按通道标准化**。这里有一条铁律：

> ⚠️ **注意**：**推理的预处理必须和训练时逐字一致。** 尺寸、`/255`、RGB 顺序、每个通道减的均值除的标准差——错一样，模型看到的就是"另一个分布"的数据，结果会莫名其妙地差。真实 ImageNet 模型常用 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`（见 [第 31 章](../p05-numeric/ch31-image-as-tensor.md)）；**我们这个小模型训练时用的是 `mean=std=0.5`**（把 `[0,1]` 映射到 `[-1,1]`），所以推理也必须用 `0.5`。用哪个模型，就配它训练时的那套。

**第 ③ 步 softmax**，把任意实数的 logits 压成一组概率。为数值稳定，实现时先减去每行最大值 \\( m \\)（否则 \\( \exp \\) 容易溢出）：

\\[ p_c = \frac{\exp(z_c - m)}{\sum_{j} \exp(z_j - m)}, \quad m = \max_j z_j \\]

减掉 \\( m \\) 不改变结果（分子分母同乘一个常数），却避免了 `exp` 爆炸——candle 的 `softmax` 内部已经这么做了，我们直接调用即可。

## 动手：建小 CNN → 存 safetensors → 加载回来 → 分类

> ⚠️ **注意**：本章代码基于 **candle 0.11.0**（`candle-core` + `candle-nn`），纯 CPU。candle 仍在演进，跨版本 API 可能变动，运行前请对照你所用版本的文档。完整独立工程见 `code/dl_labs/ch42_candle_classify/`，**`cd` 进去 `cargo run --release`** 即可复现（首次编译 candle 较久，耐心等）。

`Cargo.toml`：

```toml
[package]
name = "ch42_candle_classify"
version = "0.1.0"
edition = "2021"

# 空的 [workspace] 让本工程成为独立工作区，不被上层 code/Cargo.toml 收编。
# （你在别处 cargo new 时不需要这行；这里因为嵌在本书 code/ 目录下才加。）
[workspace]

[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
image = "0.25"
```

### 第一步：定义小 CNN（骨干 + 分类头）

网络分两部分：**骨干（backbone）** `conv1 → conv2 → fc1` 负责"看图、抽特征"，输出一个 64 维特征向量；**分类头（head）** 一个线性层，把特征映射成每类的分数。这个"骨干 + 头"的切分是下一章微调的关键，先记住它。

```rust
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::softmax;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

const SZ: usize = 28;                              // 模型输入边长
const CLASSES: [&str; 3] = ["竖条", "横条", "方块"]; // 类别索引→名称（ImageNet 1000 类标签表的迷你版）

struct SmallCnn { conv1: Conv2d, conv2: Conv2d, fc1: Linear, head: Linear }

impl SmallCnn {
    fn load(num_classes: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() }; // padding=1 → 卷积不改变边长
        Ok(Self {
            conv1: conv2d(3, 8, 3, cfg, vb.pp("conv1"))?,   // 3→8 通道，3x3 卷积
            conv2: conv2d(8, 16, 3, cfg, vb.pp("conv2"))?,  // 8→16 通道
            fc1:   linear(16 * 7 * 7, 64, vb.pp("fc1"))?,   // 展平 784 → 64 维特征
            head:  linear(64, num_classes, vb.pp("head"))?, // 64 → 类别数（输出 logits）
        })
    }
    // 骨干：输出 64 维特征
    fn features(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?; // [N,3,28,28] → [N,8,28,28]
        let x = x.max_pool2d(2)?;               // → [N,8,14,14]
        let x = self.conv2.forward(&x)?.relu()?;// → [N,16,14,14]
        let x = x.max_pool2d(2)?;               // → [N,16,7,7]
        let x = x.flatten_from(1)?;             // 展平 → [N,784]
        self.fc1.forward(&x)?.relu()            // → [N,64]
    }
    // 分类头：特征 → logits
    fn forward(&self, x: &Tensor) -> Result<Tensor> { self.head.forward(&self.features(x)?) }
}
```

注意 candle 的惯用法：`conv2d(in, out, k, cfg, vb.pp("名字"))` 和 `linear(in, out, vb.pp("名字"))`。这里的 `vb.pp("conv1")` 是**给这层权重起个前缀名**——它会对应到 safetensors 文件里叫 `conv1.weight` / `conv1.bias` 的张量（马上你会亲眼看到）。`Module` trait 提供了 `.forward()`，`max_pool2d(2)` 是 2×2 下采样，`flatten_from(1)` 把第 1 维起全部拉平。

> 📖 **术语**：**VarBuilder 与 VarMap**。`VarMap` 是"参数仓库"，装着网络所有**可训练权重**（每个是一个 `Var`）；`VarBuilder` 是"取参数的工具"，`.pp(前缀)` 用来给不同层的权重分门别类地命名/定位。构建网络时，`VarBuilder` 从 `VarMap`（新建随机权重）或从 `.safetensors` 文件（加载现成权重）里把每层的 weight/bias 取出来——**这正是"加载预训练模型"的核心机关**。

### 第二步：快速训练，得到"真会认形状"的权重

要演示"加载预训练权重"，得先有权重。我们把这个小 CNN 在合成的三类形状上**快速训几十轮**（训练循环的原理见 [第 40 章](../p07-dl-basics/ch40-burn-train.md) 和 [第 41 章](../p07-dl-basics/ch41-training-tricks.md)，这里不再展开——本章的主角是随后的"存 → 取 → 推理"）。合成数据、随机数发生器 `Lcg`、确定性初始化 `det_init` 等辅助函数比较朴素，完整版见配套工程；训练主循环长这样：

```rust
let dev = Device::Cpu;
let vm = VarMap::new();                                     // 参数仓库
let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);    // 从仓库取参数来搭网络（随机初始化）
let net = SmallCnn::load(CLASSES.len(), vb)?;
det_init(&vm, 42)?;                                         // 确定性初始化（见下方"坑"）

let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
for epoch in 1..=40 {
    let loss = cross_entropy(&net.forward(&xtr)?, &ytr)?;  // 交叉熵内部含 softmax；标签是 U32
    opt.backward_step(&loss)?;                             // 反向求梯度 + 更新，一行搞定
    if epoch % 8 == 0 { println!("  epoch {epoch:>2}  loss = {:.4}", loss.to_scalar::<f32>()?); }
}
```

`AdamW::new(vm.all_vars(), ...)` 把仓库里**所有**参数交给优化器；`opt.backward_step(&loss)` 一行完成"反向 + 更新"（[第 41 章](../p07-dl-basics/ch41-training-tricks.md) 讲过这个用法）。训练输出：

```text
开始训练小 CNN（合成三类形状，AdamW，lr=1e-3）...
  epoch  8  loss = 0.9669
  epoch 16  loss = 0.6182
  epoch 24  loss = 0.2267
  epoch 32  loss = 0.0429
  epoch 40  loss = 0.0070
训练完成：训练准确率 = 100.0%   测试准确率 = 100.0%
```

### 第三步：存成 safetensors，看看里面有什么

```rust
vm.save("cnn.safetensors")?; // VarMap::save 内部就是 candle_core::safetensors::save
// 直接把文件读成 name→Tensor 字典，打印每个张量的名字和形状：
let dict = candle_core::safetensors::load("cnn.safetensors", &dev)?;
```

输出（这就是"模型权重"的真身）：

```text
文件内张量：
  conv1.bias    形状 [8]
  conv1.weight  形状 [8, 3, 3, 3]
  conv2.bias    形状 [16]
  conv2.weight  形状 [16, 8, 3, 3]
  fc1.bias      形状 [64]
  fc1.weight    形状 [64, 784]
  head.bias     形状 [3]
  head.weight   形状 [3, 64]
```

看到没——`vb.pp("conv1")` 就对应文件里的 `conv1.weight`（形状 `[输出通道, 输入通道, 核高, 核宽]`）和 `conv1.bias`。**一个 safetensors 文件，本质就是"一堆带名字的张量"**。真实 ResNet 的文件也是这样，只是名字更多、张量更大。

### 第四步：加载回来 + 完整推理

加载用 `VarBuilder::from_mmaped_safetensors`——**这就是加载 HF 预训练模型的标准姿势**：

```rust
// ⚠️ from_mmaped_safetensors 是 unsafe（它把文件内存映射进来）
let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["cnn.safetensors"], DType::F32, &dev)? };
let model = SmallCnn::load(CLASSES.len(), vb2)?; // 同一个 load，只是权重来自文件而非随机
```

完整预处理函数（对照第 ① 步的流水线，每一小步都在这）：

```rust
fn preprocess(img: &RgbImage, dev: &Device) -> Result<Tensor> {
    let small = image::imageops::resize(img, SZ as u32, SZ as u32, FilterType::Triangle); // ① resize
    let (w, h) = (small.width() as usize, small.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = small.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * h * w + y * w + x] = p[c] as f32 / 255.0; // ② HWC→CHW 同时 /255 归一化
            }
        }
    }
    let t = Tensor::from_vec(data, (1, 3, h, w), dev)?; // ③ 加 batch 维 → [1,3,H,W]
    normalize(&t)                                       // ④ 按通道标准化（mean=std=0.5）
}
```

其中 `normalize` 用**广播**做按通道标准化，也是一段值得记住的 candle 惯用法：

```rust
fn normalize(t: &Tensor) -> Result<Tensor> {
    let dev = t.device();
    let mean = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    let std  = Tensor::new(&[0.5f32, 0.5, 0.5], dev)?.reshape((1, 3, 1, 1))?;
    t.broadcast_sub(&mean)?.broadcast_div(&std) // [N,3,H,W] 广播减/除 [1,3,1,1]
}
```

最后跑推理——**logits → softmax → argmax → 类别名**：

```rust
let input = preprocess(&img, &dev)?;                                // [1,3,28,28]
let logits = model.forward(&input)?;                                // [1,3]
let probs = softmax(&logits, D::Minus1)?.to_vec2::<f32>()?[0].clone(); // 和为 1 的概率
let top = logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0] as usize;  // 最大分数的下标
println!("top-1 = {} (prob={:.3})", CLASSES[top], probs[top]);      // 索引→名称
```

对每类各合成一张 64×64 的干净测试图（先 resize 到 28×28），实测输出：

```text
推理（每类合成一张 64x64 测试图，先 resize 到 28x28）：
  真实=竖条   → top-1=竖条   (prob=0.919)   分布 [竖条=0.919, 横条=0.000, 方块=0.080]
  真实=横条   → top-1=横条   (prob=0.975)   分布 [竖条=0.000, 横条=0.975, 方块=0.025]
  真实=方块   → top-1=方块   (prob=0.991)   分布 [竖条=0.009, 横条=0.000, 方块=0.991]
```

三张图全部识别正确，概率都在 0.9 以上，而且 softmax 分布加起来正好是 1。**这条从"一张图"到"一个类别名 + 置信度"的完整链路，就是所有图像分类推理的通用骨架**——换成真实 ResNet，你改的只是网络结构、类别表和那组 mean/std，代码结构分毫不变。

## 常见坑与提示

> 🕳️ **坑**：**candle 0.11 的 CPU 后端不能 `set_seed`（会直接报错）。** 想让结果可复现，不能指望框架的随机初始化。本章的 `det_init` 就是**自己用固定种子的 `Lcg` 把每个权重确定性地写一遍**（bias 置 0，weight 取 `[-1/√fan_in, 1/√fan_in]` 均匀分布）。这样每次 `cargo run` 的 loss、准确率、概率都逐字一致。这是 candle 纯 CPU 下做可复现实验时必须注意的一点。

> 🕳️ **坑**：**HWC 和 CHW 千万别搞反。** `image` 库给的是 HWC（像素里 R/G/B 交错），模型要的是 CHW（整片 R、整片 G、整片 B）。`preprocess` 里 `data[c*h*w + y*w + x]` 这个下标就是在做 HWC→CHW 的搬运。搞反了模型不报错，只会给你一堆乱概率（[第 31 章](../p05-numeric/ch31-image-as-tensor.md) 详细踩过这个坑）。

> ⚠️ **注意**：**别忘了 batch 维。** 模型吃的是四维 `[N, C, H, W]`，哪怕只推理一张图，也要凑成 `[1, 3, 28, 28]`。代码里 `Tensor::from_vec(data, (1, 3, h, w), dev)` 的那个 `1` 就是它。少一维，`forward` 里的卷积/矩阵乘会直接报形状错。

> ⚠️ **注意**：**类别索引和名称的映射要对齐。** 模型输出的是数字下标（0、1、2……），`argmax` 给你的也是下标。真实 ImageNet 模型有一份 1000 行的**标签表**（`0 → "tench", 1 → "goldfish", ...`），下标错一位，"金鱼"就成了"丁鱲"。本章的 `CLASSES` 数组就是这份标签表的迷你版，**它的顺序必须和训练时的类别编号完全一致**。

> 💡 **提示**：**`from_mmaped_safetensors` 为什么是 `unsafe`？** 因为它用**内存映射（mmap）**把文件直接映射进进程地址空间（快、省内存）。若文件在读取期间被别的进程改写，映射的数据就可能失效——这类"编译器无法保证安全"的操作在 Rust 里就要标 `unsafe`。正常使用（文件稳定存在）是安全的，照写 `unsafe { ... }` 即可。

## 本章小结

- **图像分类推理**是一条通用流水线：**预处理 → 前向 → softmax → argmax → 查标签表**。预处理必须和训练时逐字一致（尺寸、归一化、CHW、mean/std）。
- candle 里，**`VarMap` 存参数、`VarBuilder` 取参数**，`vb.pp("名字")` 给每层权重定位；`.safetensors` 文件本质是**一堆带名字的张量**（`conv1.weight` 等）。
- **存**用 `VarMap::save`（内部即 `candle_core::safetensors::save`）；**加载**用 `unsafe { VarBuilder::from_mmaped_safetensors(&[路径], DType::F32, &dev)? }`——**这套加载 API，跑我们的小 CNN 和跑 HF 上的真实大模型完全一样**。
- 我们实测：小 CNN 训练后三类形状识别 **100%**，加载回来推理 top-1 概率 **0.92 / 0.98 / 0.99**，全部正确，且数字可复现。
- 诚实提醒：本章用小模型 + 合成图是为了**免下载、即刻可跑**；机制与真实预训练模型一致，唯一区别是模型大小和数据来源。

## 动手练习

1. **换预处理、看崩坏**：把 `normalize` 里的 `mean/std` 从 `0.5` 改成 ImageNet 的 `[0.485,0.456,0.406]/[0.229,0.224,0.225]`（但**训练时仍用 0.5**），重跑推理，观察概率怎么变——亲身体会"推理预处理和训练不一致"的后果。
2. **打印 logits**：在 softmax 之前把原始 `logits` 也打印出来，对照 softmax 后的概率，直观感受"负数、大数的分数"是怎么被压成"和为 1 的概率"的。
3. **加噪声考验**：给测试图叠加高斯噪声（[第 41 章](../p07-dl-basics/ch41-training-tricks.md) 的 `gaussian_noise`），看 top-1 概率下降多少、会不会误判——真实世界的图可没这么干净。
4. **想一想**：如果你要加载一个真实的 HF 分类模型，除了把 `SmallCnn` 换成对应的网络结构，你还必须弄清楚哪三件事，才能保证结果正确？（提示：输入尺寸、那组 mean/std、还有那份标签表。）

## 专家补强：把"能出类别"升级为"可交付的推理契约"

打印一个类别只是起点。真正上线的分类器，要把**模型权重、网络结构、类别表、预处理参数、数值后处理**当成**同一个版本化制品**一起交付——任何一项对不上，都会产生"数值正常、语义全错"的结果。进阶时请建立这几条纪律：

- **黄金样本回归**：固定若干张输入，存下它们在每个阶段的形状、最小/最大值、通道均值和最终 logits。换设备（CPU/Metal/CUDA）或换后端时，逐阶段比对，能迅速区分"预处理错了"还是"算子数值有差异"。
- **拒识（开放集）**：softmax 永远会给某类较高的相对概率，哪怕输入根本不属于任何训练类别。生产系统需要用**最大概率阈值 / 能量分数 / 特征距离**来拒识未知输入，阈值要在**代表线上分布的验证集**上按误判成本来定，而不是拍脑袋设 0.5。
- **性能分段**：只在 `--release`、预热之后计时，并把"解码 / 预处理 / 拷贝 / 模型 / 后处理"分段统计；首次推理含权重加载与内核预热，别和稳态延迟混为一谈。

这几条的完整验收标准见 [附录 G](../appendix/g-expert-standard.md)。

下一章，我们把这个"识图大脑"改造成**你自己的专才**：冻结骨干、换一个新分类头，用少量数据**微调**出一个全新任务的分类器。👉 [第 43 章](ch43-finetune.md)
