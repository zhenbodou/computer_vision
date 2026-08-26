# 第 40 章 用 burn 从零训练一个 CNN（MNIST）

> **本章导读**
> - 学完你能：用深度学习框架 **burn**（0.21.0）搭一个真正的**卷积神经网络（CNN）**，并**完整跑通一次训练**——前向、损失、反向、优化器更新、多轮迭代、最后评估准确率。
> - 你会亲眼看到 loss 逐轮下降（从 `1.04` 一路掉到 `0.0001`），这次**不用手写一行反向传播**——框架的**自动微分（autodiff）**替你把 [第 36 章](ch36-neural-nets.md) 手推到崩溃的链式法则全自动化了。
> - 搞懂 burn 的五个核心概念：**Backend 泛型、`Tensor<B, D>`、`#[derive(Module)]`、Autodiff 后端、优化器与训练循环**。
> - 本章代码**在本机 `cargo run` 实测通过**，正文所有数字都是真实运行结果。

## 场景：手写反向传播，到此为止

还记得 [第 36 章](ch36-neural-nets.md) 那个学会 XOR 的两层网络吗？它的核心逻辑挤在 `train_step` 里：前向 4 行、**反向 8 行**、更新 4 行。那 8 行反向传播，是我们咬着牙、照着链式法则一项一项手推出来的。

问题来了：那还只是一个 `2 → 8 → 1` 的迷你 MLP。真实的 CNN 动辄几十层、上百万参数、卷积池化交错——**你还打算手推每一层的梯度公式吗？** 一个 3×3 卷积对输入的偏导怎么写？连着两层卷积再接池化，链式法则要套多少层？光是想想就头皮发麻，更别说写对、还不出 bug。

这正是**深度学习框架**存在的理由。[第 39 章](ch39-dl-ecosystem.md) 我们盘点过 Rust 的 DL 生态，其中 **burn** 主打"从零训练 + 自动微分 + 多后端"，体验最接近 PyTorch。本章我们就用它把 [第 37 章](ch37-cnn.md) 讲的 CNN 原理**真正训练起来**：你只管搭好"前向怎么算"，**反向的梯度框架全自动帮你求**。手写反向传播的苦日子，到此为止。

> 📖 **术语**：**自动微分（automatic differentiation，autodiff）**。你正向搭建计算（`a * b + c` 这种），框架在背后**默默记下每一步是怎么算的**（构成一张计算图）；当你对最终的 loss 调用一次 `backward()`，它就沿着这张图**自动反向套用链式法则**，一次性算出每个参数的梯度。这就是 [第 36 章](ch36-neural-nets.md) 那 8 行反向代码的自动化版本——只不过它对**任意深、任意复杂**的网络都成立。

## 原理：burn 的五个核心概念

在动手前，先把 burn 的"世界观"讲清楚。记住这五个词，你就看懂全书后面所有 burn 代码了。

**1. Backend（后端）泛型。** burn 最大的设计特点：几乎所有代码都对一个 `Backend` trait **泛型**。CPU、GPU、WebGPU 只是不同的"后端实现"，而你的网络、训练逻辑**一份代码、随意换后端**。本章用纯 Rust 的 CPU 后端 `NdArray`；哪天想上 GPU，把后端类型换成 `Wgpu` 即可，网络代码一个字都不用改。

**2. `Tensor<B, D>`（张量）。** burn 的张量带两个泛型参数：`B` 是后端，`D` 是**维度数**，而且 `D` 是**编译期常量（const generic）**。`Tensor<B, 2>` 是二维（矩阵），`Tensor<B, 4>` 是四维——图像批量正好是四维 `[batch, channel, height, width]`（简称 NCHW）。维度写进了类型里，形状对不上，很多错误编译期就能发现。张量还有第三个默认参数"种类"：默认是 `Float`，另有 `Int`、`Bool`（标签就是 `Tensor<B, 1, Int>`）。

**3. `Module` 与 `#[derive(Module)]`。** 一个网络，在 burn 里就是一个 **struct**，字段是各层（`Conv2d`、`Linear` 等）。你给它标上 `#[derive(Module)]`，框架就能"看穿"这个结构体、**自动收集里面所有可训练参数**，从而统一地对它们求梯度、更新、保存/加载。这对应 PyTorch 里继承 `nn.Module` 的那个类。

**4. Autodiff 后端（装饰器）。** 这是最妙的一招：**自动微分本身也是一个"后端"**，但它是**装饰器（decorator）**——把普通后端包一层：`Autodiff<NdArray>`。包上之后，所有张量运算都会被记录进计算图；`loss.backward()` 就能反向求梯度。**训练时用带 Autodiff 的后端，推理/评估时用不带的**（省掉记录计算图的开销）。

**5. 优化器与训练循环。** 有了梯度，就交给**优化器（optimizer）**按梯度下降更新参数（[第 36 章](ch36-neural-nets.md) 我们手写过 `w -= lr * grad`，这里换成现成的 SGD 或 Adam）。整个训练循环就是把这几步套起来：

```text
每个 epoch，每个 batch：
  前向 forward  →  算损失 loss  →  loss.backward() 反向求梯度
        ↑                                    │
        └──────  optimizer.step 更新参数  ◀──┘
```

## 动手：训练一个能识别图形的 CNN

> ⚠️ **注意**：本章代码基于 **burn 0.21.0**（后端 feature `ndarray`、训练 feature `autodiff`）。burn 仍在快速演进，跨版本 API 可能变动，运行前请对照你所用版本的官方文档。

**关于数据：为什么不是真 MNIST？** 章节名带了"MNIST"，但 burn 生态里现成的 MNIST 加载器**需要联网下载**约 11 MB 数据。为了让例子**自包含、免下载、一键可复现**，我们改用几行代码**合成**一个小型图像分类数据集：12×12 的灰度小图，共 **3 类**——竖条、横条、方块，再撒一点噪声。**训练流程和真 MNIST 一模一样**；等你跑通了，把数据换成 MNIST 或你自己的图片即可。

> 💡 **提示**：**burn 依赖较重、首次编译较久**（要拉几十个 crate，本机首次约 50 秒）。像 [第 33 章](../p06-ml/ch33-linfa.md) 的 linfa 一样，我们把它作为一个**独立小工程**呈现，不塞进本书主工程，以免拖慢日常编译。

新建工程 `cargo new burn_lab`，`Cargo.toml` 依赖：

```toml
[dependencies]
burn = { version = "0.21.0", features = ["ndarray", "autodiff"] }
```

### 第一步：选定后端与合成数据

```rust
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;

const IMG: usize = 12;    // 图像边长 12x12
const N_CLASS: usize = 3; // 竖条 / 横条 / 方块

type B = Autodiff<NdArray>; // 训练后端 = 自动微分 装饰 纯 Rust CPU 后端
```

`type B = Autodiff<NdArray>` 就是把 CPU 后端 `NdArray` 包进 `Autodiff` 装饰器——这一层就是"能反向传播"的来源。合成数据的代码（画图 + 加噪声）比较朴素，这里给出核心（完整版见配套独立工程 `code/dl_labs/ch40_burn_train/`（含 `Cargo.toml`，`cargo run --release` 即可复现））：

```rust
// 根据类别画一张 12x12 图：类别 0 竖条、1 横条、2 方块，最后加均匀噪声
fn make_sample(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; IMG * IMG];
    match class {
        0 => { let c = 2 + rng.range(IMG - 4);          // 竖条：某两列涂白
               for y in 0..IMG { for x in c..c + 2 { img[y * IMG + x] = 1.0; } } }
        1 => { let r = 2 + rng.range(IMG - 4);          // 横条：某两行涂白
               for x in 0..IMG { for y in r..r + 2 { img[y * IMG + x] = 1.0; } } }
        _ => { let (ox, oy) = (rng.range(IMG - 3), rng.range(IMG - 3)); // 方块
               for y in oy..oy + 3 { for x in ox..ox + 3 { img[y * IMG + x] = 1.0; } } }
    }
    for v in img.iter_mut() { *v += (rng.unit() - 0.5) * 0.4; } // 加噪声
    img
}
```

这里的 `Lcg` 是 [第 36 章](ch36-neural-nets.md) 用过的极简随机数发生器（固定种子、不引 `rand`，保证复现）。`make_dataset` 把每类各生成若干张、**类别交替排列**（天然打乱），返回拉平的像素 `Vec<f32>` 和标签 `Vec<i64>`。

### 第二步：定义 CNN（`#[derive(Module)]` 是关键）

```rust
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;

#[derive(Module, Debug)]          // ← 这一行让框架能收集/训练里面所有参数
struct Cnn<B: Backend> {          // ← 网络对 Backend 泛型
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> Cnn<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            // 每个层都用 XxxConfig::new(...).init(device) 构造
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),   // 1→8 通道，3x3 卷积核
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),  // 8→16 通道
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),       // 自适应平均池化到 4x4
            fc: LinearConfig::new(16 * 4 * 4, N_CLASS).init(device), // 全连接 256→3
        }
    }

    // 前向：输入 [B,1,12,12] → 输出 logits [B,3]
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x)); // [B,1,12,12] → [B,8,10,10]
        let x = relu(self.conv2.forward(x)); // → [B,16,8,8]
        let x = self.pool.forward(x);        // → [B,16,4,4]
        let x = x.flatten::<2>(1, 3);        // 展平成 [B,256]
        self.fc.forward(x)                   // → [B,3]（每类一个分数 logit）
    }
}
```

注意 burn 的惯用法：**每个层都有一个配套的 `Config`**，`XxxConfig::new(...)` 设定超参、`.init(device)` 才真正建出带随机初始权重的层。`forward` 里张量形状一路变换（右侧注释是每步的形状），最后 `fc` 输出 `[batch, 3]` 的**原始分数（logits）**——注意还没做 softmax，因为交叉熵损失会替我们做。

> 📖 **术语**：**卷积层的输出尺寸**。`Conv2d` 默认不补零（Valid padding），一个 3×3 卷积核扫过 12×12 的图，输出是 `12 - 3 + 1 = 10`，即 10×10（原理见 [第 13 章](../p03-classic/ch13-convolution.md) 与 [第 37 章](ch37-cnn.md)）。**自适应平均池化（adaptive average pooling）** 则相反：你只说"我要输出 4×4"，它自动算好怎么下采样——省得你手算池化窗口。

### 第三步：训练循环——前向、损失、反向、更新

这是全章的心脏。对照上面"训练循环"那张图看：

```rust
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

fn main() {
    let device = Default::default();
    B::seed(&device, 42);                    // 固定随机种子，权重初始化可复现
    let mut rng = Lcg::new(12345);
    let (train_x, train_y) = make_dataset(80, &mut rng); // 240 张训练
    let (test_x, test_y) = make_dataset(20, &mut rng);   // 60 张测试
    let n_train = train_y.len();

    let mut model: Cnn<B> = Cnn::new(&device);
    let mut optim = AdamConfig::new().init::<B, Cnn<B>>(); // Adam 优化器
    let loss_cfg = CrossEntropyLossConfig::new().init(&device);
    let (lr, batch, epochs) = (0.005, 32, 20);            // 学习率、批大小、轮数

    for epoch in 1..=epochs {
        let (mut running, mut steps, mut start) = (0.0f32, 0usize, 0usize);
        while start < n_train {
            let end = (start + batch).min(n_train);
            let bs = end - start;
            // 取一个 batch，拼成 [bs,1,12,12] 的四维张量
            let xb = Tensor::<B, 1>::from_floats(
                &train_x[start * IMG * IMG..end * IMG * IMG], &device
            ).reshape([bs, 1, IMG, IMG]);
            let yb = Tensor::<B, 1, Int>::from_ints(&train_y[start..end], &device);

            let logits = model.forward(xb);            // ① 前向
            let loss = loss_cfg.forward(logits, yb);   // ② 交叉熵损失
            running += loss.clone().into_scalar();     //    记录 loss 数值
            steps += 1;

            let grads = loss.backward();               // ③ 反向：一键自动求梯度！
            let grads = GradientsParams::from_grads(grads, &model); // 梯度对齐到各参数
            model = optim.step(lr, model, grads);      // ④ 优化器更新，返回新模型
            start = end;
        }
        println!("epoch {epoch:>2}  loss = {:.4}", running / steps as f32);
    }
    // ……评估见下……
}
```

逐句看这四步：

- **① 前向** `model.forward(xb)`：把一批图片喂进网络，得到 `[bs, 3]` 的分数。
- **② 损失** `CrossEntropyLoss`：分类任务的标配。它内部先对 logits 做 softmax 再算负对数似然，衡量"预测分布"和"真实类别"差多远。输入是 logits `[bs, 3]` 和整数标签 `[bs]`。
- **③ 反向** `loss.backward()`：**全章的高潮**。就这一行，Autodiff 后端沿着前面记录的计算图，把**每一个参数**（两层卷积核、全连接权重和偏置……）的梯度全部算出来。这就是 [第 36 章](ch36-neural-nets.md) 那 8 行反向代码的"全自动、任意深度"版本。
- **④ 更新** `optim.step(lr, model, grads)`：Adam 按梯度更新所有参数，**返回更新后的新模型**（burn 里 `step` 是消费并返回 `model`，所以写成 `model = optim.step(...)`）。

`GradientsParams::from_grads(grads, &model)` 这一步是把"按计算图组织的梯度"重新**对齐到模型的各个参数**上，好让优化器认领——记住这个固定写法即可。

> 📖 **术语**：**交叉熵损失（cross-entropy loss）**。[第 36 章](ch36-neural-nets.md) 教 XOR 时我们用的是均方误差 MSE，那更适合回归；**多分类**任务的标配则是交叉熵。它先用 **softmax** 把网络输出的原始分数（logits）变成一组和为 1 的"概率"，再看**真实类别那一项的概率**有多高——概率越接近 1，损失越小。写成公式就是（\\( N \\) 是样本数，\\( p_{i,y_i} \\) 是第 \\( i \\) 个样本在其真实类别 \\( y_i \\) 上的预测概率）：
>
> \\[ L = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i} \\]
>
> burn 的 `CrossEntropyLoss` 把 softmax 和取对数**打包在内部**，所以网络最后一层**只输出原始 logits、不用自己加 softmax**——这也是为什么我们的 `forward` 末尾直接返回 `fc` 的结果。

### 第四步：评估准确率（切回普通后端）

训练完，我们要看它到底学得怎么样。评估**不需要**梯度，所以用 `model.valid()` 把模型从 `Autodiff<NdArray>` **切回普通的 `NdArray` 后端**（更省）：

```rust
    let eval = model.valid(); // AutodiffModule::valid()：切到不带自动微分的后端
    let acc_train = accuracy(&eval, &train_x, &train_y, &device);
    let acc_test = accuracy(&eval, &test_x, &test_y, &device);
    println!("训练准确率 = {:.1}%   测试准确率 = {:.1}%",
             acc_train * 100.0, acc_test * 100.0);
```

`accuracy` 函数对任意后端泛型：前向拿到 logits 后，用 `logits.argmax(1)` 取每行最大分数的下标当预测类别，再和真实标签逐个比对：

```rust
fn accuracy<B: Backend>(model: &Cnn<B>, xs: &[f32], ys: &[i64], device: &B::Device) -> f32 {
    let n = ys.len();
    let x = Tensor::<B, 1>::from_floats(xs, device).reshape([n, 1, IMG, IMG]);
    let pred = model.forward(x).argmax(1).reshape([n]);        // 每张图预测的类别
    let pv = pred.into_data().to_vec::<i64>().unwrap();        // 取回普通 Vec
    (0..n).filter(|&i| pv[i] == ys[i]).count() as f32 / n as f32
}
```

### 跑起来

```bash
cargo run --release
```

真实输出（数字可复现，为省版面中间几轮略去）：

```text
开始训练（合成数据，CNN，Adam，lr=0.005）...

epoch  1  loss = 1.0433
epoch  2  loss = 0.7855
epoch  3  loss = 0.4195
epoch  4  loss = 0.1426
epoch  5  loss = 0.0272
epoch  6  loss = 0.0050
epoch  7  loss = 0.0016
   ……
epoch 19  loss = 0.0002
epoch 20  loss = 0.0001

训练准确率 = 100.0%   测试准确率 = 100.0%
```

**读懂这条曲线**：第 1 轮 loss 是 `1.0433`——非常接近"三类瞎猜"的理论基线 \\( \ln 3 \approx 1.0986 \\)（[第 36 章](ch36-neural-nets.md) 我们说过刚初始化时网络就是在瞎猜）。随后它一路平滑下滑：`0.79 → 0.42 → 0.14 → 0.027`，到第 6 轮已经掉进 `0.005`，最终稳稳落在 `0.0001`。**这条下降曲线，就是"CNN 在学习"最直观的证据**——而这一次，你没有手写任何一行反向传播。

## 常见坑与提示

> 🕳️ **坑**：**学习率偏大，loss 会突然"炸"一下。** 我把 `lr` 从 `0.005` 调到 `0.01` 实测：训练大体正常，但**第 16 轮 loss 会突然从 `0.0000` 跳回 `2.04`、下一轮又掉回去**。原因是当 loss 已逼近 0、梯度极小时，Adam 的自适应缩放（除以梯度平方的滑动平均）容易数值抖动。表现就是曲线上一根刺。对策：**调小学习率**（我们最终用 `0.005`，曲线干净单调），或用学习率衰减（下一章讲）。学习率永远是最该调的旋钮。

> ⚠️ **注意**：**训练用 Autodiff 后端，评估记得切回去。** 网络对 `B: Backend` 泛型，`type B = Autodiff<NdArray>` 让训练能反向传播。但评估只前向、不求梯度，应调 `model.valid()` 切到 `NdArray`——否则白白背着"记录计算图"的开销。这也是 burn"一套代码、多后端"设计的直接好处：`Cnn<B>` 和 `accuracy<B>` 都不关心具体是哪个后端。

> 🕳️ **坑**：**张量维度 `D` 是编译期常量，形状要对得上。** `flatten::<2>(1, 3)` 里的 `::<2>` 是在告诉编译器"展平后是二维"；`reshape([bs, 1, IMG, IMG])` 的元素个数必须和数据长度严丝合缝，否则运行期 panic。好处是很多形状错误在编译期（维度数 `D` 对不上）就被拦下——这正是 `Tensor<B, D>` 把维度写进类型的价值。

> 🕳️ **坑**：**合成数据太"干净"，准确率会虚高。** 我们的三类图形区分度很大，网络轻松拿到 **100% / 100%**——**别被这个满分冲昏头**。真实的 MNIST、真实业务数据，类内千变万化、类间还常常长得像，绝不会这么容易。[第 16 章](../p03-classic/ch16-threshold.md) 讲指标时强调过：**一个漂亮的数字，要先问它是在什么数据上、怎么算出来的。** 训练集和测试集都 100%，更要警惕——很可能是任务太简单，而非模型有多强。把数据换成 MNIST，你才会看到有意义的、略低于训练集的测试准确率，那才是检验模型泛化能力的地方。

> ⚠️ **注意**：**burn 版本演进快，API 会变。** 本章基于 **0.21.0**。此版本的一些细节：`Backend::seed(&device, seed)` 带 `device` 参数；`nn` 层拆在 `burn-nn`、优化器在 `burn-optim`（都由门面 crate `burn` 重新导出，用 `burn::nn::*`、`burn::optim::*` 即可）；`Tensor::from_data` 的第二参已变成"创建选项"（但传 `&device` 仍然可用）。换版本时，请以官方文档为准。

## 本章小结

- burn 的世界观是五个词：**Backend 泛型**（一套代码切后端）、**`Tensor<B, D>`**（维度写进类型）、**`#[derive(Module)]`**（框架自动收集参数）、**Autodiff 后端**（装饰器，带来自动微分）、**优化器 + 训练循环**。
- 训练循环的四步：**前向 `forward` → 损失 `CrossEntropyLoss` → 反向 `loss.backward()` → 更新 `optim.step`**。其中 `backward()` 一行，就替代了 [第 36 章](ch36-neural-nets.md) 手写的全部反向传播，且对任意深的网络都成立。
- 我们用两层 `Conv2d` + 自适应池化 + `Linear` 搭了个真 CNN，在合成的三类小图上训练，**loss 从 `1.04` 平滑降到 `0.0001`**，代码在 burn 0.21.0 下 `cargo run` 实测通过。
- 评估用 `model.valid()` 切回普通后端；准确率 100% 是因为**合成数据太简单**——务必记住 [第 16 章](../p03-classic/ch16-threshold.md) 的告诫，别被虚高的数字骗了。

## 动手练习

1. **看学习率制造"尖刺"**：把 `lr` 从 `0.005` 改成 `0.01`、`0.05`，各跑一遍，观察 loss 曲线什么时候开始抖动甚至发散，体会"步子太大"的后果（呼应 [第 36 章](ch36-neural-nets.md) 的学习率实验）。
2. **换优化器**：把 `AdamConfig` 换成 `SgdConfig`（`use burn::optim::SgdConfig;`），相同轮数下比较收敛速度，感受 Adam 为什么常被当默认选择。
3. **加难度**：把 `make_sample` 里的噪声系数从 `0.4` 调到 `1.0`，或增加类别（比如再加"斜线""空心框"），看测试准确率会不会终于跌破 100%——这才是更真实的分类。
4. **进阶——换成真 MNIST**：查阅 burn 文档里的数据集加载器，把合成数据换成真正的 MNIST 手写数字（10 类、28×28）。你需要把网络输入尺寸和 `N_CLASS` 相应改大，其余训练逻辑几乎不用动——这正是"一套训练循环走天下"的威力。

下一章，我们把训练里那些"让模型更强、更稳"的实战技巧讲透：**数据增强、优化器与学习率调度、迁移学习**。👉 [第 41 章 训练技巧：数据增强、优化器、迁移学习](ch41-training-tricks.md)

## 专家补强：从“能收敛”到可信实验

上面的合成任务只验证计算图能工作，不能证明泛化。进入真实任务前先冻结实验契约：
类别语义、输入尺寸、RGB/BGR、NCHW/NHWC、归一化、划分单位、主指标、资源预算、
随机种子和停止条件。训练、验证、测试必须按主体、视频、设备、批次或时间分组；把同一
视频的相邻帧随机拆分会造成严重泄漏。

交叉熵直接接收 logits：

\[
L=-\frac1N\sum_i\log\frac{\exp(z_{i,y_i}-m_i)}
{\sum_c\exp(z_{i,c}-m_i)},\quad m_i=\max_c z_{i,c}.
\]

减去最大 logit 保证数值稳定；不要先做 softmax 再送入交叉熵。部署端必须逐项复现
训练预处理，并用一组黄金输入比较两端 logits，而不能只比较最终类别。

### 训练系统最小验收

- 先让网络过拟合 8 个样本；做不到通常是标签、损失、梯度或预处理错误。
- 增加随机 batch、验证阶段、最佳 checkpoint、resume、梯度裁剪和非有限值检查。
- 固定数据清单与依赖锁文件，运行至少三个种子，报告均值和标准差。
- 除 accuracy 外报告混淆矩阵、macro-F1、每类召回与概率校准 ECE。
- 保存配置、环境、逐轮指标、模型校验和、数据划分与失败样本。
- 做消融实验，每次只改变一个因素，证明收益来自哪里。

| 现象 | 优先怀疑 | 最小验证实验 |
|---|---|---|
| 训练 loss 不降 | 标签、归一化、梯度、学习率 | 过拟合 8 个样本 |
| 训练好而验证差 | 泄漏、过拟合、分布偏移 | 按来源切片并检查近重复样本 |
| loss 出现 NaN | 溢出、坏样本、学习率过大 | 记录梯度范数和首个异常样本 id |
| 离线好而部署差 | 预处理、类别表、算子差异 | 比较中间张量与 logits |

### 开放式考核

1. 用一段视频构造分类集，对比“随机按帧”和“按视频片段”划分的分数，用最近邻证明
   前者存在近重复泄漏。
2. 为示例实现 checkpoint/resume。验收线：中断恢复与连续训练在固定容差内一致；注入
   NaN 样本时能报告样本 id 并安全停止。
3. 把模型移到另一推理后端，比较 20 个黄金输入的最大绝对误差、余弦相似度和预测一致率；
   再故意交换 RGB/BGR，解释误差如何传播。

本节按[附录 G](../appendix/g-expert-standard.md)验收：代码运行只是第一关，实验可复现、
失败可解释、部署可核对才算真正掌握。
