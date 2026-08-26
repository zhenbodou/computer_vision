# 第 33 章 用 linfa 做分类、回归与聚类

> **本章导读**
> - 认识纯 Rust 的机器学习库 **linfa**（可以理解成 Rust 版的 scikit-learn），掌握它"一招吃遍所有模型"的通用范式。
> - 亲手跑通三大经典任务：**KMeans 聚类**（无监督）、**逻辑回归分类**、**线性回归**，每个都有可复现的实测结果。
> - 直面并解决一个**真实的生态摩擦**：linfa 锁定的 `ndarray` 版本和本书主线不一致，学会用"独立小工程"干净利落地绕开。

## 场景：有了特征向量，然后呢

[第 32 章](ch32-ml-intro.md) 我们想通了机器学习的思路，[第 27 章](../p04-features/ch27-handcrafted-features.md) 我们会把图像压成特征向量。现在这两条线要汇合了：**手上有一堆特征向量（可能带标签），怎么真正训练出一个能用的模型？**

自己像上一章那样手搓梯度下降，教学上很棒，工程上却不现实——每换一个模型都要重推公式、重写优化器。好在 Rust 有 **linfa**：它把聚类、分类、回归、降维等一大票经典算法封装好，接口统一、纯 Rust、无需 OpenCV 或 Python。这一章我们就用它把三件最常见的事各做一遍。

> 📖 **术语**：**linfa** 是 Rust 的传统机器学习工具箱，对标 Python 的 scikit-learn。它由一个核心 crate `linfa` 加一堆算法子 crate（`linfa-clustering`、`linfa-linear`、`linfa-logistic`、`linfa-trees` …）组成，共享同一套数据结构和 `fit / predict` 接口。

## linfa 的通用范式：一次学会，处处适用

linfa 最省心的地方是**所有模型都长一个样**，记住这四步走天下：

```text
① use linfa::prelude::*;                 // 引入 Fit / Predict 等 trait
② 把数据装进 Dataset::new(records, targets) // records: Array2, targets: Array1
③ let model = 某模型::params(...).…….fit(&dataset)?;  // 训练
④ let 预测 = model.predict(&新数据);       // 推理
```

- **records（特征矩阵）**：一个 `Array2<f64>`，形状 `(样本数, 特征数)`——**一行一个样本**。
- **targets（标签）**：一个 `Array1`，长度等于样本数。回归任务里是 `f64`，分类任务里是类别（我们用 `usize` 表示 0/1）。无监督的聚类不需要标签。

记牢这个骨架，下面三个例子你会发现"换汤不换药"。

## 先解决版本冲突：为什么要单开一个小工程

动手前必须说清一件事，否则你的代码根本编译不过：

> 🕳️ **坑**：**linfa 0.8 锁定 `ndarray 0.16`，和本书主工程 `code/` 用的 `ndarray 0.17` 对不上。** 如果你往主工程的 `Cargo.toml` 里同时写 `ndarray = "0.17"` 和 `linfa = "0.8"`，cargo 会**同时拉入两个版本的 ndarray**。此时你用 0.17 造出来的 `Array2` 和 linfa（0.16）想要的 `Array2` 在编译器眼里是**两个不同的类型**，报错 `expected ArrayBase..., found ArrayBase...`，一头雾水。

解决办法很简单，也很诚实：**给 linfa 的示例单独开一个小工程**，在里面把 `ndarray` 也钉到 `0.16`，和 linfa 内部保持一致。基础 API（`array!`、`Array2::zeros`、`.dot`）在 0.16 / 0.17 之间几乎没差别，读者用起来无感。这就是 Rust 机器学习生态目前真实存在的一点小摩擦——**知道原因，就不慌**。

新建工程 `cargo new linfa_lab`，它的 `Cargo.toml` 这样写：

```toml
[package]
name = "linfa_lab"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.16"          # 关键：与 linfa 内部一致，不要用 0.17
linfa = "0.8"
linfa-clustering = "0.8"  # KMeans
linfa-linear = "0.8"      # 线性回归
linfa-logistic = "0.8"    # 逻辑回归
```

本章所有代码都放在这个工程的 `src/main.rs` 里，`cargo run` 一把跑完三个任务。先放一个**确定性随机数小工具**（保证你每次运行结果和书上一致），后面三段都用它造数据：

```rust
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2};

/// 极简确定性随机数发生器（线性同余），让结果可复现
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed) }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64   // 映射到 [0,1)
    }
    /// Box-Muller 生成标准正态分布
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
```

## 一、聚类：KMeans 把散点自动分堆

**聚类**是无监督学习：给一堆**没有标签**的点，让算法自己把相近的归成一簇。KMeans 的思路极朴素——先随机撒 `k` 个"质心"，反复做两件事：①每个点归到最近的质心；②每个质心挪到自己那簇点的中心。如此往复直到稳定。

我们合成三簇二维点（中心分别在 `(2,2)`、`(8,3)`、`(5,9)`，各 50 个，带高斯噪声），交给 KMeans 分 3 簇：

```rust
fn part1_kmeans() {
    println!("========== 一、聚类 KMeans ==========");
    let mut rng = Lcg::new(1);
    let centers = [(2.0, 2.0), (8.0, 3.0), (5.0, 9.0)]; // 三个真实簇中心
    let per = 50usize;
    let n = centers.len() * per;
    let mut data = Array2::<f64>::zeros((n, 2));         // records: (150, 2)
    let mut row = 0;
    for &(cx, cy) in &centers {
        for _ in 0..per {
            data[[row, 0]] = cx + 0.6 * rng.next_gauss();
            data[[row, 1]] = cy + 0.6 * rng.next_gauss();
            row += 1;
        }
    }

    // 聚类无标签，DatasetBase::from 直接吃特征矩阵
    let model = KMeans::params(3)          // 分 3 簇（内部已用固定种子，可复现）
        .max_n_iterations(200)
        .tolerance(1e-4)
        .fit(&DatasetBase::from(data.clone()))
        .expect("KMeans 训练失败");

    let centroids = model.centroids();     // &Array2<f64>，形状 (3, 2)
    let counts = model.cluster_count();    // &Array1<f64>，各簇样本数
    for k in 0..3 {
        println!("簇 {}: 质心 = ({:.2}, {:.2}), 样本数 = {}",
            k, centroids[[k, 0]], centroids[[k, 1]], counts[k] as u32);
    }
    println!("inertia(各点到质心的平均平方距离) = {:.2}", model.inertia());
}
```

实测输出：

```text
========== 一、聚类 KMeans ==========
簇 0: 质心 = (4.86, 8.96), 样本数 = 50
簇 1: 质心 = (1.96, 2.01), 样本数 = 50
簇 2: 质心 = (8.00, 3.23), 样本数 = 50
inertia(各点到质心的平均平方距离) = 0.67
```

三个质心 `(4.86,8.96)`、`(1.96,2.01)`、`(8.00,3.23)` 分别精准命中了真实中心 `(5,9)`、`(2,2)`、`(8,3)`，每簇 50 个样本不多不少。注意**簇的编号（0/1/2）是随机初始化决定的，本身没有含义**——KMeans 只负责"分堆"，不负责给堆起名。

## 二、分类：逻辑回归划出决策边界

**分类**是监督学习：给带标签的点，学一条**决策边界**把不同类分开。**逻辑回归（logistic regression）**是最经典的二分类模型，它本质上就是上一章的线性模型外面套一个把输出压到 `0~1`（当作概率）的 S 形函数，再用梯度下降训练。

我们合成两类可分的点（类 0 围绕 `(2,2)`，类 1 围绕 `(6,6)`），按 8:2 切成训练集和测试集，训练后在测试集上报准确率：

```rust
fn part2_logistic() {
    println!("\n========== 二、分类 逻辑回归 ==========");
    let mut rng = Lcg::new(7);
    let per = 100usize;
    let n = per * 2;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<usize>::zeros(n);   // 标签用 usize：0 / 1
    for i in 0..per {                        // 两类交替填入，切分后训练/测试都含两类
        let (a, b) = (2 * i, 2 * i + 1);
        x[[a, 0]] = 2.0 + rng.next_gauss(); x[[a, 1]] = 2.0 + rng.next_gauss(); y[a] = 0;
        x[[b, 0]] = 6.0 + rng.next_gauss(); x[[b, 1]] = 6.0 + rng.next_gauss(); y[b] = 1;
    }

    let ds = Dataset::new(x, y);                       // 有标签，用 Dataset::new
    let (train, valid) = ds.split_with_ratio(0.8);     // 8:2 切分
    println!("训练集 {} 条, 测试集 {} 条", train.nsamples(), valid.nsamples());

    let model = LogisticRegression::default()
        .max_iterations(200)
        .fit(&train)
        .expect("逻辑回归训练失败");

    let pred = model.predict(&valid);                  // 返回 Array1<usize>
    let cm = pred.confusion_matrix(&valid).expect("混淆矩阵失败");
    println!("混淆矩阵:\n{:?}", cm);
    println!("测试集准确率 = {:.1}%", cm.accuracy() * 100.0);

    // 用训练好的模型预测两个全新的点
    let newcomers = Array2::from_shape_vec((2, 2), vec![2.2, 1.8, 5.7, 6.3]).unwrap();
    let labels = model.predict(&newcomers);
    println!("新点 (2.2,1.8) -> 类 {}", labels[0]);
    println!("新点 (5.7,6.3) -> 类 {}", labels[1]);
}
```

实测输出：

```text
========== 二、分类 逻辑回归 ==========
训练集 160 条, 测试集 40 条
混淆矩阵:

classes    | 1          | 0
1          | 20         | 0
0          | 0          | 20

测试集准确率 = 100.0%
新点 (2.2,1.8) -> 类 0
新点 (5.7,6.3) -> 类 1
```

因为两类离得远、线性可分，模型在测试集上拿到 **100% 准确率**，混淆矩阵是漂亮的对角阵（20 + 20 全判对）。最后两个从没参与训练的新点也各归其类——这就是**泛化**。`confusion_matrix` 和 `accuracy()` 来自 linfa 内置的评估工具，比自己数对错省事得多。

## 三、回归：线性回归拟合带噪声的直线

**回归**预测的是**连续值**（而非类别）。这回让 linfa 来干上一章我们手搓梯度下降干的活——拟合 \\( y = kx + b \\)，看它和我们的手写结果是否一致：

```rust
fn part3_linear() {
    println!("\n========== 三、回归 线性回归 ==========");
    let mut rng = Lcg::new(42);
    let n = 80usize;
    let mut x = Array2::<f64>::zeros((n, 1));   // 单特征：(80, 1)
    let mut y = Array1::<f64>::zeros(n);        // 回归标签是 f64
    for i in 0..n {
        let xi = i as f64 * 0.1;
        x[[i, 0]] = xi;
        y[i] = 3.0 * xi + 2.0 + 0.5 * rng.next_gauss(); // 真实 y = 3x + 2 + 噪声
    }

    let ds = Dataset::new(x, y);
    let model = LinearRegression::new().fit(&ds).expect("线性回归训练失败");
    println!("真实: 斜率 k=3.0, 截距 b=2.0");
    println!("拟合: 斜率 k={:.3}, 截距 b={:.3}", model.params()[0], model.intercept());

    let pred = model.predict(&ds);
    let r2 = pred.r2(&ds).expect("r2 失败");     // 拟合优度，越接近 1 越好
    println!("R^2 = {:.4}", r2);
}

fn main() {
    part1_kmeans();
    part2_logistic();
    part3_linear();
}
```

实测输出：

```text
========== 三、回归 线性回归 ==========
真实: 斜率 k=3.0, 截距 b=2.0
拟合: 斜率 k=3.028, 截距 b=1.882
R^2 = 0.9943
```

linfa 用**最小二乘法**（不是逐步梯度下降，而是一步解出解析解）拟合出 `k=3.028, b=1.882`，和真实的 `3.0 / 2.0` 高度吻合；\\( R^2 = 0.9943 \\)（拟合优度，`1` 为完美）说明这条线几乎解释了数据的全部变化。对比上一章手写梯度下降的 `k≈1.99`，你会体会到：**同一个问题，库用更成熟的解法给出了同样可靠的结果**，而你已经懂了它背后的原理。

## 常见坑与提示

> 🕳️ **坑**：**版本冲突（再强调一遍）。** 千万别把 linfa 加进本书主工程 `code/` 的 `Cargo.toml`——它会拽进第二个 `ndarray`，把其它所有示例的编译一起搞崩。linfa 的实验请**永远在独立小工程里、用 `ndarray = "0.16"`** 跑。

> ⚠️ **注意**：**元素类型统一用 `f64`。** linfa 的算法对元素类型是泛型 `F: Float`，但整条链路的类型必须一致。本书统一用 `f64`（数值稳定、官方示例也都用它）。别一处 `f32`、一处 `f64`，否则编译器会用一串 trait bound 报错糊你一脸。

> 🕳️ **坑**：**`records` 和 `targets` 的形状别搞反。** `records` 必须是 `(样本数, 特征数)` 的 **二维** `Array2`——哪怕只有一个特征，也要是 `(n, 1)` 而不是长度 `n` 的一维数组（见回归例子里的 `Array2::zeros((n, 1))`）。`targets` 则是长度为样本数的一维 `Array1`。形状错了，`fit` 要么编译不过，要么 panic。

> 💡 **提示**：**`predict` 返回什么，取决于你喂它什么。** 传 `&Dataset` 或 `&Array2` 时，分类/聚类模型返回 `Array1<标签>`（本章分类是 `Array1<usize>`、KMeans 是簇编号），回归模型返回 `Array1<f64>`。想换个模型试试？把 `LogisticRegression` 换成 `linfa_trees::DecisionTree`（记得在 `Cargo.toml` 加 `linfa-trees`），`fit / predict` 的用法几乎原样不变——这就是统一范式的威力。

## 本章小结

- **linfa** 是纯 Rust 的传统机器学习库，所有模型共享 `use linfa::prelude::*;` → `Dataset::new(records, targets)` → `模型::params(..).fit(&ds)?` → `model.predict(..)` 的**统一四步范式**。
- **数据结构**：`records` 是 `(样本数, 特征数)` 的 `Array2<f64>`，`targets` 是 `Array1`（回归用 `f64`、分类用 `usize`），聚类无需标签。
- 三个实测跑通：**KMeans** 把三簇点精准分开（质心命中真值）；**逻辑回归**在可分两类上拿到 100% 测试准确率并能预测新点；**线性回归**用最小二乘拟合出 `k=3.028, b=1.882`、\\( R^2=0.9943 \\)。
- **版本冲突是本章第一坑**：linfa 0.8 绑定 `ndarray 0.16`，务必在**独立工程**里用 `ndarray = "0.16"`，别污染主工程。类型统一 `f64`，形状别搞反。

## 动手练习

1. **换个分类器**：把 `LogisticRegression` 换成 `linfa_trees::DecisionTree`（`Cargo.toml` 加 `linfa-trees = "0.8"`），对比它在同一份数据上的准确率。感受"范式不变，模型可换"。
2. **让分类变难**：把两类中心从 `(2,2)`/`(6,6)` 挪近到 `(3,3)`/`(4.5,4.5)`，两类开始重叠，看准确率怎么从 100% 掉下来——这才是真实数据的常态。
3. **选簇数 k**：给 KMeans 传 `k=2` 或 `k=4`（真实是 3 簇），观察 `inertia` 和分堆结果怎么变。想想：只看 `inertia` 能不能判断"分几簇最合适"？
4. **进阶**：把 [第 27 章](../p04-features/ch27-handcrafted-features.md) 算出的 LBP 直方图当特征向量，随手造几个"纹理样本"喂给逻辑回归，做一个玩具级"纹理分类器"——这正是下一章的思路雏形。

下一章起，我们就把这套机器学习和前面的图像特征**正式接起来**：先用 [第 34 章 滑动窗口 + HOG + SVM](ch34-sliding-window.md) 做传统目标检测，再在 [第 35 章](ch35-digit-recognition.md) 实战手写数字识别。👉 [第 34 章 传统目标检测：滑动窗口 + HOG + SVM](ch34-sliding-window.md)
