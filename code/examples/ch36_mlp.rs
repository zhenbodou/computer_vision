//! 第 36 章 神经网络与反向传播：从零手写一个两层 MLP 学会 XOR
//! 运行：cargo run --example ch36_mlp
//! 依赖：ndarray 0.17.2（不借助任何深度学习框架，前向/反向/梯度下降全手写）
//!
//! 网络结构：2 输入 → 隐藏层(H 个神经元, sigmoid) → 1 输出(sigmoid)
//! 任务：拟合 XOR（异或）——线性模型永远学不会，正好证明"非线性"的价值。

use ndarray::{array, Array1, Array2, Axis};

/// 极简线性同余随机数发生器：不引入 rand 依赖，保证每次运行结果完全一致（可复现）
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    /// 返回一个落在 [-1.0, 1.0] 的伪随机数，用来初始化权重
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 33) as u32;
        (u as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// sigmoid 激活函数：把任意实数"压"到 (0, 1) 区间，形状像个拉长的 S
fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// 两层多层感知机（一个隐藏层）
struct Mlp {
    w1: Array2<f32>, // 输入→隐藏 的权重，形状 (输入维=2, 隐藏维=H)
    b1: Array1<f32>, // 隐藏层偏置，形状 (H,)
    w2: Array2<f32>, // 隐藏→输出 的权重，形状 (H, 1)
    b2: Array1<f32>, // 输出层偏置，形状 (1,)
}

impl Mlp {
    /// 用小随机数初始化权重（打破对称性），偏置置 0
    fn new(n_in: usize, n_hidden: usize, seed: u64) -> Self {
        let mut r = Lcg::new(seed);
        Mlp {
            w1: Array2::from_shape_fn((n_in, n_hidden), |_| r.next_f32()),
            b1: Array1::zeros(n_hidden),
            w2: Array2::from_shape_fn((n_hidden, 1), |_| r.next_f32()),
            b2: Array1::zeros(1),
        }
    }

    /// 前向传播：输入 x (N,2) → 隐藏激活 a1 (N,H)、输出 a2 (N,1)
    /// 返回 (a1, a2)，反向传播时要用到中间量 a1
    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let z1 = x.dot(&self.w1) + &self.b1; // 加权和 + 偏置，广播加 b1
        let a1 = sigmoid(&z1); // 隐藏层激活（非线性！去掉它网络就退化成线性）
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = sigmoid(&z2); // 输出层激活，落在 (0,1) 当作"属于类别1的概率"
        (a1, a2)
    }

    /// 一步训练：前向 → 算损失 → 反向传播求梯度 → 梯度下降更新参数，返回本轮损失
    fn train_step(&mut self, x: &Array2<f32>, y: &Array2<f32>, lr: f32) -> f32 {
        let n = x.shape()[0] as f32; // 样本数
        let (a1, a2) = self.forward(x);

        // 损失：均方误差 MSE = 平均((预测 - 真实)^2)
        let loss = (&a2 - y).mapv(|v| v * v).sum() / n;

        // ---------- 反向传播：链式法则，误差一层层往回传 ----------
        // 输出层：dL/dz2 = dL/da2 * da2/dz2
        //   dL/da2 = 2(a2 - y)/n ；da2/dz2 = a2(1-a2)（sigmoid 的导数）
        let d_a2 = (&a2 - y).mapv(|v| v * 2.0 / n);
        let dz2 = &d_a2 * &a2.mapv(|v| v * (1.0 - v)); // (N,1)
        let dw2 = a1.t().dot(&dz2); // (H,N)·(N,1) = (H,1)
        let db2 = dz2.sum_axis(Axis(0)); // (1,)

        // 隐藏层：误差先经 w2 传回来，再乘隐藏层 sigmoid 的导数
        let da1 = dz2.dot(&self.w2.t()); // (N,1)·(1,H) = (N,H)
        let dz1 = &da1 * &a1.mapv(|v| v * (1.0 - v)); // (N,H)
        let dw1 = x.t().dot(&dz1); // (2,N)·(N,H) = (2,H)
        let db1 = dz1.sum_axis(Axis(0)); // (H,)

        // ---------- 梯度下降：参数沿"负梯度"方向挪一小步（步长 = lr） ----------
        self.w1.scaled_add(-lr, &dw1); // w1 += -lr * dw1
        self.b1.scaled_add(-lr, &db1);
        self.w2.scaled_add(-lr, &dw2);
        self.b2.scaled_add(-lr, &db2);

        loss
    }
}

fn main() {
    // XOR 数据集：输入两位，相同为 0、不同为 1
    let x: Array2<f32> = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y: Array2<f32> = array![[0.0], [1.0], [1.0], [0.0]];

    let mut net = Mlp::new(2, 8, 20240826); // 2 输入、8 个隐藏神经元、固定随机种子
    let lr = 1.0; // 学习率
    let epochs = 20000; // 训练轮数

    println!("开始训练 XOR（2-8-1 的 MLP，学习率 {lr}）...\n");
    for epoch in 0..=epochs {
        let loss = net.train_step(&x, &y, lr);
        // 只在几个关键节点打印，观察 loss 是否稳步下降
        if epoch == 0
            || epoch == 100
            || epoch == 1000
            || epoch == 5000
            || epoch == 10000
            || epoch == epochs
        {
            println!("第 {epoch:>5} 轮  loss = {loss:.6}");
        }
    }

    // 训练完，看看网络学成什么样
    let (_, pred) = net.forward(&x);
    println!("\n最终预测（>0.5 记为类别 1）：");
    let mut correct = 0;
    for i in 0..4 {
        let p = pred[[i, 0]];
        let cls = if p > 0.5 { 1 } else { 0 };
        let truth = y[[i, 0]] as i32;
        if cls == truth {
            correct += 1;
        }
        println!(
            "  输入 ({}, {}) → 预测 {:.4} → 类别 {}  (正确答案 {})",
            x[[i, 0]] as i32,
            x[[i, 1]] as i32,
            p,
            cls,
            truth
        );
    }
    println!("\n准确率：{}/4", correct);
}
