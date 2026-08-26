// 第 40 章：用 burn 0.21.0 从零训练一个 CNN（合成小数据集）
//
// ⚠️ 这是一个【独立工程】的示例，不属于主 code/（cv_examples）工程。
//    burn 依赖较重、编译较久，为不拖慢主工程，未把它加进 code/Cargo.toml。
//    请单独新建工程运行：
//        cargo new burn_lab && cd burn_lab
//        # 把本文件内容复制到 src/main.rs
//        # Cargo.toml 里加：
//        #   burn = { version = "0.21.0", features = ["ndarray", "autodiff"] }
//        cargo run --release
//
// 后端：NdArray（纯 Rust CPU）+ Autodiff 装饰器（自动微分）
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::relu;

const IMG: usize = 12; // 图像边长（12x12 灰度）
const N_CLASS: usize = 3; // 类别数：竖条 / 横条 / 方块

// ---------- 一个极简可复现随机数发生器（不引 rand，保证每次运行一致）----------
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    } // [0,1]
    fn range(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

// ---------- 合成数据：根据类别画一张 12x12 图，再加噪声 ----------
fn make_sample(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut img = vec![0.0f32; IMG * IMG];
    match class {
        0 => {
            // 竖条：随机某两列涂白
            let c = 2 + rng.range(IMG - 4);
            for y in 0..IMG {
                for x in c..c + 2 {
                    img[y * IMG + x] = 1.0;
                }
            }
        }
        1 => {
            // 横条：随机某两行涂白
            let r = 2 + rng.range(IMG - 4);
            for x in 0..IMG {
                for y in r..r + 2 {
                    img[y * IMG + x] = 1.0;
                }
            }
        }
        _ => {
            // 方块：随机位置一个 3x3 实心块
            let sz = 3;
            let ox = rng.range(IMG - sz);
            let oy = rng.range(IMG - sz);
            for y in oy..oy + sz {
                for x in ox..ox + sz {
                    img[y * IMG + x] = 1.0;
                }
            }
        }
    }
    // 加一点均匀噪声，别让任务太"干净"
    for v in img.iter_mut() {
        *v += (rng.unit() - 0.5) * 0.4;
    }
    img
}

// 生成数据集：类别交替排列（天然打乱），返回 (拉平的像素, 标签)
fn make_dataset(n_per_class: usize, rng: &mut Lcg) -> (Vec<f32>, Vec<i64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n_per_class {
        for c in 0..N_CLASS {
            xs.extend(make_sample(c, rng));
            ys.push(c as i64);
        }
    }
    (xs, ys)
}

// ---------- 网络定义：两层卷积 + 自适应池化 + 全连接 ----------
#[derive(Module, Debug)]
struct Cnn<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> Cnn<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device), // 1→8 通道，3x3
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device), // 8→16 通道
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),      // 自适应池化到 4x4
            fc: LinearConfig::new(16 * 4 * 4, N_CLASS).init(device), // 256→3
        }
    }

    // 输入 [B,1,12,12] → 输出 logits [B,3]
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = relu(self.conv1.forward(x)); // [B,8,10,10]
        let x = relu(self.conv2.forward(x)); // [B,16,8,8]
        let x = self.pool.forward(x); // [B,16,4,4]
        let x = x.flatten::<2>(1, 3); // [B,256]
        self.fc.forward(x) // [B,3]
    }
}

// ---------- 通用准确率评估（对任意后端） ----------
fn accuracy<B: Backend>(model: &Cnn<B>, xs: &[f32], ys: &[i64], device: &B::Device) -> f32 {
    let n = ys.len();
    let x = Tensor::<B, 1>::from_floats(xs, device).reshape([n, 1, IMG, IMG]);
    let logits = model.forward(x);
    let pred = logits.argmax(1).reshape([n]); // [n] Int
    let pv = pred.into_data().to_vec::<i64>().unwrap();
    let correct = (0..n).filter(|&i| pv[i] == ys[i]).count();
    correct as f32 / n as f32
}

type B = Autodiff<NdArray>; // 训练用带自动微分的后端

fn main() {
    let device = Default::default();
    B::seed(&device, 42); // 固定随机种子：让权重初始化可复现
    let mut rng = Lcg::new(12345);
    let (train_x, train_y) = make_dataset(80, &mut rng); // 80*3 = 240 张训练
    let (test_x, test_y) = make_dataset(20, &mut rng); // 20*3 = 60 张测试
    let n_train = train_y.len();

    let mut model: Cnn<B> = Cnn::new(&device);
    let mut optim = AdamConfig::new().init::<B, Cnn<B>>();
    let loss_cfg = CrossEntropyLossConfig::new().init(&device);

    let lr = 0.005; // 学习率
    let batch = 32; // 批大小
    let epochs = 20; // 训练轮数

    println!("开始训练（合成数据，CNN，Adam，lr={lr}）...\n");
    for epoch in 1..=epochs {
        let (mut running, mut steps, mut start) = (0.0f32, 0usize, 0usize);
        while start < n_train {
            let end = (start + batch).min(n_train);
            let bs = end - start;
            let xb =
                Tensor::<B, 1>::from_floats(&train_x[start * IMG * IMG..end * IMG * IMG], &device)
                    .reshape([bs, 1, IMG, IMG]);
            let yb = Tensor::<B, 1, Int>::from_ints(&train_y[start..end], &device);

            let logits = model.forward(xb); // 前向
            let loss = loss_cfg.forward(logits, yb); // 交叉熵损失
            let l: f32 = loss.clone().into_scalar(); // 取标量记录
            running += l;
            steps += 1;

            let grads = loss.backward(); // 反向：自动微分
            let grads = GradientsParams::from_grads(grads, &model); // 收集到参数上
            model = optim.step(lr, model, grads); // 优化器更新，返回新模型
            start = end;
        }
        println!("epoch {epoch:>2}  loss = {:.4}", running / steps as f32);
    }

    let eval = model.valid(); // 切到普通后端做评估
    let acc_train = accuracy(&eval, &train_x, &train_y, &device);
    let acc_test = accuracy(&eval, &test_x, &test_y, &device);
    println!(
        "\n训练准确率 = {:.1}%   测试准确率 = {:.1}%",
        acc_train * 100.0,
        acc_test * 100.0
    );
}
