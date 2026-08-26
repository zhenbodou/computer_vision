//! 第 37 章 卷积神经网络 CNN 原理：手写"卷积核滑动 + ReLU + 最大池化"的数值过程
//! 运行：cargo run --example ch37_cnn
//! 依赖：ndarray 0.17.2（不训练网络，只把一层 conv-relu-pool 的数字算给你看）
//!
//! 承接第 13 章：那里我们手工设计卷积核做边缘检测；这里同样的操作，
//! 只不过在 CNN 里核的数值是"学出来"的。本例仍用一个手工的竖直边缘核，
//! 目的是把 CNN 一层里发生的算术过程看清楚。

use ndarray::{array, Array2};

/// "有效"卷积（valid，无 padding、stride=1）：输出尺寸 = 输入 - 核 + 1
/// 这里做的是"相关"（不翻转核），与第 13 章、与主流深度学习框架一致。
fn conv2d_valid(input: &Array2<f32>, kernel: &Array2<f32>) -> Array2<f32> {
    let (ih, iw) = (input.shape()[0], input.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let (oh, ow) = (ih - kh + 1, iw - kw + 1);
    Array2::from_shape_fn((oh, ow), |(r, c)| {
        let mut sum = 0.0;
        for i in 0..kh {
            for j in 0..kw {
                // 核盖住输入的 (r+i, c+j)，逐格相乘再累加
                sum += input[[r + i, c + j]] * kernel[[i, j]];
            }
        }
        sum
    })
}

/// ReLU：把负数拍成 0，正数原样保留
fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// 2×2 最大池化，步长 2：每个不重叠的 2×2 小块只留最大值，尺寸减半
fn maxpool2x2(x: &Array2<f32>) -> Array2<f32> {
    let (ih, iw) = (x.shape()[0], x.shape()[1]);
    let (oh, ow) = (ih / 2, iw / 2);
    Array2::from_shape_fn((oh, ow), |(r, c)| {
        let (r0, c0) = (r * 2, c * 2);
        x[[r0, c0]]
            .max(x[[r0, c0 + 1]])
            .max(x[[r0 + 1, c0]])
            .max(x[[r0 + 1, c0 + 1]])
    })
}

fn main() {
    // 6×6 输入"图像"：左上角 3×3 是亮块(9)，其余为暗(0)——制造一条竖直边缘
    let input = Array2::from_shape_fn((6, 6), |(r, c)| if r < 3 && c < 3 { 9.0 } else { 0.0 });
    println!("输入图像 (6×6)：\n{input}\n");

    // 竖直边缘检测核（左正右负）：与第 13 章的手工核同款，只是这里假装它是"学出来的"
    let kernel = array![[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]];
    println!("卷积核 (3×3，竖直边缘检测)：\n{kernel}\n");

    // ① 卷积：核在图上滑动，valid 模式输出 (6-3+1)=4，即 4×4 特征图
    let feat = conv2d_valid(&input, &kernel);
    println!("① 卷积后特征图 (4×4)：\n{feat}");
    println!("   尺寸公式：输出 = (输入 - 核)/步长 + 1 = (6-3)/1 + 1 = 4 ✓\n");

    // ② ReLU：滤掉负响应，只留"检测到的"正响应
    let activated = relu(&feat);
    println!("② ReLU 之后 (4×4)：\n{activated}\n");

    // ③ 最大池化 2×2：尺寸减半到 2×2，保留每个区域最强的响应
    let pooled = maxpool2x2(&activated);
    println!("③ 最大池化 2×2 之后 (2×2)：\n{pooled}");
    println!("   尺寸减半：4×4 → 2×2，边缘的强响应被保留了下来\n");

    // ---------- 为什么 CNN 比全连接省参数？用数字说话 ----------
    // 假设处理一张 224×224×3 的彩色图：
    let (h, w, c) = (224usize, 224, 3);
    let fc_units = 1000usize; // 全连接到 1000 个神经元
    let fc_params = h * w * c * fc_units; // 每个输入像素都连到每个神经元
    let n_filters = 64usize; // 一个卷积层用 64 个 3×3×3 的卷积核
    let conv_params = n_filters * 3 * 3 * c;
    println!("===== 参数量对比（处理 224×224×3 图像）=====");
    println!("全连接层 → {fc_units} 神经元：{fc_params} 个权重（约 {:.1} 亿）", fc_params as f64 / 1e8);
    println!("卷积层   → {n_filters} 个 3×3 核 ：{conv_params} 个权重");
    println!("差距约 {} 倍——这就是权值共享省下的参数", fc_params / conv_params);
}
