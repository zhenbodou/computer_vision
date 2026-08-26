//! 第 29 章 ndarray：多维数组/张量的基本功
//! 运行：cargo run --example ch29_ndarray
//! 依赖：ndarray 0.17.2
//!
//! 演示：构造/索引/切片/按轴求和/广播/矩阵乘/reshape/mapv，
//! 以及“图像即数组”——把一小张灰度图读进 Array2<f32> 做归一化并按行求均值。

use ndarray::{array, s, Array1, Array2, Axis};

fn main() {
    // ---------- ① 构造数组 ----------
    // array! 宏：像写字面量一样直接写出数组（注意 f32 后缀，统一用 f32）
    let v: Array1<f32> = array![1.0, 2.0, 3.0, 4.0]; // 一维，形状 (4,)
    let m: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 二维，形状 (2,3)
    println!("v = {v}  形状 {:?}", v.shape());
    println!("m =\n{m}\n形状 {:?}", m.shape());

    // 全 0 / 全 1：形状用元组给出（行, 列）
    let zeros: Array2<f32> = Array2::zeros((2, 3));
    let ones: Array2<f32> = Array2::ones((2, 3));
    println!("zeros 形状 {:?}, ones.sum() = {}", zeros.shape(), ones.sum());

    // 从一维 Vec + 形状拼出多维数组（行优先填充：先填满第一行，再第二行）
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let a: Array2<f32> = Array2::from_shape_vec((3, 4), data).unwrap();
    println!("a =\n{a}");

    // ---------- ② 索引与切片 s![] ----------
    println!("a[[1, 2]] = {}", a[[1, 2]]); // 第 1 行第 2 列（从 0 数）
    let col1 = a.slice(s![.., 1]); // 所有行、第 1 列 -> 一维
    let sub = a.slice(s![0..2, 1..3]); // 前两行、第 1~2 列 -> 子块
    println!("第 1 列 col1 = {col1}");
    println!("左上子块 sub =\n{sub}");

    // ---------- ③ 按轴运算 ----------
    // Axis(0) 是“行方向”（跨行、逐列汇总）；Axis(1) 是“列方向”（跨列、逐行汇总）
    let col_sum = a.sum_axis(Axis(0)); // 每列求和 -> 形状 (4,)
    let row_mean = a.mean_axis(Axis(1)).unwrap(); // 每行求均值 -> 形状 (3,)
    println!("每列之和 col_sum = {col_sum}");
    println!("每行均值 row_mean = {row_mean}");

    // ---------- ④ 逐元素运算与广播 ----------
    let scaled = &a * 2.0 + 1.0; // 标量广播：每个元素 *2 再 +1
    println!("a*2+1 =\n{scaled}");
    // 数组间广播：a 形状 (3,4)，bias 形状 (4,)，自动按行广播相加
    let bias: Array1<f32> = array![100.0, 200.0, 300.0, 400.0];
    let added = &a + &bias; // (3,4) + (4,) -> (3,4)
    println!("a + bias(按行广播) =\n{added}");

    // ---------- ⑤ 矩阵乘 dot ----------
    // (2,3) dot (3,2) -> (2,2)
    let x: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let z = x.dot(&y);
    println!("x.dot(y) =\n{z}");

    // ---------- ⑥ reshape：0.17 里用 to_shape / into_shape_with_order ----------
    // to_shape 借用原数组，返回可能借用的 CowArray；用 .to_owned() 拿到独立副本
    let flat = a.to_shape(12).unwrap().to_owned(); // (3,4) -> (12,)
    println!("拉平成一维 flat = {flat}");
    let reshaped = a.clone().into_shape_with_order((2, 6)).unwrap(); // 消耗所有权
    println!("重排成 (2,6) =\n{reshaped}");

    // ---------- ⑦ mapv：对每个元素套一个函数 ----------
    let squared = a.mapv(|e| e * e);
    println!("逐元素平方后 a[[2,3]]^2 = {}", squared[[2, 3]]);

    // ---------- ⑧ 图像即数组：灰度图 -> Array2<f32> -> 归一化 -> 按行均值 ----------
    // 一张 2 行 4 列的“灰度小图”，像素是 0~255 的 u8
    let pixels: Vec<u8> = vec![
        0, 64, 128, 255, // 第 0 行
        10, 20, 30, 40, // 第 1 行
    ];
    // 先转 f32，再按 (高, 宽) = (2, 4) 塞进 Array2
    let img_f: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    let img: Array2<f32> = Array2::from_shape_vec((2, 4), img_f).unwrap();
    // 归一化：整幅图除以 255，把 [0,255] 映射到 [0,1]——深度学习喂图前的标准动作
    let norm = &img / 255.0;
    // 按行求均值：Axis(1) 跨列汇总，得到每一行的平均亮度
    let row_brightness = norm.mean_axis(Axis(1)).unwrap();
    println!("\n===== 图像即数组 =====");
    println!("原始像素矩阵 =\n{img}");
    println!("归一化后 =\n{norm}");
    println!("每行平均亮度 = {row_brightness}");
}
