// 第 35 章：手写数字识别（合成数据 + 8x8 特征 + 手写 KNN）
// 运行：cargo run --example ch35_digit_recognition
// 依赖：image、imageproc
//
// 全自包含：用"七段数码管"程序化合成 0-9 的数字图，加噪声/随机偏移生成数据集，
// 每张缩放到 8x8 拉平成 64 维特征(Vec<f64>)，手写 KNN 分类，报准确率 + 混淆矩阵。
// 特征刻意落成 Vec<f64> —— 与 ndarray 版本无关，也是喂给独立 linfa 工程的中间形式。
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::noise::gaussian_noise;
use imageproc::rect::Rect;

// 简易可复现伪随机（xorshift）
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i32
    }
}

/// 七段数码管：每个数字点亮哪几段（a b c d e f g）
/// 段位布局：a上, b右上, c右下, d下, e左下, f左上, g中
fn segments(digit: u8) -> [bool; 7] {
    // 顺序 [a, b, c, d, e, f, g]
    match digit {
        0 => [true, true, true, true, true, true, false],
        1 => [false, true, true, false, false, false, false],
        2 => [true, true, false, true, true, false, true],
        3 => [true, true, true, true, false, false, true],
        4 => [false, true, true, false, false, true, true],
        5 => [true, false, true, true, false, true, true],
        6 => [true, false, true, true, true, true, true],
        7 => [true, true, true, false, false, false, false],
        8 => [true, true, true, true, true, true, true],
        _ => [true, true, true, true, false, true, true], // 9
    }
}

/// 把一个数字画成 24x36 的灰度图（白底黑字），可带偏移 (dx, dy)
fn draw_digit(digit: u8, dx: i32, dy: i32) -> GrayImage {
    let (w, h, t) = (24u32, 36u32, 4u32); // 画布宽高 + 段粗
    let mut img: GrayImage = ImageBuffer::from_pixel(w, h, Luma([255u8]));
    let seg = segments(digit);
    let ink = Luma([0u8]);
    let ti = t as i32;
    let (wi, hi) = (w as i32, h as i32);
    // 三条横段 (a, g, d)
    let horiz = |y: i32| Rect::at(ti + dx, y + dy).of_size(w - 2 * t, t);
    // 四条竖段 (f, b, e, c)
    let vert = |x: i32, y: i32| Rect::at(x + dx, y + dy).of_size(t, (h - 3 * t) / 2 + t);
    let rects = [
        (seg[0], horiz(0)),                 // a 上
        (seg[6], horiz((hi - ti) / 2)),     // g 中
        (seg[3], horiz(hi - ti)),           // d 下
        (seg[5], vert(0, ti)),              // f 左上
        (seg[1], vert(wi - ti, ti)),        // b 右上
        (seg[4], vert(0, hi / 2)),          // e 左下
        (seg[2], vert(wi - ti, hi / 2)),    // c 右下
    ];
    for (on, r) in rects {
        if on {
            draw_filled_rect_mut(&mut img, r, ink);
        }
    }
    img
}

/// 特征提取：缩放到 8x8，拉平成 64 维，像素归一化到 [0,1]
fn extract_feature(img: &GrayImage) -> Vec<f64> {
    let small = image::imageops::resize(img, 8, 8, image::imageops::FilterType::Triangle);
    small.pixels().map(|p| p.0[0] as f64 / 255.0).collect()
}

/// 生成数据集：每个数字 n_per 张（不同偏移 + 高斯噪声），返回 (特征, 标签)
fn make_dataset(n_per: usize, seed: u64) -> Vec<(Vec<f64>, u8)> {
    let mut rng = Rng(seed);
    let mut data = Vec::new();
    for digit in 0u8..=9 {
        for k in 0..n_per {
            let dx = rng.range_i32(-2, 2);
            let dy = rng.range_i32(-2, 2);
            let img = draw_digit(digit, dx, dy);
            // 每张用不同 seed 的高斯噪声，制造类内差异
            let noisy = gaussian_noise(&img, 0.0, 55.0, seed.wrapping_add((digit as u64) * 131 + k as u64));
            data.push((extract_feature(&noisy), digit));
        }
    }
    data
}

/// 欧氏距离平方
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// KNN 预测：在训练集里找 k 个最近邻，多数投票
fn knn_predict(train: &[(Vec<f64>, u8)], x: &[f64], k: usize) -> u8 {
    let mut dists: Vec<(f64, u8)> = train.iter().map(|(f, y)| (dist2(f, x), *y)).collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut votes = [0u32; 10];
    for &(_, y) in dists.iter().take(k) {
        votes[y as usize] += 1;
    }
    votes.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap().0 as u8
}

/// 把数据集写成 CSV：每行 = label,f0,f1,...,f63  —— 与 ndarray 版本无关的中间形式
fn write_csv(path: &str, data: &[(Vec<f64>, u8)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for (feat, label) in data {
        write!(f, "{label}")?;
        for v in feat {
            write!(f, ",{v:.6}")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn main() {
    // 训练集与测试集用不同 seed 独立生成（关键：绝不能用训练集报准确率）
    let train = make_dataset(30, 0x1111); // 每数字 30 张 → 300 张
    let test = make_dataset(15, 0x7777); //  每数字 15 张 → 150 张
    println!("训练集 {} 张，测试集 {} 张，特征维度 {}", train.len(), test.len(), train[0].0.len());

    let k = 3;
    let mut confusion = [[0u32; 10]; 10]; // confusion[真实][预测]
    let mut correct = 0usize;
    for (feat, truth) in &test {
        let pred = knn_predict(&train, feat, k);
        confusion[*truth as usize][pred as usize] += 1;
        if pred == *truth {
            correct += 1;
        }
    }
    let acc = correct as f64 / test.len() as f64;
    println!("KNN(k={k}) 测试集准确率 = {:.3}（{}/{}）", acc, correct, test.len());

    // 打印 10x10 混淆矩阵（行=真实，列=预测）
    println!("\n混淆矩阵（行=真实数字，列=预测数字）：");
    print!("     ");
    for p in 0..10 {
        print!("{p:3}");
    }
    println!();
    for t in 0..10 {
        print!("真{t} |");
        for p in 0..10 {
            print!("{:3}", confusion[t][p]);
        }
        println!();
    }

    // 演示"用训练集自评"的误导：在训练集上再测一次（会虚高）
    let mut train_correct = 0usize;
    for (feat, truth) in &train {
        if knn_predict(&train, feat, k) == *truth {
            train_correct += 1;
        }
    }
    println!(
        "\n[对照] 若错误地用训练集自评：准确率 = {:.3}（明显偏高，是作弊）",
        train_correct as f64 / train.len() as f64
    );

    // 把特征落成 CSV：这两个文件就是喂给独立 linfa 工程训练的中间数据
    write_csv("out_ch35_train.csv", &train).unwrap();
    write_csv("out_ch35_test.csv", &test).unwrap();
    println!("\n已写出 out_ch35_train.csv / out_ch35_test.csv（label + 64 维特征），供独立 linfa 工程读取训练");
}
