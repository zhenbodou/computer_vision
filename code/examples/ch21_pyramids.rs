//! 第 21 章 图像金字塔与多尺度 —— 正文代码验证
//! 运行：cargo run --example ch21_pyramids
#![allow(unused)]

use image::imageops::{self, FilterType};
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::gaussian_blur_f32;

/// 合成一张灰度测试图：左右渐变背景 + 一个亮圆 + 一个暗圆 + 右上角细棋盘纹理
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        // 背景：从左到右的灰度渐变
        let mut v = (x * 180 / (w - 1)) as i32 + 20;
        // 左上一个亮圆
        let (cx, cy, r) = (w as i32 / 3, h as i32 / 3, 40);
        if (xi - cx).pow(2) + (yi - cy).pow(2) < r * r {
            v = 240;
        }
        // 右下一个暗圆
        let (cx2, cy2, r2) = (2 * w as i32 / 3, 2 * h as i32 / 3, 28);
        if (xi - cx2).pow(2) + (yi - cy2).pow(2) < r2 * r2 {
            v = 30;
        }
        // 右上细棋盘纹理（高频细节，金字塔顶层会被逐渐抹平）
        if xi > 2 * w as i32 / 3 && yi < h as i32 / 3 && ((xi / 3 + yi / 3) % 2 == 0) {
            v = 200;
        }
        Luma([v.clamp(0, 255) as u8])
    })
}

/// 上采样（放大）到指定尺寸——构建与重建拉普拉斯金字塔时都用它，保证一致
fn expand(img: &GrayImage, w: u32, h: u32) -> GrayImage {
    imageops::resize(img, w, h, FilterType::Triangle)
}

/// 构建高斯金字塔：每层 = 上一层「先高斯模糊、再降采样到 1/2」
fn gaussian_pyramid(base: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyr = vec![base.clone()];
    for _ in 1..levels {
        let prev = pyr.last().unwrap();
        // ① 先高斯模糊：抗混叠（anti-aliasing）的关键一步
        let blurred = gaussian_blur_f32(prev, 1.0);
        // ② 再降采样到一半（宽高各取 1/2）
        let (w, h) = ((prev.width() / 2).max(1), (prev.height() / 2).max(1));
        let down = imageops::resize(&blurred, w, h, FilterType::Triangle);
        pyr.push(down);
    }
    pyr
}

/// 构建拉普拉斯金字塔：L_i = G_i - expand(G_{i+1})，顶层直接存最小的高斯层
/// 用 f32 存储：相邻层之差有正有负，且 u8 之差恰好是精确整数
fn laplacian_pyramid(gauss: &[GrayImage]) -> Vec<Vec<f32>> {
    let n = gauss.len();
    let mut lap: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let (w, h) = (gauss[i].width(), gauss[i].height());
        let up = expand(&gauss[i + 1], w, h); // 把上一层放大回本层尺寸
        let diff: Vec<f32> = gauss[i]
            .as_raw()
            .iter()
            .zip(up.as_raw().iter())
            .map(|(&a, &b)| a as f32 - b as f32) // 本层 - 放大后的粗糙版 = 丢失的细节
            .collect();
        lap.push(diff);
    }
    // 顶层（最小、最模糊的高斯层）作为重建的「种子」
    lap.push(gauss[n - 1].as_raw().iter().map(|&p| p as f32).collect());
    lap
}

/// 从拉普拉斯金字塔重建原图：自顶向下，G_i = L_i + expand(G_{i+1})
fn reconstruct(lap: &[Vec<f32>], sizes: &[(u32, u32)]) -> GrayImage {
    let n = lap.len();
    // 从最顶层（最小）开始：种子就是最小的高斯层
    let (tw, th) = sizes[n - 1];
    let seed = lap[n - 1]
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let mut cur: GrayImage = ImageBuffer::from_raw(tw, th, seed).unwrap();

    for i in (0..n - 1).rev() {
        let (w, h) = sizes[i];
        let up = expand(&cur, w, h); // 把当前重建结果放大回本层尺寸
        let out = up
            .as_raw()
            .iter()
            .zip(lap[i].iter())
            .map(|(&u, &l)| (u as f32 + l).round().clamp(0.0, 255.0) as u8) // 补回细节
            .collect();
        cur = ImageBuffer::from_raw(w, h, out).unwrap();
    }
    cur
}

/// 把有正有负的拉普拉斯层偏移 +128 后 clamp，存成可视化灰度图
fn lap_to_vis(lap: &[f32], w: u32, h: u32) -> GrayImage {
    let buf: Vec<u8> = lap
        .iter()
        .map(|&v| (v + 128.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    ImageBuffer::from_raw(w, h, buf).unwrap()
}

/// 一维统计：均值
fn mean(img: &GrayImage) -> f32 {
    let s: u64 = img.as_raw().iter().map(|&p| p as u64).sum();
    s as f32 / img.as_raw().len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ============ 1. 高斯金字塔 ============
    let base = make_scene(256, 192);
    let levels = 4;
    let gauss = gaussian_pyramid(&base, levels);
    println!("== 高斯金字塔（{} 层）==", levels);
    for (i, g) in gauss.iter().enumerate() {
        println!("  第 {} 层尺寸 = {}x{}", i, g.width(), g.height());
        g.save(format!("out_ch21_gauss_l{}.png", i))?;
    }

    // ============ 2. 拉普拉斯金字塔 + 无损重建 ============
    let sizes: Vec<(u32, u32)> = gauss.iter().map(|g| (g.width(), g.height())).collect();
    let lap = laplacian_pyramid(&gauss);
    for i in 0..lap.len() - 1 {
        lap_to_vis(&lap[i], sizes[i].0, sizes[i].1).save(format!("out_ch21_lap_l{}.png", i))?;
    }
    let recon = reconstruct(&lap, &sizes);
    recon.save("out_ch21_reconstructed.png")?;

    // 逐像素比较重建结果与原图
    let (mut max_diff, mut sum_diff) = (0i32, 0i64);
    for (&a, &b) in base.as_raw().iter().zip(recon.as_raw().iter()) {
        let d = (a as i32 - b as i32).abs();
        max_diff = max_diff.max(d);
        sum_diff += d as i64;
    }
    let mean_diff = sum_diff as f64 / base.as_raw().len() as f64;
    println!("== 拉普拉斯金字塔重建 ==");
    println!("  重建 vs 原图：最大误差 = {max_diff}，平均误差 = {mean_diff:.4}");

    // ============ 3. 抗混叠对照实验：细条纹降采样 ============
    // 1 像素宽的竖条纹（周期 2），是能表示的最高频信号，降采样最容易翻车
    let stripes: GrayImage = ImageBuffer::from_fn(256, 64, |x, _y| {
        Luma([if x % 2 == 0 { 255 } else { 0 }])
    });
    println!("== 抗混叠对照（周期 2 竖条纹，原图均值 = {:.1}）==", mean(&stripes));

    // 方法 A：直接用最近邻降采样，不做任何模糊
    let alias = imageops::resize(&stripes, 128, 32, FilterType::Nearest);
    // 方法 B：先高斯模糊，再最近邻降采样
    let blurred = gaussian_blur_f32(&stripes, 1.0);
    let clean = imageops::resize(&blurred, 128, 32, FilterType::Nearest);

    println!("  A. 无预模糊 + 最近邻 → 均值 = {:.1}（条纹被混叠成纯色，亮度信息丢失）", mean(&alias));
    println!("  B. 先模糊 + 最近邻 → 均值 = {:.1}（正确保留整体亮度）", mean(&clean));
    alias.save("out_ch21_alias.png")?;
    clean.save("out_ch21_antialias.png")?;

    Ok(())
}
