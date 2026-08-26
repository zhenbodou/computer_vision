//! 第 80 章 表面缺陷检测（黄金模板差分 + 形态学 + Blob 分析）
//! 运行：cargo run --example ch80_defect
//! 依赖：image 0.25.10, imageproc 0.27.0
//!
//! 造一张"良品"纹理图 → 注入已知位置/大小的缺陷（划痕、斑点）+ 轻微噪声 →
//! 黄金模板差分 → 统计阈值(均值+kσ) → 形态学去噪 → 连通域 Blob 分析 →
//! 打印检出缺陷数/面积/位置 vs 注入真值，并存标注 PNG。

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::distance_transform::Norm;
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::morphology::open;
use imageproc::noise::{gaussian_noise, salt_and_pepper_noise};
use imageproc::rect::Rect;
use imageproc::region_labelling::{connected_components, Connectivity};

const W: u32 = 240;
const H: u32 = 200;

/// 合成"黄金良品"：水平渐变 + 温和正弦纹理（确定性、可复现）。
fn make_golden() -> GrayImage {
    GrayImage::from_fn(W, H, |x, y| {
        let base = 90.0 + 40.0 * (x as f64 / W as f64); // 渐变 90..130
        let tex = 12.0 * ((x as f64 * 0.15).sin() + (y as f64 * 0.13).sin());
        Luma([(base + tex).round().clamp(0.0, 255.0) as u8])
    })
}

/// 画一道"粗"划痕（3×3 笔刷扫过，约 3 像素宽，才不会被开运算抹掉）。
fn draw_scratch(img: &mut GrayImage, a: (f32, f32), b: (f32, f32), val: u8) {
    for ox in -1..=1 {
        for oy in -1..=1 {
            draw_line_segment_mut(
                img,
                (a.0 + ox as f32, a.1 + oy as f32),
                (b.0 + ox as f32, b.1 + oy as f32),
                Luma([val]),
            );
        }
    }
}

fn main() {
    let golden = make_golden();
    golden.save("out_ch80_golden.png").ok();

    // —— 注入已知缺陷（真值）——
    let mut test = golden.clone();
    // 1 道划痕（很暗，保证沿全长的差分都稳稳高过阈值，不被开运算截断）
    draw_scratch(&mut test, (55.0, 35.0), (135.0, 120.0), 15);
    // 3 个斑点：(圆心, 半径, 灰度)
    let spots = [((200i32, 60i32), 6i32, 35u8), ((55, 145), 5, 235), ((170, 155), 8, 35)];
    for &((cx, cy), r, v) in &spots {
        draw_filled_circle_mut(&mut test, (cx, cy), r, Luma([v]));
    }
    // 叠加轻微高斯噪声 + 少量椒盐噪声（模拟真实采集：传感器噪声 + 灰尘/坏点）
    let test = gaussian_noise(&test, 0.0, 3.0, 20260827);
    let test = salt_and_pepper_noise(&test, 0.004, 12345);
    test.save("out_ch80_test.png").ok();

    // 注入真值一览（用于自验证）
    println!("注入缺陷真值（共 4 个）：");
    println!("  划痕: 中点≈(95, 78)");
    for (i, &((cx, cy), r, _)) in spots.iter().enumerate() {
        println!("  斑点{}: 圆心=({cx},{cy}) 半径={r} 面积≈{:.0}", i + 1, std::f64::consts::PI * (r * r) as f64);
    }

    // —— 黄金模板差分：逐像素 |test - golden| ——
    let diff = GrayImage::from_fn(W, H, |x, y| {
        let d = (test.get_pixel(x, y)[0] as i32 - golden.get_pixel(x, y)[0] as i32).abs();
        Luma([d.min(255) as u8])
    });
    diff.save("out_ch80_diff.png").ok();

    // —— 统计阈值：均值 + kσ（对光照/噪声的整体波动自适应）——
    let vals: Vec<f64> = diff.as_raw().iter().map(|&v| v as f64).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
    let k = 4.0;
    let thr = mean + k * std;
    println!(
        "\n差分统计：均值={mean:.2}，σ={std:.2}，阈值(均值+{k}σ)={thr:.2}"
    );

    let mask = GrayImage::from_fn(W, H, |x, y| {
        if diff.get_pixel(x, y)[0] as f64 > thr {
            Luma([255])
        } else {
            Luma([0])
        }
    });

    // —— 形态学开运算去掉零星噪点（引第 17 章）——
    let clean = open(&mask, Norm::LInf, 1);
    clean.save("out_ch80_mask.png").ok();
    let raw_on = mask.as_raw().iter().filter(|&&v| v > 0).count();
    let clean_on = clean.as_raw().iter().filter(|&&v| v > 0).count();
    // 对比：不去噪时连通域会炸出一大堆假缺陷
    let comp = |img: &GrayImage| {
        connected_components(img, Connectivity::Eight, Luma([0u8]))
            .as_raw()
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    };
    println!("阈值后前景像素={raw_on}，开运算去噪后={clean_on}");
    println!("阈值后连通域数={}（含噪点），开运算后={}", comp(&mask), comp(&clean));

    // —— 连通域 Blob 分析：数缺陷、算面积/质心/包围盒（引第 18 章）——
    let labels = connected_components(&clean, Connectivity::Eight, Luma([0u8]));
    let n = labels.as_raw().iter().copied().max().unwrap_or(0) as usize;
    let mut area = vec![0u32; n + 1];
    let (mut sx, mut sy) = (vec![0u64; n + 1], vec![0u64; n + 1]);
    let (mut mnx, mut mny) = (vec![u32::MAX; n + 1], vec![u32::MAX; n + 1]);
    let (mut mxx, mut mxy) = (vec![0u32; n + 1], vec![0u32; n + 1]);
    for (x, y, p) in labels.enumerate_pixels() {
        let l = p[0] as usize;
        if l == 0 {
            continue;
        }
        area[l] += 1;
        sx[l] += x as u64;
        sy[l] += y as u64;
        mnx[l] = mnx[l].min(x);
        mny[l] = mny[l].min(y);
        mxx[l] = mxx[l].max(x);
        mxy[l] = mxy[l].max(y);
    }

    let min_area = 20u32; // 小于它的当残余噪点丢弃
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(test.clone()).to_rgb8();
    let mut found = 0;
    println!("\n检出缺陷（面积 ≥ {min_area}）：");
    for l in 1..=n {
        if area[l] < min_area {
            continue;
        }
        found += 1;
        let (cx, cy) = ((sx[l] / area[l] as u64) as i32, (sy[l] / area[l] as u64) as i32);
        let (bx, by, bw, bh) = (mnx[l], mny[l], mxx[l] - mnx[l] + 1, mxy[l] - mny[l] + 1);
        println!("  缺陷#{found}: 面积={:>4} 质心=({cx},{cy}) 包围盒=({bx},{by},{bw}x{bh})", area[l]);
        draw_hollow_rect_mut(&mut vis, Rect::at(bx as i32, by as i32).of_size(bw, bh), Rgb([255, 0, 0]));
    }
    vis.save("out_ch80_defects.png").ok();
    println!("\n检出缺陷数 = {found} / 注入 4");
    println!("已保存：out_ch80_diff.png / out_ch80_mask.png / out_ch80_defects.png");
}
