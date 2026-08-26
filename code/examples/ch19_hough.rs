//! 第 19 章 霍夫变换 —— 正文代码验证
#![allow(unused)]

use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut};
use imageproc::edges::canny;
use imageproc::hough::{detect_lines, draw_polar_lines_mut, LineDetectionOptions, PolarLine};
use imageproc::rect::Rect;

/// 合成一张“照片”：白底上有一个黑色矩形和一条粗斜线
fn make_scene(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    // 黑色实心正方形：左右两条竖边、上下两条横边（等长，票数对称）
    draw_filled_rect_mut(&mut img, Rect::at(60, 70).of_size(150, 150), Rgb([30, 30, 30]));
    // 一条较粗的黑色斜线（画多条相邻线段模拟粗细）
    for d in -1..=1 {
        let dx = d as f32;
        draw_line_segment_mut(
            &mut img,
            (250.0 + dx, 40.0),
            (370.0 + dx, 260.0),
            Rgb([20, 20, 20]),
        );
    }
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(420, 300);
    let gray: GrayImage = image::DynamicImage::ImageRgb8(scene.clone()).to_luma8();

    // 关键：先用 Canny 得到边缘图，再喂给霍夫
    let edges = canny(&gray, 50.0, 100.0);

    for vt in [50u32, 120, 200] {
        let opts = LineDetectionOptions {
            vote_threshold: vt,
            suppression_radius: 8,
        };
        let lines = detect_lines(&edges, opts);
        println!("vote_threshold={vt} -> 检出 {} 条直线", lines.len());
        for l in &lines {
            println!("   r={:.1}, angle={}°", l.r, l.angle_in_degrees);
        }
    }

    // 用一个合适的阈值画回原图
    let opts = LineDetectionOptions {
        vote_threshold: 120,
        suppression_radius: 8,
    };
    let lines = detect_lines(&edges, opts);
    println!("最终用于绘制的直线数 = {}", lines.len());
    let mut out = scene.clone();
    draw_polar_lines_mut(&mut out, &lines, Rgb([255, 0, 0]));

    edges.save("out_ch19_edges.png")?;
    out.save("out_ch19_lines.png")?;
    Ok(())
}
