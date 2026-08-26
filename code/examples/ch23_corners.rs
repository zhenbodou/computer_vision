//! 第 23 章 角点检测（FAST）—— 正文代码验证
#![allow(unused)]

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::corners::{corners_fast9, Corner};
use imageproc::drawing::{draw_cross_mut, draw_filled_rect_mut};
use imageproc::rect::Rect;
use imageproc::suppress::suppress_non_maximum;

/// 合成一张灰度“照片”：白底 + 三个不同对比度的方块 + 一小片棋盘格。
/// 方块颜色越浅、与背景对比越弱，角点“分数”越低，越容易被高阈值滤掉。
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([235])); // 近白背景
    // 三个方块：对比度从强到弱（角点分数依次降低）
    draw_filled_rect_mut(&mut img, Rect::at(40, 40).of_size(70, 70), Luma([30])); // 强对比
    draw_filled_rect_mut(&mut img, Rect::at(150, 55).of_size(55, 55), Luma([130])); // 中对比
    draw_filled_rect_mut(&mut img, Rect::at(60, 165).of_size(55, 55), Luma([185])); // 弱对比
    // 一小片高对比棋盘格：贡献大量密集角点
    let (ox, oy, cell, n) = (250i32, 150i32, 18i32, 4);
    for gy in 0..n {
        for gx in 0..n {
            if (gx + gy) % 2 == 0 {
                let rx = ox + gx * cell;
                let ry = oy + gy * cell;
                draw_filled_rect_mut(
                    &mut img,
                    Rect::at(rx, ry).of_size(cell as u32, cell as u32),
                    Luma([20]),
                );
            }
        }
    }
    img
}

/// 把角点列表画成十字标记，输出到彩色图上便于观察
fn draw_corners(gray: &GrayImage, corners: &[Corner], color: Rgb<u8>) -> RgbImage {
    let mut canvas = RgbImage::new(gray.width(), gray.height());
    for (x, y, p) in gray.enumerate_pixels() {
        let v = p[0];
        canvas.put_pixel(x, y, Rgb([v, v, v]));
    }
    for c in corners {
        draw_cross_mut(&mut canvas, color, c.x as i32, c.y as i32);
    }
    canvas
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gray = make_scene(360, 260);

    // ① 阈值高低对角点数量的影响
    for t in [20u8, 70, 120, 170] {
        let corners = corners_fast9(&gray, t);
        println!("threshold={:>3} -> FAST 角点数 = {}", t, corners.len());
    }

    // ② 取一个中等阈值，看看角点“扎堆”现象
    let threshold = 40u8;
    let corners = corners_fast9(&gray, threshold);
    println!("\nthreshold={} 时原始角点数 = {}", threshold, corners.len());

    // ③ 非极大值抑制（NMS）：把角点分数写进一张响应图，再抑制局部非极大
    let mut response = GrayImage::new(gray.width(), gray.height());
    for c in &corners {
        // corners_fast9 的 score 本质是 u8 阈值，可无损放进 Luma<u8>
        response.put_pixel(c.x, c.y, Luma([c.score as u8]));
    }
    let radius = 6u32;
    let suppressed = suppress_non_maximum(&response, radius);
    let kept: Vec<Corner> = suppressed
        .enumerate_pixels()
        .filter(|(_, _, p)| p[0] > 0)
        .map(|(x, y, p)| Corner::new(x, y, p[0] as f32))
        .collect();
    println!("NMS(radius={}) 后保留角点数 = {}", radius, kept.len());

    // ④ 输出可视化：原始角点 vs NMS 后角点
    draw_corners(&gray, &corners, Rgb([255, 0, 0])).save("out_ch23_fast_raw.png")?;
    draw_corners(&gray, &kept, Rgb([0, 200, 0])).save("out_ch23_fast_nms.png")?;
    gray.save("out_ch23_scene.png")?;
    Ok(())
}
