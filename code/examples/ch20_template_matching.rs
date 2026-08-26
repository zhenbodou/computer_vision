//! 第 20 章 模板匹配 —— 正文代码验证
#![allow(unused)]

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};

/// 合成一张大图：灰底 + 左上角“靶心”图案（待匹配目标）+ 右下一块很亮的干扰方块 + 若干填充物
fn make_big(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([90]));
    // 目标图案：靶心（外圈亮、内芯暗），位于 (50,40)-(90,80) 区域
    draw_filled_circle_mut(&mut img, (70, 60), 18, Luma([150]));
    draw_filled_circle_mut(&mut img, (70, 60), 8, Luma([30]));
    // 干扰：一块很亮（接近白）的方块——专门用来“骗”非归一化的互相关
    draw_filled_rect_mut(&mut img, Rect::at(200, 120).of_size(46, 46), Luma([250]));
    // 其它填充物
    draw_filled_circle_mut(&mut img, (150, 150), 16, Luma([120]));
    draw_filled_rect_mut(&mut img, Rect::at(30, 150).of_size(40, 30), Luma([60]));
    img
}

/// 整体“变亮”：像素乘以系数后饱和截断到 [0,255]（模拟增益/曝光变化）
fn brighten(img: &GrayImage, factor: f32) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let v = img.get_pixel(x, y)[0] as f32 * factor;
        Luma([v.min(255.0) as u8])
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let big = make_big(300, 200);
    let (tx, ty, tw, th) = (50u32, 40u32, 40u32, 40u32); // 目标真实位置与模板大小

    // 从大图里裁一小块当模板
    let template = image::imageops::crop_imm(&big, tx, ty, tw, th).to_image();

    // 方法一：误差平方和（SSE），越小越匹配
    let sse = match_template(&big, &template, MatchTemplateMethod::SumOfSquaredErrors);
    let e = find_extremes(&sse);
    println!(
        "[SSE] 最佳(最小)位置 = {:?}, 最小误差 = {}",
        e.min_value_location, e.min_value
    );

    // 方法二：非归一化互相关，越大越匹配——会被“亮”骗
    let cc = match_template(&big, &template, MatchTemplateMethod::CrossCorrelation);
    let e_cc = find_extremes(&cc);
    println!(
        "[CC 非归一化] 最佳(最大)位置 = {:?}, 最大值 = {:.0}",
        e_cc.max_value_location, e_cc.max_value
    );

    // 方法三：归一化互相关，越大越匹配——只看“形状像不像”
    let ncc = match_template(&big, &template, MatchTemplateMethod::CrossCorrelationNormalized);
    let e_ncc = find_extremes(&ncc);
    println!(
        "[NCC 归一化] 最佳(最大)位置 = {:?}, 最大值 = {:.4}",
        e_ncc.max_value_location, e_ncc.max_value
    );

    // 整体变亮后再比一次：非归一化 vs 归一化
    let bright = brighten(&big, 1.4);
    let cc_b = match_template(&bright, &template, MatchTemplateMethod::CrossCorrelation);
    let ncc_b = match_template(&bright, &template, MatchTemplateMethod::CrossCorrelationNormalized);
    let e_cc_b = find_extremes(&cc_b);
    let e_ncc_b = find_extremes(&ncc_b);
    println!(
        "[变亮后 CC] 位置 = {:?}, 最大值 = {:.0}",
        e_cc_b.max_value_location, e_cc_b.max_value
    );
    println!(
        "[变亮后 NCC] 位置 = {:?}, 最大值 = {:.4}",
        e_ncc_b.max_value_location, e_ncc_b.max_value
    );

    // 画框：绿=NCC 找到的正确位置，红=非归一化被骗到的位置
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(big.clone()).to_rgb8();
    let (gx, gy) = e_ncc.max_value_location;
    draw_hollow_rect_mut(&mut vis, Rect::at(gx as i32, gy as i32).of_size(tw, th), Rgb([0, 255, 0]));
    let (rx, ry) = e_cc.max_value_location;
    draw_hollow_rect_mut(&mut vis, Rect::at(rx as i32, ry as i32).of_size(tw, th), Rgb([255, 0, 0]));

    template.save("out_ch20_template.png")?;
    vis.save("out_ch20_matched.png")?;
    Ok(())
}
