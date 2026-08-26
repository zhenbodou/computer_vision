//! 第 9 章 颜色空间与灰度化 —— 正文代码验证
#![allow(unused)]

use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

/// 经典 Rec.601 加权灰度公式
fn gray_601(r: u8, g: u8, b: u8) -> u8 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8
}

/// image 的 to_luma8 内部用的 Rec.709/sRGB 整数权重：[2126,7152,722]/10000，向下取整
fn gray_709(r: u8, g: u8, b: u8) -> u8 {
    ((2126 * r as u32 + 7152 * g as u32 + 722 * b as u32) / 10000) as u8
}

/// RGB -> HSV。H 单位为度 [0,360)，S、V 为 [0,1]
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let (rf, gf, bf) = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;

    let mut h = if delta == 0.0 {
        0.0
    } else if max == rf {
        60.0 * (((gf - bf) / delta).rem_euclid(6.0))
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    if h < 0.0 {
        h += 360.0;
    }

    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;
    (h, s, v)
}

/// 合成场景：灰度渐变背景 + 左侧红圆 + 右侧绿方块
fn make_scene(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        let (cx, cy, rad) = (w as i32 / 4, h as i32 / 2, 45); // 左侧红圆
        if (xi - cx).pow(2) + (yi - cy).pow(2) < rad * rad {
            return Rgb([220, 30, 30]);
        }
        if xi > 3 * w as i32 / 5 && xi < 4 * w as i32 / 5 // 右侧绿方块
            && yi > h as i32 / 3 && yi < 2 * h as i32 / 3
        {
            return Rgb([40, 200, 60]);
        }
        let bg = (x * 200 / (w - 1)) as u8; // 背景灰度渐变
        Rgb([bg, bg, bg])
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(320, 200);

    // 手写灰度（两种权重）与 to_luma8 对比
    let mut my601: GrayImage = ImageBuffer::new(scene.width(), scene.height());
    let mut my709: GrayImage = ImageBuffer::new(scene.width(), scene.height());
    for (x, y, px) in scene.enumerate_pixels() {
        let [r, g, b] = px.0;
        my601.put_pixel(x, y, Luma([gray_601(r, g, b)]));
        my709.put_pixel(x, y, Luma([gray_709(r, g, b)]));
    }
    let lib_gray = DynamicImage::ImageRgb8(scene.clone()).to_luma8();

    let max_diff = |a: &GrayImage, b: &GrayImage| -> i32 {
        a.as_raw().iter().zip(b.as_raw().iter())
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max().unwrap_or(0)
    };
    println!("Rec.601 手写 vs to_luma8 最大差异 = {}", max_diff(&my601, &lib_gray));
    println!("Rec.709 手写 vs to_luma8 最大差异 = {}", max_diff(&my709, &lib_gray));

    // 用 HSV 阈值提取红色区域
    let mut mask: GrayImage = ImageBuffer::new(scene.width(), scene.height());
    let mut red_count = 0u32;
    for (x, y, px) in scene.enumerate_pixels() {
        let [r, g, b] = px.0;
        let (hh, ss, vv) = rgb_to_hsv(r, g, b);
        let is_red = (hh <= 15.0 || hh >= 345.0) && ss >= 0.4 && vv >= 0.2;
        if is_red {
            red_count += 1;
        }
        mask.put_pixel(x, y, Luma([if is_red { 255 } else { 0 }]));
    }
    println!("被判为红色的像素数 = {}", red_count);

    my601.save("out_ch09_gray601.png")?;
    lib_gray.save("out_ch09_luma8.png")?;
    mask.save("out_ch09_red_mask.png")?;
    Ok(())
}
