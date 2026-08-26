//! 第 5 章 图像到底是什么：用 image 造灰度图与彩色图，读取并打印像素数值。
//! 验证：cargo check --example ch05_image_basics / cargo run --example ch05_image_basics
#![allow(unused)]

use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 1. 一张 4×3 的灰度图，从左到右由暗变亮 ----------
    let (w, h) = (4u32, 3u32);
    let gray: GrayImage = ImageBuffer::from_fn(w, h, |x, _y| {
        // x=0..3 → 0,80,160,240，越往右越亮
        Luma([(x * 80) as u8])
    });

    println!("== 灰度图 {w}×{h} ==");
    // get_pixel 返回 &Luma<u8>，.0 取出内部的 [u8; 1] 数组
    println!("像素 (0,0) = {:?}", gray.get_pixel(0, 0).0); // [0]
    println!("像素 (3,0) = {:?}", gray.get_pixel(3, 0).0); // [240]

    // ---------- 2. 一张 2×2 的彩色图（RGB） ----------
    let mut rgb: RgbImage = ImageBuffer::new(2, 2);
    rgb.put_pixel(0, 0, Rgb([255, 0, 0])); // 红
    rgb.put_pixel(1, 0, Rgb([0, 255, 0])); // 绿
    rgb.put_pixel(0, 1, Rgb([0, 0, 255])); // 蓝
    rgb.put_pixel(1, 1, Rgb([255, 255, 255])); // 白

    println!("\n== 彩色图 2×2 ==");
    println!("像素 (0,0) = {:?}", rgb.get_pixel(0, 0).0); // [255, 0, 0]
    println!("像素 (1,1) = {:?}", rgb.get_pixel(1, 1).0); // [255,255,255]

    let (width, height) = rgb.dimensions();
    println!("尺寸 = {width}×{height}，通道数 = 3");

    // ---------- 3. 把图像看成一维字节数组 ----------
    // 行优先：先第 0 行的所有像素，再第 1 行……每像素 3 个字节 R,G,B
    let raw: &[u8] = rgb.as_raw();
    println!("底层字节数 = {} (= 2×2×3)", raw.len());
    println!("前 6 个字节 = {:?}", &raw[0..6]); // 红 + 绿

    // ---------- 4. 存盘（PNG 无损） ----------
    gray.save("gray.png")?;
    rgb.save("color.png")?;
    println!("\n已保存 gray.png / color.png");
    Ok(())
}
