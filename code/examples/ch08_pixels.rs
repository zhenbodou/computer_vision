//! 第 8 章 访问和修改像素 —— 正文代码验证
#![allow(unused)]

use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};

fn make_gradient(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let r = (x * 255 / (w - 1)) as u8;
        let g = (y * 255 / (h - 1)) as u8;
        Rgb([r, g, 128])
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut img = make_gradient(200, 150);

    // 读单个像素：get_pixel 返回 &Rgb<u8>，解引用得到值
    let p = *img.get_pixel(0, 0);
    println!("(0,0) 的 RGB = {:?}", p.0); // p.0 是内部的 [u8; 3]
    println!("红色分量 = {}", p[0]); // 也可以像数组一样索引

    // 越界安全读取：get_pixel_checked 返回 Option，不会 panic
    match img.get_pixel_checked(999, 999) {
        Some(px) => println!("{:?}", px),
        None => println!("(999,999) 越界，返回 None"),
    }

    // 反色：遍历每个可变像素，new = 255 - old
    for (_x, _y, px) in img.enumerate_pixels_mut() {
        px.0[0] = 255 - px.0[0];
        px.0[1] = 255 - px.0[1];
        px.0[2] = 255 - px.0[2];
    }

    // 在左上角画一个 40x40 的红色实心方块
    let side = 40u32;
    for y in 0..side.min(img.height()) {
        for x in 0..side.min(img.width()) {
            img.put_pixel(x, y, Rgb([255, 0, 0]));
        }
    }

    // 底层缓冲区：as_raw 拿到 &Vec<u8>，长度 = 宽 x 高 x 通道数
    let raw: &Vec<u8> = img.as_raw();
    println!(
        "缓冲区长度 = {}，w*h*3 = {}",
        raw.len(),
        img.width() * img.height() * 3
    );

    img.save("out_ch08_result.png")?;
    println!("已保存 out_ch08_result.png");
    Ok(())
}
