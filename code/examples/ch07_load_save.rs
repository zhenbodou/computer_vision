//! 第 7 章 读取、显示与保存图像 —— 正文代码验证
#![allow(unused)]

use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};

/// 合成一张 256x256 的彩色渐变图：横向红色渐变、纵向绿色渐变
fn make_gradient(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let r = (x * 255 / (w - 1)) as u8;
        let g = (y * 255 / (h - 1)) as u8;
        Rgb([r, g, 128])
    })
}

/// 用 match 稳妥地处理打开失败的情况
fn open_and_report(path: &str) {
    match image::open(path) {
        Ok(img) => println!("打开成功：{}，尺寸 {:?}", path, img.dimensions()),
        Err(e) => eprintln!("打开失败：{} —— {}", path, e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 先合成一张图并存盘，作为后面读取的输入（让示例自包含）
    let grad = make_gradient(256, 256);
    grad.save("out_gradient.png")?;

    // 2. 读图：image::open 返回 Result，用 ? 传播错误
    let img: DynamicImage = image::open("out_gradient.png")?;

    // 3. 尺寸：dimensions() 来自 GenericImageView
    let (w, h) = img.dimensions();
    println!("图像尺寸：{} x {}，色彩类型：{:?}", w, h, img.color());

    // 4. 转灰度 / 转 RGB
    let gray = img.to_luma8();
    let rgb = img.to_rgb8();
    gray.save("out_gray.png")?;

    // 5. 格式转换：按扩展名自动选格式（png -> jpg）
    img.save("out_gradient.jpg")?;

    // 6. 扩展名与真实格式不一致时，用 save_with_format 显式指定
    img.save_with_format("out_forced.bin", ImageFormat::Png)?;

    // 7. 演示错误处理
    open_and_report("out_gradient.png");
    open_and_report("no_such_file.png");

    println!("完成，已生成 out_*.png / .jpg 等文件。");
    Ok(())
}
