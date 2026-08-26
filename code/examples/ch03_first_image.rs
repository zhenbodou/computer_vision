//! 第 3 章：第一个能跑的程序——生成一张彩色渐变图并保存为 PNG。
//! 运行： cargo run --example ch03_first_image
//! 依赖： image = "0.25"

use image::{ImageBuffer, Rgb, RgbImage};

fn main() {
    // 图像尺寸：宽 256、高 256（正好让每个方向的渐变从 0 铺到 255）
    let width: u32 = 256;
    let height: u32 = 256;

    // from_fn 会对每一个像素坐标 (x, y) 调用一次闭包，返回该像素的颜色。
    // Rgb([r, g, b])：红、绿、蓝三个通道，每个取值 0..=255。
    let img: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
        let r = x as u8; // 红色随 x 从左到右由暗变亮
        let g = y as u8; // 绿色随 y 从上到下由暗变亮
        let b = 128u8; // 蓝色固定，让整体偏紫
        Rgb([r, g, b])
    });

    // 按扩展名自动选择 PNG 编码写盘；出错就直接 panic（第一个程序，简单点）。
    img.save("gradient.png").expect("保存图片失败");

    println!("已生成 gradient.png（{} x {}）", width, height);
}
