//! 第 13 章 卷积与线性滤波 —— 正文代码验证
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::{box_filter, filter_clamped, gaussian_blur_f32, separable_filter};
use imageproc::kernel::Kernel;

/// 合成灰度测试图：水平灰度渐变背景 + 中央亮方块 + 顶部一组细竖线（高频细节）
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        // 背景：从左到右 40 -> 200 的灰度渐变
        let mut v = 40 + (x * 160 / (w - 1)) as i32;
        // 中央亮方块
        if x > w / 3 && x < 2 * w / 3 && y > h / 3 && y < 2 * h / 3 {
            v = 235;
        }
        // 顶部每隔 8 像素一条黑竖线，制造细节，便于观察模糊
        if y < h / 4 && x % 8 == 0 {
            v = 20;
        }
        Luma([v.clamp(0, 255) as u8])
    })
}

/// 手写 3×3 均值卷积：邻域 9 个像素求和再除以 9。边界用"复制边缘像素"处理。
fn manual_mean3x3(src: &GrayImage) -> GrayImage {
    let (w, h) = src.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    // 越界坐标夹回图内 = 复制最边缘的像素（边界处理）
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    sum += src.get_pixel(sx, sy).0[0] as u32;
                }
            }
            out.put_pixel(x, y, Luma([(sum / 9) as u8])); // 除以 9 = 归一化
        }
    }
    out
}

fn mean_brightness(img: &GrayImage) -> f32 {
    let total: u64 = img.as_raw().iter().map(|&p| p as u64).sum();
    total as f32 / img.as_raw().len() as f32
}

fn max_diff(a: &GrayImage, b: &GrayImage) -> i32 {
    a.as_raw()
        .iter()
        .zip(b.as_raw().iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0)
}

/// 与 imageproc 内部一致的一维高斯核：半径 = ceil(2σ)，最后归一化
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (2.0 * sigma).ceil() as usize;
    let mut k = vec![0.0f32; 2 * radius + 1];
    for i in 0..=radius {
        let x = i as f32;
        let v = (-(x * x) / (2.0 * sigma * sigma)).exp();
        k[radius + i] = v;
        k[radius - i] = v;
    }
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(256, 256);

    // ① 手写 3×3 均值卷积 vs 库函数 box_filter(半径=1，即 3×3)
    let mine = manual_mean3x3(&scene);
    let boxed = box_filter(&scene, 1, 1);
    println!("手写均值 vs box_filter 最大差异 = {}", max_diff(&mine, &boxed));

    // ② 坑：核未归一化。3×3 全 1 核（权重和=9），不除以 9 → 整体亮度约 ×9 而饱和
    let ones = [1i32; 9];
    let bad_kernel = Kernel::new(&ones, 3, 3);
    let over_bright = filter_clamped::<_, i32, u8>(&scene, bad_kernel);
    println!("原图平均亮度         = {:.1}", mean_brightness(&scene));
    println!("未归一化(全1核)后亮度 = {:.1}", mean_brightness(&over_bright));

    // ③ 高斯模糊（imageproc）
    let g = gaussian_blur_f32(&scene, 2.0);

    // ④ 可分离卷积：两次一维高斯 == 一次二维高斯
    let k1d = gaussian_kernel_1d(2.0);
    let sep = separable_filter(&scene, &k1d, &k1d);
    println!(
        "可分离(两次1D) vs 二维 gaussian_blur_f32 最大差异 = {}",
        max_diff(&sep, &g)
    );
    println!(
        "高斯核长度 = {}（朴素二维需 {} 次乘加/像素，可分离仅 {} 次）",
        k1d.len(),
        k1d.len() * k1d.len(),
        2 * k1d.len()
    );

    // ⑤ 自定义锐化核（权重和=1，保持整体亮度）
    let sharpen_data = [0i32, -1, 0, -1, 5, -1, 0, -1, 0];
    let sharpen = Kernel::new(&sharpen_data, 3, 3);
    let sharp = filter_clamped::<_, i32, u8>(&scene, sharpen);
    println!(
        "锐化后平均亮度 = {:.1}（与原图相近说明核已归一化）",
        mean_brightness(&sharp)
    );

    scene.save("out_ch13_scene.png")?;
    boxed.save("out_ch13_box.png")?;
    g.save("out_ch13_gauss.png")?;
    sharp.save("out_ch13_sharp.png")?;
    Ok(())
}
