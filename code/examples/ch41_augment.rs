//! 第 41 章 数据增强（data augmentation）：对一张合成图做一整套增强并存盘
//! 运行：cargo run --example ch41_augment
//! 依赖：image 0.25、imageproc 0.27（全部纯 Rust；不训练模型，只演示"免费扩充数据"）
//!
//! 思路：数据增强 = 在不改变"标签语义"的前提下，对原图做各种随机变换，
//! 让一张图变出许多张"看起来不一样、但还是同一个东西"的样本，从而提升泛化。
//! 本例把 翻转/旋转/裁剪/亮度·对比度抖动/加噪/缩放 各做一遍，
//! 打印每种增强前后的图像统计（尺寸 + 亮度均值/标准差），并把结果存成 PNG。

use image::{ImageBuffer, Rgb, RgbImage};
use image::imageops::{self, FilterType};
use imageproc::geometric_transformations::{rotate_about_center, Border, Interpolation};
use imageproc::noise::gaussian_noise;

/// 合成一张"有明显朝向"的彩色场景：灰度渐变背景 + 左上红矩形 + 右下绿圆 + 中间偏上蓝方块。
/// 故意做得左右/上下都不对称，这样翻转、旋转的效果一眼就能看出来。
fn make_scene(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        // 左上角红矩形
        if (15..55).contains(&xi) && (15..45).contains(&yi) {
            return Rgb([210, 40, 40]);
        }
        // 中间偏上的蓝方块
        if (60..74).contains(&xi) && (18..32).contains(&yi) {
            return Rgb([50, 80, 220]);
        }
        // 右下角绿圆
        let (cx, cy, r) = (96, 70, 18);
        if (xi - cx).pow(2) + (yi - cy).pow(2) < r * r {
            return Rgb([40, 190, 70]);
        }
        // 背景：从左到右由暗到亮的灰度渐变
        let g = (40 + x * 120 / (w - 1)) as u8;
        Rgb([g, g, g])
    })
}

/// 用 Rec.601 权重把 RGB 折算成亮度，统计整幅图的亮度均值与标准差。
/// 均值反映"整体明暗"，标准差反映"对比度/噪声量"——增强前后一比就有直觉。
fn luma_stats(img: &RgbImage) -> (f32, f32) {
    let n = (img.width() * img.height()) as f32;
    let lumas: Vec<f32> = img
        .pixels()
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();
    let mean = lumas.iter().sum::<f32>() / n;
    let var = lumas.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// 亮度抖动：每个通道整体加上 delta（可正可负），越界钳到 [0,255]。
fn brightness_jitter(img: &RgbImage, delta: i32) -> RgbImage {
    let clamp = |c: u8| (c as i32 + delta).clamp(0, 255) as u8;
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        Rgb([clamp(p[0]), clamp(p[1]), clamp(p[2])])
    })
}

/// 对比度抖动：以 128 为中心把每个像素值"拉开"(factor>1)或"压扁"(factor<1)。
fn contrast_jitter(img: &RgbImage, factor: f32) -> RgbImage {
    let adj = |c: u8| (((c as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0) as u8;
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        Rgb([adj(p[0]), adj(p[1]), adj(p[2])])
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (128u32, 96u32);
    let src = make_scene(w, h);
    let (m0, s0) = luma_stats(&src);
    println!("原图: {}x{}  亮度均值={:.1}  标准差={:.1}", w, h, m0, s0);
    src.save("out_ch41_00_src.png")?;
    println!();

    // 用来存放"增强后"的样本，最后统一报数
    let mut generated = 0u32;
    let mut report = |name: &str, img: &RgbImage, note: &str| -> Result<(), Box<dyn std::error::Error>> {
        let (m, s) = luma_stats(img);
        println!(
            "{name:<10} {:>3}x{:<3}  亮度均值={:>5.1}  标准差={:>5.1}   {note}",
            img.width(),
            img.height(),
            m,
            s
        );
        img.save(format!("out_ch41_{name}.png"))?;
        generated += 1;
        Ok(())
    };

    // ① 水平翻转：像素只换位置、不换数值 → 亮度均值/标准差与原图完全一致
    let flipped = imageops::flip_horizontal(&src);
    report("01_flip", &flipped, "左右镜像；统计量与原图相同（只是位置变了）")?;

    // ② 旋转 15°：绕中心顺时针旋转，转出画面的角落用黑色填充
    let rotated = rotate_about_center(
        &src,
        15f32.to_radians(),
        Interpolation::Bilinear,
        Border::Constant(Rgb([0, 0, 0])),
    );
    report("02_rotate", &rotated, "转 15°，角落黑边拉低了亮度均值")?;

    // ③ 随机裁剪 + 缩放回原尺寸：截取内部 80% 的窗口再放大，等价于"镜头拉近"
    let (cw, ch) = (w * 4 / 5, h * 4 / 5);
    let (ox, oy) = (12u32, 8u32); // 固定偏移，保证结果可复现
    let cropped = imageops::crop_imm(&src, ox, oy, cw, ch).to_image();
    let crop_resized = imageops::resize(&cropped, w, h, FilterType::Triangle);
    report("03_crop", &crop_resized, "裁掉边缘再放大回原尺寸（zoom-in）")?;

    // ④ 亮度抖动：整体调亮 50 个灰阶 → 亮度均值明显上升
    let brighter = brightness_jitter(&src, 50);
    report("04_bright", &brighter, "整体调亮 +50，均值上升")?;

    // ⑤ 对比度抖动：以 128 为中心拉开 1.4 倍 → 标准差变大
    let contrasted = contrast_jitter(&src, 1.4);
    report("05_contrast", &contrasted, "对比度 ×1.4，标准差变大")?;

    // ⑥ 加高斯噪声：模拟传感器/低光噪点 → 均值几乎不变、标准差上升
    let noisy = gaussian_noise(&src, 0.0, 25.0, 42);
    report("06_noise", &noisy, "高斯噪声 σ=25，均值不变、标准差上升")?;

    // ⑦ 缩放：直接缩到 0.75 倍尺寸（内容不变，尺寸变小）
    let (sw, sh) = (w * 3 / 4, h * 3 / 4);
    let scaled = imageops::resize(&src, sw, sh, FilterType::Lanczos3);
    report("07_scale", &scaled, "缩放到 0.75x，尺寸变小")?;

    println!();
    println!(
        "一张原图 → 额外生成了 {generated} 张增强样本（数据集直接放大到 {} 倍，全部免费）",
        generated + 1
    );
    Ok(())
}
