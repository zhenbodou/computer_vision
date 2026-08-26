//! 第 16 章 阈值分割与二值化 —— 正文代码验证（与书中代码块一致）
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::contrast::{adaptive_threshold, otsu_level, threshold, ThresholdType};

const W: u32 = 240;
const H: u32 = 180;

/// 背景光照（左暗右亮的线性渐变）：40 → 210
fn background(x: u32) -> i32 {
    40 + (x as i32 * 170 / (W as i32 - 1))
}

/// 三根竖直暗条（宽约 12 像素），模拟"局部比背景暗"的目标
fn is_bar(x: u32, y: u32) -> bool {
    if !(30..150).contains(&y) {
        return false;
    }
    (40..52).contains(&x) || (110..122).contains(&x) || (185..197).contains(&x)
}

/// 合成"光照不均 + 局部暗目标"的灰度图
fn make_scene() -> GrayImage {
    ImageBuffer::from_fn(W, H, |x, y| {
        let bg = background(x);
        if is_bar(x, y) {
            Luma([(bg - 60).clamp(0, 255) as u8]) // 比"当地背景"暗 60
        } else {
            Luma([bg.clamp(0, 255) as u8])
        }
    })
}

/// 前景 IoU（交并比）：前景=暗条
fn iou<F: Fn(u32, u32) -> bool>(detected: F) -> f32 {
    let (mut inter, mut union) = (0u32, 0u32);
    for y in 0..H {
        for x in 0..W {
            let (truth, det) = (is_bar(x, y), detected(x, y));
            if truth && det { inter += 1; }
            if truth || det { union += 1; }
        }
    }
    if union == 0 { 0.0 } else { inter as f32 / union as f32 * 100.0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene();

    // 打印几处像素，看"暗条亮度"与"别处背景亮度"如何重叠
    println!("左/中/右暗条像素值 ≈ {} / {} / {}",
        scene.get_pixel(45, 90)[0], scene.get_pixel(116, 90)[0], scene.get_pixel(190, 90)[0]);
    println!("最右侧背景像素值 ≈ {}", scene.get_pixel(238, 90)[0]);

    // ① 全局固定阈值：暴力扫描 0..=255，找"最优"全局阈值能达到的最高 IoU
    let (mut best_t, mut best_iou) = (0u8, 0.0f32);
    for t in 0u16..=255 {
        let v = iou(|x, y| scene.get_pixel(x, y)[0] as u16 <= t); // 暗于阈值 → 前景
        if v > best_iou { best_iou = v; best_t = t as u8; }
    }
    println!("最优全局固定阈值 T={}，前景 IoU={:.1}%", best_t, best_iou);

    // ② Otsu 自动阈值
    let level = otsu_level(&scene);
    println!("Otsu 阈值={}，前景 IoU={:.1}%",
        level, iou(|x, y| scene.get_pixel(x, y)[0] <= level));

    // ③ 自适应阈值（block 半径 15，delta 10）
    // 规则：pixel >= 局部均值-delta → 白(255)；"比当地背景暗"的暗条 → 黑(0)=前景
    let adap = adaptive_threshold(&scene, 15, 10);
    println!("自适应阈值 前景 IoU={:.1}%", iou(|x, y| adap.get_pixel(x, y)[0] == 0));

    // ④ ThresholdType 选反 → 黑白颠倒（两者白像素数互补）
    let bin = threshold(&scene, level, ThresholdType::Binary);
    let bin_inv = threshold(&scene, level, ThresholdType::BinaryInverted);
    println!("Binary 白像素={}, BinaryInverted 白像素={}（互补）",
        bin.iter().filter(|&&p| p == 255).count(),
        bin_inv.iter().filter(|&&p| p == 255).count());

    scene.save("out_ch16_scene.png")?;
    bin_inv.save("out_ch16_global.png")?;
    adap.save("out_ch16_adaptive.png")?;
    Ok(())
}
