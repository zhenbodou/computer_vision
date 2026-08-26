//! 第 112 章 集成测试（tests/ 目录）——把 `cv_examples` 库当外部使用者来测。
//! 运行：cargo test --test ch112_integration
//!
//! 视觉算法怎么测？关键是用**合成图**做**可复现的精确/容差断言**，绝不依赖外部图片或网络。

use cv_examples::{count_bright_pixels, otsu_binarize, VisionError};
use image::{GrayImage, ImageBuffer, Luma};

/// 合成一张已知白像素数的图：画一个 w_box × h_box 的白色矩形在纯黑背景上。
fn image_with_white_rect(w: u32, h: u32, bw: u32, bh: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        if x < bw && y < bh { Luma([255u8]) } else { Luma([0u8]) }
    })
}

#[test]
fn white_pixel_count_matches_rect_area() {
    // 精确断言：白矩形 40×30 → 白像素恰好 1200 个。合成输入让「预期值」可手算。
    let img = image_with_white_rect(128, 128, 40, 30);
    assert_eq!(count_bright_pixels(&img, 128), 40 * 30);
}

#[test]
fn otsu_foreground_within_tolerance() {
    // 数值容差比较：矩形前景约 1200 像素，允许 Otsu 边界有少量偏差。
    let img = image_with_white_rect(128, 128, 40, 30);
    let (_bin, fg) = otsu_binarize(&img).expect("非空图不应失败");
    let expected = 1200i64;
    assert!((fg as i64 - expected).abs() <= 5, "前景 {fg} 偏离预期 {expected} 过多");
}

#[test]
fn zero_sized_image_returns_error_not_panic() {
    let empty: GrayImage = ImageBuffer::new(0, 0);
    // 契约：库层遇到坏输入返回 Err，而不是 panic。这样调用方可控地处理。
    assert!(matches!(otsu_binarize(&empty), Err(VisionError::BadSize { .. })));
}
