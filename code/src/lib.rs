//! `cv_examples` 库 crate —— 可复用的视觉逻辑放这里（第 112 章：lib + bin 分离）。
//!
//! 设计原则：
//! - **纯函数优先**：核心算法不做 I/O、不 `println!`、不 `panic!`，只吃数据、吐结果或 `Result`。
//! - **库层不 `unwrap`**：所有可能失败的路径都返回 `Result<_, VisionError>`，把「怎么处理错误」的
//!   决定权交给调用方（应用层再用 anyhow 兜底，见 `examples/ch112_engineering.rs`）。

use image::GrayImage;
use thiserror::Error;

/// 库层自定义错误：用 `thiserror` 把「可能出什么错」显式建模成一个枚举。
/// 每个变体带上下文，`#[from]` 让 `?` 能自动把底层错误（如 `image::ImageError`）转过来。
#[derive(Error, Debug)]
pub enum VisionError {
    #[error("读写图像失败: {0}")]
    Io(#[from] image::ImageError),

    #[error("图像尺寸非法: {w}x{h}（宽高都必须 > 0）")]
    BadSize { w: u32, h: u32 },
}

/// 统计灰度图中「亮像素」（值 > thresh）的个数。
///
/// 这是一个**纯函数**：给定同样的输入永远得到同样的输出，因此极易测试
/// （见 `tests/ch112_integration.rs` 与本 crate 的单元测试）。
pub fn count_bright_pixels(img: &GrayImage, thresh: u8) -> u64 {
    img.pixels().filter(|p| p.0[0] > thresh).count() as u64
}

/// 用 Otsu 自动阈值把灰度图二值化，并返回（二值图, 前景像素数）。
///
/// 会失败的地方都用 `Result` 显式表达，绝不 `unwrap`：
/// - 空图（宽或高为 0）没有意义 → `BadSize`。
pub fn otsu_binarize(img: &GrayImage) -> Result<(GrayImage, u64), VisionError> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(VisionError::BadSize { w, h });
    }
    use imageproc::contrast::{otsu_level, threshold, ThresholdType};
    let level = otsu_level(img);
    let bin = threshold(img, level, ThresholdType::Binary);
    let fg = count_bright_pixels(&bin, 0); // 二值图里 > 0 即前景（255）
    Ok((bin, fg))
}

/// 从文件读图 → 转灰度 → Otsu 二值化 → 返回前景像素数。
///
/// 这条链路串起了多种错误来源：文件读失败（`Io`）、尺寸非法（`BadSize`）。
/// 注意函数体里全是 `?`——一旦某步出错就带着上下文向上返回，没有一个 `unwrap`。
pub fn foreground_ratio_from_path(path: &str) -> Result<f64, VisionError> {
    let img = image::open(path)?.to_luma8(); // ImageError 经 #[from] 自动转成 VisionError::Io
    let (w, h) = img.dimensions();
    let (_bin, fg) = otsu_binarize(&img)?;
    Ok(fg as f64 / (w as u64 * h as u64) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    /// 合成一张「一半黑、一半白」的灰度图：可复现，不依赖任何外部图片。
    fn half_black_half_white(w: u32, h: u32) -> GrayImage {
        ImageBuffer::from_fn(w, h, |x, _| {
            if x < w / 2 { Luma([0u8]) } else { Luma([255u8]) }
        })
    }

    #[test]
    fn bright_count_is_exact() {
        // 100x10 图，右半（x>=50）50 列全白 → 白像素 = 50*10 = 500。可精确断言。
        let img = half_black_half_white(100, 10);
        assert_eq!(count_bright_pixels(&img, 128), 500);
    }

    #[test]
    fn otsu_splits_bimodal_image() {
        let img = half_black_half_white(100, 10);
        let (_bin, fg) = otsu_binarize(&img).expect("非空图不该失败");
        // 双峰图（纯黑+纯白）Otsu 阈值会落在中间，前景应恰为白色那一半。
        assert_eq!(fg, 500);
    }

    #[test]
    fn empty_image_is_rejected() {
        let empty: GrayImage = ImageBuffer::new(0, 0);
        // 不该 panic，而应返回 Err —— 这正是「库层不 unwrap」的价值。
        assert!(matches!(otsu_binarize(&empty), Err(VisionError::BadSize { .. })));
    }
}
