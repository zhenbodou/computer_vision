//! 核心 API 验证：用 `cargo check --example verify_core` 类型检查。
//! 目的是确认本书正文中引用的 image / imageproc / ndarray / nalgebra API 真实存在且签名正确。
#![allow(unused)]

use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::contrast::{otsu_level, threshold, ThresholdType};
use imageproc::corners::corners_fast9;
use imageproc::distance_transform::Norm;
use imageproc::drawing::{draw_filled_rect_mut, draw_line_segment_mut};
use imageproc::edges::canny;
use imageproc::filter::{box_filter, gaussian_blur_f32, median_filter};
use imageproc::geometric_transformations::{
    rotate_about_center, warp, Border, Interpolation, Projection,
};
use imageproc::gradients::sobel_gradients;
use imageproc::hough::{detect_lines, draw_polar_lines_mut, LineDetectionOptions, PolarLine};
use imageproc::morphology::{close, dilate, erode, open};
use imageproc::rect::Rect;
use imageproc::region_labelling::{connected_components, Connectivity};
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};

fn image_basics() -> Result<(), Box<dyn std::error::Error>> {
    // 打开 -> 尺寸 -> 转灰度/RGB -> 缩放 -> 裁剪 -> 旋转 -> 存盘
    let img: DynamicImage = image::open("in.png")?;
    let (w, h) = img.dimensions();
    let gray: GrayImage = img.to_luma8();
    let rgb: RgbImage = img.to_rgb8();
    let small = img.resize(w / 2, h / 2, image::imageops::FilterType::Lanczos3);
    let cropped = img.crop_imm(0, 0, 100, 100);
    let rotated = img.rotate90();
    let flipped = img.fliph();
    small.save("out.png")?;

    // 逐像素读写
    let mut canvas: RgbImage = ImageBuffer::from_fn(64, 64, |_x, _y| Rgb([0u8, 0, 0]));
    let p = *rgb.get_pixel(0, 0);
    canvas.put_pixel(0, 0, p);
    for (x, y, px) in rgb.enumerate_pixels() {
        let _ = (x, y, px);
    }
    Ok(())
}

fn imageproc_pipeline() {
    let gray: GrayImage = ImageBuffer::from_pixel(128, 128, Luma([120u8]));

    // 滤波
    let blurred = gaussian_blur_f32(&gray, 2.0);
    let boxed = box_filter(&gray, 2, 2);
    let med = median_filter(&gray, 2, 2);

    // 边缘 & 梯度
    let edges = canny(&gray, 40.0, 100.0);
    let g: ImageBuffer<Luma<u16>, Vec<u16>> = sobel_gradients(&gray);

    // 阈值
    let t = otsu_level(&gray);
    let bin = threshold(&gray, t, ThresholdType::Binary);

    // 形态学
    let d = dilate(&bin, Norm::LInf, 1);
    let e = erode(&bin, Norm::LInf, 1);
    let o = open(&bin, Norm::L1, 1);
    let c = close(&bin, Norm::L1, 1);

    // 连通域
    let labels = connected_components(&bin, Connectivity::Eight, Luma([0u8]));

    // 霍夫直线
    let opts = LineDetectionOptions { vote_threshold: 40, suppression_radius: 8 };
    let lines: Vec<PolarLine> = detect_lines(&edges, opts);
    let mut rgb = image::DynamicImage::ImageLuma8(gray.clone()).to_rgb8();
    draw_polar_lines_mut(&mut rgb, &lines, Rgb([255, 0, 0]));

    // 角点
    let corners = corners_fast9(&gray, 20);

    // 模板匹配
    let tmpl: GrayImage = ImageBuffer::from_pixel(16, 16, Luma([200u8]));
    let score = match_template(&gray, &tmpl, MatchTemplateMethod::CrossCorrelationNormalized);
    let ext = find_extremes(&score);

    // 绘制
    draw_filled_rect_mut(&mut rgb, Rect::at(5, 5).of_size(20, 20), Rgb([0, 255, 0]));
    draw_line_segment_mut(&mut rgb, (0.0, 0.0), (50.0, 50.0), Rgb([0, 0, 255]));

    // 几何变换
    let rot = rotate_about_center(&rgb, 0.3, Interpolation::Bilinear, Border::Constant(Rgb([0, 0, 0])));
    let proj = Projection::translate(10.0, 20.0).and_then(Projection::scale(1.2, 1.2));
    let warped = warp(&rgb, proj, Interpolation::Bilinear, Border::Constant(Rgb([0, 0, 0])));
}

fn ndarray_nalgebra() {
    use nalgebra::{Matrix3, Vector3};
    use ndarray::{array, Array2};

    let a: Array2<f32> = Array2::zeros((3, 4));
    let b = array![[1.0f32, 2.0], [3.0, 4.0]];
    let s = b.sum();

    let m = Matrix3::<f32>::identity();
    let v = Vector3::new(1.0f32, 2.0, 3.0);
    let mv = m * v;
}

fn main() {
    println!("verify only; run `cargo check --example verify_core`");
}
