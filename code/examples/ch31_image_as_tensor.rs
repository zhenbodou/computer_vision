// 第 31 章：把图像当数据 —— 图像与张量互转、批处理
// 运行：cargo run --example ch31_image_as_tensor
// 依赖：image 0.25、ndarray 0.17
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::{Array3, Array4, Axis};

/// 合成一张彩色小图：红色随 x 渐变、绿色随 y 渐变、蓝色随对角线渐变。
/// 每个像素的 RGB 组合几乎都不同，往返或通道搞反都能立刻看出来。
fn make_image(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let r = (255 * x / (w - 1)) as u8; // 水平方向红色渐变
        let g = (255 * y / (h - 1)) as u8; // 垂直方向绿色渐变
        let b = (255 * (x + y) / (w + h - 2)) as u8; // 对角蓝色渐变
        Rgb([r, g, b])
    })
}

/// 图像 → CHW 张量：u8[0,255] 归一化到 f32[0,1]，并从 HWC 转成 CHW。
fn image_to_chw_tensor(img: &RgbImage) -> Array3<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    // 第一步：先按图像天然的 HWC 布局(高, 宽, 通道)塞进数组，顺手除以 255 归一化
    let hwc = Array3::from_shape_fn((h, w, 3), |(y, x, c)| {
        img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    });
    // 第二步：HWC → CHW。permuted_axes 只改 strides、不搬数据（零拷贝）
    // [2,0,1] 含义：原轴 2(C)→新轴 0，原轴 0(H)→新轴 1，原轴 1(W)→新轴 2
    let chw = hwc.permuted_axes([2, 0, 1]);
    // 第三步：此刻内存仍是 HWC 顺序，只是"看起来"像 CHW。
    // 交给模型或做 flatten 前，用 as_standard_layout 拷成真正的 CHW 连续内存
    chw.as_standard_layout().to_owned()
}

/// CHW 张量 → 图像：反归一化 f32[0,1]→u8[0,255]（round+clamp），并从 CHW 转回 HWC。
fn chw_tensor_to_image(t: &Array3<f32>) -> RgbImage {
    let (c, h, w) = (t.shape()[0], t.shape()[1], t.shape()[2]);
    assert_eq!(c, 3, "期望 3 通道 CHW 张量");
    // 反归一化：*255 后必须 round + clamp，否则 f32→u8 的直接截断会累积误差
    let to_u8 = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    let mut img = RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            // 注意索引顺序：CHW 张量用 [通道, 行(y), 列(x)]；图像 put_pixel 用 (列 x, 行 y)
            let px = Rgb([
                to_u8(t[[0, y, x]]), // 通道 0 = R
                to_u8(t[[1, y, x]]), // 通道 1 = G
                to_u8(t[[2, y, x]]), // 通道 2 = B
            ]);
            img.put_pixel(x as u32, y as u32, px);
        }
    }
    img
}

/// 按通道做 ImageNet 标准化：(x - mean) / std，三个通道的 mean/std 各不相同。
fn standardize_imagenet(t: &Array3<f32>) -> Array3<f32> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut out = t.clone();
    for c in 0..3 {
        // index_axis_mut(Axis(0), c) 取出第 c 个通道的二维可变视图，就地标准化
        out.index_axis_mut(Axis(0), c)
            .mapv_inplace(|v| (v - mean[c]) / std[c]);
    }
    out
}

fn main() {
    let img = make_image(64, 48);
    println!("原图尺寸: {}x{} (宽x高)", img.width(), img.height());

    // 1) 图像 → CHW 张量
    let chw = image_to_chw_tensor(&img);
    println!("CHW 张量形状: {:?}  (C, H, W)", chw.shape());
    println!("转换后是否为标准(连续)布局: {}", chw.is_standard_layout());

    // 抽查一个像素，确认 CHW 索引 [c, y, x] 的取值 == 原像素 / 255
    let (tx, ty) = (10usize, 5usize);
    let p = img.get_pixel(tx as u32, ty as u32).0;
    println!("像素(x={tx}, y={ty}) 原始 RGB = {p:?}");
    println!(
        "  CHW 取值   : R={:.4} G={:.4} B={:.4}",
        chw[[0, ty, tx]],
        chw[[1, ty, tx]],
        chw[[2, ty, tx]]
    );
    println!(
        "  期望(各/255): R={:.4} G={:.4} B={:.4}",
        p[0] as f32 / 255.0,
        p[1] as f32 / 255.0,
        p[2] as f32 / 255.0
    );

    // 2) 加 batch 维：CHW → NCHW（一张图也要凑成"1 张的批"）
    let nchw: Array4<f32> = chw.clone().insert_axis(Axis(0));
    println!("NCHW 张量形状: {:?}  (N, C, H, W)", nchw.shape());

    // 3) 往返验证：CHW → 图像，与原图逐像素比最大误差
    let back = chw_tensor_to_image(&chw);
    let max_err = img
        .as_raw()
        .iter()
        .zip(back.as_raw().iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    println!("往返(图像→张量→图像)最大像素误差 = {max_err}");

    // 4) 按通道 ImageNet 标准化，打印每个通道标准化后的均值与范围
    let norm = standardize_imagenet(&chw);
    for c in 0..3 {
        let ch = norm.index_axis(Axis(0), c);
        let lo = ch.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = ch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "标准化后通道{} 均值={:.3} 范围=[{:.3}, {:.3}]",
            c,
            ch.mean().unwrap(),
            lo,
            hi
        );
    }
}
