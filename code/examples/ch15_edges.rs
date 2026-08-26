//! 第 15 章 梯度与边缘检测 —— 正文代码验证（与书中代码块一致）
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use imageproc::gradients::{horizontal_sobel, sobel_gradients, vertical_sobel};

/// 极简可复现伪随机数（线性同余），用来加噪声，避免引入 rand 依赖
struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// 合成灰度场景：中灰背景(100) + 亮矩形(200) + 暗圆(30)
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        if (40..110).contains(&xi) && (40..130).contains(&yi) {
            return Luma([200]); // 亮矩形
        }
        let (cx, cy, rad) = (175, 90, 35);
        if (xi - cx).pow(2) + (yi - cy).pow(2) < rad * rad {
            return Luma([30]); // 暗圆
        }
        Luma([100]) // 背景
    })
}

/// 给灰度图加均匀噪声（范围约 ±amp），结果 clamp 到 [0,255]
fn add_noise(img: &GrayImage, amp: i32, seed: u64) -> GrayImage {
    let mut rng = Lcg(seed);
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let base = img.get_pixel(x, y)[0] as i32;
        let noise = (rng.next_u8() as i32 % (2 * amp + 1)) - amp;
        Luma([(base + noise).clamp(0, 255) as u8])
    })
}

/// 把 u16 梯度幅值图归一化到 u8 便于可视化（关键：类型转换！）
fn u16_to_u8_normalized(g: &ImageBuffer<Luma<u16>, Vec<u16>>) -> GrayImage {
    let max = g.iter().copied().max().unwrap_or(1).max(1) as f32;
    ImageBuffer::from_fn(g.width(), g.height(), |x, y| {
        let v = g.get_pixel(x, y)[0] as f32;
        Luma([(v / max * 255.0) as u8])
    })
}

/// 统计二值边缘图中值为 255 的像素数
fn count_edges(img: &GrayImage) -> u32 {
    img.iter().filter(|&&p| p == 255).count() as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(240, 180);

    // ① Sobel 梯度幅值：imageproc 内部已算好 sqrt(Gx^2+Gy^2)，返回 Luma<u16>
    let grad = sobel_gradients(&scene);
    println!("Sobel 梯度幅值最大值 = {}", grad.iter().copied().max().unwrap_or(0));

    // ② 分别取水平/垂直 Sobel（Luma<i16>，带符号！），手动合成幅值与方向
    let gx = horizontal_sobel(&scene);
    let gy = vertical_sobel(&scene);
    let (px, py) = (40u32, 85u32); // 矩形左边缘上一点：一条竖直边缘
    let dx = gx.get_pixel(px, py)[0] as f32;
    let dy = gy.get_pixel(px, py)[0] as f32;
    println!(
        "竖直边缘点(40,85): Gx={}, Gy={}, 幅值={:.1}, 方向={:.1}°",
        dx as i32, dy as i32, (dx * dx + dy * dy).sqrt(), dy.atan2(dx).to_degrees()
    );

    // ③ Canny：同一张含噪图，阈值不同结果天差地别
    let noisy = add_noise(&scene, 25, 12345);
    let e_lo = canny(&noisy, 10.0, 30.0); // 阈值太低：噪声也被当成边缘
    let e_ok = canny(&noisy, 40.0, 80.0); // 阈值合适：干净地勾出形状
    let e_hi = canny(&noisy, 100.0, 200.0); // 阈值太高：弱边缘丢失
    println!(
        "含噪图 Canny：低(10,30)={}, 合适(40,80)={}, 高(100,200)={}",
        count_edges(&e_lo), count_edges(&e_ok), count_edges(&e_hi)
    );

    // ④ "先降噪"的重要性：低阈值下，先高斯降噪能压掉海量噪声边缘
    let denoised = gaussian_blur_f32(&noisy, 2.0);
    println!(
        "低阈值(10,30)下：含噪直接={}, 先降噪再 Canny={}",
        count_edges(&e_lo), count_edges(&canny(&denoised, 10.0, 30.0))
    );

    u16_to_u8_normalized(&grad).save("out_ch15_sobel_mag.png")?;
    e_ok.save("out_ch15_canny_ok.png")?;
    e_lo.save("out_ch15_canny_lowthresh.png")?;
    Ok(())
}
