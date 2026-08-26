//! 第 14 章 非线性滤波：中值与双边 —— 正文代码验证
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::bilateral::GaussianEuclideanColorDistance;
use imageproc::filter::{bilateral_filter, box_filter, gaussian_blur_f32, median_filter};

/// 极简可复现伪随机数（xorshift32）
struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// 返回 [0,1) 的浮点数
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Box-Muller 生成标准差为 sigma 的高斯噪声
    fn gauss(&mut self, sigma: f32) -> f32 {
        let u1 = self.unit().max(1e-6);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * sigma
    }
}

/// 干净场景：灰底(128) + 亮圆盘(210) + 一根暗竖条(40)
fn make_shapes(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xi, yi) = (x as i32, y as i32);
        let (cx, cy, r) = (w as i32 / 2, h as i32 / 2, 45);
        if (xi - cx).pow(2) + (yi - cy).pow(2) < r * r {
            return Luma([210]);
        }
        if xi > 3 * w as i32 / 4 && xi < 3 * w as i32 / 4 + 12 {
            return Luma([40]);
        }
        Luma([128])
    })
}

/// 加椒盐噪声：每个像素以概率 prob 变成纯黑(0,胡椒)或纯白(255,盐)
fn add_salt_pepper(src: &GrayImage, rng: &mut Rng, prob: f32) -> GrayImage {
    let mut out = src.clone();
    for p in out.pixels_mut() {
        let r = rng.unit();
        if r < prob / 2.0 {
            p.0[0] = 0;
        } else if r < prob {
            p.0[0] = 255;
        }
    }
    out
}

/// 竖直阶跃边缘：左半暗(70) 右半亮(190)
fn make_step(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, _| {
        if x < w / 2 {
            Luma([70])
        } else {
            Luma([190])
        }
    })
}

/// 加高斯噪声
fn add_gaussian_noise(src: &GrayImage, rng: &mut Rng, sigma: f32) -> GrayImage {
    let mut out = src.clone();
    for p in out.pixels_mut() {
        let v = p.0[0] as f32 + rng.gauss(sigma);
        p.0[0] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// 平均绝对误差（相对某张参考图）
fn mae(a: &GrayImage, b: &GrayImage) -> f32 {
    let s: f64 = a
        .as_raw()
        .iter()
        .zip(b.as_raw().iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs() as f64)
        .sum();
    (s / a.as_raw().len() as f64) as f32
}

/// 统计取到极值(0 或 255)的像素个数——用来数"残留的椒盐点"
fn count_extremes(img: &GrayImage) -> u32 {
    img.as_raw().iter().filter(|&&v| v == 0 || v == 255).count() as u32
}

/// 某一列(x 固定)的平均灰度——用来量边缘
fn col_mean(img: &GrayImage, x: u32) -> f32 {
    let h = img.height();
    let s: u32 = (0..h).map(|y| img.get_pixel(x, y).0[0] as u32).sum();
    s as f32 / h as f32
}

/// 左侧平坦区(远离边缘)的灰度标准差——衡量噪声大小
fn flat_std(img: &GrayImage) -> f32 {
    let (w, h) = img.dimensions();
    let mut vals = vec![];
    for y in 0..h {
        for x in 0..(w / 4) {
            // 只取最左 1/4，确保在平坦暗区内
            vals.push(img.get_pixel(x, y).0[0] as f32);
        }
    }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    var.sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng(0x1234_5678);

    // ===== 第一部分：椒盐噪声下，中值 vs 均值 =====
    let clean = make_shapes(240, 200);
    let noisy = add_salt_pepper(&clean, &mut rng, 0.08); // 8% 椒盐

    let med = median_filter(&noisy, 1, 1); // 3×3 中值
    let avg = box_filter(&noisy, 1, 1); // 3×3 均值
    let gau = gaussian_blur_f32(&noisy, 1.0); // 高斯

    println!("== 椒盐噪声：谁能救回原图（相对干净图的 MAE，越小越好）==");
    println!("加噪后    MAE = {:.2}", mae(&noisy, &clean));
    println!("均值模糊  MAE = {:.2}", mae(&avg, &clean));
    println!("高斯模糊  MAE = {:.2}", mae(&gau, &clean));
    println!("中值滤波  MAE = {:.2}", mae(&med, &clean));
    println!(
        "残留椒盐点(值=0或255)：加噪 {} -> 中值 {} / 均值 {}",
        count_extremes(&noisy),
        count_extremes(&med),
        count_extremes(&avg)
    );

    // ===== 第二部分：双边滤波保边去噪 =====
    let step = make_step(240, 200);
    let step_noisy = add_gaussian_noise(&step, &mut rng, 15.0);

    let gb = gaussian_blur_f32(&step_noisy, 2.0);
    let bf = bilateral_filter(&step_noisy, 3, 2.0, GaussianEuclideanColorDistance::new(30.0));

    // 边缘落差：紧贴分界线左右两列的灰度差（越大 = 边缘越陡越清晰）
    let edge = 240 / 2;
    let local_step = |img: &GrayImage| col_mean(img, edge) - col_mean(img, edge - 1);
    println!("\n== 高斯噪声：双边保边、高斯抹边 ==");
    println!("干净阶跃  边缘落差 = {:.1}", local_step(&step));
    println!("加噪后    边缘落差 = {:.1}", local_step(&step_noisy));
    println!("高斯模糊  边缘落差 = {:.1}（边缘被抹平）", local_step(&gb));
    println!("双边滤波  边缘落差 = {:.1}（边缘保住了）", local_step(&bf));
    println!(
        "左侧平坦区噪声(标准差)：加噪 {:.1} -> 高斯 {:.1} / 双边 {:.1}",
        flat_std(&step_noisy),
        flat_std(&gb),
        flat_std(&bf)
    );

    clean.save("out_ch14_clean.png")?;
    noisy.save("out_ch14_saltpepper.png")?;
    med.save("out_ch14_median.png")?;
    step_noisy.save("out_ch14_step_noisy.png")?;
    bf.save("out_ch14_step_bilateral.png")?;
    Ok(())
}
