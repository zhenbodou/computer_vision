//! 第 22 章 频域处理：傅里叶变换与频域滤波 —— 正文代码验证
//! 依赖 rustfft 6.4.1（纯 Rust FFT），num-complex 0.4.6（由 rustfft 再导出）
//! 运行：cargo run --example ch22_frequency
#![allow(unused)]

use image::{GrayImage, ImageBuffer, Luma};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// 合成灰度图：平滑亮斑（低频）+ 一组竖向正弦条纹（一个明确的高频）
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        // 平滑亮斑：中心亮、四周暗（低频成分）
        let blob = (-((fx - 0.5).powi(2) + (fy - 0.5).powi(2)) * 8.0).exp() * 120.0;
        // 竖向正弦条纹：整幅图内出现 12 个周期（一个确定的高频）
        let stripes = (2.0 * std::f32::consts::PI * 12.0 * fx).cos() * 60.0;
        let v = 128.0 + blob * 0.5 + stripes;
        Luma([v.round().clamp(0.0, 255.0) as u8])
    })
}

/// 就地 2D FFT：先对每一行做 1D FFT，再对每一列做 1D FFT。
/// inverse=false 为正变换，true 为逆变换（rustfft 的逆变换不含 1/N 归一化）。
fn fft_2d(data: &mut [Complex<f32>], w: usize, h: usize, planner: &mut FftPlanner<f32>, inverse: bool) {
    // ① 对每一行（长度 w）做 1D FFT
    let fft_row = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for row in data.chunks_mut(w) {
        fft_row.process(row);
    }
    // ② 对每一列（长度 h）做 1D FFT：先把一列抠出来，变换后再写回
    let fft_col = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex::new(0.0f32, 0.0); h];
    for x in 0..w {
        for y in 0..h {
            col[y] = data[y * w + x];
        }
        fft_col.process(&mut col);
        for y in 0..h {
            data[y * w + x] = col[y];
        }
    }
}

/// 折叠频率下标：返回该下标离直流（DC）分量的「周期性距离」。
/// DC 在角落 (0,0)，最高频在正中间 n/2；u 与 n-u 是同一个频率的正负两支。
fn folded(u: usize, n: usize) -> f32 {
    u.min(n - u) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (128usize, 128usize); // FFT 建议用 2 的幂，最快
    let scene = make_scene(w as u32, h as u32);
    scene.save("out_ch22_input.png")?;

    let mut planner = FftPlanner::<f32>::new();

    // 灰度像素 -> 复数矩阵（实部为像素值，虚部 0）
    let mut freq: Vec<Complex<f32>> = scene
        .as_raw()
        .iter()
        .map(|&p| Complex::new(p as f32, 0.0))
        .collect();

    // ============ 1. 2D 正变换 ============
    fft_2d(&mut freq, w, h, &mut planner, false);

    // ============ 2. 逆变换往返验证（应几乎无损）============
    let mut roundtrip = freq.clone();
    fft_2d(&mut roundtrip, w, h, &mut planner, true);
    let n = (w * h) as f32;
    let mut max_err = 0f32;
    for (c, &orig) in roundtrip.iter().zip(scene.as_raw().iter()) {
        let val = c.re / n; // rustfft 逆变换要手动除以 N=w*h
        max_err = max_err.max((val - orig as f32).abs());
    }
    println!("== 往返验证：FFT -> IFFT ->/N ==");
    println!("  与原图最大误差 = {max_err:.6}（浮点级，可视为无损）");

    // ============ 3. 幅度谱（log 尺度 + fftshift 居中）============
    // 幅度 = 复数的模；能量集中在少数几个频率上，需取 log 才看得清
    let mag: Vec<f32> = freq.iter().map(|c| (1.0 + c.norm()).ln()).collect();
    let maxmag = mag.iter().cloned().fold(0.0f32, f32::max);
    // fftshift：把 DC 从角落搬到画面正中央，四象限对调
    let mut spec: GrayImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let sx = (x + w / 2) % w;
            let sy = (y + h / 2) % h;
            let v = (mag[y * w + x] / maxmag * 255.0) as u8;
            spec.put_pixel(sx as u32, sy as u32, Luma([v]));
        }
    }
    spec.save("out_ch22_spectrum.png")?;

    // 找出「非 DC」的最强频率——它就是那组正弦条纹留下的孤立亮点
    let (mut peak, mut peak_uv) = (0f32, (0usize, 0usize));
    for y in 0..h {
        for x in 0..w {
            if x == 0 && y == 0 {
                continue; // 跳过 DC（直流/平均亮度），它总是最大
            }
            let m = freq[y * w + x].norm();
            if m > peak {
                peak = m;
                peak_uv = (x, y);
            }
        }
    }
    println!("== 幅度谱 ==");
    println!(
        "  最强的非 DC 峰在 (u={}, v={})，横向折叠频率 = {} 个周期",
        peak_uv.0,
        peak_uv.1,
        folded(peak_uv.0, w)
    );

    // ============ 4. 频域低通滤波：只保留中心低频，其余置零 ============
    let radius = 8.0; // 截止半径（折叠频率）
    let mut low = freq.clone();
    let mut kept = 0usize;
    for y in 0..h {
        for x in 0..w {
            let (du, dv) = (folded(x, w), folded(y, h));
            if (du * du + dv * dv).sqrt() > radius {
                low[y * w + x] = Complex::new(0.0, 0.0); // 去掉高频
            } else {
                kept += 1;
            }
        }
    }
    fft_2d(&mut low, w, h, &mut planner, true);
    let lp: Vec<u8> = low
        .iter()
        .map(|c| (c.re / n).round().clamp(0.0, 255.0) as u8) // 取实部并归一化
        .collect();
    let lp_img = ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, lp).unwrap();
    lp_img.save("out_ch22_lowpass.png")?;

    // ============ 5. 频域高通滤波：只保留高频（去掉中心低频）============
    let mut high = freq.clone();
    for y in 0..h {
        for x in 0..w {
            let (du, dv) = (folded(x, w), folded(y, h));
            if (du * du + dv * dv).sqrt() <= radius {
                high[y * w + x] = Complex::new(0.0, 0.0); // 去掉低频，只留边缘/细节
            }
        }
    }
    fft_2d(&mut high, w, h, &mut planner, true);
    let hp: Vec<u8> = high
        .iter()
        .map(|c| (c.re / n + 128.0).round().clamp(0.0, 255.0) as u8) // 高通结果有正负，偏移 128 可视化
        .collect();
    let hp_img = ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, hp).unwrap();
    hp_img.save("out_ch22_highpass.png")?;

    // 用「一行像素的标准差」量化条纹是否被低通抹掉
    let std_row = |img: &GrayImage, row: u32| -> f32 {
        let vals: Vec<f32> = (0..img.width()).map(|x| img.get_pixel(x, row).0[0] as f32).collect();
        let m = vals.iter().sum::<f32>() / vals.len() as f32;
        (vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32).sqrt()
    };
    println!("== 低通滤波（截止半径 {radius}，保留 {kept} 个频率）==");
    println!("  中间一行像素标准差：原图 = {:.1} -> 低通后 = {:.1}（条纹被抹平）", std_row(&scene, h as u32 / 2), std_row(&lp_img, h as u32 / 2));

    Ok(())
}
