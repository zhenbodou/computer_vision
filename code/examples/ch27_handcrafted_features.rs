// 第 27 章：传统全局特征 HOG 与 LBP
// 运行：cargo run --example ch27_handcrafted_features
// 依赖：image、imageproc
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::hog::{cell_histograms, hog, render_hist_grid, HogOptions, HogSpec};
use imageproc::local_binary_patterns::{count_transitions, local_binary_pattern, min_shift};

/// 合成一个 64x128 的"行人剪影"灰度图：浅灰背景 + 深色人形（头 + 躯干 + 两条腿）
fn make_person(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as i32, y as i32);
        let cx = w as i32 / 2;
        let head_cy = h as i32 / 6;
        let head_r = w as i32 / 6;
        let in_head = (xf - cx).pow(2) + (yf - head_cy).pow(2) < head_r * head_r;
        let in_body =
            xf > cx - w as i32 / 5 && xf < cx + w as i32 / 5 && yf > h as i32 / 4 && yf < 3 * h as i32 / 4;
        let in_legs = yf >= 3 * h as i32 / 4
            && ((xf > cx - w as i32 / 5 && xf < cx - w as i32 / 20)
                || (xf > cx + w as i32 / 20 && xf < cx + w as i32 / 5));
        if in_head || in_body || in_legs {
            Luma([40u8]) // 深色前景
        } else {
            Luma([200u8]) // 浅灰背景
        }
    })
}

/// 竖条纹纹理
fn make_stripes(w: u32, h: u32, period: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, _y| {
        if (x / period) % 2 == 0 { Luma([220]) } else { Luma([40]) }
    })
}

/// 方格棋盘纹理
fn make_checker(w: u32, h: u32, period: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        if ((x / period) + (y / period)) % 2 == 0 { Luma([220]) } else { Luma([40]) }
    })
}

/// 整幅图的 256 维 LBP 直方图（跳过一圈边界像素，那里 LBP 无定义）
fn lbp_histogram(img: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let (w, h) = (img.width(), img.height());
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                hist[code as usize] += 1;
            }
        }
    }
    hist
}

/// 旋转不变版：用 min_shift 把每个码归一化到"最小循环移位"
fn lbp_histogram_ri(img: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    let (w, h) = (img.width(), img.height());
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                hist[min_shift(code) as usize] += 1;
            }
        }
    }
    hist
}

/// 两直方图先各自归一化为频率，再算 L1 距离（越小越像）
fn l1_distance(a: &[u32; 256], b: &[u32; 256]) -> f32 {
    let sa = a.iter().sum::<u32>().max(1) as f32;
    let sb = b.iter().sum::<u32>().max(1) as f32;
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f32 / sa - y as f32 / sb).abs())
        .sum()
}

/// uniform 模式（0/1 跳变 <= 2 次）的像素占比
fn uniform_ratio(img: &GrayImage) -> f32 {
    let (w, h) = (img.width(), img.height());
    let (mut total, mut uni) = (0u32, 0u32);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if let Some(code) = local_binary_pattern(img, x, y) {
                total += 1;
                if count_transitions(code) <= 2 {
                    uni += 1;
                }
            }
        }
    }
    uni as f32 / total.max(1) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 一、HOG：描述形状/轮廓 ----------
    let person = make_person(64, 128);
    // Dalal-Triggs 行人检测经典设置：9 个无符号方向、8x8 的 cell、2x2 cell 的 block、步长 1 cell
    let opts = HogOptions::new(9, false, 8, 2, 1);
    let feat = hog(&person, opts)?;
    println!("[HOG] 图像 64x128, cell=8px, block=2cell, stride=1cell, orientations=9(unsigned)");
    println!("[HOG] 特征向量维度 = {}", feat.len());

    // 手工核对维度：cells 8x16 → blocks 7x15 → 每 block 9*2*2=36 → 7*15*36
    let spec = HogSpec::from_options(64, 128, opts).unwrap();
    println!("[HOG] 公式核对维度 = 7*15*36 = {}", spec.descriptor_length());

    // 可视化每个 cell 的方向直方图（"星形"图），存 PNG
    let mut grid = cell_histograms(&person, spec);
    let vis = render_hist_grid(16, &grid.view_mut(), opts.signed);
    vis.save("out_ch27_hog_vis.png")?;
    person.save("out_ch27_person.png")?;

    // 故意给一组不合法的参数，看它如何报错（cell 边长必须整除宽高）
    match hog(&person, HogOptions::new(9, false, 7, 2, 1)) {
        Ok(_) => println!("[HOG] （意外）竟然通过了"),
        Err(e) => println!("[HOG] 非法参数报错示例：{e}"),
    }

    // ---------- 二、LBP：描述纹理 ----------
    let stripes = make_stripes(96, 96, 6);
    let checker = make_checker(96, 96, 8);

    let h_stripes = lbp_histogram(&stripes);
    let h_checker = lbp_histogram(&checker);
    println!(
        "\n[LBP] 竖条纹 vs 方格：直方图 L1 距离 = {:.3}（越大越不同 → 纹理可区分）",
        l1_distance(&h_stripes, &h_checker)
    );
    println!(
        "[LBP] uniform 模式占比：竖条纹 {:.1}% , 方格 {:.1}%",
        uniform_ratio(&stripes) * 100.0,
        uniform_ratio(&checker) * 100.0
    );

    // 旋转不变：竖条纹旋转 90° 变横条纹
    let rot = image::imageops::rotate90(&stripes);
    let d_plain = l1_distance(&lbp_histogram(&stripes), &lbp_histogram(&rot));
    let d_ri = l1_distance(&lbp_histogram_ri(&stripes), &lbp_histogram_ri(&rot));
    println!(
        "[LBP] 竖条纹 vs 其旋转90°：普通 LBP 距离 = {:.3} , min_shift 旋转不变距离 = {:.3}",
        d_plain, d_ri
    );

    stripes.save("out_ch27_stripes.png")?;
    Ok(())
}
