// 第 98 章 项目：二维码识别 —— 纯 Rust 生成→解码闭环 + 鲁棒性压力测试
//
// 依赖（均为纯 Rust，cargo tree 查无 *-sys / cc / bindgen / cxx）：
//   qrcode 0.14.1 —— QR 码「生成」
//   rqrr   0.10.1 —— QR 码「定位 + 解码」（ZXing 之外的纯 Rust 实现，零 C/C++ 依赖）
//   image  0.25.10 —— 图像读写 / 几何变换配套
//   imageproc 0.27.0 —— 旋转、加椒盐噪声、中值滤波（合成扰动测试图 + 预处理）
//
// 运行： cd code/projects/ch98_qrcode && cargo run --release

use image::{GrayImage, ImageBuffer, Luma};
use imageproc::filter::median_filter;
use imageproc::geometric_transformations::{rotate_about_center, Border, Interpolation};
use imageproc::noise::salt_and_pepper_noise;
use qrcode::{EcLevel, QrCode};
use rqrr::PreparedImage;

/// 用 qrcode 把文本编码成 QR 图（灰度、含 quiet zone 静默区）。
fn make_qr(text: &str, ec: EcLevel, min_side: u32) -> GrayImage {
    let code = QrCode::with_error_correction_level(text.as_bytes(), ec).expect("QR 编码失败");
    code.render::<Luma<u8>>()
        .min_dimensions(min_side, min_side)
        .quiet_zone(true) // 保留 QR 四周白边（静默区），解码定位需要它
        .build()
}

/// 用 rqrr 解码灰度图里的 QR。成功返回 (读回文本, 版本模块数, 纠错等级原始位)。
fn decode(img: &GrayImage) -> Option<(String, usize, u16)> {
    let mut prep = PreparedImage::prepare(img.clone());
    for grid in prep.detect_grids() {
        if let Ok((meta, content)) = grid.decode() {
            return Some((content, meta.version.to_size(), meta.ecc_level));
        }
    }
    None
}

/// rqrr 的 ecc_level 是 QR「格式信息」里的原始 2 bit（不是 L/M/Q/H 的序号）：
/// QR 标准里 M=00、L=01、H=10、Q=11，故原始位 0→M、1→L、2→H、3→Q。
fn ecc_name(raw: u16) -> &'static str {
    match raw {
        0 => "M(15%)",
        1 => "L(7%)",
        2 => "H(30%)",
        3 => "Q(25%)",
        _ => "?",
    }
}

/// 解码成功且读回文本与原文逐字一致 → true。
fn decodes_to(img: &GrayImage, text: &str) -> bool {
    decode(img).map(|(b, ..)| b == text).unwrap_or(false)
}

/// 把 QR 贴到更大的白色画布正中央，避免旋转时四角（含定位图案）被裁掉。
fn embed_center(qr: &GrayImage, side: u32) -> GrayImage {
    let mut canvas: GrayImage = ImageBuffer::from_pixel(side, side, Luma([255]));
    let (ox, oy) = ((side - qr.width()) / 2, (side - qr.height()) / 2);
    for (x, y, p) in qr.enumerate_pixels() {
        canvas.put_pixel(ox + x, oy + y, *p);
    }
    canvas
}

fn main() {
    // 指定要编码的文本：英文 URL + 中文（验证 UTF-8 字节模式往返无损）
    let text = "https://rust-cv.example/ch98 | 用 Rust 玩转机器视觉";

    println!("========== 第 98 章 QR 码 生成→解码 闭环 ==========");
    println!("原始文本：{text}");
    println!("UTF-8 字节数：{}", text.len());

    // ---------- 1. 编解码闭环 ----------
    let qr = make_qr(text, EcLevel::M, 360); // 默认纠错等级 M（15%）
    qr.save("qr_clean.png").expect("保存失败");
    println!("\n[1] 编解码闭环（纯 Rust：qrcode 生成 → rqrr 解码）");
    println!("  生成 QR：{}×{} 像素，已存 qr_clean.png", qr.width(), qr.height());
    match decode(&qr) {
        Some((back, size, ecc)) => {
            println!("  解码成功：{size}×{size} 模块，纠错等级 {}", ecc_name(ecc));
            println!("  读回文本：{back}");
            println!("  与原文逐字一致：{}", if back == text { "是 ✓" } else { "否 ✗" });
        }
        None => println!("  解码失败（不应发生）"),
    }

    // ---------- 2. 旋转鲁棒性 ----------
    println!("\n[2] 旋转鲁棒性（QR 靠三个「回」字定位图案自纠正朝向）");
    for (name, rotated) in [
        ("90°", image::imageops::rotate90(&qr)),
        ("180°", image::imageops::rotate180(&qr)),
        ("270°", image::imageops::rotate270(&qr)),
    ] {
        println!("  直角 {name:>4}：{}", if decodes_to(&rotated, text) { "成功 ✓" } else { "失败 ✗" });
    }
    let padded = embed_center(&qr, (qr.width() as f32 * 1.5) as u32);
    for deg in [5.0f32, 15.0, 30.0, 45.0] {
        let rot = rotate_about_center(
            &padded,
            deg.to_radians(),
            Interpolation::Bilinear,
            Border::Constant(Luma([255])),
        );
        println!("  任意 {deg:>4}°：{}", if decodes_to(&rot, text) { "成功 ✓" } else { "失败 ✗" });
    }

    // ---------- 3. 缩放鲁棒性 ----------
    println!("\n[3] 缩放鲁棒性（360px 原图缩到更小，看多小还能读出）");
    let mut smallest_ok = 0u32;
    for target in [300u32, 240, 180, 150, 120, 90, 75, 60, 45] {
        let small =
            image::imageops::resize(&qr, target, target, image::imageops::FilterType::Triangle);
        let ok = decodes_to(&small, text);
        if ok {
            smallest_ok = target;
        }
        println!("  缩到 {target:>3}px：{}", if ok { "成功 ✓" } else { "失败 ✗" });
    }
    println!("  → 仍能解码的最小尺寸 ≈ {smallest_ok}px（约每模块 {:.1}px）", smallest_ok as f32 / 33.0);

    // ---------- 4. 椒盐噪声 × 纠错等级 × 中值滤波（每档 30 个随机种子求成功率） ----------
    println!("\n[4] 椒盐噪声鲁棒性：纠错等级(L/M/H) 与 中值滤波预处理的对照");
    println!("    （每档 30 个不同随机种子，统计「解码成功且读回一致」的比例）");
    let qr_l = make_qr(text, EcLevel::L, 360);
    let qr_h = make_qr(text, EcLevel::H, 360);
    println!(
        "    三档模块数：L={}×{}  M={}×{}  H={}×{}（纠错越强、冗余越多、码越大）",
        decode(&qr_l).unwrap().1, decode(&qr_l).unwrap().1,
        decode(&qr).unwrap().1, decode(&qr).unwrap().1,
        decode(&qr_h).unwrap().1, decode(&qr_h).unwrap().1,
    );
    let trials = 30u64;
    let rate_of = |img: &GrayImage, rate: f64, med: bool| -> f64 {
        let mut ok = 0u64;
        for seed in 0..trials {
            let noisy = salt_and_pepper_noise(img, rate, seed);
            let cand = if med { median_filter(&noisy, 1, 1) } else { noisy };
            if decodes_to(&cand, text) {
                ok += 1;
            }
        }
        ok as f64 / trials as f64 * 100.0
    };
    println!("    噪声率   L(7%)裸   M(15%)裸  H(30%)裸  M+中值3×3");
    for rate in [0.01f64, 0.02, 0.03, 0.05, 0.08] {
        println!(
            "    {:>5.0}%   {:>6.1}%   {:>6.1}%   {:>6.1}%   {:>6.1}%",
            rate * 100.0,
            rate_of(&qr_l, rate, false),
            rate_of(&qr, rate, false),
            rate_of(&qr_h, rate, false),
            rate_of(&qr, rate, true),
        );
    }
    // 存一对样图：3% 椒盐噪声原样 vs 中值滤波后
    let demo = salt_and_pepper_noise(&qr, 0.03, 0);
    demo.save("qr_noise3.png").ok();
    median_filter(&demo, 1, 1).save("qr_noise3_median.png").ok();

    println!("\n全部测试完成。产物：qr_clean.png / qr_noise3.png / qr_noise3_median.png");
}
