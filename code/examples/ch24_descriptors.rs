//! 第 24 章 特征描述子（BRIEF）—— 正文代码验证
#![allow(unused)]

use image::imageops;
use image::{GrayImage, Luma};
use imageproc::binary_descriptors::brief::{brief, TestPair};
use imageproc::binary_descriptors::BinaryDescriptor;
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::point::Point;
use imageproc::rect::Rect;

/// 合成一张灰度“照片”：白底 + 两个深色方块 + 一小片棋盘格，制造角点。
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([235]));
    draw_filled_rect_mut(&mut img, Rect::at(60, 55).of_size(70, 70), Luma([30]));
    draw_filled_rect_mut(&mut img, Rect::at(170, 70).of_size(55, 55), Luma([45]));
    let (ox, oy, cell, n) = (250i32, 150i32, 16i32, 4);
    for gy in 0..n {
        for gx in 0..n {
            if (gx + gy) % 2 == 0 {
                draw_filled_rect_mut(
                    &mut img,
                    Rect::at(ox + gx * cell, oy + gy * cell).of_size(cell as u32, cell as u32),
                    Luma([20]),
                );
            }
        }
    }
    img
}

/// 把整幅图平移 (dx, dy)，空出来的地方填 fill。内容原样搬家，用于验证平移不变性。
fn translate(src: &GrayImage, dx: i32, dy: i32, fill: u8) -> GrayImage {
    let (w, h) = (src.width(), src.height());
    let mut out = GrayImage::from_pixel(w, h, Luma([fill]));
    for (x, y, p) in src.enumerate_pixels() {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
            out.put_pixel(nx as u32, ny as u32, *p);
        }
    }
    out
}

/// 用一个确定性小 PRNG 生成固定的 256 组测试点对（都落在 31x31 patch 内）。
/// 真实项目里 brief(.., None) 会自动随机生成；这里固定下来是为了：
/// ① 输出可复现；② 跨图匹配本就必须复用同一组点对，否则比特无从对应。
fn fixed_test_pairs(length: usize) -> Vec<TestPair> {
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || -> u32 {
        // 经典 LCG 常数
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32) % 31 // 落在 [0, 31)
    };
    let mut pairs = Vec::with_capacity(length);
    for _ in 0..length {
        let (x0, y0, x1, y1) = (next(), next(), next(), next());
        pairs.push(TestPair {
            p0: Point::new(x0, y0),
            p1: Point::new(x1, y1),
        });
    }
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (360u32, 260u32);
    let gray = make_scene(w, h);

    // ① FAST 检测角点
    let corners = corners_fast9(&gray, 40);
    println!("FAST 检出角点数 = {}", corners.len());

    // ② 转成 BRIEF 需要的 Point<u32>；BRIEF patch 为 31x31，
    //    关键点必须离四边 >=16px，否则 brief() 会整体报错。这里先过滤。
    let margin = 16u32;
    let keypoints: Vec<Point<u32>> = corners
        .iter()
        .filter(|c| c.x >= margin && c.x + margin < w && c.y >= margin && c.y + margin < h)
        .map(|c| (*c).into()) // Corner -> Point<u32>
        .collect();
    println!("离边缘 >= {}px、可用于 BRIEF 的关键点数 = {}", margin, keypoints.len());

    // ③ 计算 BRIEF 描述子（256 位）。用固定测试点对，保证可复现、跨图可比。
    let test_pairs = fixed_test_pairs(256);
    let (descriptors, _) = brief(&gray, &keypoints, 256, Some(&test_pairs))?;
    let d0 = &descriptors[0];
    println!(
        "每个描述子 = {} 位 = {} 个 u128 打包",
        d0.get_size(),
        d0.bits.len()
    );

    // ④ Hamming 距离：同一个点对自己 = 0；两个不同角点通常差很多
    let da = &descriptors[0];
    let bi = descriptors.len() / 2;
    let db = &descriptors[bi];
    println!("Hamming(点0, 点0)         = {}", da.hamming_distance(da));
    println!(
        "Hamming(点0, 点{})         = {} / {} 位",
        bi,
        da.hamming_distance(db),
        da.get_size()
    );

    // ⑤ 挑一个关键点，验证“平移不变”与“旋转敏感”。用同一组 test_pairs 才可比。
    let kp = keypoints[0];
    println!("\n选定关键点 kp = ({}, {})", kp.x, kp.y);

    // 平移：整幅图搬 (12, 9)，同一个物理点搬到 (kp.x+12, kp.y+9)
    let (dx, dy) = (12i32, 9i32);
    let shifted = translate(&gray, dx, dy, 235);
    let kp_shift = Point::new((kp.x as i32 + dx) as u32, (kp.y as i32 + dy) as u32);
    let (desc_shift, _) = brief(&shifted, &[kp_shift], 256, Some(&test_pairs))?;
    let (desc_orig, _) = brief(&gray, &[kp], 256, Some(&test_pairs))?;
    println!(
        "平移后 Hamming(原图 kp, 平移图 kp) = {}   <- 平移几乎不变",
        desc_orig[0].hamming_distance(&desc_shift[0])
    );

    // 旋转 180°：同一个物理点映射到 (w-1-x, h-1-y)，邻域被转了 180°
    let rotated = imageops::rotate180(&gray);
    let kp_rot = Point::new(w - 1 - kp.x, h - 1 - kp.y);
    let (desc_rot, _) = brief(&rotated, &[kp_rot], 256, Some(&test_pairs))?;
    println!(
        "旋转后 Hamming(原图 kp, 旋转图 kp) = {}   <- 旋转后指纹大变",
        desc_orig[0].hamming_distance(&desc_rot[0])
    );

    Ok(())
}
