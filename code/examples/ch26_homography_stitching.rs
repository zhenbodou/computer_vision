// 第 26 章：单应矩阵、图像对齐与全景拼接
// 运行： cargo run --example ch26_homography_stitching
use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};
use nalgebra::{DMatrix, DVector};

// ---------- 合成一张“宽世界”，两台相机各看一部分（有重叠）----------
fn make_world(w: u32, h: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([235, 235, 235]));
    // 网格线：透视变形后一眼能看出来
    for y in 0..h {
        for x in 0..w {
            if x % 40 == 0 || y % 40 == 0 {
                img.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
    }
    // 沿宽度撒一排不同颜色的地标（保证左右两半内容不同，重叠区可辨认）
    let marks = [
        (80u32, 120u32, Rgb([200, 40, 40])),
        (180, 250, Rgb([40, 140, 40])),
        (300, 100, Rgb([40, 40, 200])),
        (330, 280, Rgb([210, 160, 20])),
        (430, 160, Rgb([160, 40, 160])),
        (560, 220, Rgb([20, 170, 170])),
        (660, 120, Rgb([120, 90, 40])),
    ];
    for (cx, cy, color) in marks {
        for y in cy.saturating_sub(26)..(cy + 26).min(h) {
            for x in cx.saturating_sub(26)..(cx + 26).min(w) {
                img.put_pixel(x, y, color);
            }
        }
    }
    img
}

fn crop(img: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    image::imageops::crop_imm(img, x, y, w, h).to_image()
}

// ---------- 教学：用 nalgebra 手写 DLT 解单应矩阵 H（打印出来看）----------
// 每对对应点 (x,y)->(x',y') 贡献 2 个方程；4 对 -> 8 方程，令 h8=1 解 8 个未知数。
fn solve_homography(from: &[(f64, f64)], to: &[(f64, f64)]) -> [f32; 9] {
    let mut a = DMatrix::<f64>::zeros(8, 8);
    let mut b = DVector::<f64>::zeros(8);
    for i in 0..4 {
        let (x, y) = from[i];
        let (xp, yp) = to[i];
        // x' = (h0 x + h1 y + h2) / (h6 x + h7 y + 1)  =>  移项后：
        a.row_mut(2 * i)
            .copy_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp]);
        b[2 * i] = xp;
        a.row_mut(2 * i + 1)
            .copy_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp]);
        b[2 * i + 1] = yp;
    }
    let h = a.lu().solve(&b).expect("控制点退化（共线？），无法解 H");
    [
        h[0] as f32, h[1] as f32, h[2] as f32,
        h[3] as f32, h[4] as f32, h[5] as f32,
        h[6] as f32, h[7] as f32, 1.0,
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (ww, wh) = (760u32, 380u32);
    let world = make_world(ww, wh);
    let (cw, ch) = (440u32, 380u32); // 每张“照片”的大小

    // 相机 A：直接看世界左半 [0,440)
    let img_a = crop(&world, 0, 0, cw, ch);

    // 相机 B：看世界右半，且带一个已知透视（模拟不同视角）
    // world_to_b：世界坐标 -> B 图坐标 = 先平移(-260)，再乘一个透视
    let shift = 260.0f32;
    let persp = Projection::from_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0008, 0.0, 1.0]).unwrap();
    let world_to_b = Projection::translate(-shift, 0.0).and_then(persp);
    // 用 warp 把世界“拍成”B（输出到 cw×ch 画布）
    let mut img_b = RgbImage::new(cw, ch);
    warp_into(
        &world,
        world_to_b,
        Interpolation::Bilinear,
        Border::Constant(Rgb([0, 0, 0])),
        &mut img_b,
    );

    // ---------- 关键一步：只用 4 对对应点求 H ----------
    // 实战中这 4+ 对点来自“特征匹配 + RANSAC”（见第 25 章）；这里用重叠区已知点。
    let world_pts = [(280.0f32, 90.0), (420.0, 90.0), (420.0, 300.0), (280.0, 300.0)];
    // 每个世界点在 B 图里的位置（= world_to_b 作用的结果）
    let b_pts: Vec<(f32, f32)> = world_pts.iter().map(|&p| world_to_b * p).collect();

    // 方式一（推荐）：库函数直接从 4 对点求投影（内部就是 SVD 解 DLT）
    let from4: [(f32, f32); 4] = [b_pts[0], b_pts[1], b_pts[2], b_pts[3]];
    let to4: [(f32, f32); 4] = world_pts;
    let h_b_to_world = Projection::from_control_points(from4, to4).expect("求 H 失败");

    // 方式二（教学）：nalgebra 手写 DLT，打印出 3x3 的 H
    let fromd: Vec<(f64, f64)> = b_pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let tod: Vec<(f64, f64)> = world_pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let h = solve_homography(&fromd, &tod);
    println!("手写 DLT 解出的单应矩阵 H（B -> 世界坐标）：");
    println!("  [{:8.4} {:8.4} {:8.4}]", h[0], h[1], h[2]);
    println!("  [{:8.4} {:8.4} {:8.4}]", h[3], h[4], h[5]);
    println!("  [{:8.4} {:8.4} {:8.4}]", h[6], h[7], h[8]);

    // 校验：拿一个没参与求解的重叠点，看 H 把它从 B 映回世界，误差多大
    let check_world = (350.0f32, 200.0);
    let check_b = world_to_b * check_world;
    let mapped = h_b_to_world * check_b;
    println!(
        "校验点：真实世界坐标({:.1},{:.1})，经 H 映回得({:.2},{:.2})，误差 {:.3}px",
        check_world.0, check_world.1, mapped.0, mapped.1,
        ((mapped.0 - check_world.0).powi(2) + (mapped.1 - check_world.1).powi(2)).sqrt()
    );

    // ---------- 拼接：把 B warp 到世界坐标系，与 A 加权融合 ----------
    let mut pano_b = RgbImage::new(ww, wh);
    warp_into(&img_b, h_b_to_world, Interpolation::Bilinear, Border::Constant(Rgb([0, 0, 0])), &mut pano_b);
    // B 的覆盖掩膜：把一张纯白按同样的 H warp 过去，非黑即“B 有内容”
    let white = RgbImage::from_pixel(cw, ch, Rgb([255, 255, 255]));
    let mut mask_b = RgbImage::new(ww, wh);
    warp_into(&white, h_b_to_world, Interpolation::Nearest, Border::Constant(Rgb([0, 0, 0])), &mut mask_b);

    // 找重叠区左边界（A 覆盖 x<cw；B 覆盖 mask_b>128）
    let b_covered = |x: u32, y: u32| mask_b.get_pixel(x, y)[0] > 128;
    let mut overlap_lo = cw;
    for x in 0..cw {
        if (0..wh).any(|y| b_covered(x, y)) {
            overlap_lo = x;
            break;
        }
    }

    let mut pano = RgbImage::from_pixel(ww, wh, Rgb([245, 245, 245]));
    for y in 0..wh {
        for x in 0..ww {
            let a_cov = x < cw;
            let b_cov = b_covered(x, y);
            let pb = *pano_b.get_pixel(x, y);
            let pixel = if a_cov && b_cov {
                // 线性羽化：在重叠带内，A 权重从 1 平滑降到 0
                let wa = ((cw - x) as f32 / (cw - overlap_lo).max(1) as f32).clamp(0.0, 1.0);
                let pa = *img_a.get_pixel(x, y);
                Rgb([
                    (pa[0] as f32 * wa + pb[0] as f32 * (1.0 - wa)) as u8,
                    (pa[1] as f32 * wa + pb[1] as f32 * (1.0 - wa)) as u8,
                    (pa[2] as f32 * wa + pb[2] as f32 * (1.0 - wa)) as u8,
                ])
            } else if a_cov {
                *img_a.get_pixel(x, y)
            } else if b_cov {
                pb
            } else {
                Rgb([245, 245, 245])
            };
            pano.put_pixel(x, y, pixel);
        }
    }

    println!("重叠区左边界 x≈{}，全景图尺寸 {}×{}", overlap_lo, ww, wh);
    img_a.save("out_ch26_a.png")?;
    img_b.save("out_ch26_b.png")?;
    pano.save("out_ch26_pano.png")?;
    Ok(())
}
