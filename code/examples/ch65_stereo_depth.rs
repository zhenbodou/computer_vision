//! 第 65 章 立体视觉与深度：合成一对「已校正」的左右灰度图
//! （同一随机纹理按不同已知视差平移；前景视差大=近，背景视差小=远），
//! 用 SAD 块匹配求视差，左右一致性检验剔除不可靠点（遮挡带），再由 Z = f·B/d 转深度。
//! 运行：cargo run --example ch65_stereo_depth
//! 依赖：image 0.25
//!
//! 自验证：真值视差已知 -> 估计视差 -> 打印「估计 vs 真值」、被检出的无效比例、
//! 由此得到的深度；并存左图/右图/视差图 PNG。

use image::{GrayImage, Luma};

const W: u32 = 256;
const H: u32 = 192;
const D0: u32 = 5; // 背景视差（远）
const D1: u32 = 20; // 前景视差（近）
const DMAX: u32 = 24; // 视差搜索上限
const R: i32 = 3; // 块半径（7×7 窗口）

// 确定性伪随机纹理：给块匹配提供丰富、唯一的花纹（真实弱纹理场景会更难，见文末坑）。
// seed 不同 => 前景/背景用两套互不相同的纹理，避免「背景花纹恰好也匹配上」的合成假象。
fn tex(x: u32, y: u32, seed: u64) -> u8 {
    let h = (x as u64)
        .wrapping_mul(73856093)
        .wrapping_add((y as u64).wrapping_mul(19349663))
        .wrapping_add(seed);
    let h = (h ^ (h >> 13)).wrapping_mul(0x2545F4914F6CDD1D);
    ((h >> 33) & 0xFF) as u8
}
fn tex_bg(x: u32, y: u32) -> u8 {
    tex(x, y, 0xB6)
}
fn tex_fg(x: u32, y: u32) -> u8 {
    tex(x, y, 0x3F)
}

// 前景矩形在「左图」中的范围（左图作参考坐标系）
fn is_foreground_left(x: u32, y: u32) -> bool {
    (110..170).contains(&x) && (70..140).contains(&y)
}
// 前景矩形在「右图」中的范围：左图范围整体左移 D1
fn is_foreground_right(x: u32, y: u32) -> bool {
    (90..150).contains(&x) && (70..140).contains(&y)
}
// 左图每个像素的真值视差
fn true_disp(x: u32, y: u32) -> u32 {
    if is_foreground_left(x, y) {
        D1
    } else {
        D0
    }
}

// SAD 块匹配求视差。dir=-1：参考=left、目标=right，匹配位置 x-d；dir=+1：反过来。
// 越界的 d 直接跳过；返回每像素视差（无合法 d 处记 0）。
fn disparity(reference: &GrayImage, target: &GrayImage, dir: i32) -> Vec<u32> {
    let mut disp = vec![0u32; (W * H) as usize];
    let rp = |img: &GrayImage, x: i32, y: i32| img.get_pixel(x as u32, y as u32)[0] as i64;
    for y in R..(H as i32 - R) {
        for x in R..(W as i32 - R) {
            let (mut best_d, mut best_cost) = (0u32, i64::MAX);
            for d in 0..=DMAX as i32 {
                let tx = x + dir * d; // 目标图中的对应列
                if tx - R < 0 || tx + R >= W as i32 {
                    continue; // 该视差会越界，跳过
                }
                let mut cost = 0i64;
                for dy in -R..=R {
                    for dx in -R..=R {
                        cost +=
                            (rp(reference, x + dx, y + dy) - rp(target, tx + dx, y + dy)).abs();
                    }
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_d = d as u32;
                }
            }
            disp[(y as u32 * W + x as u32) as usize] = best_d;
        }
    }
    disp
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ==================== ① 合成已校正的左右图 ====================
    // 左图：前景矩形贴前景纹理，其余贴背景纹理。
    // 右图：同一表面点左移各自的视差 d，即 right(x)=surface(x+d)；前景在右图位于 [90,150)。
    let left = GrayImage::from_fn(W, H, |x, y| {
        if is_foreground_left(x, y) {
            Luma([tex_fg(x, y)])
        } else {
            Luma([tex_bg(x, y)])
        }
    });
    let right = GrayImage::from_fn(W, H, |x, y| {
        if is_foreground_right(x, y) {
            Luma([tex_fg(x + D1, y)]) // 前景（近，视差大）
        } else {
            Luma([tex_bg(x + D0, y)]) // 背景（远，视差小）
        }
    });

    // ==================== ② 双向 SAD 块匹配 ====================
    let disp_l = disparity(&left, &right, -1); // 以左图为参考
    let disp_r = disparity(&right, &left, 1); // 以右图为参考（用于一致性检验）

    // ==================== ③ 左右一致性检验：剔除遮挡/误匹配 ====================
    // 左图视差 dL 处，回查右图对应点的视差 dR；两者差 >1 判为不可靠（无效）。
    // 评估区取「两个方向都合法」的公共区域 x∈[DMAX+R, W-DMAX-R)。
    let mut valid = vec![false; (W * H) as usize];
    let (x_lo, x_hi) = ((DMAX as i32 + R) as u32, (W as i32 - DMAX as i32 - R) as u32);
    let (y_lo, y_hi) = (R as u32, (H as i32 - R) as u32);
    let (mut total, mut consistent, mut correct_all, mut correct_valid) = (0u64, 0u64, 0u64, 0u64);
    for y in y_lo..y_hi {
        for x in x_lo..x_hi {
            let dl = disp_l[(y * W + x) as usize];
            let xr = x as i32 - dl as i32; // 右图中的对应列
            let dr = disp_r[(y * W + xr as u32) as usize];
            let ok = (dl as i32 - dr as i32).abs() <= 1;
            valid[(y * W + x) as usize] = ok;
            let truth = true_disp(x, y);
            total += 1;
            if dl == truth {
                correct_all += 1;
            }
            if ok {
                consistent += 1;
                if dl == truth {
                    correct_valid += 1;
                }
            }
        }
    }
    println!("评估区像素 {} 个", total);
    println!(
        "  原始视差准确率（全部像素）        = {:.2}%",
        100.0 * correct_all as f64 / total as f64
    );
    println!(
        "  左右一致性判为有效的像素          = {} 个（{:.2}%），无效 {:.2}%",
        consistent,
        100.0 * consistent as f64 / total as f64,
        100.0 * (total - consistent) as f64 / total as f64
    );
    println!(
        "  有效像素中的视差准确率            = {:.2}%（一致性检验成功滤掉了遮挡带的错点）",
        100.0 * correct_valid as f64 / consistent as f64
    );

    // ==================== ④ 分区平均视差 vs 真值 ====================
    let region_avg = |x0: u32, x1: u32, y0: u32, y1: u32| -> f64 {
        let (mut s, mut n) = (0.0, 0.0);
        for y in y0..y1 {
            for x in x0..x1 {
                s += disp_l[(y * W + x) as usize] as f64;
                n += 1.0;
            }
        }
        s / n
    };
    let bg = region_avg(40, 90, 30, 60); // 纯背景区
    let fg = region_avg(120, 152, 95, 135); // 纯前景区（避开遮挡边界）
    println!("\n背景区平均视差：估计 {:.3} vs 真值 {}", bg, D0);
    println!("前景区平均视差：估计 {:.3} vs 真值 {}", fg, D1);

    // ==================== ⑤ 视差转深度 Z = f·B/d ====================
    let f = 500.0_f64; // 焦距（像素）
    let b = 0.10_f64; // 基线（米）
    let depth = |d: f64| f * b / d;
    println!("\n相机：f={f} px，基线 B={b} m");
    println!("背景深度：估计 {:.3} m（真值 {:.3} m）", depth(bg), depth(D0 as f64));
    println!("前景深度：估计 {:.3} m（真值 {:.3} m）", depth(fg), depth(D1 as f64));
    println!("=> 前景更近、背景更远，符合「视差越大越近」直觉 ✓");

    // ==================== ⑥ 存图 ====================
    left.save("out_ch65_left.png")?;
    right.save("out_ch65_right.png")?;
    // 视差图：0..DMAX 拉伸到 0..255（前景亮、背景暗）；一致性判为无效的点涂黑
    let vis = GrayImage::from_fn(W, H, |x, y| {
        let idx = (y * W + x) as usize;
        if x >= x_lo && x < x_hi && y >= y_lo && y < y_hi && !valid[idx] {
            Luma([0]) // 无效（遮挡/误匹配）
        } else {
            Luma([(disp_l[idx] * 255 / DMAX) as u8])
        }
    });
    vis.save("out_ch65_disparity.png")?;
    println!("\n已存 out_ch65_left.png / out_ch65_right.png / out_ch65_disparity.png");
    Ok(())
}
