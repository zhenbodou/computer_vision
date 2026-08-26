//! 第 59 章 光流（Lucas-Kanade 与稠密光流）—— 正文代码验证
//! 运行：cargo run --example ch59_optical_flow
//! 依赖：image 0.25.10、imageproc 0.27.0、nalgebra 0.35.0（主工程已配好）
//!
//! 演示：
//! ① 合成两帧（把带纹理的场景整体平移已知 (dx,dy)），在角点处用 Lucas-Kanade
//!    解 2×2 法方程恢复运动矢量 (u,v)，与真值对比；
//! ② 单次 LK 对“大位移”会低估，迭代 LK（金字塔/迭代思想的雏形）可纠正；
//! ③ 孔径问题：直边上法方程矩阵 M 近奇异（最小特征值≈0），无法唯一求解；
//! ④ 自实现稠密光流，用 imageproc drawing 把光流箭头画出来存 PNG。

use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use nalgebra::{Matrix2, Vector2};

// ---------- 自包含 PRNG（xorshift64*）：给帧加可复现的高斯噪声，不引入 rand ----------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------- 合成一张带纹理的灰度场景：灰底 + 若干亮/暗方块（角点丰富，适合 LK）----------
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::from_pixel(w, h, Luma([100]));
    // (x0, y0, 宽, 高, 灰度)；方块的边角提供“两个方向”的梯度，是 LK 能解的前提
    let blocks = [
        (40u32, 40u32, 36u32, 28u32, 210u8),
        (140, 55, 30, 40, 55),
        (90, 130, 44, 30, 185),
        (200, 105, 34, 34, 40),
        (170, 150, 26, 26, 230),
    ];
    for &(x0, y0, bw, bh, v) in &blocks {
        for y in y0..(y0 + bh).min(h) {
            for x in x0..(x0 + bw).min(w) {
                img.put_pixel(x, y, Luma([v]));
            }
        }
    }
    img
}

// ---------- 双线性采样：在浮点坐标处取灰度，越界夹到边界 ----------
fn sample(img: &GrayImage, fx: f32, fy: f32) -> f32 {
    let (w, h) = img.dimensions();
    let x = fx.clamp(0.0, (w - 1) as f32);
    let y = fy.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let ax = x - x0 as f32;
    let ay = y - y0 as f32;
    let p = |xx, yy| img.get_pixel(xx, yy)[0] as f32;
    let top = p(x0, y0) * (1.0 - ax) + p(x1, y0) * ax;
    let bot = p(x0, y1) * (1.0 - ax) + p(x1, y1) * ax;
    top * (1.0 - ay) + bot * ay
}

// ---------- 把整幅图平移 (dx,dy)：frame2(x,y)=frame1(x-dx,y-dy) ----------
// 于是 frame1 中位于 p 的特征，在 frame2 中出现在 p+(dx,dy)，光流真值即 (dx,dy)。
fn shift(img: &GrayImage, dx: f32, dy: f32) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let v = sample(img, x as f32 - dx, y as f32 - dy);
        Luma([v.round().clamp(0.0, 255.0) as u8])
    })
}

fn add_noise(img: &GrayImage, sigma: f32, rng: &mut Rng) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let v = img.get_pixel(x, y)[0] as f32 + sigma * rng.normal() as f32;
        Luma([v.round().clamp(0.0, 255.0) as u8])
    })
}

// ---------- 归一化梯度场：imageproc 的 Sobel 是“真导数的 8 倍”，必须 /8 才与帧差 I_t 单位一致 ----------
fn grad_field(img: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = img.dimensions();
    let gx = horizontal_sobel(img);
    let gy = vertical_sobel(img);
    let mut ix = vec![0.0f32; (w * h) as usize];
    let mut iy = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            ix[i] = gx.get_pixel(x, y)[0] as f32 / 8.0;
            iy[i] = gy.get_pixel(x, y)[0] as f32 / 8.0;
        }
    }
    (ix, iy)
}

/// 在点 (px,py) 处做迭代 Lucas-Kanade。
/// 返回 (u, v, M 的最小特征值)。最小特征值反映“这个点好不好跟”：
/// 越大越好；≈0 说明是无纹理区或直边（孔径问题），解不可靠。
/// iters=1 就是经典单次 LK；iters>1 会“把 frame2 按当前估计采样回来重算残差”，纠正大位移低估。
fn lk_track(
    f1: &GrayImage,
    f2: &GrayImage,
    ix: &[f32],
    iy: &[f32],
    px: i32,
    py: i32,
    radius: i32,
    iters: usize,
) -> Option<(f32, f32, f32)> {
    let (iw, ih) = f1.dimensions();
    let w = iw as usize;
    // ---- ① 结构张量 M：只依赖 frame1 的梯度，窗口固定，只需算一次 ----
    let (mut a11, mut a12, mut a22) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (x, y) = (px + dx, py + dy);
            if x < 0 || y < 0 || x >= iw as i32 || y >= ih as i32 {
                continue;
            }
            let idx = (y as usize) * w + x as usize;
            let (gx, gy) = (ix[idx] as f64, iy[idx] as f64);
            a11 += gx * gx;
            a12 += gx * gy;
            a22 += gy * gy;
        }
    }
    // 最小特征值（2×2 对称阵闭式解）
    let tr = a11 + a22;
    let det = a11 * a22 - a12 * a12;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let min_eig = (tr / 2.0 - disc) as f32;
    let m = Matrix2::new(a11, a12, a12, a22);
    let inv = m.try_inverse()?; // M 奇异（纯直边/无纹理）则放弃：这正是孔径问题
    // ---- ② 迭代：用当前 (u,v) 把 frame2 采样回来算残差 It，再解增量 ----
    let (mut u, mut v) = (0.0f32, 0.0f32);
    for _ in 0..iters {
        let (mut b1, mut b2) = (0.0f64, 0.0f64);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (x, y) = (px + dx, py + dy);
                if x < 0 || y < 0 || x >= iw as i32 || y >= ih as i32 {
                    continue;
                }
                let idx = (y as usize) * w + x as usize;
                let (gx, gy) = (ix[idx] as f64, iy[idx] as f64);
                // 残差 It = f2(x+u, y+v) - f1(x, y)（亮度恒定下应趋于 0）
                let it =
                    (sample(f2, x as f32 + u, y as f32 + v) - f1.get_pixel(x as u32, y as u32)[0] as f32) as f64;
                b1 += gx * it;
                b2 += gy * it;
            }
        }
        // 解 M·[du,dv]^T = -[b1,b2]^T
        let d = inv * Vector2::new(-b1, -b2);
        u += d.x as f32;
        v += d.y as f32;
        if d.norm() < 1e-3 {
            break;
        }
    }
    Some((u, v, min_eig))
}

// 在画布上画一支带箭头的矢量（shaft + 两条箭羽）
fn draw_arrow(canvas: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgb<u8>) {
    draw_line_segment_mut(canvas, (x0, y0), (x1, y1), color);
    let ang = (y1 - y0).atan2(x1 - x0);
    let head = 4.0f32;
    for da in [2.6f32, -2.6] {
        let hx = x1 - head * (ang + da).cos();
        let hy = y1 - head * (ang + da).sin();
        draw_line_segment_mut(canvas, (x1, y1), (hx, hy), color);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (260u32, 200u32);
    let radius = 6i32; // LK 窗口半径 -> 13×13 窗口

    // ================= ① 稀疏 LK：整体平移已知 (u,v)，在角点处恢复运动 =================
    let (u_true, v_true) = (2.0f32, 1.0f32);
    let frame1 = make_scene(w, h);
    let mut rng = Rng::new(59);
    let frame1n = add_noise(&frame1, 1.5, &mut rng); // 两帧各加一点独立噪声，更接近真实
    let frame2 = add_noise(&shift(&frame1, u_true, v_true), 1.5, &mut rng);
    let (ix, iy) = grad_field(&frame1n);
    // 干净（无噪声）版本：留给 ②孔径问题 / ③大位移 用，好让“直边严格奇异”看得清清楚楚
    let frame2c = shift(&frame1, u_true, v_true);
    let (ixc, iyc) = grad_field(&frame1);
    let (ix2, iy2) = grad_field(&frame2); // 反向跟踪(frame2->frame1)用第二帧的梯度

    // 用 FAST 角点当跟踪点（第 23 章），过滤掉离边太近的
    let mut corners = corners_fast9(&frame1n, 30);
    let margin = (radius + 2) as u32;
    corners.retain(|c| c.x >= margin && c.y >= margin && c.x + margin < w && c.y + margin < h);
    corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    corners.truncate(10);

    println!("== ① 稀疏 Lucas-Kanade（真值 u={:.1}, v={:.1}）==", u_true, v_true);
    println!("角点数 {}，窗口 {}×{}", corners.len(), 2 * radius + 1, 2 * radius + 1);
    println!("  点(x,y)     单次LK(u,v)        迭代LK(u,v)       误差(迭代)");
    let (mut se1, mut se_it, mut se_fb, mut cnt) = (0.0f32, 0.0f32, 0.0f32, 0);
    for c in &corners {
        let (px, py) = (c.x as i32, c.y as i32);
        let one = lk_track(&frame1n, &frame2, &ix, &iy, px, py, radius, 1);
        let itr = lk_track(&frame1n, &frame2, &ix, &iy, px, py, radius, 8);
        if let (Some((u1, v1, _)), Some((u2, v2, _))) = (one, itr) {
            let e1 = ((u1 - u_true).powi(2) + (v1 - v_true).powi(2)).sqrt();
            let e2 = ((u2 - u_true).powi(2) + (v2 - v_true).powi(2)).sqrt();
            // 前后向一致性：再从 frame2 的落点往回跟一次，正向+反向位移应互相抵消（≈0）
            let (bx, by) = ((px as f32 + u2).round() as i32, (py as f32 + v2).round() as i32);
            if let Some((ub, vb, _)) = lk_track(&frame2, &frame1n, &ix2, &iy2, bx, by, radius, 8) {
                se_fb += ((u2 + ub).powi(2) + (v2 + vb).powi(2)).sqrt();
            }
            se1 += e1;
            se_it += e2;
            cnt += 1;
            println!(
                "  ({:>3},{:>3})   ({:>5.2},{:>5.2})     ({:>5.2},{:>5.2})     {:>5.3}",
                px, py, u1, v1, u2, v2, e2
            );
        }
    }
    println!(
        "平均误差：单次 LK = {:.3} px，迭代 LK = {:.3} px（迭代把大位移的低估纠了回来）",
        se1 / cnt as f32,
        se_it / cnt as f32
    );
    println!(
        "前后向一致性平均残差 = {:.3} px（无需真值也能自检：残差大就该丢弃该点）\n",
        se_fb / cnt as f32
    );

    // ================= ② 孔径问题：直边 vs 角点，看最小特征值 =================
    println!("== ② 孔径问题：直边只能定“法向”一维，M 奇异；角点两维都能定 ==");
    // 直边点：大方块(40..76, 40..68)的左边中点(40,54)，窗口里只有竖直边 -> Iy≡0 -> M 严格奇异
    match lk_track(&frame1, &frame2c, &ixc, &iyc, 40, 54, radius, 8) {
        Some((u, v, e)) => println!("  直边点(40,54): (u,v)=({:.2},{:.2}) min_eig={:.2}", u, v, e),
        None => println!("  直边点(40,54): M 奇异，无法唯一求解（沿边方向那一维彻底说不清）"),
    }
    // 角点：同一方块左上角(40,40)
    if let Some((u, v, e)) = lk_track(&frame1, &frame2c, &ixc, &iyc, 40, 40, radius, 8) {
        println!(
            "  角点  (40,40): (u,v)=({:.2},{:.2}) min_eig={:.2}  <- 两个方向都有梯度，稳",
            u, v, e
        );
    }
    println!();

    // ================= ③ 单次 LK 的“大位移”失效，与迭代对比 =================
    println!("== ③ 大位移：单次 LK 低估，迭代（金字塔思想的雏形）纠正 ==");
    for big in [1.0f32, 3.0, 6.0] {
        let f2 = shift(&frame1, big, 0.0); // 只在 x 方向平移 big 像素
        let (bx, by) = (40, 40); // 用那个强角点
        let one = lk_track(&frame1, &f2, &ixc, &iyc, bx, by, radius, 1).unwrap();
        let itr = lk_track(&frame1, &f2, &ixc, &iyc, bx, by, radius, 20).unwrap();
        println!(
            "  真值 u={:>3.0}: 单次 u={:>5.2}（低估），迭代 u={:>5.2}",
            big, one.0, itr.0
        );
    }
    println!();

    // ================= ④ 稠密光流：逐网格点跑 LK，画箭头存 PNG =================
    println!("== ④ 稠密光流（自实现）：网格采样 + 画箭头 ==");
    let mut canvas: RgbImage = image::DynamicImage::ImageLuma8(frame1n.clone()).to_rgb8();
    let step = 14i32;
    let scale = 4.0f32; // 位移太小，放大 4 倍才看得见
    let (mut ok, mut skip) = (0, 0);
    let mut y = step;
    while y < h as i32 - step {
        let mut x = step;
        while x < w as i32 - step {
            match lk_track(&frame1n, &frame2, &ix, &iy, x, y, radius, 5) {
                // 只在纹理够强（min_eig 够大）处画，否则平坦区/单边只是噪声乱解
                Some((u, v, e)) if e > 120.0 => {
                    draw_arrow(
                        &mut canvas,
                        x as f32,
                        y as f32,
                        x as f32 + u * scale,
                        y as f32 + v * scale,
                        Rgb([255, 40, 40]),
                    );
                    ok += 1;
                }
                _ => skip += 1,
            }
            x += step;
        }
        y += step;
    }
    println!("画了 {} 支箭头，跳过 {} 个弱纹理点（min_eig≤120）", ok, skip);
    frame1n.save("out_ch59_frame1.png")?;
    frame2.save("out_ch59_frame2.png")?;
    canvas.save("out_ch59_flow.png")?;
    println!("已存 out_ch59_frame1.png / out_ch59_frame2.png / out_ch59_flow.png");
    Ok(())
}
