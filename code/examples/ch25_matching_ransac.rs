// 第 25 章：特征匹配（FAST+BRIEF + 比率测试）与手写 RANSAC
// 运行： cargo run --example ch25_matching_ransac
use image::{GrayImage, Rgb, RgbImage};
use imageproc::binary_descriptors::brief::brief;
use imageproc::binary_descriptors::{match_binary_descriptors, BinaryDescriptor};
use imageproc::corners::corners_fast9;
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use imageproc::point::Point;

// ---------- 一个自包含的小 RNG（xorshift64*），避免引入 rand 依赖 ----------
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
    // [0,1) 均匀分布
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    // 标准正态分布（Box-Muller）
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo) as u64) as u32
    }
}

// ---------- 合成一张“布满角点”的宽场景，方便裁两张有重叠的图 ----------
fn make_scene(w: u32, h: u32, seed: u64) -> RgbImage {
    let mut img = RgbImage::from_pixel(w, h, Rgb([230, 230, 230]));
    let mut rng = Rng::new(seed);
    // 撒一堆不同灰度/颜色的实心矩形——矩形的四个角就是天然的 FAST 角点
    for _ in 0..40 {
        let rx = rng.range(5, w - 45);
        let ry = rng.range(5, h - 45);
        let rw = rng.range(12, 40);
        let rh = rng.range(12, 40);
        let color = Rgb([
            rng.range(0, 200) as u8,
            rng.range(0, 200) as u8,
            rng.range(0, 200) as u8,
        ]);
        for y in ry..(ry + rh).min(h) {
            for x in rx..(rx + rw).min(w) {
                img.put_pixel(x, y, color);
            }
        }
    }
    img
}

fn crop(img: &RgbImage, x: u32, y: u32, w: u32, h: u32) -> RgbImage {
    image::imageops::crop_imm(img, x, y, w, h).to_image()
}

// FAST 角点 -> 过滤掉离边太近的（BRIEF 需要 15px 半径的 patch）-> 按分数取前 n 个
fn good_keypoints(gray: &GrayImage, thresh: u8, margin: u32, n: usize) -> Vec<Point<u32>> {
    let (w, h) = gray.dimensions();
    let mut corners = corners_fast9(gray, thresh);
    corners.retain(|c| c.x >= margin && c.x + margin < w && c.y >= margin && c.y + margin < h);
    corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    corners.truncate(n);
    corners.into_iter().map(|c| Point::new(c.x, c.y)).collect()
}

// ---------- 手写“最近邻 + Lowe 比率测试”匹配（教学版，暴力 O(Na*Nb)）----------
// 对 d1 里每个描述子，在 d2 里找最近和次近；若 最近 < ratio * 次近 则保留。
fn match_with_ratio<'a, T: BinaryDescriptor>(
    d1: &'a [T],
    d2: &'a [T],
    ratio: f32,
) -> Vec<(&'a T, &'a T)> {
    let mut out = Vec::new();
    for a in d1 {
        let (mut best, mut second) = (u32::MAX, u32::MAX);
        let mut best_j = 0usize;
        for (j, b) in d2.iter().enumerate() {
            let dist = a.hamming_distance(b);
            if dist < best {
                second = best;
                best = dist;
                best_j = j;
            } else if dist < second {
                second = dist;
            }
        }
        // 比率测试：最近邻要明显比次近邻更近，才认为是可靠匹配
        if (best as f32) < ratio * (second as f32) {
            out.push((a, &d2[best_j]));
        }
    }
    out
}

fn feature_matching_demo() -> Result<(), Box<dyn std::error::Error>> {
    // 一张宽场景，裁两张有重叠、彼此平移的图（“同一图的两个版本”）
    let scene = make_scene(520, 400, 20250826);
    let (cw, ch) = (380u32, 360u32);
    let (shift_x, shift_y) = (24u32, 12u32); // B 相对 A 的已知平移
    let img_a = crop(&scene, 0, 0, cw, ch);
    let img_b = crop(&scene, shift_x, shift_y, cw, ch);
    let gray_a = image::DynamicImage::ImageRgb8(img_a.clone()).to_luma8();
    let gray_b = image::DynamicImage::ImageRgb8(img_b.clone()).to_luma8();

    // 1) 两图各自检测 FAST 角点
    let kp_a = good_keypoints(&gray_a, 25, 16, 60);
    let kp_b = good_keypoints(&gray_b, 25, 16, 60);
    println!("A 角点 {} 个，B 角点 {} 个", kp_a.len(), kp_b.len());

    // 2) 计算 BRIEF 描述子：两图必须用【同一套】测试点对，才有可比性！
    let (desc_a, pairs) = brief(&gray_a, &kp_a, 256, None)?;
    let (desc_b, _) = brief(&gray_b, &kp_b, 256, Some(&pairs))?;

    // 3a) 手写“比率测试”匹配
    let matches = match_with_ratio(&desc_a, &desc_b, 0.8);
    // 3b) 库自带的匹配（Hamming 阈值 + 局部敏感哈希）作对照
    let lib_matches = match_binary_descriptors(&desc_a, &desc_b, 64, Some(7));

    // 用“已知平移”判断哪些匹配是几何内点（真对）哪些是外点（误匹配）
    let is_inlier = |a: &Point<u32>, b: &Point<u32>| -> bool {
        let dx = a.x as i32 - b.x as i32;
        let dy = a.y as i32 - b.y as i32;
        (dx - shift_x as i32).abs() <= 3 && (dy - shift_y as i32).abs() <= 3
    };
    let inliers = matches
        .iter()
        .filter(|(a, b)| is_inlier(&a.position(), &b.position()))
        .count();
    println!(
        "比率测试匹配 {} 对，其中几何正确(内点) {} 对，误匹配(外点) {} 对",
        matches.len(),
        inliers,
        matches.len() - inliers
    );
    println!("库 match_binary_descriptors 匹配 {} 对", lib_matches.len());

    // 4) 画到并排大图上：绿=正确匹配，红=误匹配
    let gap = 30u32;
    let mut canvas = RgbImage::from_pixel(cw * 2 + gap, ch, Rgb([255, 255, 255]));
    for (x, y, p) in img_a.enumerate_pixels() {
        canvas.put_pixel(x, y, *p);
    }
    for (x, y, p) in img_b.enumerate_pixels() {
        canvas.put_pixel(x + cw + gap, y, *p);
    }
    for (a, b) in &matches {
        let (pa, pb) = (a.position(), b.position());
        let color = if is_inlier(&pa, &pb) {
            Rgb([0, 180, 0])
        } else {
            Rgb([230, 0, 0])
        };
        draw_line_segment_mut(
            &mut canvas,
            (pa.x as f32, pa.y as f32),
            ((pb.x + cw + gap) as f32, pb.y as f32),
            color,
        );
    }
    canvas.save("out_ch25_matches.png")?;
    Ok(())
}

// ---------- 手写 RANSAC：从“带外点的散点”里稳健拟合直线 ----------

// 最小二乘拟合 y = m*x + c（会被外点带偏）
fn least_squares(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pts {
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let m = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let c = (sy - m * sx) / n;
    (m, c)
}

// 点到“过 p1,p2 两点直线”的垂直距离
fn point_line_dist(q: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let (dx, dy) = (p2.0 - p1.0, p2.1 - p1.1);
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    ((q.0 - p1.0) * dy - (q.1 - p1.1) * dx).abs() / len
}

fn ransac_line(pts: &[(f64, f64)], thresh: f64, iters: usize, rng: &mut Rng) -> (f64, f64, usize) {
    let n = pts.len();
    let mut best_inliers: Vec<usize> = Vec::new();
    for _ in 0..iters {
        // ① 随机取最小样本：拟合一条直线只需 2 个点
        let i = rng.range(0, n as u32) as usize;
        let mut j = rng.range(0, n as u32) as usize;
        while j == i {
            j = rng.range(0, n as u32) as usize;
        }
        let (p1, p2) = (pts[i], pts[j]);
        if (p1.0 - p2.0).abs() < 1e-6 && (p1.1 - p2.1).abs() < 1e-6 {
            continue; // 退化：两点重合，跳过
        }
        // ② 数内点：离这条线足够近的点
        let inliers: Vec<usize> = (0..n)
            .filter(|&k| point_line_dist(pts[k], p1, p2) < thresh)
            .collect();
        // ③ 保留内点最多的模型
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
        }
    }
    // ④ 用全部内点做最小二乘精修
    let inlier_pts: Vec<(f64, f64)> = best_inliers.iter().map(|&k| pts[k]).collect();
    let (m, c) = least_squares(&inlier_pts);
    (m, c, best_inliers.len())
}

fn ransac_demo() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::new(12345);
    let (true_m, true_c) = (0.5, 20.0); // 真实直线
    let mut pts: Vec<(f64, f64)> = Vec::new();
    // 内点：贴着真实直线，加小噪声
    for _ in 0..80 {
        let x = rng.uniform() * 200.0;
        let y = true_m * x + true_c + rng.normal() * 3.0;
        pts.push((x, y));
    }
    // 外点：满屏乱撒
    let n_out = 40;
    for _ in 0..n_out {
        pts.push((rng.uniform() * 200.0, rng.uniform() * 200.0));
    }
    let w = 80.0 / pts.len() as f64; // 内点比例估计

    // 理论迭代次数 N = log(1-p) / log(1 - w^s)
    let (p, s) = (0.99_f64, 2.0_f64);
    let n_theory = (1.0 - p).ln() / (1.0 - w.powf(s)).ln();
    println!(
        "内点比例 w≈{:.2}，置信度 p={}，最小样本 s={} → 理论迭代 N≈{:.1} 次",
        w, p, s, n_theory
    );

    let (lm, lc) = least_squares(&pts);
    let (rm, rc, n_in) = ransac_line(&pts, 5.0, 200, &mut rng);
    println!("真实直线:      y = {:.3} x + {:.2}", true_m, true_c);
    println!("最小二乘(全部): y = {:.3} x + {:.2}  <- 被外点带偏", lm, lc);
    println!(
        "RANSAC + 精修:  y = {:.3} x + {:.2}  (找到 {} 个内点)",
        rm, rc, n_in
    );

    // 可视化：散点 + 两条拟合线
    let (wd, ht) = (220u32, 160u32);
    let mut vis = RgbImage::from_pixel(wd, ht, Rgb([255, 255, 255]));
    let ty = |y: f64| (ht as f64 - y).clamp(0.0, ht as f64 - 1.0) as i32; // 图像 y 轴朝下，翻一下
    for (idx, &(x, y)) in pts.iter().enumerate() {
        let color = if idx < 80 {
            Rgb([0, 150, 0])
        } else {
            Rgb([220, 0, 0])
        };
        draw_filled_circle_mut(&mut vis, (x as i32, ty(y)), 1, color);
    }
    let draw_fit = |vis: &mut RgbImage, m: f64, c: f64, color: Rgb<u8>| {
        for xi in 0..wd {
            let y = m * xi as f64 + c;
            let yy = ty(y);
            if yy >= 0 && yy < ht as i32 {
                vis.put_pixel(xi, yy as u32, color);
            }
        }
    };
    draw_fit(&mut vis, lm, lc, Rgb([0, 0, 255])); // 蓝：最小二乘（被带偏）
    draw_fit(&mut vis, rm, rc, Rgb([255, 140, 0])); // 橙：RANSAC（贴合内点）
    vis.save("out_ch25_ransac.png")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==== 特征匹配 ====");
    feature_matching_demo()?;
    println!("\n==== RANSAC 拟合直线 ====");
    ransac_demo()?;
    Ok(())
}
