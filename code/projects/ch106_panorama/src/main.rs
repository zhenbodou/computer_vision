// 第 106 章 项目：全景图拼接工具 —— 纯 Rust 全流程
//
// 流水线：合成带重叠的两张图 → FAST 角点 + BRIEF 描述子匹配（imageproc，呼应第四部分）
//        → RANSAC + DLT 估单应矩阵（nalgebra 解线性系统，呼应 ch14/ch64）
//        → 透视 warp + 线性羽化融合 → 输出全景 PNG
//
// 依赖（均为纯 Rust）：image 0.25.10 / imageproc 0.27.0 / nalgebra 0.35.0
// 运行： cd code/projects/ch106_panorama && cargo run --release

use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::binary_descriptors::brief::{brief, TestPair};
use imageproc::binary_descriptors::match_binary_descriptors;
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_line_segment_mut;
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};
use imageproc::point::Point;
use imageproc::rect::Rect;
use nalgebra::{DMatrix, Matrix3, Vector3};

// ---------------- 确定性随机数：LCG，取「高位」（全书通用，见 §18） ----------------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 // 取高位
    }
    fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + self.unit() * (b - a)
    }
}

const H: u32 = 420; // 图高
const WS: u32 = 1000; // 场景总宽
const WV: u32 = 600; // 单张视图宽

/// 合成一张「局部处处可区分」的场景图：中性底 + 少量大色块（全局结构）
/// + 密集随机彩色斑点（局部纹理）。
/// 关键教训：BRIEF 靠局部像素对比来区分，如果只画大矩形，「所有直角corner长得一模一样」
/// 会让描述子无区分度、误匹配爆炸。撒一层随机斑点，让每个 31×31 邻域都独一无二，
/// 描述子才有辨识度、匹配才干净。
fn make_scene() -> RgbImage {
    let mut img: RgbImage = ImageBuffer::from_pixel(WS, H, Rgb([235, 235, 230]));
    let mut rng = Lcg::new(20240826);
    // 少量大色块：提供全局结构、便于肉眼看拼接是否对齐
    for _ in 0..24 {
        let cx = rng.range(30.0, WS as f64 - 60.0) as i32;
        let cy = rng.range(30.0, H as f64 - 60.0) as i32;
        let col = Rgb([rng.below(210) as u8, rng.below(210) as u8, rng.below(210) as u8]);
        if rng.unit() < 0.5 {
            let (w, h) = (rng.range(30.0, 70.0) as u32, rng.range(30.0, 70.0) as u32);
            imageproc::drawing::draw_filled_rect_mut(&mut img, Rect::at(cx, cy).of_size(w, h), col);
        } else {
            imageproc::drawing::draw_filled_circle_mut(&mut img, (cx, cy), rng.range(14.0, 30.0) as i32, col);
        }
    }
    // 密集随机斑点：让局部邻域处处不同 → BRIEF 描述子有区分度
    for _ in 0..2600 {
        let cx = rng.range(3.0, WS as f64 - 3.0) as i32;
        let cy = rng.range(3.0, H as f64 - 3.0) as i32;
        let col = Rgb([rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8]);
        imageproc::drawing::draw_filled_circle_mut(&mut img, (cx, cy), rng.range(1.0, 3.0) as i32, col);
    }
    img
}

fn to_luma(img: &RgbImage) -> GrayImage {
    let mut g = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        // BT.601 亮度
        let v = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        g.put_pixel(x, y, Luma([v.round() as u8]));
    }
    g
}

/// 生成确定性 BRIEF 测试点对（坐标落在 31×31 patch 内）。
/// 必须两张图共用同一组测试对，否则描述子不可比（imageproc 默认走系统随机，不可复现！）。
fn make_test_pairs(n: usize, seed: u64) -> Vec<TestPair> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| TestPair {
            p0: Point::new(rng.below(31) as u32, rng.below(31) as u32),
            p1: Point::new(rng.below(31) as u32, rng.below(31) as u32),
        })
        .collect()
}

/// FAST 角点检测 + 边界过滤 + 按响应取前 max_kp 个 + BRIEF 描述。
fn detect_and_describe(
    gray: &GrayImage,
    pairs: &[TestPair],
    max_kp: usize,
) -> Vec<imageproc::binary_descriptors::brief::BriefDescriptor> {
    let (w, h) = (gray.width(), gray.height());
    let mut corners = corners_fast9(gray, 25);
    // BRIEF 需要关键点离边至少 16px（patch 半径 15），否则整批报错
    corners.retain(|c| c.x > 16 && c.x + 16 < w && c.y > 16 && c.y + 16 < h);
    corners.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    corners.truncate(max_kp);
    let kps: Vec<Point<u32>> = corners.iter().map(|c| Point::new(c.x, c.y)).collect();
    let (desc, _) = brief(gray, &kps, 256, Some(&pairs.to_vec())).expect("BRIEF 失败");
    desc
}

// ---------------- DLT 单应估计（nalgebra，含 Hartley 归一化） ----------------

/// 点集归一化：平移到质心、缩放到平均距离 √2。返回 (归一化矩阵 T, 归一化后点)。
fn normalize(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
    let n = pts.len() as f64;
    let (cx, cy) = pts.iter().fold((0.0, 0.0), |(sx, sy), p| (sx + p[0], sy + p[1]));
    let (cx, cy) = (cx / n, cy / n);
    let mean_d = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_d > 1e-9 { 2f64.sqrt() / mean_d } else { 1.0 };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let normed = pts.iter().map(|p| [s * (p[0] - cx), s * (p[1] - cy)]).collect();
    (t, normed)
}

/// 由 ≥4 对点 (src→dst) 用 DLT 估计单应矩阵 H（H 把 src 映到 dst）。
fn estimate_homography(pairs: &[([f64; 2], [f64; 2])]) -> Option<Matrix3<f64>> {
    if pairs.len() < 4 {
        return None;
    }
    let src: Vec<[f64; 2]> = pairs.iter().map(|(s, _)| *s).collect();
    let dst: Vec<[f64; 2]> = pairs.iter().map(|(_, d)| *d).collect();
    let (ts, sn) = normalize(&src);
    let (td, dn) = normalize(&dst);
    // 每对点贡献 2 行，堆成 2N×9 的 A，解 A h = 0（h 为 H 拉平的 9 维）
    let mut a = DMatrix::<f64>::zeros(2 * pairs.len(), 9);
    for i in 0..pairs.len() {
        let [x, y] = sn[i];
        let [u, v] = dn[i];
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }
    // 解 A h = 0：取 A 的最小奇异向量。用 9×9 的 AᵀA 做对称特征分解最稳，
    // 且行数（点对数）无论 4 还是几百都得到 9×9 系统 —— 避开「瘦 SVD 的 V 不足 9 行」的坑。
    let ata = &a.transpose() * &a;
    let se = ata.symmetric_eigen();
    // 最小特征值对应的特征向量（eigenvectors 的列），即 Ah=0 的最小二乘解
    let mut min_i = 0usize;
    for i in 1..se.eigenvalues.len() {
        if se.eigenvalues[i] < se.eigenvalues[min_i] {
            min_i = i;
        }
    }
    let hv = se.eigenvectors.column(min_i);
    let hn = Matrix3::new(
        hv[0], hv[1], hv[2],
        hv[3], hv[4], hv[5],
        hv[6], hv[7], hv[8],
    );
    // 反归一化：H = Td^{-1} · Hn · Ts
    let h = td.try_inverse()? * hn * ts;
    if h[(2, 2)].abs() < 1e-12 {
        return None;
    }
    Some(h / h[(2, 2)]) // 固定尺度，令 h33 = 1
}

/// 用 H 把点 (x,y) 投影到目标平面。
fn apply_h(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
    let p = h * Vector3::new(x, y, 1.0);
    (p[0] / p[2], p[1] / p[2])
}

/// RANSAC：反复取 4 对点估 H、统计内点，留下内点最多的模型，再用全部内点重估。
/// 返回 (最终 H, 内点索引)。
fn ransac(
    pairs: &[([f64; 2], [f64; 2])],
    iters: usize,
    thresh: f64,
    rng: &mut Lcg,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let n = pairs.len();
    if n < 4 {
        return None;
    }
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..iters {
        // 随机取 4 个互异下标（LCG 走高位）
        let mut idx = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            let c = rng.below(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let sample: Vec<_> = idx.iter().map(|&i| pairs[i]).collect();
        if let Some(h) = estimate_homography(&sample) {
            let inl: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (sx, sy) = (pairs[i].0[0], pairs[i].0[1]);
                    let (px, py) = apply_h(&h, sx, sy);
                    let (dx, dy) = (pairs[i].1[0], pairs[i].1[1]);
                    ((px - dx).powi(2) + (py - dy).powi(2)).sqrt() < thresh
                })
                .collect();
            if inl.len() > best.len() {
                best = inl;
            }
        }
    }
    if best.len() < 4 {
        return None;
    }
    let inlier_pairs: Vec<_> = best.iter().map(|&i| pairs[i]).collect();
    let h = estimate_homography(&inlier_pairs)?;
    Some((h, best))
}

fn mean_reproj_err(h: &Matrix3<f64>, pairs: &[([f64; 2], [f64; 2])], idx: &[usize]) -> f64 {
    let mut s = 0.0;
    for &i in idx {
        let (px, py) = apply_h(h, pairs[i].0[0], pairs[i].0[1]);
        s += ((px - pairs[i].1[0]).powi(2) + (py - pairs[i].1[1]).powi(2)).sqrt();
    }
    s / idx.len() as f64
}

fn main() {
    println!("========== 第 106 章 纯 Rust 全景拼接 ==========");

    // ---------- 1. 合成场景 + 两张有重叠的视图 ----------
    let scene = make_scene();
    // 左视图：场景 [0,600) 的直接裁剪 → 左图坐标 = 场景坐标
    let left: RgbImage = scene.view(0, 0, WV, H).to_image();
    // 右视图：把场景「右半 + 轻微透视」映射到 600×420。
    // 右图 4 角对应的场景四边形（含平移 +360 与轻微梯形形变，模拟相机转了个小角度）：
    let scene_quad = [(360.0, 12.0), (958.0, 0.0), (952.0, 414.0), (366.0, 404.0)];
    let r_rect = [(0.0, 0.0), (WV as f32, 0.0), (WV as f32, H as f32), (0.0, H as f32)];
    // 用「场景→右图」的投影，从场景反采样生成右图
    let scene_to_r = Projection::from_control_points(scene_quad, r_rect).expect("控制点求投影失败");
    let mut right: RgbImage = ImageBuffer::from_pixel(WV, H, Rgb([245, 245, 245]));
    warp_into(&scene, scene_to_r, Interpolation::Bilinear, Border::Constant(Rgb([245, 245, 245])), &mut right);
    left.save("pano_left.png").ok();
    right.save("pano_right.png").ok();
    println!("[1] 合成 2 张 {WV}×{H} 视图（重叠区约 240px），已存 pano_left/right.png");

    // 真值单应（右图→左图/场景 平面）：由 4 对控制点自行 DLT 得到，用于最后核验精度
    let gt_pairs: Vec<([f64; 2], [f64; 2])> = r_rect
        .iter()
        .zip(scene_quad.iter())
        .map(|(r, s)| ([r.0 as f64, r.1 as f64], [s.0 as f64, s.1 as f64]))
        .collect();
    let h_gt = estimate_homography(&gt_pairs).unwrap();

    // ---------- 2. 特征检测 + 描述 + 匹配 ----------
    let pairs_tp = make_test_pairs(256, 12345); // 两图共用的确定性 BRIEF 测试对
    let gl = to_luma(&left);
    let gr = to_luma(&right);
    let desc_l = detect_and_describe(&gl, &pairs_tp, 500);
    let desc_r = detect_and_describe(&gr, &pairs_tp, 500);
    println!("[2] FAST 角点：左图 {} 个、右图 {} 个（各取响应最强的前 500）", desc_l.len(), desc_r.len());
    // 匹配（d1=右, d2=左）→ (.0=右点/src, .1=左点/dst)。seed 固定 → 可复现
    let matches = match_binary_descriptors(&desc_r, &desc_l, 40, Some(2024));
    let pairs: Vec<([f64; 2], [f64; 2])> = matches
        .iter()
        .map(|(r, l)| {
            ([r.corner.x as f64, r.corner.y as f64], [l.corner.x as f64, l.corner.y as f64])
        })
        .collect();
    println!("    BRIEF 匹配对（Hamming<40）：{} 对（含误匹配，交给 RANSAC 筛）", pairs.len());

    // ---------- 3. RANSAC + DLT 估单应 ----------
    let mut rng = Lcg::new(777);
    let (h_est, inliers) = ransac(&pairs, 3000, 3.0, &mut rng).expect("RANSAC 失败");
    let err = mean_reproj_err(&h_est, &pairs, &inliers);
    println!("[3] RANSAC 估单应（3000 次迭代，内点阈值 3px）：");
    println!(
        "    内点 {} / {} 对（内点率 {:.1}%），内点平均重投影误差 {:.3}px",
        inliers.len(),
        pairs.len(),
        inliers.len() as f64 / pairs.len() as f64 * 100.0,
        err
    );
    // 与真值单应对比：把右图 4 角分别用 估计H / 真值H 投影，比较落点
    let corner_err: f64 = r_rect
        .iter()
        .map(|c| {
            let (ex, ey) = apply_h(&h_est, c.0 as f64, c.1 as f64);
            let (gx, gy) = apply_h(&h_gt, c.0 as f64, c.1 as f64);
            ((ex - gx).powi(2) + (ey - gy).powi(2)).sqrt()
        })
        .sum::<f64>()
        / 4.0;
    println!("    估计单应 vs 真值单应：右图四角平均落点误差 {:.3}px（越小越准）", corner_err);

    // ---------- 4. 透视 warp + 羽化融合 → 全景 ----------
    let canvas_w = 970u32;
    // 4a. 把右图 warp 到画布（左图坐标系）。from_matrix 传「源→目标」矩阵，warp 内部自动求逆
    let hm = h_est;
    let h9: [f32; 9] = [
        hm[(0, 0)] as f32, hm[(0, 1)] as f32, hm[(0, 2)] as f32,
        hm[(1, 0)] as f32, hm[(1, 1)] as f32, hm[(1, 2)] as f32,
        hm[(2, 0)] as f32, hm[(2, 1)] as f32, hm[(2, 2)] as f32,
    ];
    let proj = Projection::from_matrix(h9).expect("from_matrix 失败");
    let mut r_on_canvas: RgbImage = ImageBuffer::from_pixel(canvas_w, H, Rgb([0, 0, 0]));
    warp_into(&right, proj, Interpolation::Bilinear, Border::Constant(Rgb([0, 0, 0])), &mut r_on_canvas);
    // 4b. 同样 warp 一张全白 mask，得到右图在画布上的真实覆盖范围
    let white_r: GrayImage = ImageBuffer::from_pixel(WV, H, Luma([255]));
    let mut mask_on_canvas: GrayImage = ImageBuffer::from_pixel(canvas_w, H, Luma([0]));
    warp_into(&white_r, proj, Interpolation::Bilinear, Border::Constant(Luma([0])), &mut mask_on_canvas);
    // 4c. 求重叠带的横向范围 [xo0, xo1]（右图覆盖的最左列 ~ 左图右边界）
    let mut xo0 = canvas_w;
    for y in 0..H {
        for x in 0..WV.min(canvas_w) {
            if mask_on_canvas.get_pixel(x, y)[0] > 127 {
                xo0 = xo0.min(x);
            }
        }
    }
    let xo1 = WV - 1; // 左图右边界
    // 4d. 逐像素融合：只左图→取左；只右图→取右；重叠→按横向位置线性羽化
    let mut pano: RgbImage = ImageBuffer::from_pixel(canvas_w, H, Rgb([0, 0, 0]));
    for y in 0..H {
        for x in 0..canvas_w {
            let lcov = x < WV;
            let rcov = mask_on_canvas.get_pixel(x, y)[0] > 127;
            let pixel = if lcov && rcov {
                let a = ((x as f64 - xo0 as f64) / (xo1 as f64 - xo0 as f64)).clamp(0.0, 1.0); // 右图权重
                let lp = left.get_pixel(x, y);
                let rp = r_on_canvas.get_pixel(x, y);
                Rgb([
                    ((1.0 - a) * lp[0] as f64 + a * rp[0] as f64).round() as u8,
                    ((1.0 - a) * lp[1] as f64 + a * rp[1] as f64).round() as u8,
                    ((1.0 - a) * lp[2] as f64 + a * rp[2] as f64).round() as u8,
                ])
            } else if lcov {
                *left.get_pixel(x, y)
            } else if rcov {
                *r_on_canvas.get_pixel(x, y)
            } else {
                Rgb([255, 255, 255])
            };
            pano.put_pixel(x, y, pixel);
        }
    }
    pano.save("pano_result.png").ok();
    println!("[4] 透视 warp + 羽化融合完成，输出全景 {canvas_w}×{H}，已存 pano_result.png");

    // ---------- 5. 匹配可视化（左|右 并排，内点连绿线） ----------
    let mut viz: RgbImage = ImageBuffer::from_pixel(WV * 2, H, Rgb([255, 255, 255]));
    for (x, y, p) in left.enumerate_pixels() {
        viz.put_pixel(x, y, *p);
    }
    for (x, y, p) in right.enumerate_pixels() {
        viz.put_pixel(x + WV, y, *p);
    }
    let inset: std::collections::HashSet<usize> = inliers.iter().copied().collect();
    for (i, (s, d)) in pairs.iter().enumerate() {
        if inset.contains(&i) {
            // s=右点（画在右半，需 +WV），d=左点（画在左半）
            draw_line_segment_mut(
                &mut viz,
                (d[0] as f32, d[1] as f32),
                (s[0] as f32 + WV as f32, s[1] as f32),
                Rgb([0, 180, 0]),
            );
        }
    }
    viz.save("pano_matches.png").ok();
    println!("[5] 匹配可视化（内点绿线）已存 pano_matches.png");

    println!("\n全部完成。产物：pano_left/right/result/matches.png");
}
