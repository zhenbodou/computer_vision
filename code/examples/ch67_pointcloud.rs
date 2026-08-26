//! 第 67 章 点云处理入门
//! 运行：cargo run --example ch67_pointcloud
//! 依赖：nalgebra 0.35
//!
//! 演示：合成“带噪声的平面 + 离群点”点云；质心、AABB 包围盒、
//! 体素下采样（哈希网格）、暴力最近邻、PCA 平面拟合、RANSAC 抗噪平面拟合。

use nalgebra::{DMatrix, Vector3};
use std::collections::HashMap;

type P3 = Vector3<f64>; // 点=3D 向量：支持点差、点积、叉积、求模

// ---- 自包含小 RNG（xorshift64*），可复现、无需 rand ----
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
    fn range_f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }
    fn normal(&mut self) -> f64 {
        // Box-Muller 标准正态
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn range_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

const TRUE_A: f64 = 0.30; // 平面真值 z = a·x + b·y + c
const TRUE_B: f64 = -0.20;
const TRUE_C: f64 = 1.50;

fn make_cloud(seed: u64) -> (Vec<P3>, usize) {
    let mut rng = Rng::new(seed);
    let mut pts = Vec::new();
    let n_in = 800; // 平面内点：贴着平面 + 2cm 噪声
    for _ in 0..n_in {
        let x = rng.range_f(-2.0, 2.0);
        let y = rng.range_f(-2.0, 2.0);
        let z = TRUE_A * x + TRUE_B * y + TRUE_C + rng.normal() * 0.02;
        pts.push(P3::new(x, y, z));
    }
    for _ in 0..120 {
        // 离群点：整个包围盒里乱撒
        pts.push(P3::new(
            rng.range_f(-2.0, 2.0),
            rng.range_f(-2.0, 2.0),
            rng.range_f(0.0, 3.0),
        ));
    }
    (pts, n_in)
}

fn centroid(pts: &[P3]) -> P3 {
    let mut s = P3::zeros();
    for p in pts {
        s += *p;
    }
    s / pts.len() as f64
}

fn aabb(pts: &[P3]) -> (P3, P3) {
    // 逐维取 min/max
    let (mut lo, mut hi) = (pts[0], pts[0]);
    for p in pts {
        for k in 0..3 {
            if p[k] < lo[k] {
                lo[k] = p[k];
            }
            if p[k] > hi[k] {
                hi[k] = p[k];
            }
        }
    }
    (lo, hi)
}

// 体素下采样：坐标除以体素尺寸取整做 key，同格点取质心，一格留一点
fn voxel_downsample(pts: &[P3], voxel: f64) -> Vec<P3> {
    let mut cells: HashMap<(i64, i64, i64), (P3, usize)> = HashMap::new();
    for p in pts {
        let key = (
            (p.x / voxel).floor() as i64,
            (p.y / voxel).floor() as i64,
            (p.z / voxel).floor() as i64,
        );
        let e = cells.entry(key).or_insert((P3::zeros(), 0));
        e.0 += *p; // 累加坐标
        e.1 += 1; // 计数
    }
    cells.values().map(|(sum, cnt)| sum / *cnt as f64).collect()
}

// 暴力最近邻：遍历所有点，O(N)
fn nearest(pts: &[P3], q: &P3) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (i, p) in pts.iter().enumerate() {
        let d = (p - q).norm();
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

// PCA 平面拟合：中心化 -> SVD -> 最小奇异向量即法向；返回 (法向, 质心)
fn fit_plane_pca(pts: &[P3]) -> (P3, P3) {
    let c = centroid(pts);
    let mut data = Vec::with_capacity(pts.len() * 3);
    for p in pts {
        let d = p - c;
        data.push(d.x);
        data.push(d.y);
        data.push(d.z);
    }
    let m = DMatrix::from_row_slice(pts.len(), 3, &data); // (N×3)
    let vt = m.svd(true, true).v_t.unwrap(); // 奇异值降序
    let normal = P3::new(vt[(2, 0)], vt[(2, 1)], vt[(2, 2)]).normalize(); // 最后一行
    (normal, c)
}

// (法向 n, 平面上一点 p0) -> 显式 z = a·x + b·y + c，便于和真值比
fn to_explicit(n: &P3, p0: &P3) -> (f64, f64, f64) {
    let d = n.dot(p0); // 平面 n·p = d
    (-n.x / n.z, -n.y / n.z, d / n.z)
}

// RANSAC 平面拟合：随机 3 点定平面、数内点、留最优，再用内点精修
fn ransac_plane(pts: &[P3], thresh: f64, iters: usize, rng: &mut Rng) -> (P3, P3, usize) {
    let n = pts.len();
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..iters {
        let (i, j, k) = (rng.range_usize(n), rng.range_usize(n), rng.range_usize(n));
        if i == j || j == k || i == k {
            continue;
        }
        let normal = (pts[j] - pts[i]).cross(&(pts[k] - pts[i]));
        if normal.norm() < 1e-9 {
            continue; // 三点共线，退化
        }
        let normal = normal.normalize();
        let d = normal.dot(&pts[i]);
        let inliers: Vec<usize> = (0..n)
            .filter(|&t| (normal.dot(&pts[t]) - d).abs() < thresh)
            .collect();
        if inliers.len() > best.len() {
            best = inliers;
        }
    }
    let inlier_pts: Vec<P3> = best.iter().map(|&t| pts[t]).collect();
    let (normal, c) = fit_plane_pca(&inlier_pts); // 用内点精修
    (normal, c, best.len())
}

fn main() {
    let (cloud, n_in) = make_cloud(20250827);
    println!(
        "合成点云：共 {} 点（内点 {} + 离群 {}）",
        cloud.len(),
        n_in,
        cloud.len() - n_in
    );

    let c = centroid(&cloud);
    let (lo, hi) = aabb(&cloud);
    println!("质心 = ({:.3}, {:.3}, {:.3})", c.x, c.y, c.z);
    println!(
        "AABB 尺寸 = ({:.2}, {:.2}, {:.2})",
        hi.x - lo.x,
        hi.y - lo.y,
        hi.z - lo.z
    );

    let down = voxel_downsample(&cloud, 0.5);
    println!("体素下采样(0.5m): {} -> {} 点", cloud.len(), down.len());

    let q = P3::new(0.5, 0.5, TRUE_A * 0.5 + TRUE_B * 0.5 + TRUE_C);
    let (idx, dist) = nearest(&cloud, &q);
    println!("最近邻: #{} 距离 {:.4} m", idx, dist);

    let (pn, pp) = fit_plane_pca(&cloud);
    let (pa, pb, pc) = to_explicit(&pn, &pp);
    println!("PCA 直接拟合(含离群): z = {:.3}x + {:.3}y + {:.3}", pa, pb, pc);

    let mut rng = Rng::new(999);
    let (rn, rp, n_inl) = ransac_plane(&cloud, 0.06, 300, &mut rng);
    let (ra, rb, rc) = to_explicit(&rn, &rp);
    println!(
        "RANSAC 拟合: z = {:.3}x + {:.3}y + {:.3} (内点 {})",
        ra, rb, rc, n_inl
    );
    println!(
        "平面真值:    z = {:.3}x + {:.3}y + {:.3}",
        TRUE_A, TRUE_B, TRUE_C
    );
}
