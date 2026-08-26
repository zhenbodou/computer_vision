//! 第 68 章 运动恢复结构 SfM 与视觉 SLAM 概览
//! 运行：cargo run --example ch68_sfm_slam
//! 依赖：nalgebra 0.35
//!
//! 核心实验：线性三角化（DLT）。已知两个相机投影矩阵 P1、P2 与它们对同一 3D 点的
//! 2D 观测，构造 A 矩阵，用 SVD 取“最小奇异值对应的右奇异向量”反解出 3D 点坐标。

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Vector2, Vector3, Vector4};

// 自包含小 RNG（xorshift64*），仅用于给观测加噪声，避免引入 rand
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
        // 标准正态
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn intrinsics() -> Matrix3<f64> {
    // 内参 K：焦距 800，主点 (320,240)
    Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0)
}

fn rot_y(theta: f64) -> Matrix3<f64> {
    // 绕 Y 轴旋转
    let (c, s) = (theta.cos(), theta.sin());
    Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
}

// 组装投影矩阵 P = K [R | t]，t = -R·C（C 为相机光心在世界系中的位置）
fn projection(k: &Matrix3<f64>, r: &Matrix3<f64>, center: &Vector3<f64>) -> Matrix3x4<f64> {
    let t = -r * center;
    let rt = Matrix3x4::new(
        r[(0, 0)], r[(0, 1)], r[(0, 2)], t[0], //
        r[(1, 0)], r[(1, 1)], r[(1, 2)], t[1], //
        r[(2, 0)], r[(2, 1)], r[(2, 2)], t[2],
    );
    k * rt // (3×3)·(3×4) = (3×4)
}

fn project(p: &Matrix3x4<f64>, x: &Vector3<f64>) -> Vector2<f64> {
    // 3D 点 -> 像素
    let hom = p * Vector4::new(x.x, x.y, x.z, 1.0);
    Vector2::new(hom.x / hom.z, hom.y / hom.z) // 透视除法
}

// 往 A 填一个视图的两行： u·P3 - P1 = 0 , v·P3 - P2 = 0
fn fill_rows(a: &mut DMatrix<f64>, row: usize, p: &Matrix3x4<f64>, uv: &Vector2<f64>) {
    for c in 0..4 {
        a[(row, c)] = uv.x * p[(2, c)] - p[(0, c)];
        a[(row + 1, c)] = uv.y * p[(2, c)] - p[(1, c)];
    }
}

// 线性三角化：两视图 -> 4 行方程，解 A·X=0 取最小奇异向量
fn triangulate(
    p1: &Matrix3x4<f64>,
    uv1: &Vector2<f64>,
    p2: &Matrix3x4<f64>,
    uv2: &Vector2<f64>,
) -> Vector3<f64> {
    let mut a = DMatrix::zeros(4, 4);
    fill_rows(&mut a, 0, p1, uv1);
    fill_rows(&mut a, 2, p2, uv2);
    let vt = a.svd(true, true).v_t.unwrap(); // v_t 最后一行 = 齐次解
    let xh = Vector4::new(vt[(3, 0)], vt[(3, 1)], vt[(3, 2)], vt[(3, 3)]);
    Vector3::new(xh.x / xh.w, xh.y / xh.w, xh.z / xh.w) // 齐次 -> 非齐次
}

fn main() {
    let k = intrinsics();
    let p1 = projection(&k, &Matrix3::identity(), &Vector3::zeros()); // 相机1 在原点
    let c2 = Vector3::new(1.0, 0.0, 0.0); // 相机2 向右 1m
    let p2 = projection(&k, &rot_y(6.0_f64.to_radians()), &c2); // 并偏转 6°

    let truth = [
        // 4 个已知 3D 点，都在两相机前方
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(0.5, 0.3, 6.0),
        Vector3::new(-0.4, 0.2, 5.5),
        Vector3::new(0.2, -0.3, 4.5),
    ];

    println!("==== 无噪声三角化 ====");
    let mut max_err = 0.0_f64;
    for (i, x) in truth.iter().enumerate() {
        let (uv1, uv2) = (project(&p1, x), project(&p2, x)); // 投影得观测
        let rec = triangulate(&p1, &uv1, &p2, &uv2); // 再反解
        let err = (rec - x).norm();
        max_err = max_err.max(err);
        println!(
            "点{i}: 真值({:+.3},{:+.3},{:+.3}) 恢复({:+.3},{:+.3},{:+.3}) 误差={err:.2e}",
            x.x, x.y, x.z, rec.x, rec.y, rec.z
        );
    }
    println!("最大误差 = {max_err:.2e} m（接近机器精度）");

    println!("\n==== 加 0.5px 观测噪声 ====");
    let mut rng = Rng::new(2024);
    let (sigma, mut sum) = (0.5, 0.0_f64);
    for x in truth.iter() {
        let (mut uv1, mut uv2) = (project(&p1, x), project(&p2, x));
        uv1.x += rng.normal() * sigma;
        uv1.y += rng.normal() * sigma; // 每个坐标加噪
        uv2.x += rng.normal() * sigma;
        uv2.y += rng.normal() * sigma;
        sum += (triangulate(&p1, &uv1, &p2, &uv2) - x).norm();
    }
    println!(
        "平均 3D 误差 = {:.4} m（{:.1} mm）",
        sum / 4.0,
        1000.0 * sum / 4.0
    );
}
