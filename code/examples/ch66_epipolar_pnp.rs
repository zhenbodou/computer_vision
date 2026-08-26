//! 第 66 章 对极几何与位姿估计：
//!  A) 合成两台已知相机对同一组 3D 点的 2D 对应，用「归一化八点法 + SVD」估计基础矩阵 F，
//!     验证对极约束 x2ᵀ F x1 ≈ 0；顺带由 E=Kᵀ F K 检查本质矩阵的奇异值结构。
//!  B) 合成 3D-2D 对应，用 DLT 求 PnP（相机位姿 R,t），打印恢复位姿/重投影误差 vs 真值。
//! 运行：cargo run --example ch66_epipolar_pnp
//! 依赖：nalgebra 0.35

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

// ---------- 自包含小 RNG（xorshift64*），造可复现的 3D 点，免引 rand ----------
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
    // [lo, hi) 均匀分布
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    }
}

// 投影：像素 (u,v) = K (R·M + t) 再做透视除法
fn project(k: &Matrix3<f64>, r: &Matrix3<f64>, t: &Vector3<f64>, m: &Vector3<f64>) -> (f64, f64) {
    let cam = r * m + t; // 相机坐标
    let p = k * cam;
    (p.x / p.z, p.y / p.z)
}

// Hartley 归一化：平移到质心、缩放到平均距离 √2；返回归一化点与归一化矩阵 T
fn normalize_2d(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in pts {
        mx += x;
        my += y;
    }
    mx /= n;
    my /= n;
    let mut md = 0.0;
    for &(x, y) in pts {
        md += ((x - mx).powi(2) + (y - my).powi(2)).sqrt();
    }
    md /= n;
    let s = 2.0_f64.sqrt() / md;
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|&(x, y)| (s * (x - mx), s * (y - my))).collect();
    (out, t)
}

// 解 A h = 0：取最小奇异值对应的右奇异向量（V^T 最后一行）
fn svd_nullspace(a: &DMatrix<f64>) -> DVector<f64> {
    let svd = a.clone().svd(true, true);
    let vt = svd.v_t.expect("已计算 V^T");
    let last = vt.nrows() - 1;
    vt.row(last).transpose()
}

// ---------- 归一化八点法估计基础矩阵 F ----------
fn eight_point(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> Matrix3<f64> {
    let (p1n, t1) = normalize_2d(pts1);
    let (p2n, t2) = normalize_2d(pts2);
    let n = pts1.len();
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let (u1, v1) = p1n[i];
        let (u2, v2) = p2n[i];
        // 对极约束 x2ᵀ F x1 = 0 展开成关于 F 九元的一行
        a.row_mut(i).copy_from_slice(&[
            u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0,
        ]);
    }
    let f = svd_nullspace(&a);
    let f_norm = Matrix3::new(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]);

    // 强制 F 秩为 2：SVD 后把最小奇异值置零再重构（F 理论上秩亏为 2）
    let svd = f_norm.svd(true, true);
    let mut s = svd.singular_values;
    s[2] = 0.0;
    let f_rank2 = svd.u.unwrap() * Matrix3::from_diagonal(&s) * svd.v_t.unwrap();

    // 反归一化：真实像素坐标下的 F = T2ᵀ · F_norm · T1
    t2.transpose() * f_rank2 * t1
}

// ---------- PnP：已知 K 与 3D-2D 对应，DLT 求相机位姿 (R,t) ----------
fn pnp_dlt(
    world: &[Vector3<f64>],
    img: &[(f64, f64)],
    k: &Matrix3<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    let k_inv = k.try_inverse().unwrap();
    let n = world.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        // 像素 -> 归一化相机坐标 (a,b,1)，使 DLT 直接解 [R|t]（K 已剥离）
        let xn = k_inv * Vector3::new(img[i].0, img[i].1, 1.0);
        let (a_, b_) = (xn.x / xn.z, xn.y / xn.z);
        let m = world[i];
        let (x, y, z) = (m.x, m.y, m.z);
        // 行 a：-Mh 放在 P 第 0 行块，a·Mh 放在第 2 行块
        a.row_mut(2 * i)
            .copy_from_slice(&[-x, -y, -z, -1.0, 0.0, 0.0, 0.0, 0.0, a_ * x, a_ * y, a_ * z, a_]);
        a.row_mut(2 * i + 1)
            .copy_from_slice(&[0.0, 0.0, 0.0, 0.0, -x, -y, -z, -1.0, b_ * x, b_ * y, b_ * z, b_]);
    }
    let mut p = svd_nullspace(&a); // 12 元，是 P=[R|t] 的行优先展开（差一个尺度与符号）

    // 定符号：投影深度 P2·Mh 应为正（点在相机前方）
    let depth_sum: f64 = world
        .iter()
        .map(|m| p[8] * m.x + p[9] * m.y + p[10] * m.z + p[11])
        .sum();
    if depth_sum < 0.0 {
        p = -p;
    }

    // 左 3×3 块 ≈ λR，用 SVD 做正交化（求最接近的旋转），并定出尺度 λ
    let m3 = Matrix3::new(p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
    let svd = m3.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        // 保证是「右手」旋转（det=+1）：翻转最后一列符号
        let d = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
        r = u * d * vt;
    }
    let lambda = svd.singular_values.sum() / 3.0; // λR 的三个奇异值都≈λ
    let t = Vector3::new(p[3], p[7], p[11]) / lambda;
    (r, t)
}

fn main() {
    // ==================== A. 八点法估计基础矩阵 F ====================
    let k = Matrix3::new(600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0);
    // 相机1 = 世界原点 (R=I, t=0)；相机2 有一段基线 + 小旋转
    let r1 = Matrix3::<f64>::identity();
    let t1 = Vector3::zeros();
    let r2 = *Rotation3::from_euler_angles(0.05, 0.10, -0.05).matrix();
    let t2 = Vector3::new(-1.0, 0.05, 0.10);

    // 造 12 个都在两台相机前方的 3D 点
    let mut rng = Rng::new(20260827);
    let mut pts3d = Vec::new();
    while pts3d.len() < 12 {
        let m = Vector3::new(rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(4.0, 8.0));
        if (r2 * m + t2).z > 0.1 {
            pts3d.push(m);
        }
    }
    let pts1: Vec<(f64, f64)> = pts3d.iter().map(|m| project(&k, &r1, &t1, m)).collect();
    let pts2: Vec<(f64, f64)> = pts3d.iter().map(|m| project(&k, &r2, &t2, m)).collect();

    let f = eight_point(&pts1, &pts2);
    println!("估计的基础矩阵 F（已按 F22=1 归一化）：");
    let fn_ = f / f[(2, 2)];
    for row in 0..3 {
        println!("  [{:10.5} {:10.5} {:10.5}]", fn_[(row, 0)], fn_[(row, 1)], fn_[(row, 2)]);
    }

    // 验证对极约束 x2ᵀ F x1 ≈ 0
    let mut max_res = 0.0_f64;
    let mut sse = 0.0;
    for i in 0..pts3d.len() {
        let x1 = Vector3::new(pts1[i].0, pts1[i].1, 1.0);
        let x2 = Vector3::new(pts2[i].0, pts2[i].1, 1.0);
        let res = (x2.transpose() * f * x1)[(0, 0)].abs();
        max_res = max_res.max(res);
        sse += res * res;
    }
    println!(
        "对极约束残差 |x2ᵀ F x1|：最大 {:.3e}，RMS {:.3e}（应≈0）",
        max_res,
        (sse / pts3d.len() as f64).sqrt()
    );

    // 本质矩阵 E = K2ᵀ F K1（此处两相机同 K）；理论上奇异值形如 (σ, σ, 0)
    let e = k.transpose() * f * k;
    let es = e.svd(false, false).singular_values;
    println!(
        "本质矩阵 E 的奇异值（除以最大值后）≈ ({:.4}, {:.4}, {:.4})，理论应为 (1, 1, 0)",
        es[0] / es[0],
        es[1] / es[0],
        es[2] / es[0]
    );

    // ==================== B. PnP：DLT 求相机位姿 ====================
    println!("\n==== PnP：由 3D-2D 对应恢复相机位姿 ====");
    let r_true = *Rotation3::from_euler_angles(0.15, -0.20, 0.10).matrix();
    let t_true = Vector3::new(0.40, -0.30, 7.0);
    let mut rng2 = Rng::new(12345);
    let mut world = Vec::new();
    for _ in 0..12 {
        world.push(Vector3::new(
            rng2.range(-1.5, 1.5),
            rng2.range(-1.5, 1.5),
            rng2.range(-1.5, 1.5),
        ));
    }
    let img: Vec<(f64, f64)> = world.iter().map(|m| project(&k, &r_true, &t_true, m)).collect();

    let (r_est, t_est) = pnp_dlt(&world, &img, &k);

    // 旋转误差：R_estᵀ R_true 越接近单位阵越好，用夹角度量
    let rt = r_est.transpose() * r_true;
    let cos = ((rt.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    println!("旋转恢复误差 = {:.3e} 度", cos.acos().to_degrees());
    println!("平移 t：估计 [{:.4}, {:.4}, {:.4}]", t_est.x, t_est.y, t_est.z);
    println!("       真值 [{:.4}, {:.4}, {:.4}]", t_true.x, t_true.y, t_true.z);

    // 重投影误差：用恢复的 (R,t) 重新投影，与观测像素比
    let mut sse = 0.0;
    for i in 0..world.len() {
        let (u, v) = project(&k, &r_est, &t_est, &world[i]);
        sse += (u - img[i].0).powi(2) + (v - img[i].1).powi(2);
    }
    let rms = (sse / world.len() as f64).sqrt();
    println!("PnP 重投影 RMS 误差 = {:.3e} px（应≈0）", rms);
    assert!(rms < 1e-6, "PnP 误差过大");
    println!("恢复的位姿与真值一致 ✓");
}
