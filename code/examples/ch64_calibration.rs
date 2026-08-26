//! 第 64 章 相机标定：合成棋盘格 + 已知 K/姿态 -> 投影出 2D 角点 ->
//! 用「归一化 DLT + nalgebra SVD」反解每张图的单应 H，
//! 再用张正友法的闭式解从多张 H 反推内参 K。
//! 运行：cargo run --example ch64_calibration
//! 依赖：nalgebra 0.35
//!
//! 自验证思路：真值 K/(R,t) 已知 -> 正向投影得角点 -> 反解 H 与 K ->
//! 打印「反解值 vs 真值」与重投影误差，误差应≈0（浮点精度级）。

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

// ---------- 工具：对一组 2D 点做 Hartley 归一化（平移到质心、缩放到平均距离 √2）----------
// 返回归一化后的点，以及归一化矩阵 T（满足 归一化点 = T * 原始齐次点）。
fn normalize_2d(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in pts {
        mx += x;
        my += y;
    }
    mx /= n;
    my /= n;
    let mut mean_d = 0.0;
    for &(x, y) in pts {
        mean_d += ((x - mx).powi(2) + (y - my).powi(2)).sqrt();
    }
    mean_d /= n;
    let s = 2.0_f64.sqrt() / mean_d; // 缩放使平均距离为 √2
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|&(x, y)| (s * (x - mx), s * (y - my))).collect();
    (out, t)
}

// ---------- 工具：解齐次方程 A h = 0，取最小奇异值对应的右奇异向量 ----------
// nalgebra 的奇异值按降序排列，故最小者在最后 -> V^T 的最后一行。
fn svd_nullspace(a: &DMatrix<f64>) -> DVector<f64> {
    let svd = a.clone().svd(true, true);
    let vt = svd.v_t.expect("已用 svd(true,true) 计算 V^T");
    let last = vt.nrows() - 1;
    vt.row(last).transpose() // 转成列向量返回
}

// ---------- 归一化 DLT 从平面点对 (X,Y)->(u,v) 估计单应 H ----------
fn dlt_homography(world: &[(f64, f64)], img: &[(f64, f64)]) -> Matrix3<f64> {
    let (wn, tw) = normalize_2d(world);
    let (imn, ti) = normalize_2d(img);
    let n = world.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for k in 0..n {
        let (x, y) = wn[k];
        let (u, v) = imn[k];
        // 每对点由叉乘 x_img × (H x_world) = 0 得两行独立方程：
        a[(2 * k, 3)] = -x;
        a[(2 * k, 4)] = -y;
        a[(2 * k, 5)] = -1.0;
        a[(2 * k, 6)] = v * x;
        a[(2 * k, 7)] = v * y;
        a[(2 * k, 8)] = v;
        a[(2 * k + 1, 0)] = x;
        a[(2 * k + 1, 1)] = y;
        a[(2 * k + 1, 2)] = 1.0;
        a[(2 * k + 1, 6)] = -u * x;
        a[(2 * k + 1, 7)] = -u * y;
        a[(2 * k + 1, 8)] = -u;
    }
    let h = svd_nullspace(&a);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    // 反归一化：真实坐标下的 H = T_img^{-1} * H_norm * T_world
    ti.try_inverse().unwrap() * hn * tw
}

// 用 H 把平面点 (X,Y) 投影为像素 (u,v)（透视除法）
fn apply_h(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
    let p = h * Vector3::new(x, y, 1.0);
    (p.x / p.z, p.y / p.z)
}

// ---------- 张正友法：由单应列构造约束向量 v_ij（对内参对偶 B 的线性约束）----------
fn v_ij(h: &Matrix3<f64>, i: usize, j: usize) -> [f64; 6] {
    // h 的第 i、j 列
    let (a0, a1, a2) = (h[(0, i)], h[(1, i)], h[(2, i)]);
    let (b0, b1, b2) = (h[(0, j)], h[(1, j)], h[(2, j)]);
    [
        a0 * b0,
        a0 * b1 + a1 * b0,
        a1 * b1,
        a2 * b0 + a0 * b2,
        a2 * b1 + a1 * b2,
        a2 * b2,
    ]
}

fn main() {
    // ==================== ① 定义真值：内参 K 与三张图的姿态 (R,t) ====================
    let (fx, fy, cx, cy) = (520.0_f64, 525.0, 325.0, 245.0); // 真实内参（skew=0）
    let k_true = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // 棋盘格：9×6 个内角点，方格边长 1.0（单位无所谓，不影响 K）。把棋盘中心挪到原点。
    let (cols, rows) = (9usize, 6usize);
    let mut board: Vec<(f64, f64)> = Vec::new();
    for j in 0..rows {
        for i in 0..cols {
            board.push((i as f64 - 4.0, j as f64 - 2.5)); // 中心化
        }
    }

    // 三张图：不同角度/位置观察棋盘（欧拉角单位弧度；平移把棋盘放到相机前方 Z>0）
    let poses = [
        (Rotation3::from_euler_angles(0.10, -0.15, 0.05), Vector3::new(-1.0, -0.5, 12.0)),
        (Rotation3::from_euler_angles(-0.20, 0.25, -0.10), Vector3::new(0.5, 0.8, 11.0)),
        (Rotation3::from_euler_angles(0.28, 0.12, 0.20), Vector3::new(1.2, -0.6, 13.0)),
    ];

    let mut h_estimated: Vec<Matrix3<f64>> = Vec::new();

    println!("真值内参 K：fx={fx}, fy={fy}, cx={cx}, cy={cy}\n");

    // ==================== ② 每张图：投影 -> 估计 H -> 对比真值 + 重投影误差 ====================
    for (idx, (rot, t)) in poses.iter().enumerate() {
        let r = rot.matrix();
        // 平面棋盘 Z=0 时，投影退化为单应：H_true = K [r1  r2  t]
        let rt = Matrix3::new(
            r[(0, 0)], r[(0, 1)], t[0],
            r[(1, 0)], r[(1, 1)], t[1],
            r[(2, 0)], r[(2, 1)], t[2],
        );
        let h_true = k_true * rt;

        // 正向投影得到 2D 角点（像素）
        let img_pts: Vec<(f64, f64)> = board.iter().map(|&(x, y)| apply_h(&h_true, x, y)).collect();

        // 归一化 DLT + SVD 反解 H
        let h_est = dlt_homography(&board, &img_pts);

        // 重投影误差：用 H_est 把棋盘点再投影，与观测像素比较（RMS）
        let mut sse = 0.0;
        for (bp, ip) in board.iter().zip(&img_pts) {
            let (u, v) = apply_h(&h_est, bp.0, bp.1);
            sse += (u - ip.0).powi(2) + (v - ip.1).powi(2);
        }
        let rms = (sse / img_pts.len() as f64).sqrt();

        // H 是齐次量（差一个尺度），比较前都除以各自的 (2,2) 元素归一化
        let h_est_n = h_est / h_est[(2, 2)];
        let h_true_n = h_true / h_true[(2, 2)];

        if idx == 0 {
            println!("视图 #1 单应矩阵对比（已按 H22=1 归一化）：");
            for row in 0..3 {
                println!(
                    "  估计 [{:9.4} {:9.4} {:9.4}]   真值 [{:9.4} {:9.4} {:9.4}]",
                    h_est_n[(row, 0)], h_est_n[(row, 1)], h_est_n[(row, 2)],
                    h_true_n[(row, 0)], h_true_n[(row, 1)], h_true_n[(row, 2)],
                );
            }
        }
        println!("视图 #{}：{} 个角点，H 重投影 RMS 误差 = {:.3e} px", idx + 1, board.len(), rms);
        h_estimated.push(h_est);
    }

    // ==================== ③ 张正友法：多张 H 叠出约束，反解内参 K ====================
    // 每张 H 给两条约束：v12·b=0 与 (v11-v22)·b=0，其中 b 是对偶矩阵 B=K^{-T}K^{-1} 的 6 个独立元。
    let mut vmat = DMatrix::<f64>::zeros(2 * h_estimated.len(), 6);
    for (n, h) in h_estimated.iter().enumerate() {
        let v01 = v_ij(h, 0, 1);
        let v00 = v_ij(h, 0, 0);
        let v11 = v_ij(h, 1, 1);
        for c in 0..6 {
            vmat[(2 * n, c)] = v01[c];
            vmat[(2 * n + 1, c)] = v00[c] - v11[c];
        }
    }
    let mut b = svd_nullspace(&vmat); // b = [B11,B12,B22,B13,B23,B33]
    if b[0] < 0.0 {
        b = -b; // B 应正定，若符号反了取负
    }
    let (b11, b12, b22, b13, b23, b33) = (b[0], b[1], b[2], b[3], b[4], b[5]);

    // 张氏闭式解：从 B 提取内参
    let v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 * b12);
    let lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    let alpha = (lambda / b11).sqrt();
    let beta = (lambda * b11 / (b11 * b22 - b12 * b12)).sqrt();
    let gamma = -b12 * alpha * alpha * beta / lambda;
    let u0 = gamma * v0 / beta - b13 * alpha * alpha / lambda;

    println!("\n==== 张正友法反解内参 K（来自 {} 张图的单应）====", h_estimated.len());
    println!("  fx: 估计 {:.4}  vs 真值 {:.1}", alpha, fx);
    println!("  fy: 估计 {:.4}  vs 真值 {:.1}", beta, fy);
    println!("  cx: 估计 {:.4}  vs 真值 {:.1}", u0, cx);
    println!("  cy: 估计 {:.4}  vs 真值 {:.1}", v0, cy);
    println!("  skew(γ): 估计 {:.6}  vs 真值 0", gamma);

    let err = ((alpha - fx).powi(2) + (beta - fy).powi(2) + (u0 - cx).powi(2) + (v0 - cy).powi(2)).sqrt();
    println!("  内参四参数总误差(欧氏) = {:.3e}", err);
    assert!(err < 1e-3, "内参反解误差过大，检查实现");
    println!("\n反解 K 与真值一致（误差在浮点精度级）✓");
}
