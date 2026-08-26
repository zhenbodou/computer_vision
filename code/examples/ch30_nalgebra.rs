//! 第 30 章 nalgebra：线性代数与几何变换
//! 运行：cargo run --example ch30_nalgebra
//! 依赖：nalgebra 0.35.0
//!
//! 演示：静态小矩阵 Matrix3/Vector3 的乘法、转置、求逆、行列式；
//! 用 LU 解方阵线性方程组；用 SVD 最小二乘解“超定方程组”（呼应第26章单应求解）；
//! 用 3×3 齐次矩阵把一个 2D 点做“旋转 + 平移”。

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

fn main() {
    // ---------- ① 创建、矩阵×向量、矩阵×矩阵、转置 ----------
    // Matrix3::new 按“行优先”写入 9 个数（但内部是列优先存储，见文末的坑）
    let a = Matrix3::new(
        2.0, 1.0, 1.0, //
        1.0, 3.0, 2.0, //
        1.0, 0.0, 0.0,
    );
    let v = Vector3::new(1.0, 2.0, 3.0);

    let av = a * v; // 矩阵 × 向量 -> Vector3
    let aa = a * a; // 矩阵 × 矩阵 -> Matrix3
    let at = a.transpose(); // 转置
    println!("A * v = {}", av.transpose()); // 转成行向量方便一行打印
    println!("A * A =\n{aa}");
    println!("Aᵀ =\n{at}");

    // ---------- ② 行列式与求逆（try_inverse 返回 Option）----------
    let det = a.determinant();
    println!("det(A) = {det}");
    match a.try_inverse() {
        Some(inv) => {
            let should_be_i = a * inv; // A * A⁻¹ 应约等于单位阵
            println!("A⁻¹ =\n{inv}");
            println!("A * A⁻¹ =\n{should_be_i}");
        }
        None => println!("A 不可逆（奇异矩阵）"),
    }
    // 一个奇异矩阵：第 3 行是前两行的组合，行列式为 0，求逆得到 None
    let singular = Matrix3::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0);
    println!(
        "奇异矩阵 det = {}, try_inverse().is_none() = {}",
        singular.determinant(),
        singular.try_inverse().is_none()
    );

    // ---------- ③ 解方阵线性方程组 A x = b（LU 分解）----------
    // 取 x = (1,2,3)，反推 b = A x，再解回来验证
    let b = a * Vector3::new(1.0, 2.0, 3.0);
    let x = a.lu().solve(&b).expect("A 可逆，必有唯一解");
    println!("解 A x = b 得 x = {}", x.transpose()); // 应为 (1, 2, 3)

    // ---------- ④ SVD 最小二乘：解“超定方程组”（方程比未知数多）----------
    // 场景：一堆带噪声的点 (x_i, y_i)，拟合直线 y = m·x + c。
    // 把每个点写成一行方程 [x_i, 1]·[m, c]ᵀ = y_i，堆成 A(n×2) · [m,c]ᵀ = b(n×1)。
    // n 个方程、2 个未知数 -> 超定，无精确解，SVD 求“误差最小”的最优解。
    // 相机标定、单应矩阵（第26章）估计，本质都是这样把一堆点对堆成超定方程再解。
    let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    // 真值 m=2, c=1，即 y=2x+1；再叠加一点点固定“噪声”，让它无精确解
    let noise = [0.20_f64, -0.10, 0.15, -0.20, 0.10, -0.15];
    let ys: Vec<f64> = xs.iter().zip(noise).map(|(x, e)| 2.0 * x + 1.0 + e).collect();

    let n = xs.len();
    let mut a_rows = Vec::with_capacity(n * 2);
    for &xi in &xs {
        a_rows.push(xi); // 第 i 行是 [x_i, 1]
        a_rows.push(1.0);
    }
    let big_a = DMatrix::from_row_slice(n, 2, &a_rows); // 动态尺寸矩阵 (n×2)
    let big_b = DVector::from_column_slice(&ys); // 动态向量 (n)

    // SVD 求解：solve 内部用伪逆思想给出最小二乘解；eps 用来滤掉极小奇异值
    let svd = big_a.svd(true, true);
    let sol = svd.solve(&big_b, 1e-12).expect("SVD solve 失败");
    println!(
        "\n最小二乘拟合 y = m·x + c 得到 m = {:.4}, c = {:.4}（真值 m=2, c=1）",
        sol[0], sol[1]
    );

    // ---------- ⑤ 用 3×3 齐次矩阵做 2D 旋转 + 平移 ----------
    // 2D 点 (x,y) 写成齐次坐标 (x, y, 1)。
    // 旋转矩阵 R（绕原点逆时针 θ）与平移矩阵 T 都是 3×3，
    // “先旋转再平移”对应组合矩阵 M = T · R，作用在点上：p_new = M · p。
    let theta = std::f64::consts::FRAC_PI_2; // 90°
    let (c, s) = (theta.cos(), theta.sin());
    let r = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0); // 旋转
    let (tx, ty) = (2.0, 1.0);
    let t = Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0); // 平移
    let m_transform = t * r; // 组合：先转后移

    let p = Vector3::new(1.0, 0.0, 1.0); // 点 (1,0) 的齐次坐标
    let p_new = m_transform * p;
    println!(
        "\n点 (1,0) 旋转90°+平移(2,1) 后 = ({:.3}, {:.3})",
        p_new.x, p_new.y
    );
    // 手算：(1,0) 逆时针转 90° -> (0,1)，再平移 (2,1) -> (2,2)
    assert!((p_new.x - 2.0).abs() < 1e-9 && (p_new.y - 2.0).abs() < 1e-9);
    println!("与手算 (2,2) 一致 ✓");
}
