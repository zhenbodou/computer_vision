# 第 66 章 对极几何、PnP 与鲁棒位姿估计

> **本章导读**
> - 理解**对极约束**：两个视角看同一场景，匹配点被牢牢约束在一条**极线**上，写成一个漂亮的等式 \\( x_2^\top F x_1 = 0 \\)。
> - 掌握**基础矩阵 F** 与**本质矩阵 E** 的关系，动手用**归一化八点法 + SVD** 估计 F 并验证对极约束。
> - 动手用 **DLT** 解 **PnP**（已知 3D 点及其像素，反求相机位姿），打印恢复的 \\( R, t \\) 与重投影误差 vs 真值。
> - 认清这套几何的**退化陷阱**与**尺度不确定性**，理解为什么真实系统必须配 RANSAC 和非线性精修。

## 场景引入：随手两张照片，能算出相机怎么动的吗

你举着手机绕着一尊雕像拍了两张照片，位置、角度都变了。人眼一看就知道"相机往右挪了、稍微转了个角度"。计算机能不能只凭这两张图里的一堆**匹配点**，反推出两次拍摄之间相机的**相对位姿**（转了多少、移了多少）？

能。这正是**运动恢复结构（SfM）**和**视觉 SLAM** 的地基：从多视角图像恢复相机轨迹与三维结构（[第 68 章](ch68-sfm-slam.md)会展开）。而这一切的数学核心，就是本章的**对极几何（epipolar geometry）**。它还有一个同样重要的近亲——**PnP**：当你已经有了物体的三维模型、又在图里认出了它的若干点，就能反算相机相对物体的位姿（AR 把虚拟物体"钉"在现实桌面上，靠的就是它）。

## 原理：极线、基础矩阵与位姿

### 对极约束：点被锁在极线上

同一个三维点 \\( P \\) 被两台相机看到，在左图成像 \\( x_1 \\)、右图成像 \\( x_2 \\)。关键事实是：**\\( P \\)、两个光心、两个像点共面**（这个面叫对极平面）。于是 \\( x_2 \\) 不可能出现在右图任意位置，它必落在一条由 \\( x_1 \\) 决定的直线——**极线（epipolar line）**上。

```text
            P (3D)
          ╱   ╲
        ╱       ╲
   x1 ·           · x2         x2 一定落在极线 l2 = F·x1 上
      ╲           ╱            （不用全图找匹配，只需沿极线搜）
   O1 ·───────────· O2
        基线baseline
```

这个约束用一个 3×3 的**基础矩阵（fundamental matrix）** \\( F \\) 精确表达：

\\[ x_2^\top F x_1 = 0 \\]

其中 \\( x_1, x_2 \\) 是齐次像素坐标。\\( F x_1 \\) 正是右图里那条极线。\\( F \\) 只依赖两相机的相对位姿和内参，秩为 2（有一个零奇异值）。

### 从 F 到位姿：本质矩阵 E

\\( F \\) 混合了内参和位姿。若两相机内参 \\( K_1, K_2 \\) 已知，剥掉内参就得到只含相对位姿的**本质矩阵（essential matrix）** \\( E \\)：

\\[ E = K_2^\top F K_1 = [\ t\ ]_\times R \\]

这里 \\( R, t \\) 是两相机的相对旋转与平移，\\( [\ t\ ]_\times \\) 是 \\( t \\) 的反对称叉乘矩阵。\\( E \\) 有个漂亮的特征结构：**奇异值形如 \\( (\sigma, \sigma, 0) \\)**（两个相等、一个为零）。对 \\( E \\) 做 SVD 就能分解出 \\( R \\) 和 \\( t \\)——但会得到 **4 组候选** \\( (R, t) \\)，需用**正深度约束（cheirality，即重建点必须在两个相机前方）**挑出唯一物理正确的那组。而且 \\( t \\) 只能定出**方向、定不出长度**——这就是单目的**尺度不确定性**：你能知道相机"往哪个方向移动了、转了多少"，却不知道"移动了几米"（除非引入已知尺寸或另一路传感器）。

### 八点法估计 F

给 8 对以上的匹配点，就能线性地解出 F：把 \\( x_2^\top F x_1 = 0 \\) 对每对点展开成一个关于 F 九个元素的线性方程，堆成矩阵后用 SVD 求零空间（取最小奇异值对应的右奇异向量），最后**强制 F 秩为 2**（把估计出的 F 再做 SVD、把最小奇异值置零重构）。

> ⚠️ **注意：八点法必须先做 Hartley 归一化。** 像素坐标动辄几百，而 \\( x_2 x_1 \\) 这类交叉项能到几万，不归一化的话线性方程组极度病态，估出的 F 在有噪声时会烂掉。归一化（平移到质心、缩放到平均距离 \\( \sqrt{2} \\)）后再变换回去，是八点法能否实用的分水岭。

### PnP：已知 3D 点反求位姿

**PnP（Perspective-n-Point）** 是另一条路：已知 \\( n \\) 个三维点的世界坐标**和**它们在图里的像素、**以及内参 \\( K \\)**，反求相机位姿 \\( (R, t) \\)。做法是先把像素用 \\( K^{-1} \\) 转成归一化相机坐标，再对 3×4 投影矩阵 \\( [R \mid t] \\) 做 DLT（12 个未知数，SVD 求零空间），最后把左 3×3 块用 SVD **正交化**成合法旋转、定出尺度与符号（同样用正深度约束）。理论最少 **3 个点（P3P）** 就能解（给出至多 4 组解，用第 4 点消歧）；工程常用 **EPnP** 等高效解法。

## 动手写代码

我们做两件事：(A) 合成两台已知相机对同一组 3D 点的投影，用归一化八点法估 F 并验证对极约束、检查 E 的奇异值结构；(B) 合成 3D-2D 对应，用 DLT 解 PnP，对比恢复的位姿与真值。依赖只需 `nalgebra`：

```toml
[dependencies]
nalgebra = "0.35"
```

新建 `code/examples/ch66_epipolar_pnp.rs`。先是小工具——可复现 RNG、投影、Hartley 归一化、SVD 零空间：

```rust
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

// 自包含小 RNG（xorshift64*），造可复现的 3D 点，免引 rand
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed | 1) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    }
}

// 投影：像素 = K (R·M + t) 再透视除法
fn project(k: &Matrix3<f64>, r: &Matrix3<f64>, t: &Vector3<f64>, m: &Vector3<f64>) -> (f64, f64) {
    let p = k * (r * m + t);
    (p.x / p.z, p.y / p.z)
}

// Hartley 归一化：平移到质心、缩放到平均距离 √2
fn normalize_2d(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in pts { mx += x; my += y; }
    mx /= n; my /= n;
    let mut md = 0.0;
    for &(x, y) in pts { md += ((x - mx).powi(2) + (y - my).powi(2)).sqrt(); }
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
```

接着是归一化八点法与 PnP-DLT 两个核心函数：

```rust
// 归一化八点法估计基础矩阵 F
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
    // 强制 F 秩为 2：把最小奇异值置零再重构
    let svd = f_norm.svd(true, true);
    let mut s = svd.singular_values;
    s[2] = 0.0;
    let f_rank2 = svd.u.unwrap() * Matrix3::from_diagonal(&s) * svd.v_t.unwrap();
    t2.transpose() * f_rank2 * t1 // 反归一化 F = T2ᵀ F_norm T1
}

// PnP：已知 K 与 3D-2D 对应，DLT 求相机位姿 (R,t)
fn pnp_dlt(world: &[Vector3<f64>], img: &[(f64, f64)], k: &Matrix3<f64>)
    -> (Matrix3<f64>, Vector3<f64>) {
    let k_inv = k.try_inverse().unwrap();
    let n = world.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        // 像素 -> 归一化相机坐标 (a,b,1)，使 DLT 直接解 [R|t]（K 已剥离）
        let xn = k_inv * Vector3::new(img[i].0, img[i].1, 1.0);
        let (a_, b_) = (xn.x / xn.z, xn.y / xn.z);
        let m = world[i];
        let (x, y, z) = (m.x, m.y, m.z);
        a.row_mut(2 * i).copy_from_slice(
            &[-x, -y, -z, -1.0, 0.0, 0.0, 0.0, 0.0, a_ * x, a_ * y, a_ * z, a_]);
        a.row_mut(2 * i + 1).copy_from_slice(
            &[0.0, 0.0, 0.0, 0.0, -x, -y, -z, -1.0, b_ * x, b_ * y, b_ * z, b_]);
    }
    let mut p = svd_nullspace(&a); // P=[R|t] 的行优先展开（差尺度与符号）
    // 定符号：投影深度 P2·Mh 应为正（点在相机前方，即 cheirality 约束）
    let depth_sum: f64 = world.iter()
        .map(|m| p[8] * m.x + p[9] * m.y + p[10] * m.z + p[11]).sum();
    if depth_sum < 0.0 { p = -p; }
    // 左 3×3 块 ≈ λR：SVD 正交化求最接近的旋转，并定出尺度 λ
    let m3 = Matrix3::new(p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
    let svd = m3.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut r = u * vt;
    if r.determinant() < 0.0 { // 保证 det=+1（右手系旋转）
        r = u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0)) * vt;
    }
    let lambda = svd.singular_values.sum() / 3.0; // λR 三个奇异值都≈λ
    let t = Vector3::new(p[3], p[7], p[11]) / lambda;
    (r, t)
}
```

最后是主流程：

```rust
fn main() {
    // ===== A. 八点法估计基础矩阵 F =====
    let k = Matrix3::new(600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0);
    let r1 = Matrix3::<f64>::identity(); // 相机1 = 世界原点
    let t1 = Vector3::zeros();
    let r2 = *Rotation3::from_euler_angles(0.05, 0.10, -0.05).matrix(); // 相机2 有基线+小旋转
    let t2 = Vector3::new(-1.0, 0.05, 0.10);

    let mut rng = Rng::new(20260827);
    let mut pts3d = Vec::new();
    while pts3d.len() < 12 { // 造 12 个都在两相机前方的 3D 点
        let m = Vector3::new(rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(4.0, 8.0));
        if (r2 * m + t2).z > 0.1 { pts3d.push(m); }
    }
    let pts1: Vec<(f64, f64)> = pts3d.iter().map(|m| project(&k, &r1, &t1, m)).collect();
    let pts2: Vec<(f64, f64)> = pts3d.iter().map(|m| project(&k, &r2, &t2, m)).collect();

    let f = eight_point(&pts1, &pts2);
    println!("估计的基础矩阵 F（已按 F22=1 归一化）：");
    let fn_ = f / f[(2, 2)];
    for row in 0..3 {
        println!("  [{:10.5} {:10.5} {:10.5}]", fn_[(row, 0)], fn_[(row, 1)], fn_[(row, 2)]);
    }
    let mut max_res = 0.0_f64;
    let mut sse = 0.0;
    for i in 0..pts3d.len() { // 验证对极约束 x2ᵀ F x1 ≈ 0
        let x1 = Vector3::new(pts1[i].0, pts1[i].1, 1.0);
        let x2 = Vector3::new(pts2[i].0, pts2[i].1, 1.0);
        let res = (x2.transpose() * f * x1)[(0, 0)].abs();
        max_res = max_res.max(res);
        sse += res * res;
    }
    println!("对极约束残差 |x2ᵀ F x1|：最大 {:.3e}，RMS {:.3e}（应≈0）",
        max_res, (sse / pts3d.len() as f64).sqrt());

    let e = k.transpose() * f * k; // 本质矩阵 E = K2ᵀ F K1（此处同 K）
    let es = e.svd(false, false).singular_values;
    println!("本质矩阵 E 的奇异值（除以最大值后）≈ ({:.4}, {:.4}, {:.4})，理论应为 (1, 1, 0)",
        es[0] / es[0], es[1] / es[0], es[2] / es[0]);

    // ===== B. PnP：DLT 求相机位姿 =====
    println!("\n==== PnP：由 3D-2D 对应恢复相机位姿 ====");
    let r_true = *Rotation3::from_euler_angles(0.15, -0.20, 0.10).matrix();
    let t_true = Vector3::new(0.40, -0.30, 7.0);
    let mut rng2 = Rng::new(12345);
    let mut world = Vec::new();
    for _ in 0..12 {
        world.push(Vector3::new(rng2.range(-1.5, 1.5), rng2.range(-1.5, 1.5), rng2.range(-1.5, 1.5)));
    }
    let img: Vec<(f64, f64)> = world.iter().map(|m| project(&k, &r_true, &t_true, m)).collect();

    let (r_est, t_est) = pnp_dlt(&world, &img, &k);
    let rt = r_est.transpose() * r_true; // 旋转误差：R_estᵀ R_true 越接近单位阵越好
    let cos = ((rt.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    println!("旋转恢复误差 = {:.3e} 度", cos.acos().to_degrees());
    println!("平移 t：估计 [{:.4}, {:.4}, {:.4}]", t_est.x, t_est.y, t_est.z);
    println!("       真值 [{:.4}, {:.4}, {:.4}]", t_true.x, t_true.y, t_true.z);
    let mut sse = 0.0;
    for i in 0..world.len() { // 重投影误差
        let (u, v) = project(&k, &r_est, &t_est, &world[i]);
        sse += (u - img[i].0).powi(2) + (v - img[i].1).powi(2);
    }
    println!("PnP 重投影 RMS 误差 = {:.3e} px（应≈0）", (sse / world.len() as f64).sqrt());
}
```

运行：

```bash
cargo run --example ch66_epipolar_pnp
```

输出：

```text
估计的基础矩阵 F（已按 F22=1 归一化）：
  [  -0.00000   -0.00004    0.02179]
  [  -0.00000    0.00002    0.22740]
  [   0.00005   -0.22476    1.00000]
对极约束残差 |x2ᵀ F x1|：最大 1.554e-15，RMS 8.283e-16（应≈0）
本质矩阵 E 的奇异值（除以最大值后）≈ (1.0000, 1.0000, 0.0000)，理论应为 (1, 1, 0)

==== PnP：由 3D-2D 对应恢复相机位姿 ====
旋转恢复误差 = 0.000e0 度
平移 t：估计 [0.4000, -0.3000, 7.0000]
       真值 [0.4000, -0.3000, 7.0000]
PnP 重投影 RMS 误差 = 3.011e-13 px（应≈0）
```

三处自验证全部通过：八点法估出的 F 让对极约束残差降到 \\( 10^{-15} \\)（机器精度），说明所有匹配点确实被约束在极线上；本质矩阵 E 的奇异值归一化后是 **(1.0000, 1.0000, 0.0000)**，完美符合 \\( (\sigma, \sigma, 0) \\) 的理论结构；PnP 恢复的旋转误差为 **0 度**、平移 **[0.4000, -0.3000, 7.0000]** 与真值逐位相同，重投影误差 \\( 10^{-13} \\) 像素。一整套"3D → 2D → 反解位姿"的链路，被数字焊得死死的。

## 常见坑与提示

> 🕳️ **坑：退化配置会让 F 悄悄失效。** 八点法在几种情况下会崩：**所有 3D 点共面**（此时点对由单应而非 F 描述，应改用[第 26 章](../p04-features/ch26-homography-stitching.md)的单应）、**两相机纯旋转没有平移**（基线为零、无对极几何）、以及**低视差/小基线**（点几乎在无穷远，F 的解方向不稳定）。这些情形下 F 未必报错，却会给出**貌似合理实则错误**的结果。专业系统会主动**检测退化**（如判断解的条件数、比较单应与 F 的拟合优度），退化时宁可拒绝输出，也不硬给一个高置信度的错位姿。

> 🕳️ **坑：内点多 ≠ 几何可观测。** RANSAC 找到一大堆满足对极约束的内点，只能说明"这些点自洽"，**并不能证明相对位姿本身是良置的**。低视差时几乎所有点都满足约束（内点率极高），可位姿方向却极不确定。判断可观测性要看**几何构型**（视差是否充分、点是否共面），而不是内点数量。

> ⚠️ **注意：外点必须靠 RANSAC。** 真实匹配（[第 25 章](../p04-features/ch25-matching-ransac.md)）总有误匹配（外点），一个外点就能把线性最小二乘的 F/位姿彻底带偏。标准流程是：**用最小样本（八点法 8 对、P3P 3 点）在 RANSAC 里反复假设-验证 → 选内点最多的模型 → 再用全部内点做非线性重投影优化精修**。RANSAC 的内点阈值应当**由像素噪声水平推导**（如 \\( 1 \sim 2 \\) 倍标准差），而不是随手拍一个数；还要检查内点的**空间分布**是否均匀（都挤在一小块同样会退化）。

> 📖 **术语**：**尺度不确定性（scale ambiguity）**。单目从 F/E 恢复的平移 \\( t \\) 只有方向、没有长度——把整个场景连同相机轨迹一起放大两倍，投影完全不变。要恢复真实尺度（metric），必须引入已知长度（如标定板尺寸）、已知运动或另一路传感器（如 IMU、轮速计）。PnP 因为**用了已知尺度的 3D 点**，恢复的 \\( t \\) 才是带真实单位的。

## 本章小结

- **对极约束** \\( x_2^\top F x_1 = 0 \\)：匹配点被基础矩阵 F 锁在极线上；F 秩为 2。
- **归一化八点法**线性估 F：展开约束 → SVD 求零空间 → 强制秩 2 → 反归一化；Hartley 归一化是稳定性的分水岭。实测对极残差 \\( 10^{-15} \\)。
- **本质矩阵** \\( E = K_2^\top F K_1 = [t]_\times R \\)，奇异值 \\( (\sigma, \sigma, 0) \\)；分解出 4 组 \\( (R, t) \\)，用正深度约束消歧，且 **\\( t \\) 尺度不定**。
- **PnP** 用已知 3D 点 + 内参 DLT 反求位姿：SVD 求 \\( [R\mid t] \\) → 正交化旋转 → 定尺度与符号。实测旋转误差 0 度、平移与真值一致。
- 退化配置（共面、纯旋转、低视差）、外点、尺度不确定是四大现实陷阱；**内点多不等于几何可观测**；工程必须配 RANSAC + 非线性精修，并主动检测退化。

## 动手练习

1. **验证 Hartley 归一化的价值**：给 `pts2` 的每个像素叠加 ±0.5px 随机噪声（复用本章的 `Rng`），分别用"归一化"和"不归一化"（把 `normalize_2d` 换成恒等）的八点法估 F，比较对极残差——亲眼看到归一化在有噪声时的碾压优势。
2. **制造共面退化**：把 12 个 3D 点全部放到同一个平面上（如令 `z` 恒为 6），重跑八点法，观察 F 的估计如何变得不稳定（此时该用单应）。
3. **PnP 最少要几点**：把 PnP 的点数从 12 逐步减到 6、5、4，看 DLT 何时开始不稳；查阅 P3P（3 点最小解）和 EPnP 的思路，理解"最小解 + RANSAC"为何比"一次性用全部点"更鲁棒。
4. **加外点 + RANSAC（进阶）**：往匹配里掺入 20% 的随机错误对应，套用[第 25 章](../p04-features/ch25-matching-ransac.md)的 RANSAC 框架（随机取最小样本、统计内点、取最优）来鲁棒地估 F 或位姿，并用内点阈值随像素噪声变化做敏感性分析。

至此，我们已能标定相机（第 64 章）、用双目测深度（第 65 章）、从多视角反推相机位姿（本章）。下一步很自然：把这些恢复出的三维点收集起来，形成**点云**，做滤波、配准与三维重建。👉 [第 67 章 点云处理入门](ch67-pointcloud.md)
