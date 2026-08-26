# 第 64 章 相机标定（棋盘格、张正友标定法）

> **本章导读**
> - 搞清楚"相机标定"到底在标什么：把[第 6 章](../p01-start/ch06-camera-model.md)里那个内参矩阵 \\( K \\)（\\( f_x, f_y, c_x, c_y \\)）和畸变系数**真正测量出来**——这是一切三维度量的前提。
> - 理解棋盘格标定的核心链条：平面标定板 → 每张图一个**单应矩阵 \\( H \\)** → 多个 \\( H \\) 叠出内参 \\( K \\)。
> - 动手用 `nalgebra` 的 **SVD** 实现"归一化 DLT"从点对反解单应，再用**张正友法闭式解**从多张图恢复 \\( K \\)。
> - 全程"合成 → 投影 → 反解 → 对比真值"，亲眼看到反解出的 \\( K \\) 和真值几乎逐位相同。

## 场景引入：为什么不能直接拿相机量东西

假设老板扔给你一个任务："用这个摄像头量一下流水线上零件的宽度。"你可能会想：拍张照，数一下零件占了多少像素，不就行了？

问题在于：**同一个像素宽度，对应的真实宽度取决于零件离镜头多远，也取决于这颗镜头的焦距**。[第 6 章](../p01-start/ch06-camera-model.md)我们推过投影公式 \\( u = f_x X / Z + c_x \\)——里面的 \\( f_x, c_x \\) 是这颗相机"天生"的参数，你不知道它们，就没法从像素反推真实尺寸。更糟的是，真实镜头还有畸变，会把直线拍弯。

**相机标定（camera calibration）** 就是把这些参数一次性测准的过程：求出内参矩阵 \\( K \\) 和畸变系数。标定做完，相机才从"能拍照"升级成"能测量"。而最经典、最实用的标定工具，就是一张**棋盘格**。

## 原理：从棋盘格到内参

### 直觉：棋盘格提供"已知答案"

标定的本质是"解方程"，而解方程需要已知量。棋盘格的妙处在于：它的角点排布是**规则、已知**的——第 \\( i \\) 行第 \\( j \\) 列的角点，在标定板自己的坐标系里就是 \\( (i \cdot s, j \cdot s) \\)（\\( s \\) 是方格边长）。把棋盘平面当作世界坐标系的 \\( Z=0 \\) 平面，我们就同时握有一批点的**世界坐标**（我们规定的）和**像素坐标**（从图里检测出来的）。有了这一大堆"世界 → 像素"的对应，就能反解相机参数。

```text
   世界(棋盘)平面 Z=0              一张照片
   ┌─────────────────┐            ┌─────────────────┐
   │ ·  ·  ·  ·  ·  · │            │  · · · · · ·    │
   │ ·  ·  ·  ·  ·  · │  ──H_i──▶  │  ·· ·· ·· ··    │  拍摄角度不同
   │ ·  ·  ·  ·  ·  · │  (单应)    │  ····· ····     │  → H 不同
   └─────────────────┘            └─────────────────┘
   已知世界坐标(X,Y)               检测出的像素(u,v)
```

### 一张平面 = 一个单应

[第 6 章](../p01-start/ch06-camera-model.md)的完整投影是 \\( s\  \tilde{m} = K[R \mid t]\tilde{M} \\)。当所有点都在 \\( Z=0 \\) 平面上时，齐次世界坐标 \\( (X, Y, 0, 1)^\top \\) 的第三列被乘没了，投影退化成一个 3×3 的矩阵乘法：

\\[ s \begin{bmatrix} u \\\\ v \\\\ 1 \end{bmatrix} = K\ [\ r_1\ \ r_2\ \ t\ ] \begin{bmatrix} X \\\\ Y \\\\ 1 \end{bmatrix} = H \begin{bmatrix} X \\\\ Y \\\\ 1 \end{bmatrix} \\]

这里 \\( r_1, r_2 \\) 是旋转矩阵 \\( R \\) 的前两列。这个 3×3 矩阵 \\( H \\) 就是**单应矩阵（homography）**（[第 26 章](../p04-features/ch26-homography-stitching.md)已见过它），它把标定板平面直接映射到像面。**每拍一张图，就得到一个 \\( H \\)**，且 \\( H = K\ [\ r_1\ r_2\ t\ ] \\)。

### 用 DLT + SVD 反解单应

给一堆对应点 \\( (X, Y) \to (u, v) \\) 反求 \\( H \\)，用的是 **DLT（直接线性变换）**：把 \\( \tilde{m} \times (H \tilde{M}) = 0 \\) 展开，每对点贡献 2 个关于 \\( H \\) 九个元素的线性方程。堆成矩阵 \\( A \\) 后求解齐次方程 \\( A h = 0 \\)——解就是 \\( A \\) 最小奇异值对应的右奇异向量（用 SVD 求，呼应[第 30 章](../p05-numeric/ch30-nalgebra.md)）。

> ⚠️ **注意**：DLT 前一定要做 **Hartley 归一化**——把点平移到质心、缩放到平均距离 \\( \sqrt{2} \\)。因为像素坐标动辄几百、而 \\( u \cdot X \\) 这类项能到几万，不归一化的话矩阵 \\( A \\) 数值尺度悬殊、病态，解出来误差很大。归一化后再把结果变换回去：\\( H = T_{\text{img}}^{-1} H_{\text{norm}} T_{\text{world}} \\)。

### 多个 H 如何叠出 K

单个 \\( H \\) 只够反解出这张图的姿态，不够定出 \\( K \\)。张正友（Zhengyou Zhang）的洞察是：**旋转的两列 \\( r_1, r_2 \\) 必须单位正交**，这给 \\( K \\) 施加了约束。记 \\( H = [\ h_1\ h_2\ h_3\ ] = \lambda K[\ r_1\ r_2\ t\ ] \\)，由 \\( r_1 \perp r_2 \\) 且 \\( \lVert r_1 \rVert = \lVert r_2 \rVert \\) 可推出两条方程：

\\[ h_1^\top B\  h_2 = 0, \qquad h_1^\top B\  h_1 = h_2^\top B\  h_2 \\]

其中 \\( B = K^{-\top} K^{-1} \\) 是一个对称矩阵（只有 6 个独立元）。**每张图给 2 个方程，\\( B \\) 有 6 个未知数，所以至少 3 张不同角度的图就能解出 \\( B \\)**（同样用 SVD 求零空间），再用闭式公式从 \\( B \\) 拆出 \\( f_x, f_y, c_x, c_y \\)。这就是"多个 \\( H \\) 叠出 \\( K \\)"的全部秘密。

## 动手写代码

完整张氏标定还要做角点检测、畸变估计与非线性优化，比较重。我们给一个**聚焦几何内核的可运行实现**：合成棋盘格 + 已知 \\( K \\) 和 3 个姿态，正向投影出角点，再用归一化 DLT 反解每张图的 \\( H \\)，最后用张氏闭式解从 3 个 \\( H \\) 反推 \\( K \\)。依赖只需 `nalgebra`：

```toml
[dependencies]
nalgebra = "0.35"
```

先是三个工具函数——Hartley 归一化、SVD 求零空间、归一化 DLT 求单应：

```rust
use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

// 对一组 2D 点做 Hartley 归一化：平移到质心、缩放到平均距离 √2。
// 返回归一化后的点与归一化矩阵 T（满足 归一化点 = T * 原始齐次点）。
fn normalize_2d(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, Matrix3<f64>) {
    let n = pts.len() as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for &(x, y) in pts { mx += x; my += y; }
    mx /= n; my /= n;
    let mut mean_d = 0.0;
    for &(x, y) in pts { mean_d += ((x - mx).powi(2) + (y - my).powi(2)).sqrt(); }
    mean_d /= n;
    let s = 2.0_f64.sqrt() / mean_d; // 缩放使平均距离为 √2
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|&(x, y)| (s * (x - mx), s * (y - my))).collect();
    (out, t)
}

// 解齐次方程 A h = 0：取最小奇异值对应的右奇异向量。
// nalgebra 的奇异值按降序排列，故最小者在最后 -> V^T 的最后一行。
fn svd_nullspace(a: &DMatrix<f64>) -> DVector<f64> {
    let svd = a.clone().svd(true, true);
    let vt = svd.v_t.expect("已用 svd(true,true) 计算 V^T");
    let last = vt.nrows() - 1;
    vt.row(last).transpose()
}

// 归一化 DLT 从平面点对 (X,Y)->(u,v) 估计单应 H
fn dlt_homography(world: &[(f64, f64)], img: &[(f64, f64)]) -> Matrix3<f64> {
    let (wn, tw) = normalize_2d(world);
    let (imn, ti) = normalize_2d(img);
    let n = world.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for k in 0..n {
        let (x, y) = wn[k];
        let (u, v) = imn[k];
        // 每对点由叉乘 x_img × (H x_world) = 0 得两行独立方程
        a[(2 * k, 3)] = -x;   a[(2 * k, 4)] = -y;   a[(2 * k, 5)] = -1.0;
        a[(2 * k, 6)] = v * x; a[(2 * k, 7)] = v * y; a[(2 * k, 8)] = v;
        a[(2 * k + 1, 0)] = x; a[(2 * k + 1, 1)] = y; a[(2 * k + 1, 2)] = 1.0;
        a[(2 * k + 1, 6)] = -u * x; a[(2 * k + 1, 7)] = -u * y; a[(2 * k + 1, 8)] = -u;
    }
    let h = svd_nullspace(&a);
    let hn = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    ti.try_inverse().unwrap() * hn * tw // 反归一化 H = Ti⁻¹ Hn Tw
}

// 用 H 把平面点 (X,Y) 投影成像素 (u,v)（透视除法）
fn apply_h(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
    let p = h * Vector3::new(x, y, 1.0);
    (p.x / p.z, p.y / p.z)
}

// 张氏法：由单应两列构造对 B 的约束向量 v_ij
fn v_ij(h: &Matrix3<f64>, i: usize, j: usize) -> [f64; 6] {
    let (a0, a1, a2) = (h[(0, i)], h[(1, i)], h[(2, i)]);
    let (b0, b1, b2) = (h[(0, j)], h[(1, j)], h[(2, j)]);
    [a0 * b0, a0 * b1 + a1 * b0, a1 * b1,
     a2 * b0 + a0 * b2, a2 * b1 + a1 * b2, a2 * b2]
}
```

然后是主流程：定义真值、投影、反解、对比。把它和上面拼成一个文件 `code/examples/ch64_calibration.rs`：

```rust
fn main() {
    // ① 真值：内参 K 与三张图的姿态 (R,t)
    let (fx, fy, cx, cy) = (520.0_f64, 525.0, 325.0, 245.0);
    let k_true = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // 棋盘格：9×6 个内角点，方格边长 1.0，中心挪到原点
    let (cols, rows) = (9usize, 6usize);
    let mut board: Vec<(f64, f64)> = Vec::new();
    for j in 0..rows { for i in 0..cols { board.push((i as f64 - 4.0, j as f64 - 2.5)); } }

    // 三张图：不同角度/位置（欧拉角弧度；平移让棋盘落在相机前方 Z>0）
    let poses = [
        (Rotation3::from_euler_angles(0.10, -0.15, 0.05), Vector3::new(-1.0, -0.5, 12.0)),
        (Rotation3::from_euler_angles(-0.20, 0.25, -0.10), Vector3::new(0.5, 0.8, 11.0)),
        (Rotation3::from_euler_angles(0.28, 0.12, 0.20), Vector3::new(1.2, -0.6, 13.0)),
    ];

    let mut h_estimated: Vec<Matrix3<f64>> = Vec::new();
    println!("真值内参 K：fx={fx}, fy={fy}, cx={cx}, cy={cy}\n");

    // ② 每张图：投影 -> 估计 H -> 对比真值 + 重投影误差
    for (idx, (rot, t)) in poses.iter().enumerate() {
        let r = rot.matrix();
        // 平面棋盘 Z=0，投影退化为单应 H_true = K [r1 r2 t]
        let rt = Matrix3::new(
            r[(0, 0)], r[(0, 1)], t[0],
            r[(1, 0)], r[(1, 1)], t[1],
            r[(2, 0)], r[(2, 1)], t[2]);
        let h_true = k_true * rt;
        let img_pts: Vec<(f64, f64)> = board.iter().map(|&(x, y)| apply_h(&h_true, x, y)).collect();

        let h_est = dlt_homography(&board, &img_pts); // 反解

        let mut sse = 0.0; // 重投影误差
        for (bp, ip) in board.iter().zip(&img_pts) {
            let (u, v) = apply_h(&h_est, bp.0, bp.1);
            sse += (u - ip.0).powi(2) + (v - ip.1).powi(2);
        }
        let rms = (sse / img_pts.len() as f64).sqrt();

        let h_est_n = h_est / h_est[(2, 2)]; // H 是齐次量，按 H22=1 归一化再比
        let h_true_n = h_true / h_true[(2, 2)];
        if idx == 0 {
            println!("视图 #1 单应矩阵对比（已按 H22=1 归一化）：");
            for row in 0..3 {
                println!("  估计 [{:9.4} {:9.4} {:9.4}]   真值 [{:9.4} {:9.4} {:9.4}]",
                    h_est_n[(row, 0)], h_est_n[(row, 1)], h_est_n[(row, 2)],
                    h_true_n[(row, 0)], h_true_n[(row, 1)], h_true_n[(row, 2)]);
            }
        }
        println!("视图 #{}：{} 个角点，H 重投影 RMS 误差 = {:.3e} px", idx + 1, board.len(), rms);
        h_estimated.push(h_est);
    }

    // ③ 张正友法：多张 H 叠出对 B 的约束，反解内参 K
    let mut vmat = DMatrix::<f64>::zeros(2 * h_estimated.len(), 6);
    for (n, h) in h_estimated.iter().enumerate() {
        let (v01, v00, v11) = (v_ij(h, 0, 1), v_ij(h, 0, 0), v_ij(h, 1, 1));
        for c in 0..6 {
            vmat[(2 * n, c)] = v01[c];            // h1ᵀ B h2 = 0
            vmat[(2 * n + 1, c)] = v00[c] - v11[c]; // h1ᵀ B h1 = h2ᵀ B h2
        }
    }
    let mut b = svd_nullspace(&vmat); // b = [B11,B12,B22,B13,B23,B33]
    if b[0] < 0.0 { b = -b; }          // B 应正定，符号反了就取负
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
    let err = ((alpha - fx).powi(2) + (beta - fy).powi(2)
             + (u0 - cx).powi(2) + (v0 - cy).powi(2)).sqrt();
    println!("  内参四参数总误差(欧氏) = {:.3e}", err);
}
```

运行：

```bash
cargo run --example ch64_calibration
```

输出：

```text
真值内参 K：fx=520, fy=525, cx=325, cy=245

视图 #1 单应矩阵对比（已按 H22=1 归一化）：
  估计 [  46.8405   -0.1272  281.6667]   真值 [  46.8405   -0.1272  281.6667]
  估计 [   5.2131   45.4598  223.1250]   真值 [   5.2131   45.4598  223.1250]
  估计 [   0.0125    0.0082    1.0000]   真值 [   0.0125    0.0082    1.0000]
视图 #1：54 个角点，H 重投影 RMS 误差 = 2.041e-13 px
视图 #2：54 个角点，H 重投影 RMS 误差 = 1.475e-13 px
视图 #3：54 个角点，H 重投影 RMS 误差 = 1.149e-13 px

==== 张正友法反解内参 K（来自 3 张图的单应）====
  fx: 估计 520.0000  vs 真值 520.0
  fy: 估计 525.0000  vs 真值 525.0
  cx: 估计 325.0000  vs 真值 325.0
  cy: 估计 245.0000  vs 真值 245.0
  skew(γ): 估计 0.000000  vs 真值 0
  内参四参数总误差(欧氏) = 3.866e-12
```

单应矩阵反解逐位对上真值，重投影误差在 \\( 10^{-13} \\) 像素量级（纯浮点误差），而**从 3 张图反解出的 \\( f_x, f_y, c_x, c_y \\) 与真值 520/525/325/245 完全一致**，总误差只有 \\( 3.9 \times 10^{-12} \\)。整条"投影 → 反解"的链路被自己验证得干干净净——这正是张正友法的威力。

## 常见坑与提示

> 🕳️ **坑：平面点求内参至少要 3 个视角，且姿态要"够不一样"。** \\( B \\) 有 6 个未知数、每张图 2 个方程，2 张图只有 4 个方程解不出。更隐蔽的是**退化配置**：如果几张图都近乎正对棋盘（旋转几乎相同），这些方程会高度相关、方程组病态，即便有 3 张，\\( K \\) 也会解得离谱。标定时务必让棋盘**大幅度地俯仰、偏转、旋转**。

> ⚠️ **注意**：本例为聚焦几何内核，直接用了"无噪声的合成角点"。真实标定中，角点是从图像里**检测**出来的，必然带噪声；此时 DLT 只是初值，工业级标定还会跟一步**非线性优化（最小化重投影误差）**来精修 \\( K \\)、\\( R \\)、\\( t \\) 和畸变。纯 Rust 里可用 `argmin`、`levenberg-marquardt` 等 crate 做这步优化（用前自行 `cargo add` 核对版本）。

> 💡 **提示：角点精度决定标定精度。** 棋盘角点必须做**亚像素级**定位（把整数像素坐标精修到小数），否则内参会系统性偏差。亚像素角点与精密测量是[第 78 章](../p17-industrial/ch78-metrology.md)的主题。

> 📖 **术语**：**畸变（distortion）**。本章只标了理想针孔的内参 \\( K \\)。真实镜头还有径向/切向畸变（[第 6 章](../p01-start/ch06-camera-model.md)），标定时一并估计畸变系数 \\( k_1, k_2, p_1, p_2 \\)，之后就能把图像"掰直"（去畸变）。张氏原论文把畸变作为非线性优化的一部分联合求解。

## 本章小结

- **相机标定**就是把内参 \\( K \\)（\\( f_x, f_y, c_x, c_y \\)）和畸变系数测量出来，是从"像素"走向"真实尺寸"的前提。
- 棋盘格提供**已知世界坐标**的角点；平面标定板在 \\( Z=0 \\) 时，投影退化为**单应 \\( H = K[r_1\ r_2\ t] \\)**，一张图一个 \\( H \\)。
- 用**归一化 DLT + SVD** 从点对反解 \\( H \\)；Hartley 归一化对数值稳定性至关重要。
- 张正友法利用 \\( r_1, r_2 \\) 单位正交，得到对 \\( B = K^{-\top}K^{-1} \\) 的线性约束；**≥3 个不同姿态的 \\( H \\)** 即可解出 \\( B \\)，再闭式拆出 \\( K \\)。
- 我们实测反解出的 \\( K \\) 与真值误差仅 \\( 10^{-12} \\) 量级，重投影误差 \\( 10^{-13} \\) 像素——自验证完美通过。

## 动手练习

1. **换内参再验证**：把真值改成 \\( f_x = 480, c_x = 300 \\)，重新运行，确认反解依然能还原新值。这说明算法不依赖具体数值。
2. **体会"≥3 视角"**：把 `poses` 减到只剩 2 个姿态，运行看看 \\( K \\) 反解是否变差甚至出 `NaN`——亲身感受为什么标定至少要拍 3 张（实践中拍 10~20 张）。
3. **制造退化**：把三张图的旋转欧拉角都改成非常接近（比如都设成 `(0.02, 0.02, 0.0)`，近乎正对棋盘），观察 \\( B \\) 方程病态、\\( K \\) 反解崩坏。这就是"退化配置"。
4. **加噪声（进阶）**：给投影出的每个像素坐标叠加 ±0.3px 的随机扰动（可复用[第 25 章](../p04-features/ch25-matching-ransac.md)里的小 RNG），观察重投影误差和 \\( K \\) 误差怎样从 \\( 10^{-12} \\) 上升到零点几，体会"角点精度 → 标定精度"。

标定拿到了 \\( K \\)，我们就有了"能测量的相机"。下一章我们用**两台**这样的相机，靠双眼视差把二维图像重新变回三维深度。👉 [第 65 章 立体视觉与深度估计](ch65-stereo-depth.md)
