# 第 67 章 点云处理入门

> **本章导读**
> - 搞懂**点云（point cloud）**到底是什么：一堆没有拓扑关系的 3D 点，是激光雷达、深度相机、三维重建的共同产物。
> - 学会点云最基础的四件事：**质心与包围盒**、**体素下采样**、**最近邻查询**、**平面拟合**。
> - 用纯 Rust + nalgebra 合成一片"带噪声的平面 + 离群点"，亲手把它下采样、查最近邻，并用 **RANSAC** 稳稳地把平面从噪声里抠出来。

## 场景引入：扫地机器人眼里的世界

想象一台扫地机器人，头顶转着一圈激光雷达。它每秒向四周打出上万束激光，测量每束光打到障碍物再返回的时间，换算成距离。把这些距离连同发射角度一换算，就得到了成千上万个 **3D 坐标点**——地面是一大片近似平面的点，墙是竖直的点墙，桌子腿是几根点柱。这一大坨 \\((x, y, z)\\) 点，就是**点云**。

机器人要做的第一件事，往往是"**把地面找出来**"：地面对应点云里那片最大的平面，找到它就能区分"可通行的地"和"要绕开的障碍"。这一章我们就从合成一片带噪声的平面点云出发，走完从"一堆散点"到"拟合出平面方程"的全过程。

## 点云是什么：一堆"自由"的 3D 点

先跟前面章节的图像对比一下，你立刻就懂了：

```text
   图像（第 5 章）                       点云
 ┌──┬──┬──┬──┐        规则网格          {(0.1, 2.3, 5.0),
 ├──┼──┼──┼──┤   每个像素有固定邻居       (0.2, 2.1, 4.9),   一堆散点
 ├──┼──┼──┼──┤   （上下左右）             (1.7, 0.4, 5.2),   谁也不挨着谁
 └──┴──┴──┴──┘                          ... 共 N 个}
```

- **图像**是**规则网格**：像素排得整整齐齐，天然知道谁挨着谁，所以卷积、滤波才好做。
- **点云**是**无组织的散点集合** \\(\{(x_i, y_i, z_i)\}_{i=1}^{N}\\)：点与点之间**没有固定的邻接关系**，顺序还可以任意打乱。这带来两个后果——**① 量大**（一帧激光雷达轻松几万到几十万点）；**② 想找"某点附近的点"要额外费劲**（不像图像直接看上下左右）。

> 📖 **术语**：**点云（point cloud）**——三维空间中一组点的集合，每个点至少有 \\((x, y, z)\\) 坐标，可能还带颜色、法向、反射强度等属性。来源包括：激光雷达（LiDAR）、深度相机（如结构光/ToF）、以及本书后面要讲的多视图三维重建（[第 68 章](ch68-sfm-slam.md)）。

### 怎么在 Rust 里表示点云

最朴素的三种表示，够用且高效：

- `Vec<[f32; 3]>`：最省内存、最贴近"一片连续的浮点数"，适合海量点。
- `Vec<nalgebra::Point3<f64>>`：`Point3` 语义上就是"一个位置"，支持点减点得向量等几何运算。
- `ndarray::Array2<f32>`（形状 `N×3`）：想对整片点云做矩阵运算（如批量变换）时方便。

本章代码用 `Vector3<f64>`（nalgebra 的 3D 向量）当点类型——它直接支持点差、点积、叉积、求模，写几何算法最顺手。

## 四个基本操作

### 质心与 AABB 包围盒

**质心（centroid）** 就是所有点的平均位置，是点云的"重心"：

\\[ \bar{p} = \frac{1}{N}\sum_{i=1}^{N} p_i \\]

**AABB（Axis-Aligned Bounding Box，轴对齐包围盒）** 是刚好把点云装下的、各边平行于坐标轴的长方体，只需逐维求最小值和最大值即可。它常用来快速判断"两片点云有没有可能相交""这堆点大概多大"。

### 体素下采样：把点云"抽稀"

一帧几十万点，直接处理又慢又占内存。**体素下采样（voxel downsampling）** 是最常用的抽稀手段：把空间切成边长为 \\(s\\) 的小立方体格子（**体素，voxel**），每个格子里的一堆点只保留**一个代表点**（常取该格点的质心）。

关键技巧是用**哈希网格**：一个点落在哪个格子，由它的坐标除以体素尺寸再向下取整决定，这个整数三元组就当哈希表的 key：

\\[ \mathrm{key}(p) = \left(\lfloor x/s \rfloor,\ \lfloor y/s \rfloor,\ \lfloor z/s \rfloor\right) \\]

```text
  一维示意（体素尺寸 s=1）：
  点:   0.2  0.7 | 1.1  1.8 | 2.3        原始 5 个点
  格子:  key=0    key=1      key=2
  代表:  0.45  |  1.45   |  2.3          下采样后 3 个点（每格取质心）
```

同一格的点 key 相同，落进同一个桶，最后每个桶输出一个质心。体素越大，抽得越狠、点越少但越粗糙；体素越小，保留越多细节但压缩越弱——这是一个**精度 vs 数据量**的权衡。

### 最近邻查询

"离查询点 \\(q\\) 最近的点是哪个"是点云里的高频操作（配准、法向估计、聚类都要用）。最直白的**暴力法**：遍历所有点算距离取最小，复杂度 \\(O(N)\\)。点多了就慢。

工程上会用 **kd-tree**（k 维二叉树）把最近邻降到平均 \\(O(\log N)\\)。纯 Rust 生态里可以用 [`kiddo`](https://crates.io/crates/kiddo) crate（高性能 kd-tree，需自行 `cargo add` 验证版本）。本章为了自包含先写暴力法，讲清楚"要算什么"，你理解后换 kd-tree 只是换个数据结构。

### 平面拟合：PCA 与 RANSAC

一个平面可以用**单位法向** \\(n\\) 和**截距** \\(d\\) 表示为 \\(n \cdot p = d\\)。点 \\(q\\) 到平面的距离就是：

\\[ \mathrm{dist}(q) = \lvert n \cdot q - d \rvert \\]

**思路一：PCA / 最小二乘。** 把所有点中心化（减去质心），点云在哪个方向上"最扁"，哪个方向就是法向。数学上，对中心化后的 \\(N\times 3\\) 矩阵做 **SVD**（奇异值分解，见 [第 30 章](../p05-numeric/ch30-nalgebra.md)），**最小奇异值对应的右奇异向量**就是法向 \\(n\\)。这等价于求点云协方差矩阵的最小特征向量。

PCA 的软肋是**对离群点敏感**：只要有些点乱飞，"最扁方向"就会被带偏。

**思路二：RANSAC，抗噪利器。** 沿用 [第 25 章](../p04-features/ch25-matching-ransac.md) 的随机抽样一致思想，套到平面上只需把"最小样本"从 2 个点（定直线）换成 **3 个点（定平面）**：

```text
  重复 N 次：
    ① 随机取 3 个点，叉积求平面法向 n = (p1-p0)×(p2-p0)，归一化
    ② 数内点：到该平面距离 |n·p - d| < 阈值 的点
    ③ 记住"内点最多"的那个平面
  最后：用全部内点做一次 PCA 精修，得到最终平面
```

## 动手写代码

下面这个例子合成 920 个点（800 个贴着真值平面 \\(z = 0.3x - 0.2y + 1.5\\)、加 2cm 噪声的内点，加 120 个满盒乱撒的离群点），然后依次做上面四件事。为了可复现，随机数用一个自包含的 xorshift RNG（不引入 `rand` 依赖）。

运行：

```bash
cargo run --example ch67_pointcloud
```

```rust
use nalgebra::{DMatrix, Vector3};
use std::collections::HashMap;

type P3 = Vector3<f64>; // 点=3D 向量：支持点差、点积、叉积、求模

// ---- 自包含小 RNG（xorshift64*），可复现、无需 rand ----
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed | 1) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range_f(&mut self, lo: f64, hi: f64) -> f64 { lo + self.uniform() * (hi - lo) }
    fn normal(&mut self) -> f64 { // Box-Muller 标准正态
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn range_usize(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
}

const TRUE_A: f64 = 0.30;  // 平面真值 z = a·x + b·y + c
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
    for _ in 0..120 { // 离群点：整个包围盒里乱撒
        pts.push(P3::new(rng.range_f(-2.0, 2.0), rng.range_f(-2.0, 2.0), rng.range_f(0.0, 3.0)));
    }
    (pts, n_in)
}

fn centroid(pts: &[P3]) -> P3 {
    let mut s = P3::zeros();
    for p in pts { s += *p; }
    s / pts.len() as f64
}

fn aabb(pts: &[P3]) -> (P3, P3) { // 逐维取 min/max
    let (mut lo, mut hi) = (pts[0], pts[0]);
    for p in pts {
        for k in 0..3 {
            if p[k] < lo[k] { lo[k] = p[k]; }
            if p[k] > hi[k] { hi[k] = p[k]; }
        }
    }
    (lo, hi)
}

// 体素下采样：坐标除以体素尺寸取整做 key，同格点取质心，一格留一点
fn voxel_downsample(pts: &[P3], voxel: f64) -> Vec<P3> {
    let mut cells: HashMap<(i64, i64, i64), (P3, usize)> = HashMap::new();
    for p in pts {
        let key = ((p.x / voxel).floor() as i64,
                   (p.y / voxel).floor() as i64,
                   (p.z / voxel).floor() as i64);
        let e = cells.entry(key).or_insert((P3::zeros(), 0));
        e.0 += *p; e.1 += 1; // 累加坐标与计数
    }
    cells.values().map(|(sum, cnt)| sum / *cnt as f64).collect()
}

// 暴力最近邻：遍历所有点，O(N)
fn nearest(pts: &[P3], q: &P3) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (i, p) in pts.iter().enumerate() {
        let d = (p - q).norm();
        if d < best.1 { best = (i, d); }
    }
    best
}

// PCA 平面拟合：中心化 -> SVD -> 最小奇异向量即法向；返回 (法向, 质心)
fn fit_plane_pca(pts: &[P3]) -> (P3, P3) {
    let c = centroid(pts);
    let mut data = Vec::with_capacity(pts.len() * 3);
    for p in pts {
        let d = p - c;
        data.push(d.x); data.push(d.y); data.push(d.z);
    }
    let m = DMatrix::from_row_slice(pts.len(), 3, &data); // (N×3)
    let vt = m.svd(true, true).v_t.unwrap();               // 奇异值降序
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
        if i == j || j == k || i == k { continue; }
        let normal = (pts[j] - pts[i]).cross(&(pts[k] - pts[i]));
        if normal.norm() < 1e-9 { continue; } // 三点共线，退化
        let normal = normal.normalize();
        let d = normal.dot(&pts[i]);
        let inliers: Vec<usize> =
            (0..n).filter(|&t| (normal.dot(&pts[t]) - d).abs() < thresh).collect();
        if inliers.len() > best.len() { best = inliers; }
    }
    let inlier_pts: Vec<P3> = best.iter().map(|&t| pts[t]).collect();
    let (normal, c) = fit_plane_pca(&inlier_pts); // 用内点精修
    (normal, c, best.len())
}

fn main() {
    let (cloud, n_in) = make_cloud(20250827);
    println!("合成点云：共 {} 点（内点 {} + 离群 {}）", cloud.len(), n_in, cloud.len() - n_in);

    let c = centroid(&cloud);
    let (lo, hi) = aabb(&cloud);
    println!("质心 = ({:.3}, {:.3}, {:.3})", c.x, c.y, c.z);
    println!("AABB 尺寸 = ({:.2}, {:.2}, {:.2})", hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);

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
    println!("RANSAC 拟合: z = {:.3}x + {:.3}y + {:.3} (内点 {})", ra, rb, rc, n_inl);
    println!("平面真值:    z = {:.3}x + {:.3}y + {:.3}", TRUE_A, TRUE_B, TRUE_C);
}
```

在本机实测输出（数字可复现）：

```text
合成点云：共 920 点（内点 800 + 离群 120）
质心 = (-0.019, -0.034, 1.519)
AABB 尺寸 = (3.99, 3.99, 2.95)
体素下采样(0.5m): 920 -> 165 点
最近邻: #58 距离 0.0974 m
PCA 直接拟合(含离群): z = 0.290x + -0.190y + 1.518
RANSAC 拟合: z = 0.301x + -0.199y + 1.501 (内点 805)
平面真值:    z = 0.300x + -0.200y + 1.500
```

三个关键结论对上了：**① 下采样** 把 920 点抽到 165 点（约 18%），因为点几乎都在一张平面上，每个 \\((x,y)\\) 立柱方向只占少数几个体素；**② PCA 直接拟合** 被 120 个离群点拉偏（\\(a\\) 从 0.300 变成 0.290、\\(c\\) 偏了约 0.018）；**③ RANSAC** 找到 805 个内点（略多于真实的 800，因为个别离群点碰巧落在阈值内），拟合结果 \\(z = 0.301x - 0.199y + 1.501\\)，三个参数误差都在千分之一量级，几乎完美命中真值。

## 常见坑与提示

> 🕳️ **坑：体素尺寸没有"万能值"。** 体素太大，平面的凹凸细节被抹平、薄壁两侧的点会被合并成一层；太小，压缩几乎无效还徒增哈希开销。实践中先按点云的物理尺度估个数量级（如室内场景 1~5 cm），再看效果调。

> ⚠️ **注意：法向有朝向歧义。** PCA/RANSAC 求出的法向 \\(n\\) 和 \\(-n\\) 都满足平面方程——它只定了"平面朝哪一面的轴线"，没定"朝里还是朝外"。需要一致朝向时（如渲染、点云拼接），要用额外信息（比如"都朝向传感器"）翻正符号。本章把结果转成 \\(z = ax+by+c\\) 的显式形式来比较，正好规避了这个歧义。

> 🕳️ **坑：离群点会毁掉最小二乘。** 只要几颗"飞点"，直接 PCA/最小二乘拟合的平面就会歪。凡是"数据里可能混入错误点"的拟合，优先想到 RANSAC。

> 💡 **提示：点云量大，内存和性能是第一道坎。** 几十万点用 `Vec<[f32;3]>` 比 `f64` 省一半内存；最近邻/半径搜索务必上 kd-tree（`kiddo`）而非暴力法；能下采样就先下采样再算。

> 📖 **点云文件格式**：最常见的是 **PLY** 和 **PCD**（点云库 PCL 的原生格式）。纯 Rust 里读写 PLY 可看 [`ply-rs`](https://crates.io/crates/ply-rs) 等 crate（版本与 API 请自行 `cargo add` 验证）；PCD 纯 Rust 支持较零散，简单场景可自己按其文本头格式解析。

## 本章小结

- **点云**是无组织的 3D 点集合 \\(\{(x_i,y_i,z_i)\}\\)，来自激光雷达、深度相机与三维重建；相比图像，它**量大、无固定邻接**。
- Rust 里可用 `Vec<[f32;3]>`（省内存）、`Point3`（重语义）或 `Array2`（重矩阵运算）表示。
- **质心/AABB** 是最基础的统计量；**体素下采样**用哈希网格抽稀，是精度与数据量的权衡；**最近邻**暴力法 \\(O(N)\\)、生产上用 kd-tree（`kiddo`）。
- **平面拟合**：PCA（中心化 + SVD 取最小奇异向量）快但怕离群点；**RANSAC**（随机 3 点定平面 + 数内点 + 精修）稳健，实测参数误差落在千分之一量级。

## 动手练习

1. **改体素尺寸**：把 `voxel_downsample` 的体素尺寸从 0.5 改成 0.2 和 1.0，观察下采样后点数怎么变，想想为什么点云"越平"、下采样比例越低。
2. **半径搜索**：仿照 `nearest`，写一个 `radius_search(pts, q, r)` 返回距 \\(q\\) 小于 \\(r\\) 的所有点下标——这是点云聚类和法向估计的基础操作。
3. **提取内点做地面分割**：RANSAC 返回内点后，把"内点（地面）"和"外点（障碍物）"分成两片点云分别统计点数，这正是扫地机器人区分"地"与"障碍"的核心一步。
4. **进阶**：给 `make_cloud` 再叠加**第二个平面**（如一面墙），思考如何用 RANSAC "拟合一个平面 → 移除其内点 → 在剩下的点里再拟合下一个平面"来依次分割出多个平面。

下一章是第十四部分的收尾：我们把标定、立体、对极几何、点云这些积木串起来，看看多张照片是怎么"长出"一片 3D 点云的——**运动恢复结构（SfM）与视觉 SLAM**。👉 [第 68 章](ch68-sfm-slam.md)
