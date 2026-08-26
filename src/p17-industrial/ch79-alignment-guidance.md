# 第 79 章 视觉定位、机器人引导与手眼标定

> **本章导读**
> - 学完你能：用视觉求出工件的**位姿 \\( (x, y, \theta) \\)**，并把它一路换算到机器人能用的坐标，去"抓/贴/拧"。
> - 掌握定位的骨架：**坐标系链**与变换记号 \\( {}^{A}T_{B} \\)、模板定位求平移、旋转搜索求角度、**平面用单应、有高度用 PnP**。
> - 讲清**手眼标定** \\( AX = XB \\) 的概念（眼在手上 vs 眼在手外），并给一个可运行的**像素→机器人**平面仿射标定，打印映射误差。
> - 建立专家心智：定位不止给位姿，还要给**匹配质量/歧义**；对称/重复/遮挡会有**多个等价解**，上层运动规划**不能盲信最高分**；验收看**闭环取放误差**。

## 场景：机械臂怎么知道去哪抓？

贴片机要把一颗芯片贴到 PCB 的焊盘上，机械臂要从料盘里抓起一个来料不定、还可能转了角度的工件。它们都面对同一个问题：**工件此刻在哪、转了多少度？** 相机拍一张，视觉算出工件的位姿 \\( (x, y, \theta) \\)，再告诉机械臂"去这个坐标、转这个角度抓"。这就是**视觉定位与引导（vision-guided robotics）**。

难点不在"拍到工件"，而在**坐标的接力**：相机看到的是**像素**，机械臂听得懂的是**它自己基座下的毫米**。中间隔着好几层坐标系，一层没对齐，机械臂就会精准地抓向一个错误的地方。所以这一章，我们先把坐标系链理清楚，再谈定位与标定。

## 原理一：坐标系链与变换记号

一个视觉引导系统里，至少站着这么几个坐标系：**图像**（像素）、**相机**、**标定板**、**工件**、**机器人基座（base）**、**末端执行器（end-effector / tool）**。定位的本质，是把"工件在图像里的位置"一路搬到"工件在机器人基座下的位姿"。

工程上用一个清爽的记号描述坐标变换：\\( {}^{A}T_{B} \\) 表示"把 B 坐标系里的点，表达到 A 坐标系里"。它们能像链条一样**串起来**：

```text
   图像 ──T──▶ 相机 ──T──▶ 基座 ──T──▶ 末端 ──T──▶ 抓取点
  (像素)      (标定得到)   (手眼标定)   (机器人正解)

   目标： {}^{base}T_{workpiece}  = 把工件位姿表达到机器人基座下
```

复合就是矩阵相乘，比如 \\( {}^{base}T_{obj} = {}^{base}T_{cam}\cdot{}^{cam}T_{obj} \\)。这里藏着两个新手必栽的坑：**乘法方向**（谁乘谁、别乘反）和**右手系/单位**（弧度还是度、毫米还是米）。

> 💡 **提示**：Rust 的类型系统是对付这个坑的利器。把变换做成**强类型**——`Transform<Base, Cam>` 只能和 `Transform<Cam, Obj>` 相乘得到 `Transform<Base, Obj>`，乘反了直接**编译不过**。再给逆、复合、单位写单元测试。这比"全用裸 `Matrix4` 靠注释提醒自己"可靠一个数量级（详见本章"专家考核"）。

## 原理二：模板定位求 (x, y, θ)

先解决"平移"。[第 20 章 模板匹配](../p03-classic/ch20-template-matching.md) 已经教过：拿工件模板在图上滑动、算**归一化互相关（NCC）**，得分最高处就是工件位置。但模板匹配**不抗旋转**——工件一转，逐像素比对就失配。

最朴素也最好懂的办法是**旋转搜索**：把模板转成一系列候选角度，每个角度都做一遍 NCC，**得分最高的那个角度 + 位置**就是估计位姿。为了不让代价爆炸，用**粗到细（coarse-to-fine）**两级：先粗步长扫一圈定个大概，再在最佳角附近细步长精修。

但专家会多问一句：**光有位姿够吗？** 不够。定位还应输出：

- **匹配质量**：NCC 峰值多高（越接近 1 越可信）、峰有多尖锐（尖锐=位置可观测性好）。
- **歧义/多解**：对称工件（如正方形、圆）、重复纹理（如网格）、局部遮挡，都会让得分图出现**多个几乎一样高**的峰——存在多个等价解。

> ⚠️ **注意**：**上层运动规划绝不能盲信"最高分"。** 若最高分和次高分咬得很近（比如差 0.5%），说明存在歧义解，此时贸然去抓可能抓错方向 180°。稳妥做法是：把 Top-K 候选连同分数一起报上去，由上层结合先验（上料方向、CAD、上一帧）裁决，或干脆判为"无法可靠定位"触发人工。

## 原理三：平面用单应，有高度用 PnP

把像素位姿变成物理位姿，要看工件"平不平"：

- **平面工件**（贴片、丝印、平板类）：工件在一个平面上，像素↔物理是一个**单应/仿射变换**就能搞定（[第 10 章 几何变换](../p02-image-basics/ch10-geometry.md)）。本章动手部分就做这个可运行的平面标定。
- **有高度/三维姿态的工件**：单一平面映射会因**透视**和高度而系统性出错，必须请出**相机模型 + PnP**（由若干 3D-2D 对应点解出物体六自由度位姿，[第 66 章 对极几何与 PnP](../p14-3d/ch66-epipolar-pnp.md)），配合相机标定（[第 64 章](../p14-3d/ch64-calibration.md)）。

## 原理四：手眼标定 AX = XB

最后一环：相机和机器人之间那个变换 \\( X \\) 怎么求？这就是**手眼标定（hand-eye calibration）**。两种安装方式：

```text
  眼在手上 (eye-in-hand)          眼在手外 (eye-to-hand)
  相机装在机械臂末端              相机固定在支架上
   [末端]──[相机]                  [相机]      [末端]──[夹爪]
      \                              (看着)        /
       ▼看                                        ▼
     工件/标定板                              工件/标定板
```

无论哪种，都归结为经典方程 \\( AX = XB \\)：让机器人摆若干个不同位姿，各拍一张标定板；\\( A \\) 是相邻两次**机器人末端**的相对运动、\\( B \\) 是相邻两次**相机看标定板**的相对运动，未知的 \\( X \\) 就是要求的手眼变换。解它需要足够多样的采样。

> 🕳️ **坑**：**采样位姿必须覆盖足够的旋转轴和工作空间，否则方程退化。** 如果你只做**纯平移**或只绕一个轴转一点点，\\( AX = XB \\) 会病态、解不稳。要让机械臂绕**不同轴**明显地转、在工作空间里**大范围**摆姿势。

纯 Rust 里目前**没有**开箱即用的 \\( AX = XB \\) 手眼标定库（这块生态还缺，诚实告知）。要落地有两条路：自己用 nalgebra 实现经典的 Tsai-Lenz 或对偶四元数解法；或对**平面**场景退化成"像素→机器人"的仿射标定——这正是下面能跑的代码。

## 动手写代码：位姿估计 + 平面手眼标定

两个实验：**A** 把工件模板转 `12.7°` 贴到已知位置，用粗到细旋转搜索定位，打印估计位姿 vs 真值；**B** 用 6 对"像素↔机器人坐标"点，最小二乘解平面仿射（平面版手眼），打印映射误差。

`Cargo.toml` 依赖：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
nalgebra = "0.35"
```

完整代码（`examples/ch79_alignment.rs`）：

```rust
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::{draw_cross_mut, draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::geometric_transformations::{rotate_about_center, Border, Interpolation};
use imageproc::rect::Rect;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};
use nalgebra::{DMatrix, DVector};

const BG: u8 = 40; // 统一背景灰度：模板、旋转填充、场景都用它，拼接无缝
const TS: u32 = 44; // 模板边长

/// 合成一个“有方向感”的工件模板：亮矩形 + 偏置的亮圆（打破旋转对称，θ 才唯一可辨）。
fn make_template() -> GrayImage {
    let mut t = GrayImage::from_pixel(TS, TS, Luma([BG]));
    draw_filled_rect_mut(&mut t, Rect::at(6, 15).of_size(32, 14), Luma([200]));
    draw_filled_circle_mut(&mut t, (13, 22), 5, Luma([255])); // 靠左端的“键”，破对称
    t
}

/// 把模板顺时针旋转 deg 度（其余部分用 BG 填充），保持 TS×TS 尺寸。
fn rotate_template(t: &GrayImage, deg: f64) -> GrayImage {
    rotate_about_center(
        t,
        (deg as f32).to_radians(),
        Interpolation::Bilinear,
        Border::Constant(Luma([BG])),
    )
}

/// 在 angles 里逐个旋转模板、做归一化互相关，返回得分最高的 (角度°, 左上角位置, 得分)。
fn search_pose(scene: &GrayImage, t: &GrayImage, angles: &[f64]) -> (f64, (u32, u32), f32) {
    let mut best = (0.0f64, (0u32, 0u32), f32::MIN);
    for &a in angles {
        let rot = rotate_template(t, a);
        let map = match_template(scene, &rot, MatchTemplateMethod::CrossCorrelationNormalized);
        let e = find_extremes(&map);
        if e.max_value > best.2 {
            best = (a, e.max_value_location, e.max_value);
        }
    }
    best
}

fn demo_pose() {
    let template = make_template();
    // —— 造真值：把模板转 θ_true 度，贴到场景 (px,py) 处 ——
    let (px, py, theta_true) = (80i64, 55i64, 12.7f64);
    let mut scene = GrayImage::from_pixel(200, 150, Luma([BG]));
    let stamped = rotate_template(&template, theta_true);
    image::imageops::overlay(&mut scene, &stamped, px, py);
    scene.save("out_ch79_scene.png").ok();

    // 工件中心真值（旋转是绕patch中心做的，故中心 = 左上角 + 半个模板）
    let (cx_true, cy_true) = (px as f64 + TS as f64 / 2.0, py as f64 + TS as f64 / 2.0);

    // —— 第 1 级：粗搜索（步长 3°）——
    let coarse: Vec<f64> = (0..=30).step_by(3).map(|d| d as f64).collect();
    let (a_c, _, s_c) = search_pose(&scene, &template, &coarse);

    // —— 第 2 级：在粗结果附近细搜索（步长 0.5°）——
    let mut fine = Vec::new();
    let mut a = a_c - 2.0;
    while a <= a_c + 2.0 + 1e-9 {
        fine.push(a);
        a += 0.5;
    }
    let (a_f, loc, s_f) = search_pose(&scene, &template, &fine);

    let (cx_est, cy_est) = (loc.0 as f64 + TS as f64 / 2.0, loc.1 as f64 + TS as f64 / 2.0);
    let pos_err = ((cx_est - cx_true).powi(2) + (cy_est - cy_true).powi(2)).sqrt();

    println!("[A 位姿估计]");
    println!("  粗搜索最佳角 = {a_c}°（得分 {s_c:.4}）");
    println!("  细搜索最佳角 = {a_f}°（得分 {s_f:.4}）");
    println!("  估计位姿 (x,y,θ) = ({cx_est:.1}, {cy_est:.1}, {a_f:.2}°)");
    println!("  真值位姿 (x,y,θ) = ({cx_true:.1}, {cy_true:.1}, {theta_true}°)");
    println!("  位置误差 = {pos_err:.3} px，角度误差 = {:.3}°\n", (a_f - theta_true).abs());

    // 可视化：在场景上框出定位结果、标中心
    let mut vis: RgbImage = image::DynamicImage::ImageLuma8(scene).to_rgb8();
    draw_hollow_rect_mut(&mut vis, Rect::at(loc.0 as i32, loc.1 as i32).of_size(TS, TS), Rgb([0, 255, 0]));
    draw_cross_mut(&mut vis, Rgb([255, 0, 0]), cx_est as i32, cy_est as i32);
    vis.save("out_ch79_located.png").ok();
}

/// 对一维目标 target = a*px + b*py + c 做最小二乘，返回 [a,b,c]。
fn solve_affine_row(pxs: &[f64], pys: &[f64], target: &[f64]) -> [f64; 3] {
    let n = pxs.len();
    let mut a = Vec::with_capacity(n * 3);
    for i in 0..n {
        a.push(pxs[i]);
        a.push(pys[i]);
        a.push(1.0);
    }
    let am = DMatrix::from_row_slice(n, 3, &a);
    let bv = DVector::from_column_slice(target);
    let sol = am.svd(true, true).solve(&bv, 1e-9).expect("SVD solve");
    [sol[0], sol[1], sol[2]]
}

fn demo_hand_eye() {
    // —— 真值仿射：机器人坐标(mm) = M · [px, py, 1] ——
    // 相机与机器人平面差一个 缩放 s、旋转 φ、平移 (tx,ty)
    let (s, phi) = (0.25f64, 8.0f64.to_radians()); // 0.25 mm/px，转 8°
    let (tx, ty) = (12.0f64, -30.0f64);
    let mx = [s * phi.cos(), -s * phi.sin(), tx];
    let my = [s * phi.sin(), s * phi.cos(), ty];
    let map_true = |px: f64, py: f64| {
        (mx[0] * px + mx[1] * py + mx[2], my[0] * px + my[1] * py + my[2])
    };

    // —— 标定点：6 个像素点 + 对应机器人读数（叠加固定小噪声模拟测量误差）——
    let pix = [(40.0, 30.0), (280.0, 35.0), (300.0, 200.0), (60.0, 210.0), (160.0, 120.0), (200.0, 80.0)];
    let noise = [(0.03, -0.04), (-0.02, 0.05), (0.04, 0.01), (-0.05, -0.02), (0.01, 0.03), (0.02, -0.03)];
    let (mut pxs, mut pys, mut rxs, mut rys) = (vec![], vec![], vec![], vec![]);
    for (i, &(px, py)) in pix.iter().enumerate() {
        let (rx, ry) = map_true(px, py);
        pxs.push(px);
        pys.push(py);
        rxs.push(rx + noise[i].0);
        rys.push(ry + noise[i].1);
    }

    // —— 最小二乘解出 M_est ——
    let ex = solve_affine_row(&pxs, &pys, &rxs);
    let ey = solve_affine_row(&pxs, &pys, &rys);
    let map_est = |px: f64, py: f64| {
        (ex[0] * px + ex[1] * py + ex[2], ey[0] * px + ey[1] * py + ey[2])
    };

    // RMS 残差（标定点上，单位 mm）
    let mut se = 0.0;
    for &(px, py) in &pix {
        let (tx_, ty_) = map_true(px, py);
        let (ex_, ey_) = map_est(px, py);
        se += (tx_ - ex_).powi(2) + (ty_ - ey_).powi(2);
    }
    let rms = (se / pix.len() as f64).sqrt();

    // 拿一个“没参与标定”的新像素点验证
    let (npx, npy) = (123.0, 156.0);
    let (gt_x, gt_y) = map_true(npx, npy);
    let (es_x, es_y) = map_est(npx, npy);
    let test_err = ((gt_x - es_x).powi(2) + (gt_y - es_y).powi(2)).sqrt();

    println!("[B 像素→机器人 仿射标定]（6 对点最小二乘）");
    println!("  真值 M 第1行 = [{:.5}, {:.5}, {:.3}]", mx[0], mx[1], mx[2]);
    println!("  估计 M 第1行 = [{:.5}, {:.5}, {:.3}]", ex[0], ex[1], ex[2]);
    println!("  标定点 RMS 残差 = {rms:.4} mm");
    println!("  新点像素({npx},{npy}) → 机器人估计=({es_x:.3},{es_y:.3}) mm，真值=({gt_x:.3},{gt_y:.3}) mm");
    println!("  新点映射误差 = {test_err:.4} mm");
}

fn main() {
    demo_pose();
    demo_hand_eye();
}
```

> ⚠️ **注意**：位姿搜索里"旋转模板 × 多角度 × 全图 NCC"计算量不小，**debug 模式约要 20 多秒**。请用 `cargo run --release --example ch79_alignment` 运行，**约 1 秒**出结果（数字与 debug 完全一致，因为算法是确定性的）。这本身就是下面要讲的"θ 搜索代价"。

运行后终端打印（数字真实可复现）：

```text
[A 位姿估计]
  粗搜索最佳角 = 12°（得分 0.9995）
  细搜索最佳角 = 12.5°（得分 1.0000）
  估计位姿 (x,y,θ) = (102.0, 77.0, 12.50°)
  真值位姿 (x,y,θ) = (102.0, 77.0, 12.7°)
  位置误差 = 0.000 px，角度误差 = 0.200°

[B 像素→机器人 仿射标定]（6 对点最小二乘）
  真值 M 第1行 = [0.24757, -0.03479, 12.000]
  估计 M 第1行 = [0.24765, -0.03489, 12.001]
  标定点 RMS 残差 = 0.0250 mm
  新点像素(123,156) → 机器人估计=(37.019,12.888) mm，真值=(37.023,12.900) mm
  新点映射误差 = 0.0129 mm
```

逐条对账：

1. **位姿估计**：粗搜索（步长 3°）先锁定到 12°、得分 0.9995；细搜索（步长 0.5°）精修到 **12.5°、得分 1.0000**。位置误差 **0.000 像素**（模板从原图裁的，NCC 峰精确）；角度误差 **0.200°**——正好卡在细搜索的 0.5° 网格上（真值 12.7° 最近的网格点是 12.5°）。**想更准就把角度步长调更细，代价是更多次 NCC**，这就是精度与算力的权衡。
2. **平面手眼标定**：估计出的 M 第一行与真值几乎一致；6 个标定点的 **RMS 残差仅 0.0250 毫米**（就是我们注入的测量噪声量级）；拿一个**没参与标定**的新点验证，映射误差 **0.0129 毫米**——说明标定出的映射能泛化到新位置，可以拿去引导机器人了。

打开 `out_ch79_located.png`：绿框稳稳套住旋转后的工件、红十字落在其中心。

## 常见坑与提示

> 🕳️ **坑**：**模板匹配不抗旋转/尺度，旋转搜索代价随精度飙升。** 每加一档角度精度，就多一轮全图 NCC。工程上用粗到细、图像金字塔（[第 21 章](../p03-classic/ch21-pyramids.md)）、或先用几何/边缘特征估角度来省算力，别一上来就全范围细扫。

> ⚠️ **注意**：**坐标系必须自始至终一致。** 像素坐标 y 轴常朝下、机器人坐标 y 轴常朝上；角度有的用度有的用弧度、有的顺时针有的逆时针。链条里任何一环方向或单位错了，机械臂就会稳稳地抓偏。用强类型 + 单元测试把它锁死。

> 🕳️ **坑**：**对称/重复/遮挡 → 多等价解。** 正方形工件转 90° 长得一样、网格纹理到处都像。得分图会有多个高峰。别只取最高分，要看**次高峰与最高峰的差距**，太近就判歧义、交给上层或人工。

> 💡 **提示**：**物理精度依赖标定，且验收要看闭环。** 映射误差以毫米计，全靠标定质量；而标定方程残差小 ≠ 抓得准。真正的验收是**让机器人真的去取放/对位，量最终的落点误差**——那才是客户在乎的数字。

## 本章小结

- 视觉定位求工件位姿 \\( (x, y, \theta) \\)，核心是把像素一路搬到机器人基座坐标，中间是**坐标系链** \\( {}^{A}T_{B} \\) 的复合，方向/单位/右手系要用强类型锁死。
- **模板定位 + 旋转搜索**求 \\( (x, y, \theta) \\)，粗到细省算力；定位还要输出**匹配质量与歧义**，上层**不能盲信最高分**。
- **平面用单应/仿射、有高度用 PnP**；本章实测旋转搜索位置误差 0.000 像素、角度误差 0.200°（受搜索步长限制）。
- **手眼标定 \\( AX = XB \\)** 求相机↔机器人变换，采样要覆盖多旋转轴与工作空间否则退化；纯 Rust 无现成库，可自实现或退化成平面仿射标定——实测平面标定 RMS 残差 0.025 毫米、新点误差 0.013 毫米。
- 验收看**闭环取放误差**，不是标定残差。

### 专家考核

1. **强类型变换库**：用 Rust 泛型/newtype 把坐标系写进类型（如 `Transform<Base, Cam>`），实现**逆、复合、点变换**，并写单元测试覆盖"乘反了编译不过""逆的逆等于自身""单位/角度约定"。
2. **位姿不确定度**：对定位结果做 **bootstrap**（对边缘点/匹配打分重采样多次求解），估计位姿的**协方差**；对对称/重复工件，报告 Top-K 候选及其分数差，量化歧义。
3. **工作空间网格验证**：在机器人工作空间里布一张网格，逐格做独立的取放/对位验证，报告**位置误差与角度误差的分布**（均值、σ、最坏值）、**失败率**，而不是只给一个平均值。
4. **碰撞安全边界**：结合定位不确定度，给运动规划留出**安全裕度**（宁可判"定位不可靠"停手，也不要带着大歧义去抓，撞坏工件/夹爪）。

下一章，我们把"对得准"用到检测上：先对齐良品，再逐像素找差异，揪出表面的划痕与斑点。👉 [第 80 章 表面缺陷检测：从成像到量产验收](ch80-defect.md)
