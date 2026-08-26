# 第 10 章 几何变换：缩放、裁剪、旋转、仿射、透视

> **本章导读**
> - 搞懂几何变换的本质：**只搬像素的位置，不改像素的颜色**——以及"搬完之后空出来的格子怎么填"（插值）。
> - 用纯 Rust 完成常用几何操作：缩放、裁剪、翻转、旋转（含任意角度），以及更通用的**仿射**与**透视**变换。
> - 建立"坐标系 + 反向映射 + 插值"这三件套的直觉，看懂 `FilterType`、`Interpolation`、`Projection` 分别在管什么。
> - 全程用**合成图像**演示，复制即可运行。📐

## 从"拍歪的文档"说起

你一定遇到过这个场景：用手机拍一张纸质合同或名片，结果拍歪了、还带点透视变形（近大远小，纸面成了梯形）。扫描类 App 却能把它"掰正"成一张端端正正的矩形图。再比如：给网页做缩略图要**缩放**，从大图里取一块车牌区域要**裁剪**，把竖着拍的照片转正要**旋转**……

这些操作有个共同的名字：**几何变换（geometric transformation）**。它们的共同点是——**只改变像素待的位置，不改变像素本身的颜色值**。把一张图想象成贴在橡皮膜上的马赛克：缩放是把膜拉大拉小，旋转是把膜转个角度，透视是把膜的四个角各自拽到新位置。马赛克小块的颜色没变，变的是它们落在哪儿。

这和下一章的"点运算"正好互补：点运算改颜色不改位置，几何变换改位置不改颜色。

## 原理：位置怎么搬，空格怎么填

### 坐标系约定

先说清楚坐标。图像里我们统一用**左上角为原点**，x 向右增大，y 向下增大（注意 y 轴朝下，和数学课本相反）。一个像素的位置就是它的整数坐标 \\( (x, y) \\)。

### 反向映射：为什么要"倒着算"

假设我们要把图旋转 30°。最直觉的想法是**前向映射**：遍历原图每个像素，算出它转到新图的哪个位置，再搬过去。但这样有个致命问题——旋转后的坐标通常不是整数，而且新图上会出现**没有任何原像素落进去的空洞**，也会有多个像素挤到同一格的冲突。

所以实际算法都用**反向映射（inverse mapping）**：遍历**输出图**的每个整数像素 \\( (x', y') \\)，用逆变换算出它对应到**原图**的哪个位置 \\( (x, y) \\)，再去原图那里取颜色。这样输出图每个格子都被填满，不留空洞。

```text
      输出图(每个格子都要填)              原图(去这里取色)
      ┌───┬───┬───┐                    ┌───┬───┬───┐
      │(x',y')│  │      逆变换          │  ╲│   │   │
      ├───┼───┼───┤   ──────────▶      ├───╲───┼───┤
      │   │   │   │   算出落点(x,y)     │   │ ╲ │   │
      └───┴───┴───┘   通常不是整数!     └───┴───┴───┘
```

### 插值：非整数落点怎么取色

反向映射算出的 \\( (x, y) \\) 落在原图格子的"缝里"（比如 x = 12.7）。到底取哪个像素的颜色？这就是**插值（interpolation）**要解决的：

- **最近邻（Nearest）**：四舍五入，直接取最近那个整数像素。最快，但放大后会有明显的"马赛克块"和锯齿。
- **双线性（Bilinear / Triangle）**：取周围 2×2 四个像素，按距离加权平均。速度和质量的平衡点，最常用。
- **Lanczos3**：用更大的窗口做加权，缩小/放大都最平滑锐利，但最慢。

一句话选择指南：**放大或要好看用双线性/Lanczos，缩小到很小或要保持硬边（如标签图、二值 mask）用最近邻**。

> 📖 **术语**：`image` 库里缩放用的枚举叫 `FilterType`（`Nearest`/`Triangle`/`CatmullRom`/`Gaussian`/`Lanczos3`，其中 `Triangle` 就是双线性）；`imageproc` 库里旋转/warp 用的插值枚举叫 `Interpolation`（`Nearest`/`Bilinear`/`Bicubic`）。名字不同，思路一致。

### 变换家族：从旋转到透视

几何变换按"能力"从弱到强排成一个家族，都可以用一个 3×3 矩阵 \\( H \\) 描述（把点写成齐次坐标 \\( [x, y, 1]^T \\)）：

| 类型 | 能做什么 | 保持什么 | 自由度 |
|------|---------|---------|--------|
| **平移** Translation | 上下左右挪 | 一切形状、大小、朝向 | 2 |
| **刚体/欧氏** Rigid | 平移 + 旋转 | 形状、大小 | 3 |
| **相似** Similarity | + 等比缩放 | 形状（角度）、平行 | 4 |
| **仿射** Affine | + 拉伸/错切 | **平行线仍平行** | 6 |
| **透视** Perspective | + 近大远小 | **直线仍是直线**（但平行会相交） | 8 |

仿射变换的口诀是"**平行线永远平行**"（正方形最多变成平行四边形）；透视变换则允许平行线相交于一点（正方形能变成任意梯形），这正是"掰正拍歪文档"需要的能力。透视用矩阵表示为：

\\[ \begin{bmatrix} u \\\\ v \\\\ w \end{bmatrix} = \begin{bmatrix} a & b & c \\\\ d & e & f \\\\ g & h & 1 \end{bmatrix} \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix} \\]

算完后再做一次"归一化"：最终像素坐标是 \\( (u/w,\ v/w) \\)——正是这个"除以 \\( w \\)"的步骤带来了"近大远小"（仿射变换里 \\( g = h = 0 \\)，\\( w \\) 恒为 1，也就没有近大远小）。好消息是：这些矩阵你**几乎不用手算**，`imageproc` 的 `Projection` 会帮你搭好。

## 动手写代码

先准备依赖（版本以本书 `code/Cargo.toml` 为准）：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

我们合成一张 256×256 的"棋盘 + 红色竖条"图像。红色竖条是个**朝向标记**，方便你一眼看出图被转/翻/拉成了什么样。

```rust
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use image::imageops::FilterType;
use imageproc::geometric_transformations::{
    rotate_about_center, warp, Border, Interpolation, Projection,
};

/// 合成一张棋盘背景 + 红色竖条（朝向标记）的图，方便观察几何变换效果
fn make_scene(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let cell = 32; // 棋盘格边长
        let checker = ((x / cell) + (y / cell)) % 2 == 0;
        let base: u8 = if checker { 60 } else { 200 };
        // 左上区域画一根红色竖条当"朝向标记"
        if x >= w / 4 && x < w / 4 + 24 && y >= h / 4 && y < h / 4 + 64 {
            Rgb([220, 40, 40])
        } else {
            Rgb([base, base, base])
        }
    })
}
```

### 缩放：resize vs resize_exact，以及插值的肉眼差异

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(256, 256);
    let dynimg = DynamicImage::ImageRgb8(scene.clone());

    // resize: 保持宽高比，把图缩进 128×128 的框内（可能不填满）
    dynimg.resize(128, 128, FilterType::Lanczos3).save("resize_fit.png")?;
    // resize_exact: 强制拉成指定尺寸，不管比例（这里会被压扁）
    dynimg.resize_exact(300, 100, FilterType::Triangle).save("resize_exact.png")?;

    // 对比插值：先缩成 8×8 的小图，再用不同插值放大到 128×128
    let tiny = dynimg.resize_exact(8, 8, FilterType::Nearest);
    tiny.resize_exact(128, 128, FilterType::Nearest).save("up_nearest.png")?;   // 硬边马赛克
    tiny.resize_exact(128, 128, FilterType::Triangle).save("up_bilinear.png")?; // 平滑过渡
```

打开 `up_nearest.png` 和 `up_bilinear.png` 对比：最近邻是清晰的方块格子，双线性则是柔和的渐变。这就是插值算法最直观的差别。

> 💡 **提示**：`resize` 和 `resize_exact` 的区别只在**要不要保持宽高比**。做缩略图用 `resize`（不变形），把图喂给固定输入尺寸的神经网络时常用 `resize_exact`（尺寸必须精确匹配，变形交给模型适应）。

### 裁剪、直角旋转与翻转

这几个是"整像素搬运"，不涉及插值，最省事也最快：

```rust
    // 裁剪：从 (64,64) 起取一块 128×128 的 ROI（感兴趣区域）
    dynimg.crop_imm(64, 64, 128, 128).save("crop.png")?;

    // 90/180/270 度旋转（无损，像素只是换位置）
    dynimg.rotate90().save("rot90.png")?;
    dynimg.rotate180().save("rot180.png")?;
    dynimg.rotate270().save("rot270.png")?;

    // 水平/垂直翻转（镜像）
    dynimg.fliph().save("flip_h.png")?;
    dynimg.flipv().save("flip_v.png")?;
```

> 📖 **术语**：**ROI（Region of Interest，感兴趣区域）**指你只想处理的那一小块画面。工业检测里常先用 `crop_imm` 把镜头固定位置的零件区域抠出来，再做后续分析——既省算力，又排除背景干扰。`crop_imm` 的 `imm` 是 immutable（不可变），它返回一张新图，不动原图。

### 任意角度旋转：rotate_about_center

`rotate90` 这类只能转直角。要转任意角度（比如把歪了 30° 的图掰正），用 `imageproc` 的 `rotate_about_center`。它绕图像中心旋转，输出尺寸不变，转出画面的部分用 `extend` 指定的边界色补上：

```rust
    // 绕中心顺时针旋转 30°，露出的角落填黑色
    let theta = 30.0_f32.to_radians(); // 别忘了角度要转成弧度！
    let rotated = rotate_about_center(
        &scene,                        // 注意：传 RgbImage 引用，不是 DynamicImage
        theta,
        Interpolation::Bilinear,       // 用双线性插值让边缘平滑
        Border::Constant(Rgb([0, 0, 0])), // 空白处填黑
    );
    rotated.save("rotate30.png")?;
```

> ⚠️ **注意**：三角函数和这些旋转 API 一律用**弧度（radian）**，不是角度（degree）。用 `30.0_f32.to_radians()` 转换，别直接传 `30.0`——那会转成大约 1719°（30 弧度）转到你怀疑人生。

### 仿射变换：平移 + 缩放 + 旋转的组合

更通用的做法是用 `Projection` 描述一个变换，再交给 `warp` 执行。多个变换可以用 `and_then` 串起来（先做前面的，再做后面的）：

```rust
    // 组合变换：先平移(20,10)，再缩放(1.2, 0.8)，再旋转 15°
    let proj = Projection::translate(20.0, 10.0)
        .and_then(Projection::scale(1.2, 0.8))
        .and_then(Projection::rotate(15.0_f32.to_radians()));
    let affine = warp(
        &scene,
        proj,                              // ← projection 按值传，不是 &proj！
        Interpolation::Bilinear,
        Border::Constant(Rgb([0, 0, 0])),
    );
    affine.save("affine.png")?;
```

注意 `scale(1.2, 0.8)` 的两个方向系数不同，正方形棋盘格会变成长方形——但**平行线依然平行**，这就是仿射的特征。

### 透视变换：把矩形掰成梯形（或反过来掰正）

回到开头"掰正拍歪文档"的场景。透视变换最方便的构造方式是**给四对对应点**：告诉算法原图的 4 个角应该分别落到输出图的哪 4 个位置，`Projection::from_control_points` 会自动解出那个 3×3 矩阵。

这里我们演示反向的效果——把一张正对的图**故意压成上窄下宽的梯形**（模拟"从下往上仰拍"）：

```rust
    let (w, h) = (scene.width() as f32, scene.height() as f32);
    // 原图四角：左上、右上、右下、左下
    let from = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    // 目标四角：把顶边两角往中间收，形成上窄下宽的梯形
    let to = [(w * 0.18, 0.0), (w * 0.82, 0.0), (w, h), (0.0, h)];
    let persp = Projection::from_control_points(from, to)
        .expect("四点不共线时一定可解");
    let warped = warp(&scene, persp, Interpolation::Bilinear, Border::Constant(Rgb([0, 0, 0])));
    warped.save("perspective.png")?;

    println!("全部完成，查看当前目录下生成的 PNG。");
    Ok(())
}
```

打开 `perspective.png`：原本平行的棋盘竖线现在向上汇聚，产生了"近大远小"的立体感。**反过来**——如果你把拍歪文档的四个角作为 `from`，把一个标准矩形的四个角作为 `to`，`warp` 出来的就是掰正后的文档。这正是扫描 App 的核心原理，我们会在 [第 26 章 单应矩阵与图像对齐](../p04-features/ch26-homography-stitching.md) 深入讲怎么自动找到那四个角。

## 常见坑与提示

> 🕳️ **坑**：`warp` 的第二个参数 `projection` 是**按值传**（`proj`），不是引用（`&proj`）。`extend`（边界填充）必须写成 `Border::Constant(Rgb([0,0,0]))` 这样的枚举，**不能直接塞一个颜色**。这是初学者最容易编译报错的两个点。

> ⚠️ **注意**：`rotate_about_center`、`warp` 作用在**具体的 buffer 类型**（`RgbImage`/`GrayImage`）上，而 `resize`/`crop_imm`/`rotate90` 这些是 `DynamicImage` 的方法。二者之间用 `DynamicImage::ImageRgb8(buf)` 和 `.to_rgb8()` 互转即可。

> 💡 **提示**：`Projection::rotate` 是**绕左上角原点**旋转，而 `rotate_about_center` 是**绕图像中心**旋转。想绕中心做复杂组合变换时，标准技巧是"先平移到中心 → 旋转/缩放 → 再平移回去"，用 `and_then` 串三步即可。

> 🕳️ **坑**：反复缩放会累积模糊。需要多种尺寸时，**从原始大图各缩一次**，别在小图基础上再放大——放大补不回已经丢失的细节，只会越来越糊。

## 本章小结

- 几何变换**只改像素位置、不改颜色**，和点运算互补。实现上用**反向映射**保证输出无空洞，用**插值**决定非整数落点取什么颜色。
- 插值三档：最近邻（快、硬边）、双线性（均衡、最常用）、Lanczos3（最平滑、慢）。`image` 用 `FilterType`，`imageproc` 用 `Interpolation`。
- 缩放 `resize`/`resize_exact`、裁剪 `crop_imm`、直角旋转 `rotate90/180/270`、翻转 `fliph/flipv` 都是 `DynamicImage` 的现成方法。
- 任意角度旋转用 `rotate_about_center`；更通用的仿射/透视用 `Projection`（`translate`/`scale`/`rotate`/`from_control_points`）配 `warp`。
- 变换家族按能力递增：平移 ⊂ 刚体 ⊂ 相似 ⊂ 仿射（平行线保持）⊂ 透视（直线保持、可近大远小）。

## 动手练习

1. **插值对比放大**：把本章 `tiny`（8×8）分别用 `Nearest`、`Triangle`、`Lanczos3` 放大到 256×256，并排比较三者的边缘差异，说说各自适合的场景。
2. **绕中心的相似变换**：用"平移到中心 → 旋转 45° → 平移回去"的三步 `and_then` 组合，让 `warp` 也实现绕中心旋转，验证效果和 `rotate_about_center` 一致。
3. **手动掰正**：合成一张带梯形边框的图（或找一张拍歪的名片），用 `from_control_points` 把梯形四角映射到标准矩形四角，把它"掰正"，观察输出。
4. **进阶**：给 `warp` 的 `extend` 分别换成 `Border::Replicate` 和 `Border::Wrap`，看看转出画面的区域填充方式有什么不同。

下一章，我们转向另一半——**不改位置、只改颜色**的点运算：亮度、对比度、Gamma 与查找表。👉 [第 11 章 点运算](ch11-point-ops.md)
