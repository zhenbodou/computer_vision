# 第 18 章 连通域与轮廓分析

> **本章导读**
> - 学完你能：给二值图里的每个白色斑块**编号**（连通域标记），从而**数出画面里有几个物体**。
> - 会算每个物体的**面积、包围盒、质心**，并据此**过滤掉小噪点**、给物体画框标注。
> - 搞懂三个高频坑：**4 连通 vs 8 连通**怎么影响个数、`background` 背景参数、`find_contours::<u32>` 的坐标类型标注。
> - 打通一条完整链路：二值化（[第 16 章](ch16-threshold.md)）→ 形态学去噪（[第 17 章](ch17-morphology.md)）→ **数物体**。

## 场景：这画面里到底有几个物体？

这是机器视觉里最朴素、也最高频的一个问题。产线上"这一帧有几个零件？"、监控里"路口有几辆车？"、显微镜下"这视野里有多少个细胞？"——统统绕不开它。

经过前两章，我们已经能把图像二值化、再用形态学把掩膜修干净，得到一张"白色物体 + 黑色背景"的清爽二值图。可对计算机来说，它看到的还只是**一大片白像素**，并不知道"这是 3 个独立物体"。本章要做的，就是把"一片白像素"翻译成"**第 1 个物体面积 3100、中心在 (164,54)；第 2 个……**"这样结构化的信息。

干这件事有两条互补的路子：**连通域标记**（给每一"块"编号，偏向区域统计）和**轮廓提取**（勾出每个物体的边界曲线，偏向形状分析）。我们两个都学。

## 连通域：相邻的白像素是"一伙的"

先看直觉。一堆白像素，只要**挨在一起**，就属于同一个物体；中间隔着黑背景的，就是不同物体。这些"连成一片"的前景像素集合，就叫一个**连通域（connected component）**，也叫连通块。

那"挨在一起"到底算多近？这就引出一个必须先讲清楚的概念——**连通性（connectivity）**：

```text
        4 连通（只认上下左右）        8 连通（连对角也认）
              . N .                       N N N
              W O E                       W O E
              . S .                       N N N
        (O=当前像素, 标出的是它的"邻居")
```

- **4 连通**：只有上、下、左、右四个方向的像素算邻居。
- **8 连通**：在 4 连通基础上，再加上四个**对角**方向。

这个选择会**直接改变数出来的物体个数**。看一个经典情形：两个方块只在**对角**上碰了一下角——

```text
        X X X . . .
        X X X . . .          ← 左上方块
        X X X . . .
        . . . Y Y Y
        . . . Y Y Y          ← 右下方块（与左上仅对角相接）
        . . . Y Y Y
```

用 **8 连通**，对角也算相连，两个方块会被判成**1 个**物体；用 **4 连通**，对角不算，它们是**2 个**独立物体。这不是 bug，是定义使然——待会儿代码里我们会亲手跑出 `4 连通 = 2、8 连通 = 1` 的结果。

> 📖 **术语**：**连通域标记（connected component labelling）**。扫描整张二值图，给每一个连通块分配一个唯一的整数编号（label）。imageproc 的 `connected_components` 返回一张 `Image<Luma<u32>>`：**背景像素标为 0，各前景块依次标为 1, 2, 3 …（连续不跳号）**。于是"数物体个数"就等于"最大的那个编号"。

## 由标记到属性：面积、包围盒、质心

拿到每个像素的编号后，只需**遍历一遍**标记图，按编号把统计量累加起来，就能得到每个物体的关键属性：

- **面积（area）**：这个编号的像素总数。噪点面积小、真物体面积大——这正是**过滤噪点**的依据。
- **包围盒（bounding box）**：能框住该物体的最小矩形，由它所有像素的 \\( x_{\min}, y_{\min}, x_{\max}, y_{\max} \\) 决定。
- **质心（centroid）**：物体的"重心"，即所有像素坐标的平均值 \\( (\bar{x}, \bar{y}) = \left(\frac{1}{N}\sum x_i,\ \frac{1}{N}\sum y_i\right) \\)，其中 \\( N \\) 是面积。

有了面积就能滤掉噪点，有了包围盒就能画框，有了质心就能标位置、做跟踪。数物体的活儿，核心就这三样。

## 轮廓：勾出物体的边界

连通域给的是"**区域**"（哪些像素属于某物体），而**轮廓（contour）**给的是"**边界**"（沿物体外沿走一圈的点序列）。imageproc 的 `find_contours` 用经典的 Suzuki-Abe 边界跟踪算法，返回一组 `Contour`，每条轮廓带三样信息：

- `points`：边界上的点序列（可用来算周长、拟合形状）。
- `border_type`：`Outer`（**外轮廓**，物体与外部背景的分界）或 `Hole`（**内孔**，物体内部那圈包着黑洞的边界）。
- `parent`：父轮廓的下标（描述"谁套着谁"的层级关系）。

所以，**外轮廓的条数 ≈ 物体个数**（每个物体一圈外沿），而每个内部黑洞会额外贡献一条 `Hole` 轮廓——这刚好呼应了上一章"闭运算填小洞"的话题。

## 动手写代码：数物体、过滤噪点、画框标注

下面这个例子一气呵成：① 合成一张有 3 个大斑块（其中一个矩形还挖了洞）加若干小噪点的二值图；② 用连通域数出总块数；③ 按面积过滤噪点、算出每个物体的面积/包围盒/质心并画框；④ 演示 4 vs 8 连通的差别；⑤ 演示 `find_contours` 的外轮廓/内孔。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch18_contours.rs`）：

```rust
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::contours::{find_contours, BorderType};
use imageproc::drawing::{draw_cross_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use imageproc::region_labelling::{connected_components, Connectivity};

/// 在二值图上画一个实心圆（前景白）
fn fill_circle(img: &mut GrayImage, cx: i32, cy: i32, r: i32) {
    for y in (cy - r).max(0)..(cy + r).min(img.height() as i32) {
        for x in (cx - r).max(0)..(cx + r).min(img.width() as i32) {
            if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                img.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
}

/// 合成主场景：3 个大斑块（1 个矩形挖了洞）+ 若干小噪点
fn make_scene(w: u32, h: u32) -> GrayImage {
    let mut img: GrayImage = ImageBuffer::new(w, h);
    fill_circle(&mut img, 55, 60, 28); // 大斑块 1：圆
    for y in 30..80 {
        for x in 130..200 {
            img.put_pixel(x, y, Luma([255])); // 大斑块 2：矩形
        }
    }
    for y in 45..65 {
        for x in 155..175 {
            img.put_pixel(x, y, Luma([0])); // 在矩形里挖一个洞
        }
    }
    fill_circle(&mut img, 175, 135, 22); // 大斑块 3：圆
    // 小噪点（应被面积过滤掉）
    for (x, y) in [(20, 150), (110, 20), (60, 155), (205, 110)] {
        img.put_pixel(x, y, Luma([255]));
    }
    for (x, y) in [(224, 160), (225, 160), (224, 161), (225, 161)] {
        img.put_pixel(x, y, Luma([255])); // 一个 2x2 噪块
    }
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = make_scene(240, 180);

    // ---- 1) 连通域标记：数出「连通块」个数 ----
    // 背景是黑色，所以 background 传 Luma([0])
    let labels = connected_components(&scene, Connectivity::Eight, Luma([0u8]));
    // 标签被压缩为 1..=n，背景为 0，所以最大标签值 = 连通块个数
    let num = labels.as_raw().iter().copied().max().unwrap_or(0);
    println!("8 连通：连通块总数（含噪点）= {}", num);

    // 遍历标记图，按标签累加：面积、包围盒、质心
    let n = num as usize;
    let mut area = vec![0u32; n + 1];
    let (mut min_x, mut min_y) = (vec![u32::MAX; n + 1], vec![u32::MAX; n + 1]);
    let (mut max_x, mut max_y) = (vec![0u32; n + 1], vec![0u32; n + 1]);
    let (mut sum_x, mut sum_y) = (vec![0u64; n + 1], vec![0u64; n + 1]);
    for (x, y, p) in labels.enumerate_pixels() {
        let l = p[0] as usize;
        if l == 0 {
            continue; // 背景
        }
        area[l] += 1;
        min_x[l] = min_x[l].min(x);
        min_y[l] = min_y[l].min(y);
        max_x[l] = max_x[l].max(x);
        max_y[l] = max_y[l].max(y);
        sum_x[l] += x as u64;
        sum_y[l] += y as u64;
    }

    // ---- 2) 按面积过滤噪点，数出「真正的物体」并画框 ----
    let min_area = 100u32;
    let mut objects = 0;
    // 把二值图铺成灰底 RGB 画布，方便看清彩色的框
    let mut canvas: RgbImage =
        ImageBuffer::from_fn(scene.width(), scene.height(), |x, y| {
            let v = scene.get_pixel(x, y)[0];
            Rgb([v, v, v])
        });
    println!("\n面积 >= {} 视为物体：", min_area);
    for l in 1..=n {
        if area[l] < min_area {
            continue; // 噪点，跳过
        }
        objects += 1;
        let (bx, by) = (min_x[l], min_y[l]);
        let (bw, bh) = (max_x[l] - min_x[l] + 1, max_y[l] - min_y[l] + 1);
        let cx = (sum_x[l] / area[l] as u64) as i32;
        let cy = (sum_y[l] / area[l] as u64) as i32;
        println!(
            "  物体#{objects}: 面积={:>4}  包围盒=({bx},{by},{bw}x{bh})  质心=({cx},{cy})",
            area[l]
        );
        draw_hollow_rect_mut(
            &mut canvas,
            Rect::at(bx as i32, by as i32).of_size(bw, bh),
            Rgb([0, 255, 0]), // 绿色包围盒
        );
        draw_cross_mut(&mut canvas, Rgb([255, 0, 0]), cx, cy); // 红色质心
    }
    println!("过滤后物体个数 = {}", objects);

    // ---- 3) 4 连通 vs 8 连通：对角相接的两块 ----
    let mut tiny: GrayImage = ImageBuffer::new(9, 9);
    for y in 1..4 {
        for x in 1..4 {
            tiny.put_pixel(x, y, Luma([255])); // 方块 A
        }
    }
    for y in 4..7 {
        for x in 4..7 {
            tiny.put_pixel(x, y, Luma([255])); // 方块 B（与 A 仅对角相接）
        }
    }
    let count = |img: &GrayImage, c| {
        connected_components(img, c, Luma([0u8]))
            .as_raw()
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    };
    println!(
        "\n对角相接的两块：4 连通 = {} 个，8 连通 = {} 个",
        count(&tiny, Connectivity::Four),
        count(&tiny, Connectivity::Eight)
    );

    // ---- 4) 轮廓提取：find_contours 需指定坐标类型 ::<u32> ----
    let contours = find_contours::<u32>(&scene);
    let outer = contours.iter().filter(|c| c.border_type == BorderType::Outer).count();
    let holes = contours.iter().filter(|c| c.border_type == BorderType::Hole).count();
    println!(
        "\n轮廓总数 = {}（外轮廓 Outer = {}，内孔 Hole = {}）",
        contours.len(),
        outer,
        holes
    );
    println!("提示：find_contours 不做面积过滤，噪点也各算一条外轮廓。");

    canvas.save("out_ch18_boxes.png")?;
    println!("\n已保存可视化：out_ch18_boxes.png");
    Ok(())
}
```

运行后终端打印（数字真实可复现）：

```text
8 连通：连通块总数（含噪点）= 8

面积 >= 100 视为物体：
  物体#1: 面积=3100  包围盒=(130,30,70x50)  质心=(164,54)
  物体#2: 面积=2451  包围盒=(27,32,56x56)  质心=(54,59)
  物体#3: 面积=1515  包围盒=(153,113,44x44)  质心=(174,134)
过滤后物体个数 = 3

对角相接的两块：4 连通 = 2 个，8 连通 = 1 个

轮廓总数 = 9（外轮廓 Outer = 8，内孔 Hole = 1）
提示：find_contours 不做面积过滤，噪点也各算一条外轮廓。
```

把结果和原理对一遍，全都严丝合缝：

1. **总块数 8** = 3 个大斑块 + 5 个噪点（4 个单像素 + 1 个 2×2 块）。可见噪点也各算一个连通域，**必须过滤**。
2. **面积过滤后剩 3 个物体**，噪点全被 `min_area` 挡在门外。
3. **面积对得上**：物体#1 是那个挖了洞的矩形，`70×50 − 20×20 = 3500 − 400 = 3100`，分毫不差；两个圆的面积也接近 \\( \pi r^2 \\)（`2451 ≈ π·28²`、`1515 ≈ π·22²`）。包围盒尺寸 `56×56`、`44×44` 正是圆的直径。
4. **4 连通 = 2、8 连通 = 1**：对角相接的两块，连通性选择直接改变了个数——这个坑亲眼所见。
5. **轮廓：8 条外轮廓 + 1 条内孔**。8 条外轮廓 = 3 大 + 5 噪（`find_contours` 同样不滤噪点），那 1 条内孔正是矩形里挖的洞。

打开 `out_ch18_boxes.png`，你会看到三个物体被绿色包围盒框住、红色十字精准落在各自中心——一张"数出 3 个物体"的可视化就完成了。

## 常见坑与提示

> 🕳️ **坑**：**忘了先二值化，直接喂灰度图。** `connected_components` 是按"像素值**是否相等**"来分组的。要是把一张有几百种灰阶的灰度图直接丢进去，它会把每一小片相同灰阶都当成一个连通块，瞬间炸出上千个"物体"。**动手前务必先阈值化成 0/255 的二值图**（见 [第 16 章](ch16-threshold.md)）。

> ⚠️ **注意**：**`background` 背景参数别传错。** `connected_components(&img, conn, background)` 的第三个参数是"哪种像素算背景"。前景白、背景黑时传 `Luma([0u8])`；要是手滑传成 `Luma([255])`，就等于宣布"白色是背景"，你的物体反而不会被标记。

> 🕳️ **坑**：**4 连通还是 8 连通，先想清楚再选。** 两个仅对角相接的物体，8 连通会把它们**并成一个**（可能少数）；而一条本该连续、但像素锯齿只在对角处搭着的细线，用 4 连通可能被**断成好几段**（可能多数）。数独立物体一般用 8 连通更符合直觉，但没有万能答案，按场景定。

> ⚠️ **注意**：**`find_contours` 必须标注坐标类型。** 要写成 `find_contours::<u32>(&gray)`——尖括号里的类型是轮廓点坐标用什么整数类型。省略它编译器无法推断，直接报错。`u32` 通常够用；除非你确定图很小，否则别用 `u16`（坐标一超 65535 就溢出）。

> 💡 **提示**：**连通域和轮廓都不会自动过滤噪点。** 两者都会把每个小白点老老实实算进去。真正"数出几个物体"的关键一步，是你**自己设一个面积（或包围盒尺寸）阈值**把小块滤掉。阈值多大要看物体和噪点的尺度差——先把各块面积打印出来看看分布，再定这个数。

## 本章小结

- **连通域**是相邻前景像素的集合，一个连通块 ≈ 一个物体；**连通域标记**给每块编号，`connected_components` 返回 `Image<Luma<u32>>`，背景 0、前景 1..=n，**最大编号 = 物体总数**。
- **4 连通**只认上下左右，**8 连通**连对角也认；对角相接的物体在两种连通下个数不同，需按场景选择。
- 遍历标记图即可算出每块的**面积、包围盒、质心**；用**面积阈值**过滤掉小噪点，才能得到真正的物体数。
- **轮廓**给的是边界曲线，`find_contours::<u32>` 返回 `Outer`/`Hole` 两类边界；外轮廓数 ≈ 物体数，内孔对应物体内部的洞。
- 三个高频坑：**先二值化**、**`background` 传对**、**`find_contours` 标注坐标类型**；连通域与轮廓都**不自动滤噪**，过滤要自己做。

## 动手练习

1. **调面积阈值**：把 `min_area` 从 `100` 一路调到 `10`、`3000`，观察"物体个数"怎么变——太小会把噪点也算进来，太大会把真物体误杀。
2. **换连通性**：把主场景的 `Connectivity::Eight` 改成 `Four`，看物体数是否变化（本例大块彼此分离，不变）；再把两个圆挪到只对角相接，重跑，亲眼见证个数从变化。
3. **按长宽比过滤**：给每个物体多打印一个长宽比 `bw as f32 / bh as f32`，只保留接近正方形（比值在 0.8~1.25）的物体，体会"多条件过滤"。
4. **进阶——算圆度**：用 `find_contours` 拿到外轮廓，把相邻点距离累加成周长 \\( P \\)，再用面积 \\( A \\) 算"圆度" \\( 4\pi A / P^2 \\)（圆≈1、方更小），据此区分场景里的圆和矩形。

数出物体、框住它们之后，我们再进一步：**怎么检测画面里的直线和圆这类规则形状？** 下一章进入经典而优雅的霍夫变换。👉 [第 19 章 霍夫变换：检测直线与圆](ch19-hough.md)
