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
