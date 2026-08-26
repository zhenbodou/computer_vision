//! 第 63 章 背景建模与运动检测
//! 运行：cargo run --example ch63_background_motion
//! 依赖：image 0.25、imageproc 0.27（形态学开运算 + 连通域）；确定性可复现。
//!
//! 合成一段「视频」：静态灰度背景 + 一个移动的亮色方块 + 少量高斯噪声，共 N 帧。
//! 演示两条最经典的运动检测思路：
//!   ① 帧差法：|I_t − I_{t-1}| 二值化（简单，但有「鬼影/拖影」）
//!   ② running-average 背景：B_t = α·I_t + (1−α)·B_{t-1}，再 |I_t − B_t| 二值化
//! 前景掩码用形态学开运算（见第 17 章）去噪，连通域（见第 18 章）框出运动目标。

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::distance_transform::Norm;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::morphology::open;
use imageproc::noise::gaussian_noise;
use imageproc::rect::Rect;
use imageproc::region_labelling::{connected_components, Connectivity};

const W: u32 = 240;
const H: u32 = 160;
const N: usize = 24; // 帧数
const SQ: i64 = 30; // 运动方块边长
const THRESH: u8 = 30; // 差分二值化阈值
const ALPHA: f32 = 0.05; // running-average 学习率（越小背景更新越慢、越"记得住"）

/// 静态背景：左→右渐变 + 两块静止的"场景家具"（深色门 + 浅色地砖）。
/// 这些静止内容不应该出现在前景里——这正是背景建模要证明的。
fn make_background() -> GrayImage {
    let mut bg: GrayImage = ImageBuffer::from_fn(W, H, |x, _| {
        Luma([(40 + x * 40 / (W - 1)) as u8]) // 背景 40→80 渐变
    });
    for y in 20..130 {
        for x in 20..55 {
            bg.put_pixel(x, y, Luma([25])); // 静止的深色"门"
        }
    }
    for y in 110..150 {
        for x in 160..220 {
            bg.put_pixel(x, y, Luma([110])); // 静止的浅色"地砖"
        }
    }
    bg
}

/// 第 f 帧：背景 + 亮方块（intensity 220，x 随帧右移）+ 高斯噪声（每帧不同 seed）
fn make_frame(bg: &GrayImage, f: usize) -> GrayImage {
    let mut img = bg.clone();
    let x0 = -SQ + 8 * f as i64; // 方块从左侧屏幕外进入，每帧 +8px（第 0 帧完全在画外）
    let y0 = 65i64;
    for dy in 0..SQ {
        for dx in 0..SQ {
            let (x, y) = (x0 + dx, y0 + dy);
            if x >= 0 && x < W as i64 && y >= 0 && y < H as i64 {
                img.put_pixel(x as u32, y as u32, Luma([220]));
            }
        }
    }
    // 叠加高斯噪声：均值 0、标准差 8；seed 随帧变化，序列整体仍可复现
    gaussian_noise(&img, 0.0, 8.0, 100 + f as u64)
}

/// 二值化差分：|a−b|>t 记前景（白 255），否则背景（黑 0），并返回白像素个数
fn diff_mask(a: &GrayImage, b: &GrayImage, t: u8) -> (GrayImage, u32) {
    let mut mask = GrayImage::new(W, H);
    let mut cnt = 0u32;
    for (p, (pa, pb)) in mask
        .pixels_mut()
        .zip(a.pixels().zip(b.pixels()))
    {
        let d = (pa[0] as i32 - pb[0] as i32).unsigned_abs();
        if d > t as u32 {
            *p = Luma([255]);
            cnt += 1;
        }
    }
    (mask, cnt)
}

fn count_white(img: &GrayImage) -> u32 {
    img.as_raw().iter().filter(|&&v| v > 0).count() as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bg_truth = make_background();
    let frames: Vec<GrayImage> = (0..N).map(|f| make_frame(&bg_truth, f)).collect();

    // running-average 背景，用 f32 累加避免 u8 舍入漂移；初值 = 第 0 帧
    let mut bg: Vec<f32> = frames[0].as_raw().iter().map(|&v| v as f32).collect();

    println!("== 逐帧前景像素数（阈值 {THRESH}）==");
    println!("帧 | 帧差法 | 背景减除   （方块面积理论值 = {}）", SQ * SQ);
    let focus = 14usize; // 挑这一帧做后续形态学 + 连通域可视化
    let mut focus_bgsub: Option<GrayImage> = None;

    for f in 0..N {
        let cur = &frames[f];

        // ① 帧差法：与上一帧比
        let diff_cnt = if f == 0 {
            0
        } else {
            diff_mask(cur, &frames[f - 1], THRESH).1
        };

        // ② 背景减除：先算 |I−B| 掩码，再更新背景（先减后学，避免把当前前景立刻学进去）
        let mut bgsub = GrayImage::new(W, H);
        let mut bgsub_cnt = 0u32;
        let raw = cur.as_raw();
        for i in 0..raw.len() {
            let px = raw[i] as f32;
            if (px - bg[i]).abs() > THRESH as f32 {
                let (x, y) = (i as u32 % W, i as u32 / W);
                bgsub.put_pixel(x, y, Luma([255]));
                bgsub_cnt += 1;
            }
            bg[i] = ALPHA * px + (1.0 - ALPHA) * bg[i]; // B_t = αI_t+(1−α)B_{t-1}
        }

        println!("{f:>2} | {diff_cnt:>5}  | {bgsub_cnt:>5}");
        if f == focus {
            focus_bgsub = Some(bgsub);
        }
    }

    // ---------- 对 focus 帧的背景减除掩码做形态学去噪 + 连通域框目标 ----------
    let raw = focus_bgsub.unwrap();
    let raw_cnt = count_white(&raw);
    // 开运算（先腐蚀后膨胀）：清掉噪声碎点，主体基本保留。LInf+k=1 => 3x3 方形结构元素
    let opened = open(&raw, Norm::LInf, 1);
    let open_cnt = count_white(&opened);

    println!("\n== 第 {focus} 帧：背景减除掩码去噪 + 连通域 ==");
    println!("原始前景白像素   = {raw_cnt}");
    println!("开运算后白像素   = {open_cnt}（噪声碎点被清掉）");

    // 连通域：背景为黑(0)，8 连通标记，按面积过滤小块
    let labels = connected_components(&opened, Connectivity::Eight, Luma([0u8]));
    let ncc = labels.as_raw().iter().copied().max().unwrap_or(0) as usize;
    let mut area = vec![0u32; ncc + 1];
    let (mut minx, mut miny) = (vec![u32::MAX; ncc + 1], vec![u32::MAX; ncc + 1]);
    let (mut maxx, mut maxy) = (vec![0u32; ncc + 1], vec![0u32; ncc + 1]);
    for (x, y, p) in labels.enumerate_pixels() {
        let l = p[0] as usize;
        if l == 0 {
            continue;
        }
        area[l] += 1;
        minx[l] = minx[l].min(x);
        miny[l] = miny[l].min(y);
        maxx[l] = maxx[l].max(x);
        maxy[l] = maxy[l].max(y);
    }
    // 把掩码铺成灰底 RGB，画绿色运动框
    let mut canvas: RgbImage = ImageBuffer::from_fn(W, H, |x, y| {
        let v = opened.get_pixel(x, y)[0];
        Rgb([v, v, v])
    });
    let min_area = 100u32;
    let mut boxes = 0;
    println!("连通块总数（含碎块）= {ncc}，面积≥{min_area} 视为运动目标：");
    for l in 1..=ncc {
        if area[l] < min_area {
            continue;
        }
        boxes += 1;
        let (bx, by) = (minx[l], miny[l]);
        let (bw, bh) = (maxx[l] - minx[l] + 1, maxy[l] - miny[l] + 1);
        println!("  运动目标#{boxes}: 包围盒=({bx},{by},{bw}x{bh}) 面积={}", area[l]);
        draw_hollow_rect_mut(
            &mut canvas,
            Rect::at(bx as i32, by as i32).of_size(bw, bh),
            Rgb([0, 255, 0]),
        );
    }
    println!("检出运动目标数 = {boxes}");

    // ---------- 存盘：输入帧 / 帧差 / 背景减除(原始) / 开运算后 / 带框可视化 ----------
    let framediff = diff_mask(&frames[focus], &frames[focus - 1], THRESH).0;
    frames[focus].save("out_ch63_frame.png")?;
    framediff.save("out_ch63_framediff.png")?;
    raw.save("out_ch63_bgsub_raw.png")?;
    opened.save("out_ch63_bgsub_open.png")?;
    canvas.save("out_ch63_boxes.png")?;
    println!(
        "\n已存: out_ch63_frame.png / _framediff.png / _bgsub_raw.png / _bgsub_open.png / _boxes.png"
    );
    println!(
        "对比帧差法({}白) vs 背景减除({}白): 帧差只在方块前后沿留下「鬼影」条带，背景减除给出完整实心前景。",
        count_white(&framediff),
        raw_cnt
    );
    Ok(())
}
