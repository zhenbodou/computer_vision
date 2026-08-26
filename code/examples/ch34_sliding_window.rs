// 第 34 章：传统目标检测 滑动窗口 + HOG + NMS
// 运行：cargo run --example ch34_sliding_window
// 依赖：image、imageproc
//
// 本例演示传统检测流水线里可以在"主工程"跑通的三块：
//   1) 手写 IoU + 非极大值抑制(NMS)：给一堆重叠框，合并成几个；
//   2) 图像金字塔 + 滑动窗口：数一数一共要扫多少个窗口（速度直觉）；
//   3) HOG 特征提取：把每个窗口缩放到 64x128 提 HOG，打印维度，
//      并把特征落成 Vec<f64> / 一行 CSV —— 这就是喂给 SVM(独立 linfa 工程) 的输入。
use image::{imageops, GrayImage, ImageBuffer, Luma};
use imageproc::hog::{hog, HogOptions};
use std::io::Write;

/// 一个候选检测框：左上角 (x, y) + 宽高 (w, h) + 分数 score
#[derive(Clone, Copy, Debug)]
struct Det {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
}

/// 交并比 IoU = 交集面积 / 并集面积，范围 [0,1]，越大越重叠
fn iou(a: &Det, b: &Det) -> f32 {
    // 各自的右下角
    let (ax2, ay2) = (a.x + a.w, a.y + a.h);
    let (bx2, by2) = (b.x + b.w, b.y + b.h);
    // 交集矩形
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih; // 交集面积（不重叠时为 0）
    let union = a.w * a.h + b.w * b.h - inter; // 并集 = 两框面积和 − 交集
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// 贪心非极大值抑制：按分数从高到低，保留一个就压制掉与它 IoU 过高的其余框
fn nms(mut dets: Vec<Det>, iou_thr: f32) -> Vec<Det> {
    // 1. 按 score 从高到低排序
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut keep = Vec::new();
    let mut suppressed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if suppressed[i] {
            continue; // 已被更高分的框压制，跳过
        }
        keep.push(dets[i]); // 当前最高分，保留
        // 2. 把与它重叠过多的后续框统统压制
        for j in (i + 1)..dets.len() {
            if !suppressed[j] && iou(&dets[i], &dets[j]) > iou_thr {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// 合成场景：浅灰背景 + 两个深色矩形"物体"（用来产生高分窗口）
fn make_scene(w: u32, h: u32) -> GrayImage {
    // 两个物体：(x, y, w, h)
    let objs = [(32u32, 40u32, 80u32, 80u32), (150, 70, 72, 88)];
    ImageBuffer::from_fn(w, h, |x, y| {
        for &(ox, oy, ow, oh) in &objs {
            if x >= ox && x < ox + ow && y >= oy && y < oy + oh {
                return Luma([40u8]); // 深色前景
            }
        }
        Luma([205u8]) // 浅灰背景
    })
}

/// 合成一个 64x128 的"行人剪影"（同 ch27）：浅灰背景 + 深色人形，边缘丰富适合 HOG
fn make_person(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as i32, y as i32);
        let cx = w as i32 / 2;
        let head_cy = h as i32 / 6;
        let head_r = w as i32 / 6;
        let in_head = (xf - cx).pow(2) + (yf - head_cy).pow(2) < head_r * head_r;
        let in_body = xf > cx - w as i32 / 5 && xf < cx + w as i32 / 5
            && yf > h as i32 / 4 && yf < 3 * h as i32 / 4;
        let in_legs = yf >= 3 * h as i32 / 4
            && ((xf > cx - w as i32 / 5 && xf < cx - w as i32 / 20)
                || (xf > cx + w as i32 / 20 && xf < cx + w as i32 / 5));
        if in_head || in_body || in_legs { Luma([40u8]) } else { Luma([205u8]) }
    })
}

/// 枚举滑动窗口左上角坐标：窗口 win_w×win_h，步长 stride
fn slide_positions(img_w: u32, img_h: u32, win_w: u32, win_h: u32, stride: u32) -> Vec<(u32, u32)> {
    let mut pos = Vec::new();
    if img_w < win_w || img_h < win_h {
        return pos;
    }
    let mut y = 0;
    while y + win_h <= img_h {
        let mut x = 0;
        while x + win_w <= img_w {
            pos.push((x, y));
            x += stride;
        }
        y += stride;
    }
    pos
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 一、IoU + NMS：一组重叠框合并成几个 ----------
    // 模拟检测器在两个物体附近各"炸出"若干重叠框（外加一个孤立误检）
    let raw = vec![
        Det { x: 30.0, y: 40.0, w: 80.0, h: 80.0, score: 0.90 }, // 物体A 附近
        Det { x: 36.0, y: 44.0, w: 80.0, h: 80.0, score: 0.82 },
        Det { x: 28.0, y: 36.0, w: 80.0, h: 80.0, score: 0.75 },
        Det { x: 150.0, y: 70.0, w: 72.0, h: 88.0, score: 0.88 }, // 物体B 附近
        Det { x: 156.0, y: 76.0, w: 72.0, h: 88.0, score: 0.79 },
        Det { x: 240.0, y: 10.0, w: 40.0, h: 40.0, score: 0.55 }, // 孤立框（不与谁重叠）
    ];
    println!("[NMS] 输入候选框 {} 个", raw.len());
    // 打印两个"该被合并"的框之间的 IoU，直观感受阈值怎么定
    println!(
        "[NMS] 框0 与框1 的 IoU = {:.3}（同一物体的两个框，应较大）",
        iou(&raw[0], &raw[1])
    );
    println!(
        "[NMS] 框0 与框3 的 IoU = {:.3}（不同物体，应为 0）",
        iou(&raw[0], &raw[3])
    );
    let kept = nms(raw.clone(), 0.5);
    println!("[NMS] IoU 阈值 0.5 → 保留 {} 个框：", kept.len());
    for d in &kept {
        println!("      score={:.2} @ ({}, {}) {}x{}", d.score, d.x as i32, d.y as i32, d.w as i32, d.h as i32);
    }

    // ---------- 二、图像金字塔 + 滑动窗口：数窗口 ----------
    let scene = make_scene(256, 192);
    let (win_w, win_h) = (64u32, 64u32);
    let scales = [1.0f32, 0.75, 0.5]; // 三层金字塔
    let mut total_windows = 0usize;
    let mut detections: Vec<Det> = Vec::new();
    for &s in &scales {
        let sw = (scene.width() as f32 * s).round() as u32;
        let sh = (scene.height() as f32 * s).round() as u32;
        let level = imageops::resize(&scene, sw, sh, imageops::FilterType::Triangle);
        let stride = 16;
        let positions = slide_positions(sw, sh, win_w, win_h, stride);
        total_windows += positions.len();
        // 每个窗口算一个"物体分数"：这里用"深色像素占比"当 SVM 分数的替身
        for (x, y) in positions {
            let crop = imageops::crop_imm(&level, x, y, win_w, win_h).to_image();
            let dark = crop.pixels().filter(|p| p.0[0] < 128).count() as f32;
            let score = dark / (win_w * win_h) as f32;
            if score > 0.6 {
                // 命中：把该层坐标按尺度换算回原图坐标
                detections.push(Det {
                    x: x as f32 / s,
                    y: y as f32 / s,
                    w: win_w as f32 / s,
                    h: win_h as f32 / s,
                    score,
                });
            }
        }
    }
    println!("\n[金字塔] {} 层 × 步长16 共扫描窗口 = {} 个", scales.len(), total_windows);
    println!("[金字塔] 分数>0.6 的原始命中 = {} 个", detections.len());
    for thr in [0.5f32, 0.3, 0.2] {
        let merged = nms(detections.clone(), thr);
        println!("[金字塔] NMS(IoU>{:.1}) 后剩下 = {} 个", thr, merged.len());
    }

    // ---------- 三、HOG 特征：窗口 → 定长向量 → Vec<f64>/CSV ----------
    // 取一个"行人剪影"窗口（边缘丰富），缩放到 64x128（HOG 经典行人尺寸）提特征
    let win64x128: GrayImage = make_person(64, 128);
    let opts = HogOptions::new(9, false, 8, 2, 1); // 与 ch27 同款经典配置
    let feat: Vec<f32> = hog(&win64x128, opts)?;
    let l2: f32 = feat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nonzero = feat.iter().filter(|&&v| v.abs() > 1e-6).count();
    println!("\n[HOG] 行人窗口 64x128 的 HOG 维度 = {}（= 7*15*36）", feat.len());
    println!("[HOG] 特征 L2 范数 = {:.3}，非零维数 = {} / {}", l2, nonzero, feat.len());

    // 落成与 ndarray 版本无关的 Vec<f64> / CSV —— 这就是喂给独立 linfa 工程的数据
    let feat_f64: Vec<f64> = feat.iter().map(|&v| v as f64).collect();
    let mut f = std::fs::File::create("out_ch34_hog_sample.csv")?;
    // 一行：label,f0,f1,...  label=1 表示正样本(有行人)
    write!(f, "1")?;
    for v in &feat_f64 {
        write!(f, ",{:.6}", v)?;
    }
    writeln!(f)?;
    println!("[HOG] 已写出一行样本到 out_ch34_hog_sample.csv（label + {} 维特征）", feat_f64.len());

    Ok(())
}
