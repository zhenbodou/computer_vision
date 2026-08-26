//! 第 74 章 视频读写与帧处理
//! 运行：cargo run --example ch74_video_io
//! 依赖：image 0.25（含默认开启的 gif 编码）；纯 Rust、确定性可复现、免外部视频文件。
//!
//! 把「一段视频」抽象成一串按时间排列的帧 `Vec<RgbImage>`（自己合成一个移动的方块）。
//! 演示视频处理最本质的两件事：
//!   ① 逐帧处理：对第 t 帧和第 t-1 帧做帧间差分，数出「运动像素」，并把运动区染绿。
//!   ② 写回视频：把原始帧序列、以及带运动叠加的帧序列，各自编码成一个 GIF（纯 Rust）。
//! 附带把两张「静帧」存成 PNG，作为书里的插图。真实的 H.264/mp4 解码不在纯 Rust 范围内，
//! 见正文的诚实说明（ffmpeg 子进程 / ffmpeg-next FFI / retina 收流）。

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, DynamicImage, Frame, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;
use std::fs::File;

const W: u32 = 160;
const H: u32 = 120;
const N: usize = 16; // 帧数
const SQ: i64 = 28; // 移动方块边长
const STEP: i64 = 8; // 每帧右移像素
const FPS: u32 = 10; // 假定帧率：10 帧/秒 → 每帧时间戳 +0.1s
const THRESH: i32 = 30; // 帧差二值化阈值

/// 合成第 f 帧「视频画面」：左→右灰度渐变背景 + 一个红色方块从左侧移入。
/// 这就是我们的「视频源」——真实项目里这一步是解码器吐出的帧，这里用代码合成，自包含可复现。
fn make_frame(f: usize) -> RgbImage {
    // 背景：横向灰度渐变（每列一个灰度），三通道相同 → 中性灰底
    let mut img: RgbImage = ImageBuffer::from_fn(W, H, |x, _| {
        let g = (40 + x * 60 / (W - 1)) as u8; // 40→100 渐变
        Rgb([g, g, g])
    });
    // 移动的红色方块：第 0 帧完全在画面左侧外，之后每帧右移 STEP 像素
    let x0 = -SQ + STEP * f as i64;
    let y0 = 46i64;
    let red = Rgb([220u8, 40, 40]);
    for dy in 0..SQ {
        for dx in 0..SQ {
            let (x, y) = (x0 + dx, y0 + dy);
            if x >= 0 && x < W as i64 && y >= 0 && y < H as i64 {
                img.put_pixel(x as u32, y as u32, red);
            }
        }
    }
    img
}

/// 逐帧处理的核心：对两帧做灰度帧差，返回 (运动掩码, 运动像素数)。
/// 掩码里运动处为 255、静止处为 0。灰度用感知加权（和第 9 章一致）。
fn motion_mask(cur: &RgbImage, prev: &RgbImage) -> (ImageBuffer<Luma<u8>, Vec<u8>>, u32) {
    let mut mask = ImageBuffer::from_pixel(W, H, Luma([0u8]));
    let mut cnt = 0u32;
    for (m, (pc, pp)) in mask
        .pixels_mut()
        .zip(cur.pixels().zip(prev.pixels()))
    {
        let gc = luma(pc);
        let gp = luma(pp);
        if (gc - gp).abs() > THRESH {
            *m = Luma([255]);
            cnt += 1;
        }
    }
    (mask, cnt)
}

/// 感知加权灰度：0.299R + 0.587G + 0.114B
fn luma(p: &Rgb<u8>) -> i32 {
    (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as i32
}

/// 生成「展示帧」：在原始帧上，把运动像素染成亮绿，并在底部画一条随帧增长的时间进度条
/// （代替文字时间戳——纯工程主依赖不含字体库，正文说明真要写文字需 ab_glyph/draw_text_mut）。
fn overlay_frame(base: &RgbImage, mask: &ImageBuffer<Luma<u8>, Vec<u8>>, f: usize) -> RgbImage {
    let mut out = base.clone();
    for (o, m) in out.pixels_mut().zip(mask.pixels()) {
        if m[0] > 0 {
            *o = Rgb([0, 255, 0]); // 运动区染绿
        }
    }
    // 底部时间进度条：宽度 ∝ 当前帧序号
    let bar_w = ((f as u32 + 1) * W / N as u32).max(1);
    draw_filled_rect_mut(
        &mut out,
        Rect::at(0, (H - 6) as i32).of_size(bar_w, 6),
        Rgb([255, 220, 0]),
    );
    out
}

/// 把一串 RgbImage 编码成 GIF（纯 Rust）。每帧 1000/FPS 毫秒。
fn save_gif(path: &str, frames: &[RgbImage]) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut enc = GifEncoder::new(file);
    enc.set_repeat(Repeat::Infinite)?; // 循环播放
    let ms = 1000 / FPS; // 10fps → 100ms/帧
    for f in frames {
        // GIF 的 Frame 要求 RgbaImage，这里把 RgbImage 提升为 RGBA
        let rgba = DynamicImage::ImageRgb8(f.clone()).to_rgba8();
        let delay = Delay::from_numer_denom_ms(ms, 1);
        enc.encode_frame(Frame::from_parts(rgba, 0, 0, delay))?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ① 合成「视频」：一串按时间排列的帧。真实场景这里是解码器吐帧，本例用代码合成。
    let frames: Vec<RgbImage> = (0..N).map(make_frame).collect();
    println!(
        "合成视频：{N} 帧，分辨率 {W}x{H}，假定帧率 {FPS} fps → 时长 {:.1}s",
        N as f32 / FPS as f32
    );

    // ② 逐帧处理：帧差运动检测。注意「逐帧」——只需相邻两帧同时在内存，无需一次性 load 整段。
    println!("\n帧 |  时间戳 | 运动像素数");
    let mut display: Vec<RgbImage> = Vec::with_capacity(N);
    let mut total_motion = 0u64;
    let (mut peak_f, mut peak_cnt) = (0usize, 0u32);
    for f in 0..N {
        let ts = f as f32 / FPS as f32; // 由帧率反推时间戳
        let (mask, cnt) = if f == 0 {
            // 第 0 帧没有「上一帧」，运动数记 0
            (ImageBuffer::from_pixel(W, H, Luma([0u8])), 0)
        } else {
            motion_mask(&frames[f], &frames[f - 1])
        };
        total_motion += cnt as u64;
        if cnt > peak_cnt {
            peak_cnt = cnt;
            peak_f = f;
        }
        println!("{f:>2} | {ts:>5.1}s | {cnt:>6}");
        display.push(overlay_frame(&frames[f], &mask, f));
    }
    println!(
        "\n合计运动像素 {total_motion}，运动最剧烈的是第 {peak_f} 帧（{peak_cnt} 像素）"
    );
    // 帧差只逮住方块「离开」与「新到」两条竖带，理论宽约 2*STEP=16、高 SQ=28 → ~{}
    println!(
        "参考：方块每帧移 {STEP}px、边长 {SQ}px，稳定期两条差分带 ≈ 2*{STEP}*{SQ} = {} 像素",
        2 * STEP * SQ
    );

    // ③ 写回视频：原始帧序列 + 带运动叠加的帧序列，各存成一个 GIF（纯 Rust）。
    save_gif("out_ch74_movie.gif", &frames)?;
    save_gif("out_ch74_motion.gif", &display)?;

    // 另存两张静帧作书里插图：运动最剧烈那帧的原图 + 叠加图
    frames[peak_f].save("out_ch74_frame.png")?;
    display[peak_f].save("out_ch74_motion.png")?;

    println!("\n已存：out_ch74_movie.gif（原始）/ out_ch74_motion.gif（运动叠加，绿=运动）");
    println!("      out_ch74_frame.png / out_ch74_motion.png（第 {peak_f} 帧静帧）");
    Ok(())
}
