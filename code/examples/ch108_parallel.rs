//! 第 108 章 性能工程：串行 vs rayon 并行 vs 算法优化（gamma 校正实测）
//! 运行（务必带 --release！）：cargo run --release --example ch108_parallel
//! 依赖：image 0.25、rayon 1.12。纯 Rust、可复现，无需外部图片。
//!
//! 目标：在一张大合成图上做逐像素 gamma 校正，对比四种写法的耗时与加速比：
//!   A) 串行 + powf          —— 最朴素
//!   B) rayon par_iter + powf —— 数据并行
//!   C) 串行 + 256 项 LUT     —— 算法优化（把 powf 从「每像素」降到「每次 256 个」）
//!   D) rayon par_chunks + LUT—— 并行 + 算法优化叠加
//! 顺带演示一个反直觉的坑：小数据上并行反而更慢（调度开销 > 计算）。

use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const W: u32 = 3000;
const H: u32 = 2000;
const GAMMA: f32 = 2.2;

/// 单像素 gamma：把 0..=255 归一化到 0..1，做幂运算，再映射回来。powf 相对昂贵。
#[inline]
fn gamma_pixel(b: u8) -> u8 {
    let x = b as f32 / 255.0;
    (x.powf(GAMMA) * 255.0 + 0.5) as u8
}

/// 预计算 gamma 查找表：只算 256 次 powf，之后每个像素只做一次数组查表（极快）。
fn build_gamma_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = gamma_pixel(i as u8);
    }
    lut
}

/// 计时助手：预热 1 次，再跑 `rounds` 次取最好成绩（最能反映无抖动时的真实算力）。
/// 返回 (最好耗时, 校验和)。校验和用 black_box 兜住，防止编译器把整段计算优化掉。
fn bench<F: FnMut(&mut [u8])>(base: &[u8], rounds: u32, mut f: F) -> (std::time::Duration, u64) {
    let mut best = std::time::Duration::MAX;
    let mut checksum = 0u64;
    for r in 0..=rounds {
        let mut buf = base.to_vec(); // 每轮都从同一份原图复制，保证公平
        let t0 = Instant::now();
        f(black_box(&mut buf));
        let dt = t0.elapsed();
        checksum = buf.iter().map(|&b| b as u64).sum();
        black_box(&buf);
        if r > 0 {
            best = best.min(dt); // 第 0 轮当预热，不计入
        }
    }
    (best, checksum)
}

fn main() {
    // 合成一张大图的原始字节缓冲（RGB，每像素 3 字节）。用简单渐变+噪声填充，可复现。
    let n = (W * H * 3) as usize;
    let base: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    let mb = n as f64 / (1024.0 * 1024.0);
    let threads = rayon::current_num_threads();
    println!("图像 {W}x{H} RGB，缓冲 {n} 字节（{mb:.1} MiB）；rayon 线程数 = {threads}\n");

    let lut = build_gamma_lut();

    // A) 串行 + powf
    let (t_a, c_a) = bench(&base, 5, |buf| {
        for b in buf.iter_mut() {
            *b = gamma_pixel(*b);
        }
    });

    // B) rayon par_iter + powf（只把 A 的 iter_mut 换成 par_iter_mut）
    let (t_b, c_b) = bench(&base, 5, |buf| {
        buf.par_iter_mut().for_each(|b| *b = gamma_pixel(*b));
    });

    // C) 串行 + LUT（算法优化：powf 只算 256 次）
    let (t_c, c_c) = bench(&base, 5, |buf| {
        for b in buf.iter_mut() {
            *b = lut[*b as usize];
        }
    });

    // D) rayon par_chunks_mut + LUT（按 4096 字节分块并行，缓存友好 + 算法优化叠加）
    let (t_d, c_d) = bench(&base, 5, |buf| {
        buf.par_chunks_mut(4096).for_each(|chunk| {
            for b in chunk.iter_mut() {
                *b = lut[*b as usize];
            }
        });
    });

    // 四种写法结果必须完全一致（校验和相等）——正确性是优化的前提。
    assert_eq!(c_a, c_b);
    assert_eq!(c_a, c_c);
    assert_eq!(c_a, c_d);
    println!("正确性校验：四种写法输出校验和一致 = {c_a} ✔\n");

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    let base_ms = ms(t_a);
    println!("{:<28} {:>10} {:>10}", "方法", "耗时(ms)", "加速比");
    println!("{}", "-".repeat(50));
    for (name, t) in [
        ("A 串行 + powf", t_a),
        ("B rayon par_iter + powf", t_b),
        ("C 串行 + LUT", t_c),
        ("D rayon par_chunks + LUT", t_d),
    ] {
        println!("{:<28} {:>10.2} {:>9.2}x", name, ms(t), base_ms / ms(t));
    }

    // —— 反直觉的坑：小数据上并行更慢 ——
    println!("\n[小数据陷阱] 只处理 256 个字节时：");
    let small: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let (ts, _) = bench(&small, 2000, |buf| {
        for b in buf.iter_mut() {
            *b = gamma_pixel(*b);
        }
    });
    let (tp, _) = bench(&small, 2000, |buf| {
        buf.par_iter_mut().for_each(|b| *b = gamma_pixel(*b));
    });
    println!("  串行   = {:.3} µs", ts.as_secs_f64() * 1e6);
    println!("  并行   = {:.3} µs  （调度开销让它反而更慢！）", tp.as_secs_f64() * 1e6);
}
