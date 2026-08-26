//! 第 108 章 criterion 基准测试：串行 vs rayon 并行 vs LUT 的 gamma 校正。
//! 运行：cargo bench --bench ch108_bench
//! criterion 0.8：dev-dependency，纯 Rust。它自动预热、多次采样、报告中位数与置信区间。
//!
//! 与 examples/ch108_parallel.rs 的手写 Instant 计时相比，criterion 更严谨：
//! 统计学去噪、自动确定采样次数、输出方差——适合「认真」的性能回归。

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rayon::prelude::*;
use std::hint::black_box;

const N: usize = 1_000_000; // 100 万字节，够大到能看出并行收益，又不至于让 bench 太慢
const GAMMA: f32 = 2.2;

#[inline]
fn gamma_pixel(b: u8) -> u8 {
    ((b as f32 / 255.0).powf(GAMMA) * 255.0 + 0.5) as u8
}

fn build_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = gamma_pixel(i as u8);
    }
    lut
}

fn gamma_benchmark(c: &mut Criterion) {
    let base: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();
    let lut = build_lut();

    let mut group = c.benchmark_group("gamma_1M");
    group.sample_size(30); // 默认 100，这里调小让 bench 更快跑完（仍有统计意义）

    // iter_batched：每次迭代前克隆一份新缓冲（setup 不计时），只测 transform 本体。
    group.bench_function("serial_powf", |b| {
        b.iter_batched(
            || base.clone(),
            |mut buf| {
                for x in buf.iter_mut() {
                    *x = gamma_pixel(*x);
                }
                black_box(buf)
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("rayon_powf", |b| {
        b.iter_batched(
            || base.clone(),
            |mut buf| {
                buf.par_iter_mut().for_each(|x| *x = gamma_pixel(*x));
                black_box(buf)
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("serial_lut", |b| {
        b.iter_batched(
            || base.clone(),
            |mut buf| {
                for x in buf.iter_mut() {
                    *x = lut[*x as usize];
                }
                black_box(buf)
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, gamma_benchmark);
criterion_main!(benches);
