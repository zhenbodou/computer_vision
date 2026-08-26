# 第 108 章 性能工程：测量、内存、并行与 SIMD

> **本章导读**
> - 建立专家级的性能第一心法：**先测量，再优化**——不靠猜，靠数据找瓶颈。
> - 学会用 `rayon` 把逐像素处理并行到多核，并用**本机实测的真实加速比**认识收益与代价。
> - 诚实看待 SIMD：稳定 Rust 上靠**自动向量化**与 `wide`，而非 nightly 的 `std::simd`。
> - 会用 `criterion` 和 `std::time::Instant` 做靠谱的基准测试（预热、多次采样、`--release`）。

## 场景引入：一条"慢半拍"的流水线

你写好了一条视觉流水线：读图 → gamma 校正 → 检测 → 画框输出。测下来单帧 80 ms，达不到 25 fps。第一反应往往是："肯定是检测慢，我去优化那个卷积！"——**打住。** 这正是新手和专家的分水岭：专家不猜，先测。你可能花两天优化了卷积，结果发现真正的瓶颈是那句不起眼的 gamma 校正里每像素调用了一次 `powf`。

本章我们就用一个能亲手跑通、打印真实数字的例子，把"测量—优化"的完整闭环走一遍。

## 原理：优化的三条铁律

### 铁律一：先测量，再优化（Amdahl 定律）

凭直觉猜瓶颈，十有八九猜错。更要命的是：**优化一段本来就不占时间的代码，等于白干。** 阿姆达尔定律（Amdahl's law）把这件事说死了：设某段代码占总时间的比例为 \\( p \\)，你把它加速 \\( s \\) 倍，那么整体加速上限是

\\( S = \dfrac{1}{(1-p) + p/s} \\)

如果某函数只占总时间的 5%（\\( p=0.05 \\)），哪怕你把它优化到**无穷快**（\\( s\to\infty \\)），整体也最多快 \\( 1/0.95 \approx 1.05 \\) 倍——5%。所以正确的顺序永远是：**先 profile，找到那个占大头的热点，再动手。**

### 铁律二：`--release` 不是可选项

Rust 的 debug 构建为了编译快、调试友好，**关掉了几乎所有优化**，还插入了溢出检查。做性能测试却忘了加 `--release`，是本章最容易踩、也最丢人的坑——你测出来的"慢"可能比真实值慢**几十倍**，据此得出的一切结论都是错的。

### 铁律三：内存与缓存决定图像处理的下限

图像在内存里是**行优先（row-major）**连续存放的：第 0 行所有像素、第 1 行所有像素……CPU 每次从内存搬数据是按"缓存行"（通常 64 字节）成块搬的。所以**顺着内存走**（外层 `y`、内层 `x`，或者干脆遍历扁平缓冲）就能吃满缓存；**跳着走**（外层 `x`、内层 `y`，逐列访问）则每读一个像素都可能触发一次缓存未命中。

```text
行优先存储（顺着读快，跳着读慢）：
  内存地址 →  [第0行: p00 p01 p02 ...][第1行: p10 p11 ...][第2行 ...]
  ✔ 顺序访问：一次缓存行搬来一串，命中率高
  ✘ 逐列访问：每次跨一整行，缓存频繁失效
```

此外还有两条省钱守则：**别做多余的 `clone()` 和内存分配**（在热循环里 `Vec::new()` 是慢的源头之一），**优先用迭代器**（`iter`/`map`/`for_each` 往往比手写索引更容易被编译器优化、还免去越界检查）。

## 动手：串行 vs rayon 并行 vs 算法优化

我们对一张 3000×2000 的 RGB 大图做 gamma 校正，用四种写法较量，全部用 `Instant` 计时并打印加速比。先加依赖（`rayon` 是纯 Rust、很轻量）：

```toml
[dependencies]
image = "0.25"
rayon = "1.12"
```

核心逻辑（完整可运行版见配套 `examples/ch108_parallel.rs`）：

```rust
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const GAMMA: f32 = 2.2;

// 单像素 gamma：归一化 → 幂运算 → 映射回来。powf 相对昂贵。
#[inline]
fn gamma_pixel(b: u8) -> u8 {
    (( b as f32 / 255.0).powf(GAMMA) * 255.0 + 0.5) as u8
}

// 预计算查找表：powf 只算 256 次，之后每像素一次数组查表（极快）。
fn build_gamma_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() { *v = gamma_pixel(i as u8); }
    lut
}

fn main() {
    let n = (3000 * 2000 * 3) as usize;               // RGB 每像素 3 字节
    let base: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let lut = build_gamma_lut();

    // A) 串行 + powf
    let mut a = base.clone();
    let t = Instant::now();
    for b in a.iter_mut() { *b = gamma_pixel(*b); }
    println!("A 串行 powf: {:?}", t.elapsed());

    // B) rayon 并行 + powf：只把 iter_mut 换成 par_iter_mut，一个词的事
    let mut b = base.clone();
    let t = Instant::now();
    b.par_iter_mut().for_each(|x| *x = gamma_pixel(*x));
    println!("B 并行 powf: {:?}", t.elapsed());

    // C) 串行 + LUT（算法优化）
    let mut c = base.clone();
    let t = Instant::now();
    for x in c.iter_mut() { *x = lut[*x as usize]; }
    println!("C 串行 LUT: {:?}", t.elapsed());

    // D) 并行 + LUT + 按块（par_chunks_mut 把缓冲切成小块并行，缓存友好）
    let mut d = base.clone();
    let t = Instant::now();
    d.par_chunks_mut(4096).for_each(|chunk| {
        for x in chunk.iter_mut() { *x = lut[*x as usize]; }
    });
    println!("D 并行 LUT: {:?}", t.elapsed());

    assert!(a == b && a == c && a == d);   // 正确性是优化的前提！
    black_box(&d);                          // 防止编译器把计算优化掉
}
```

运行（**务必带 `--release`**）：

```bash
cargo run --release --example ch108_parallel
```

本机（10 核 CPU，`rayon` 自动用满 10 线程）实测：

| 方法 | 耗时 | 加速比 |
|------|------|--------|
| A 串行 + `powf` | 71.50 ms | 1.00× |
| B `rayon` 并行 + `powf` | 9.05 ms | **7.90×** |
| C 串行 + LUT | 5.13 ms | 13.94× |
| D 并行 + LUT + 分块 | 0.71 ms | **101×** |

三个专家级洞见全在这张表里：

1. **rayon 几乎白送 8 倍**：一句 `iter_mut → par_iter_mut`，10 核拿到 7.9× 加速（差的那点是任务切分与合并的开销）。这就是[第 2 章](../p01-start/ch02-why-rust.md)说的"无畏并发"的实惠。
2. **好算法比蛮力并行更狠**：串行 LUT（13.9×）反而**超过**了并行 powf（7.9×）——因为它把每像素一次 `powf` 换成一次数组查表（gamma 与查找表见[第 11 章](../p02-image-basics/ch11-point-ops.md)）。**先想算法，再谈并行。**
3. **两者叠加才是王道**：好算法 + 并行 = 101×。

## SIMD：诚实地聊聊"单指令多数据"

SIMD（Single Instruction, Multiple Data，单指令多数据）让一条 CPU 指令同时处理 8 个、16 个数，是逐像素运算的加速利器。但在 Rust 里要说清现状：

- **`std::simd` 仍是 nightly**（`#![feature(portable_simd)]`），稳定版还用不了。别为了它把整个项目绑死在 nightly 工具链上。
- **稳定路线一：自动向量化（auto-vectorization）**。编译器在 `--release` 下会自动把"整齐的循环"编译成 SIMD 指令——你上面那个 LUT 循环，编译器很可能已经帮你矢量化了。写出"向量化友好"的代码即可白嫖：
  - 循环边界简单、无提前 `break`；
  - 连续内存、定长切片（给编译器"这里有一串同类型数据"的信号，比如先 `let s: &mut [u8] = &mut buf[..]`）；
  - 循环体内无分支、无函数调用副作用；
  - 想验证是否被矢量化，可开 `RUSTFLAGS="-C target-cpu=native"` 让编译器针对本机指令集（如 AVX2）生成代码，或用 `cargo asm` 看汇编。
- **稳定路线二：`wide` crate**（纯 Rust）。它提供 `f32x8`、`u8x16` 等类型，在稳定版就能显式写 SIMD，跨平台回退安全。需要手动向量化又不想上 nightly 时，它是务实选择。

> 💡 **提示**：99% 的情况下，你该做的是"写好循环 + 开 `--release` + 让编译器自动向量化"，而不是手写 SIMD。手写 SIMD 属于榨最后 20% 性能的专家活，且必须始终保留一份**标量参考实现**做差分测试，否则极易在尾部元素、对齐、饱和转换上出错。

## 基准测试：从 Instant 到 criterion

`std::time::Instant` 适合快速看个大概，但要点必须做对：**预热**（第一次跑有冷缓存、CPU 未升频，要丢弃）、**多次取最好/中位数**、**用 `black_box` 兜住结果**防止编译器把"没人用的计算"整段删掉。

要"认真"做性能回归，用 `criterion`（dev-dependency，纯 Rust）——它自动预热、自适应采样、给出**中位数与置信区间**，还能对比历史结果告诉你"这次改动是快了还是变慢了"。

```toml
[dev-dependencies]
criterion = "0.8"

[[bench]]
name = "ch108_bench"
harness = false          # 关掉默认 test 框架，交给 criterion
```

```rust
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

fn gamma_benchmark(c: &mut Criterion) {
    let base: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
    let mut group = c.benchmark_group("gamma_1M");
    group.sample_size(30);
    // iter_batched：每次迭代前克隆输入（setup 不计时），只测处理本体
    group.bench_function("serial_powf", |b| {
        b.iter_batched(|| base.clone(),
            |mut buf| { for x in buf.iter_mut() { *x = gamma_pixel(*x); } buf },
            BatchSize::LargeInput)
    });
    group.finish();
}
criterion_group!(benches, gamma_benchmark);
criterion_main!(benches);
```

```bash
cargo bench --bench ch108_bench
```

本机实测（100 万字节），criterion 报出中位数：`serial_powf` 4.81 ms、`rayon_powf` 634 µs（≈7.6×）、`serial_lut` 295 µs（≈16×）——与前面 `Instant` 的结论互相印证。方括号里的 `[下界 中位 上界]` 就是置信区间，波动大说明测量环境不稳。

## 常见坑与提示

> 🕳️ **坑：忘了 `--release`。** debug 构建能比 release 慢几十倍。任何性能数字，先确认是 `--release` 跑出来的，否则一切结论作废。

> 🕳️ **坑：小数据上 rayon 反而更慢。** 本机实测：处理 256 个字节时，串行 0.96 µs，而 `par_iter_mut` 要 7.4 µs——**慢了 7 倍**！因为切分任务、唤醒线程、合并结果的固定开销远超那点计算量。**并行只对"足够大的独立任务"划算**，小数据老老实实串行。

> ⚠️ **注意：过早优化是万恶之源。** 先把代码写清楚、跑正确，再 profile 出真瓶颈去优化。别一上来就手写 SIMD、到处并行，既难维护又常常优化错地方。

> ⚠️ **注意：false sharing（伪共享）。** 多线程各写各的，但如果它们写的数据挤在同一条 64 字节缓存行里，CPU 会被迫反复同步这条缓存行，性能暴跌。`rayon` 的 `par_chunks_mut` 按大块切分能天然规避；自己手动分线程时要让各线程的写入区域隔开。

> 💡 **提示：benchmark 要控制变量。** 关掉后台大程序、插电（笔记本省电模式会降频）、多跑几次看方差。`criterion` 的置信区间就是帮你判断"这次真的变快了，还是只是噪声"。

## 本章小结

- **先测量再优化**：靠 profile 找热点，阿姆达尔定律告诉你优化非热点等于白干。
- 性能数字必须在 `--release` 下测；debug 慢几十倍。
- 图像处理受内存/缓存制约：**行优先顺序访问**、少 `clone`/少分配、多用迭代器。
- `rayon` 把 `iter_mut` 换成 `par_iter_mut`/`par_chunks_mut` 即可数据并行，本机实测 10 核拿到 7.9× 加速；但**好算法（LUT 13.9×）常胜过蛮力并行**，二者叠加可达 101×。
- SIMD：`std::simd` 仍 nightly；稳定路线是**自动向量化**（写向量化友好的循环 + `--release`）或纯 Rust 的 `wide`。
- 基准测试：`Instant` 要预热 + `black_box`；`criterion` 更严谨（自适应采样 + 置信区间）。

## 动手练习

1. **验证 `--release` 的威力**：把本章例子分别用 `cargo run` 和 `cargo run --release` 跑，记录 A 方法的耗时差多少倍。
2. **换一个算子**：把 gamma 换成阈值二值化或简单 3×3 均值卷积（见[第 13 章](../p03-classic/ch13-convolution.md)），重测串行 vs `rayon` 的加速比，看看不同计算量下并行收益如何变化。
3. **找并行的盈亏平衡点**：把图像尺寸从 64×64 一路加到 4000×4000，画出"串行/并行耗时随尺寸的变化"，找出并行开始划算的临界大小。
4. **（专家挑战）建立分段基线**：给你自己的一条真实流水线（读图→预处理→推理→后处理）分段计时，找出占比最大的一段；提出一个优化假设，实施后用 `criterion` 对比前后，并用输出等价断言证明"变快了但结果没变"。记住：只报"快了 X 倍"不算数，要同时报端到端耗时、峰值内存与最坏输入表现。

> 📖 **下一章预告**：CPU 并行到头了怎么办？[第 109 章](ch109-gpu.md) 我们上 **GPU 加速**（Metal / CUDA / wgpu），把逐像素运算搬到成千上万个核心上跑。工程化的另一半——项目结构、错误处理、测试与 CI/CD——见[第 112 章](ch112-engineering.md)。
