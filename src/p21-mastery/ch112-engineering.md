# 第 112 章 生产工程：架构、测试、观测与发布

> **本章导读**
> - 学会用 **lib + bin 分离**组织视觉项目：把可复用的算法沉到库里，应用只管拼装。
> - 掌握 Rust 的错误处理分工：**库层用 `thiserror`、应用层用 `anyhow`，库里绝不 `unwrap`**。
> - 会用**合成图**写可复现的视觉算法测试（`#[test]` + `tests/` 集成测试），并配上 `fmt`/`clippy`/`doc` 质量门。
> - 拿到一份**可直接用的 GitHub Actions CI 配置**，让每次提交都自动跑格式、静态检查和测试。

## 场景引入：能跑的 demo 和能交付的系统

你的检测 demo 在自己机器上跑得好好的。可一旦交给同事、上了服务器、跑上三个月，问题就来了：一张损坏的图片让程序直接 `panic` 崩溃、重构时不小心改错了阈值却没人发现、新人不敢动你那坨 800 行的 `main.rs`……

从"能跑的 demo"到"能交付的系统"，差的就是本章这套**工程素养**：清晰的结构、诚实的错误处理、自动化的测试与 CI。这也是[第 2 章](../p01-start/ch02-why-rust.md)所说 Rust "又快又稳"里"稳"字的落地方式。

## 原理与实践一：项目结构（lib + bin 分离）

新手爱把所有代码堆在 `main.rs`。专家的第一刀，是把**可复用的算法逻辑**从**应用的拼装逻辑**里切出来：

```text
code/
├── Cargo.toml
├── src/
│   ├── lib.rs      ← 可复用视觉逻辑（库 crate）：算法、错误类型、纯函数
│   └── main.rs     ← 可执行入口（bin crate）：读配置、调库、输出
├── examples/       ← 每章示例 chNN_xxx.rs（cargo run --example）
├── tests/          ← 集成测试（把库当外部使用者来测）
└── benches/        ← 基准测试（见第 108 章）
```

好处立竿见影：库里的函数能被 `main`、`examples`、`tests`、`benches` 共享；单元测试测库、集成测试从外部测库；将来要做成 Web 服务（[第 96 章](../p19-deploy/ch96-web-service.md)）或 WASM，直接复用同一个库。**核心算法应该是"纯"的**——只吃数据、吐结果，不掺杂读文件、打印、`panic` 这些副作用。

## 原理与实践二：错误处理（库层 thiserror，应用层 anyhow）

这是 Rust 工程最见功力的地方，一句话：**库返回错误，让调用方决定怎么办；应用兜住错误，决定崩还是降级。**

- **库层用 `thiserror`**：把"可能出什么错"显式建模成一个错误枚举，每个变体带清晰信息。**库里严禁 `unwrap()`/`panic!`**——你不知道调用方是想重试、降级还是记日志，凭什么替它决定崩溃？
- **应用层用 `anyhow`**：`main`、CLI、服务入口用 `anyhow::Result`，`?` 一路上抛，`.context()` 补充"当时在干嘛"，最后统一打印/记录。

先看**库层**（`src/lib.rs`，本书配套代码实测可编译）：

```rust
use image::GrayImage;
use thiserror::Error;

// 用 thiserror 把「可能出什么错」建模成枚举，每个变体自带可读信息。
#[derive(Error, Debug)]
pub enum VisionError {
    #[error("读写图像失败: {0}")]
    Io(#[from] image::ImageError),        // #[from] 让 ? 能自动转换底层错误
    #[error("图像尺寸非法: {w}x{h}（宽高都必须 > 0）")]
    BadSize { w: u32, h: u32 },
}

// 纯函数：给定输入永远得到相同输出，极易测试。
pub fn count_bright_pixels(img: &GrayImage, thresh: u8) -> u64 {
    img.pixels().filter(|p| p.0[0] > thresh).count() as u64
}

// 会失败的地方一律返回 Result，绝不 unwrap——把处理权交给调用方。
pub fn otsu_binarize(img: &GrayImage) -> Result<(GrayImage, u64), VisionError> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Err(VisionError::BadSize { w, h });   // 坏输入 → 返回 Err，不 panic
    }
    use imageproc::contrast::{otsu_level, threshold, ThresholdType};
    let level = otsu_level(img);
    let bin = threshold(img, level, ThresholdType::Binary);
    let fg = count_bright_pixels(&bin, 0);
    Ok((bin, fg))
}
```

再看**应用层**（`examples/ch112_engineering.rs`，`cargo run --example ch112_engineering` 实测通过）：

```rust
use anyhow::{Context, Result};
use cv_examples::{otsu_binarize, foreground_ratio_from_path};

fn main() -> Result<()> {                    // main 返回 anyhow::Result，? 可直接用
    // ……合成一张演示图并存盘（略）……
    let (_, fg) = otsu_binarize(&img).context("对演示图做 Otsu 二值化时出错")?;
    println!("Otsu 前景像素 = {fg}");

    // 优雅处理错误路径：库返回 Err，应用捕获后不崩溃
    match foreground_ratio_from_path("no_such_file.png") {
        Ok(r)  => println!("占比 = {r}"),
        Err(e) => println!("按预期失败，已捕获（不崩溃）：{e}"),
    }
    Ok(())
}
```

实测输出（节选）：`Otsu 前景像素 = 10000`、`按预期失败，已被捕获（程序不崩溃）：读写图像失败: No such file or directory (os error 2)`。坏文件没让程序崩，而是被规规矩矩地接住了——这就是"库返回、应用兜底"的价值。

## 原理与实践三：视觉算法怎么测

"图像算法没法写单元测试"是个误区。诀窍是**用合成图构造已知答案**，做可复现断言，**绝不依赖外部图片或网络**。

放在 `src/lib.rs` 里的**单元测试**（可测私有实现）：

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    #[test]
    fn bright_count_is_exact() {
        // 100x10 图，右半 50 列全白 → 白像素恰好 500 个，可精确断言
        let img = ImageBuffer::from_fn(100, 10, |x, _| {
            if x < 50 { Luma([0u8]) } else { Luma([255u8]) }
        });
        assert_eq!(count_bright_pixels(&img, 128), 500);
    }
}
```

放在 `tests/` 里的**集成测试**（把库当外部使用者，`cargo test --test ch112_integration`）：

```rust
use cv_examples::{otsu_binarize, VisionError};
use image::{GrayImage, ImageBuffer, Luma};

#[test]
fn otsu_foreground_within_tolerance() {
    let img = ImageBuffer::from_fn(128, 128, |x, y|
        if x < 40 && y < 30 { Luma([255u8]) } else { Luma([0u8]) });
    let (_bin, fg) = otsu_binarize(&img).expect("非空图不该失败");
    // 数值容差比较：白矩形约 1200 像素，允许 Otsu 边界少量偏差
    assert!((fg as i64 - 1200).abs() <= 5);
}

#[test]
fn zero_sized_image_returns_error_not_panic() {
    let empty: GrayImage = ImageBuffer::new(0, 0);
    // 契约：坏输入返回 Err 而非 panic
    assert!(matches!(otsu_binarize(&empty), Err(VisionError::BadSize { .. })));
}
```

`cargo test` 本机实测全绿：

```text
running 3 tests (src/lib.rs 单元测试)
test tests::bright_count_is_exact ... ok
test tests::otsu_splits_bimodal_image ... ok
test tests::empty_image_is_rejected ... ok
test result: ok. 3 passed; 0 failed

running 3 tests (tests/ch112_integration.rs 集成测试)
test white_pixel_count_matches_rect_area ... ok
test otsu_foreground_within_tolerance ... ok
test zero_sized_image_returns_error_not_panic ... ok
test result: ok. 3 passed; 0 failed
```

三类断言覆盖了视觉测试的典型套路：**精确计数**（白像素 == 500）、**数值容差**（前景 ≈ 1200 ± 5）、**错误契约**（坏输入返回 `Err` 而非崩溃）。Otsu 阈值原理见[第 16 章](../p03-classic/ch16-threshold.md)。

> 💡 **提示**：想更进一步，可用 `proptest`（属性测试，property-based testing）自动生成成百上千张随机合成图，验证"不变量"——比如"二值图的前景像素数永远 ≤ 总像素数""旋转 360° 后图像应与原图一致（在容差内）"。它能替你想出你没料到的边界用例。

## 原理与实践四：质量门（fmt / clippy / doc）

三个几乎零成本、却能拉开专业度的命令：

```bash
cargo fmt --all              # 统一代码格式，消灭"风格之争"
cargo clippy --all-targets -- -D warnings   # 静态检查，把 warning 当 error 卡死
cargo doc --no-deps --open   # 为 pub 项生成 API 文档（写好 /// 注释即自动成文档）
```

`clippy` 是 Rust 官方的"资深评审"，能揪出 `x.clone()` 多余、`unwrap` 滥用、可简化的迭代器等一大票问题。养成"提交前先 `fmt` + `clippy`"的肌肉记忆，代码质量自然上一个台阶。

## 原理与实践五：CI/CD（一份可直接用的 GitHub Actions）

把上面的质量门交给机器：每次 push / PR 自动跑格式、静态检查、测试。在仓库根建 `.github/workflows/ci.yml`：

```yaml
name: CI
on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  check:
    name: fmt + clippy + test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2      # 缓存依赖，二次 CI 快很多
      - name: 格式检查（不改文件，只报错）
        run: cargo fmt --all -- --check
      - name: Clippy（warning 当 error）
        run: cargo clippy --all-targets -- -D warnings
      - name: 测试（release，贴近真实性能与优化路径）
        run: cargo test --release
      - name: 文档能否构建
        run: cargo doc --no-deps

  # 可选：交叉编译矩阵，确认能编到别的平台/架构
  cross:
    name: cross-compile
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [x86_64-unknown-linux-musl, aarch64-unknown-linux-gnu]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - run: cargo install cross --locked
      - run: cross build --release --target ${{ matrix.target }}
```

> 💡 **提示**：如果你的 crate 在子目录（如本书代码在 `code/`），给每个 `cargo` 步骤加 `working-directory: code`，或用 `--manifest-path code/Cargo.toml`。`musl` 目标能编出**静态链接、零依赖**的单文件，正是[第 2 章](../p01-start/ch02-why-rust.md)说的"拷过去就能跑"。

## 常见坑与提示

> 🕳️ **坑：库里 `unwrap()` / `panic!`。** 库是给别人用的，你替调用方决定"出错就崩"是越权。一张坏图能让整条服务挂掉。库层一律返回 `Result`，把决策权交出去。

> 🕳️ **坑：吞掉错误（`let _ = do_something();`）。** 用 `let _ =` 把 `Result` 丢进黑洞，出了问题连日志都没有，排查起来欲哭无泪。要么 `?` 上抛，要么显式处理并记录，别装作没发生。

> ⚠️ **注意：测试依赖外部文件 / 网络。** 测试里 `image::open("my_photo.jpg")` 或去下载模型，换台机器、断了网就全红，还不可复现。**一律用合成图**（`ImageBuffer::from_fn`）构造已知答案。

> ⚠️ **注意：CI 不跑 clippy、不跑 `--release`。** 只跑 `cargo test`（debug）会漏掉 clippy 能发现的一堆问题，也测不到 release 下才会触发的优化相关行为。CI 里 `clippy -D warnings` + `test --release` 缺一不可。

> 🕳️ **坑：`edition` 不一致。** 本地 2024、CI 或队友 2021，会冒出诡异的编译差异。在 `Cargo.toml` 里钉死 `edition`，并在 CI 固定工具链版本（如用 `rust-toolchain.toml`）。

## 本章小结

- **结构**：lib + bin 分离，可复用算法沉到 `lib.rs`（保持"纯"），应用只做拼装；`tests/` 从外部测库。
- **错误处理**：库层 `thiserror` 定义错误枚举 + 返回 `Result`、**绝不 `unwrap`**；应用层 `anyhow` + `?` + `.context()` 兜底。
- **测试**：用**合成图**做可复现断言（精确计数 / 数值容差 / 错误契约），避免依赖外部文件与网络；进阶用 `proptest`。本章 6 个测试本机 `cargo test` 全绿。
- **质量门**：`cargo fmt` + `cargo clippy -D warnings` + `cargo doc`，提交前必跑。
- **CI/CD**：GitHub Actions 自动跑 fmt + clippy + test（`--release`）+ doc，可选交叉编译矩阵；`musl` 目标产出静态单文件。

## 动手练习

1. **补一个错误变体**：给 `VisionError` 加一个 `BadThreshold(i32)` 变体，写一个"阈值必须在 0..=255"的校验函数返回 `Result`，并加一个测试断言越界时返回该错误。
2. **反面教材改造**：找一段自己写过的、库里带 `unwrap()` 的代码，改成返回 `Result`，体会调用方多了哪些选择。
3. **合成图测旋转不变量**：写一个测试，把一张合成图旋转 90° 四次应回到原图，用像素级容差断言（引出属性测试的思路）。
4. **（专家挑战）把 demo 产品化**：挑一个你写过的单文件 demo，按本章切成 lib + bin，补齐 `thiserror`/`anyhow`、合成图测试与 CI；再往流水线里注入坏帧、超时，证明程序不崩溃、错误被分类处理。这正是[附录 G](../appendix/g-expert-standard.md) 里"专家级"的验收标准。

> 📖 **下一章预告**：代码工程做扎实了，数据工程呢？[第 113 章](ch113-datasets-mlops.md) 我们聊**数据集、标注与 MLOps**——模型的上限，往往由数据的质量与流程决定。
