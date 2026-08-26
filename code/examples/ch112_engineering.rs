//! 第 112 章 生产工程：应用层错误处理（anyhow）+ 调用可复用库逻辑
//! 运行：cargo run --example ch112_engineering
//! 依赖：image 0.25、anyhow 1.0，以及本 crate 的库（src/lib.rs：thiserror 定义的 VisionError）。
//!
//! 分工示范（第 112 章核心）：
//!   - **库层**（src/lib.rs）：用 thiserror 定义 `VisionError`，函数返回 `Result`，不 unwrap。
//!   - **应用层**（这里）：用 anyhow 兜底，`?` 一路上抛，用 `.context()` 补充「在做什么」。

use anyhow::{Context, Result};
use cv_examples::{count_bright_pixels, foreground_ratio_from_path, otsu_binarize};
use image::{ImageBuffer, Luma};

fn main() -> Result<()> {
    // 1) 合成一张一半黑一半白的灰度图并存盘（不依赖外部素材，可复现）。
    let img = ImageBuffer::from_fn(200, 100, |x, _| {
        if x < 100 { Luma([20u8]) } else { Luma([230u8]) }
    });
    let path = std::env::temp_dir().join("ch112_demo.png");
    let path = path.to_string_lossy().to_string();
    img.save(&path).with_context(|| format!("保存演示图到 {path} 失败"))?;
    println!("已写入演示图: {path}");

    // 2) 直接调用库里的纯函数（不会失败，无需 Result）。
    println!("亮像素(>128)个数 = {}", count_bright_pixels(&img, 128));

    // 3) 调用返回 Result 的库函数，用 ? 上抛；.context 让错误信息说清「当时在干嘛」。
    let (_, fg) = otsu_binarize(&img).context("对演示图做 Otsu 二值化时出错")?;
    println!("Otsu 前景像素 = {fg}");

    let ratio = foreground_ratio_from_path(&path).context("统计前景占比时出错")?;
    println!("前景占比 = {:.1}%", ratio * 100.0);

    // 4) 演示错误路径：故意读一个不存在的文件。库层返回 Err，应用层「优雅处理」而非崩溃。
    println!("\n--- 故意触发一个错误（读不存在的文件）---");
    match foreground_ratio_from_path("no_such_file_zzz.png") {
        Ok(r) => println!("竟然成功了？{r}"),
        Err(e) => {
            // anyhow / thiserror 的错误可以打印完整信息；生产中这里应写日志并返回明确状态。
            println!("按预期失败，已被捕获（程序不崩溃）：{e}");
        }
    }

    println!("\n完成。");
    Ok(())
}
