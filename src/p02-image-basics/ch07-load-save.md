# 第 7 章 读取、显示与保存图像

> **本章导读**
> - 学完你能：用纯 Rust 把磁盘上的图片**读进内存**、查看它的**尺寸与色彩类型**、在 PNG/JPEG 等格式间**互相转换**、再**存回磁盘**。
> - 搞懂三个关键角色：`image::open`（读）、`DynamicImage`（内存里的图像容器）、`save`（写）。
> - 说清楚一个新手常问的问题：**纯 Rust 里怎么"显示"一张图**？
> - 全程用**合成的渐变图**做输入，示例自包含，复制粘贴就能跑。

## 从一个真实的小需求说起

假设你手机里存了一堆度假照片，都是 `.jpg`。现在你想写个小工具：把它们统一转成 `.png`，顺便生成一张灰度版留档。

这个需求听起来简单，却把图像处理最基础的三个动作全用上了——**读进来、改一改、写出去**。几乎所有机器视觉程序的第一行和最后一行，做的都是这两件事：把图像从磁盘搬进内存，处理完再搬回去。这一章我们就先把这条"搬运通道"打通。

> 📖 **术语**：**编码格式 vs 内存表示**。磁盘上的 `.jpg`、`.png` 是**编码格式**——为了省空间，它们把像素压缩成一串二进制字节。但计算机没法直接对压缩数据做运算，所以读图时必须先**解码**成一格一格的像素（还记得 [第 5 章](../p01-start/ch05-what-is-an-image.md) 说的"图像就是一堆数字"吗？）。`image::open` 帮你做的就是"解码"，`save` 帮你做的则是"编码"。

## 三个关键角色

### 1. `image::open`：一行代码读图

`image` 库把读图简化成了一行：

```rust
let img = image::open("photo.jpg")?;
```

有两点要记住：

- 它**根据文件内容**（文件头的"魔数"）自动判断格式，而**不是**看扩展名。所以哪怕你把一张 PNG 错误地命名成 `photo.jpg`，它照样能读对。
- 它返回的是 `Result<DynamicImage, ImageError>`——**读图是可能失败的**（文件不存在、格式不支持、文件损坏……）。所以你必须处理这个 `Result`，要么用 `?` 往上抛，要么用 `match` 就地处理。这正是 Rust 强迫你"把错误当回事"的地方。

### 2. `DynamicImage`：内存里的万能容器

读进来的 `img` 是一个 `DynamicImage`。你可以把它理解成一个**"不管里面装的是灰度、RGB 还是带透明通道的 RGBA，都能统一拿在手里"的盒子**。

为什么要有这么个盒子？因为你 `open` 之前并不知道图片到底是什么色彩类型。`DynamicImage` 用一个枚举把各种可能都包了起来，让你先拿到手，再决定怎么"落实"成具体的像素缓冲区：

```rust
let gray = img.to_luma8();   // 落实成 8 位灰度：ImageBuffer<Luma<u8>, Vec<u8>>
let rgb  = img.to_rgb8();    // 落实成 8 位 RGB： ImageBuffer<Rgb<u8>,  Vec<u8>>
let rgba = img.to_rgba8();   // 落实成 8 位 RGBA
```

拿到 `rgb`/`gray` 这样的具体缓冲区后，才能逐像素读写（下一章 [第 8 章](ch08-pixels.md) 细讲）。查看尺寸则任何时候都行：

```rust
use image::GenericImageView;         // dimensions() 来自这个 trait，必须 use
let (w, h) = img.dimensions();        // 返回 (u32, u32)
println!("{} x {}，色彩类型 {:?}", w, h, img.color());
```

> ⚠️ **注意**：`dimensions()`、`get_pixel()` 等方法定义在 `GenericImageView` trait 上。如果你忘了 `use image::GenericImageView;`，编译器会报"方法不存在"，很容易让人一头雾水。记住这个"trait 要显式引入"的规则。

### 3. `save`：按扩展名写回磁盘

存盘同样是一行，格式由**扩展名**决定：

```rust
rgb.save("out.png")?;    // 存成 PNG（无损）
rgb.save("out.jpg")?;    // 存成 JPEG（有损压缩，文件更小）
```

如果你想让"文件名"和"格式"解耦（比如存成 `.bin` 但内容其实是 PNG），用 `save_with_format`：

```rust
use image::ImageFormat;
rgb.save_with_format("out.bin", ImageFormat::Png)?;
```

### 那……怎么"显示"图像？

这是纯 Rust 新手最容易卡住的地方，所以专门说清楚：**`image` 库本身不带任何 GUI 窗口，没有类似 OpenCV 里 `imshow` 那样"弹一个窗口"的函数。** 这在服务器、嵌入式等无桌面环境里其实是优点（不强依赖图形界面）。

本书的**主线方案**很朴素：**把结果 `save` 到文件，然后用系统自带的看图器打开**。开发时甚至可以让程序帮你自动打开（macOS 用 `open`、Windows 用 `start`、Linux 用 `xdg-open`）。

如果你确实想要一个能实时刷新的窗口（比如后面处理视频时），Rust 生态里有 `show_image`、`minifb` 这类窗口显示 crate 可选。它们能开窗口、把像素缓冲区画上去。本书为保持"最小依赖、跨平台省心"，正文一律用"存盘 + 系统查看器"，需要窗口时再单独引入——这里先知道有这么回事即可。

## 动手写代码：读一张图，转格式并存盘

下面是一个完整、可运行的例子。为了**自包含**（不依赖你手头有没有图片），我们先用代码合成一张彩色渐变图存盘，再把它当"输入图"读回来走完整流程。

`Cargo.toml` 依赖：

```toml
[dependencies]
image = "0.25"
```

完整代码（可保存为 `examples/ch07_load_save.rs`，用 `cargo run --example ch07_load_save` 运行）：

```rust
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Rgb, RgbImage};

/// 合成一张彩色渐变图：横向红色渐变、纵向绿色渐变、蓝色固定为 128
fn make_gradient(w: u32, h: u32) -> RgbImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let r = (x * 255 / (w - 1)) as u8;
        let g = (y * 255 / (h - 1)) as u8;
        Rgb([r, g, 128])
    })
}

/// 用 match 稳妥地处理"打开可能失败"的情况
fn open_and_report(path: &str) {
    match image::open(path) {
        Ok(img) => println!("打开成功：{}，尺寸 {:?}", path, img.dimensions()),
        Err(e) => eprintln!("打开失败：{} —— {}", path, e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 先合成一张图并存盘，作为后面读取的输入（让示例自包含）
    let grad = make_gradient(256, 256);
    grad.save("out_gradient.png")?;

    // 2. 读图：image::open 返回 Result，用 ? 把错误往上抛
    let img: DynamicImage = image::open("out_gradient.png")?;

    // 3. 查看尺寸与色彩类型（dimensions/color 需要 GenericImageView）
    let (w, h) = img.dimensions();
    println!("图像尺寸：{} x {}，色彩类型：{:?}", w, h, img.color());

    // 4. 转灰度、转 RGB，各存一份
    let gray = img.to_luma8();
    gray.save("out_gray.png")?;

    // 5. 格式转换：按扩展名自动选格式（png -> jpg）
    img.save("out_gradient.jpg")?;

    // 6. 扩展名与真实格式不一致时，用 save_with_format 显式指定
    img.save_with_format("out_forced.bin", ImageFormat::Png)?;

    // 7. 演示错误处理：一个存在、一个不存在
    open_and_report("out_gradient.png");
    open_and_report("no_such_file.png");

    Ok(())
}
```

运行后，你会在当前目录看到 `out_gradient.png`、`out_gray.png`、`out_gradient.jpg` 等文件，终端还会打印：

```text
图像尺寸：256 x 256，色彩类型：Rgb8
打开成功：out_gradient.png，尺寸 (256, 256)
打开失败：no_such_file.png —— No such file or directory (os error 2)
```

用你系统的看图器打开 `out_gradient.png`，就能看到一张左上偏暗、向右变红、向下变绿的渐变图——这就是"显示"。注意最后那行：打开不存在的文件时，程序**没有崩溃**，而是优雅地打印了错误——这正是我们用 `match` 处理 `Result` 换来的稳健。

> 💡 **提示**：想用真实照片替换合成图？把第 2 步换成 `image::open("assets/your_photo.jpg")?` 即可，后面的流程完全不用改。这就是 `DynamicImage` 作为"万能容器"的好处——上层代码不关心输入到底是什么格式。

## 常见坑与提示

> 🕳️ **坑**：**相对路径是相对"运行目录"，不是相对源文件。** `image::open("photo.jpg")` 里的 `photo.jpg` 是相对你敲 `cargo run` 时所在的目录（通常是项目根），而不是 `.rs` 文件所在目录。找不到文件时，先用 `std::env::current_dir()` 打印一下当前目录，或者干脆用绝对路径排查。

> ⚠️ **注意**：**别用 `.unwrap()` 图省事。** `image::open("x.png").unwrap()` 在文件不存在时会直接 panic 让程序崩溃。学习时无所谓，但真实程序里请用 `?` 或 `match`，把"文件损坏""格式不支持"这些情况处理好。

> 🕳️ **坑**：**格式支持取决于编译进来的 feature。** `image` 通过 feature 开关决定支持哪些格式，默认开启了 PNG、JPEG、GIF、BMP 等常见格式。如果 `save("x.webp")` 报错说格式不支持，多半是对应 feature 没启用——查一下 `image` 的文档，在 `Cargo.toml` 里打开对应 feature 即可。

> 💡 **提示**：**PNG 无损、JPEG 有损。** 把图存成 `.jpg` 会有压缩损失（尤其反复"读-存-读-存"会越来越糊）。做中间结果、需要精确像素时，一律用 PNG；只有最终交付、追求小体积时才用 JPEG。

## 本章小结

- 读图用 `image::open(path)?`，它按**文件内容**（而非扩展名）识别格式，返回 `Result<DynamicImage, _>`，**必须**处理这个 `Result`。
- `DynamicImage` 是内存里的**万能图像容器**；用 `to_luma8()` / `to_rgb8()` / `to_rgba8()` 把它落实成可逐像素操作的具体缓冲区。
- `dimensions()` 取宽高（记得 `use image::GenericImageView`）；`save(path)?` 按扩展名写盘，`save_with_format` 可显式指定格式。
- 纯 Rust **没有内置显示窗口**，主线做法是"存盘 + 系统查看器"；需要实时窗口时可用 `show_image`/`minifb` 等 crate。
- PNG 无损、JPEG 有损；相对路径相对运行目录；格式支持取决于编译 feature——这三点是新手最常踩的坑。

## 动手练习

1. **跑通它**：把上面的完整例子跑起来，用看图器逐个打开生成的文件，确认 `out_gradient.jpg` 和 `out_gradient.png` 看起来一样、但文件大小不同（对比 JPEG 与 PNG 的体积）。
2. **批量转换**：写一个函数 `to_png(src: &str, dst: &str)`，读入任意图片、存成 PNG。用它把你手头的一张 `.jpg` 转成 `.png`。
3. **稳健读取**：改造练习 2，当 `src` 不存在或不是合法图片时，不 panic，而是打印一条友好的错误信息后继续。
4. **进阶**：给程序加一个"读图后自动用系统查看器打开"的功能（提示：用 `std::process::Command` 调用 macOS 的 `open` / Windows 的 `cmd /c start` / Linux 的 `xdg-open`）。

读进来、存出去都会了，下一章我们钻进图像内部，学会**一个像素一个像素地读和改**——并亲手把一张图做成"反色 + 角落红方块"的效果。👉 [第 8 章 访问和修改像素](ch08-pixels.md)
