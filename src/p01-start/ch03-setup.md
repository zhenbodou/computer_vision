# 第 3 章 搭建开发环境，跑通第一个程序

> **本章导读**
> - 用官方工具 `rustup` 在 macOS / Linux / Windows 上装好 Rust，并验证装成功了。
> - 配好最舒服的编辑器组合：VS Code + rust-analyzer。
> - 学会用 `cargo` 新建项目、看懂 `Cargo.toml`、添加第三方库。
> - **亲手跑通第一个视觉程序**：不依赖任何外部图片，用代码"画"出一张彩色渐变图并存成 PNG。

## 先把工具备齐，再谈算法

学游泳不能只在岸上看教材。这一章我们把"泳池"建好——一套能编译、能运行、能自动补全的 Rust 环境，然后立刻下水，写出你的第一张图。

整个流程就四步，跟着敲，十几分钟搞定：

```text
① 装 Rust (rustup)  →  ② 装编辑器 (VS Code + rust-analyzer)
        →  ③ 新建项目 (cargo new)  →  ④ 加依赖 + 写代码 + cargo run
```

## 第一步：用 rustup 安装 Rust

Rust 官方推荐用 **`rustup`** 这个工具来安装和管理 Rust 版本（它会一并装好编译器 `rustc` 和构建工具 `cargo`）。

**macOS / Linux**：打开终端，粘贴这行命令，一路回车选默认即可：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows**：到官网 [https://rustup.rs](https://rustup.rs) 下载 `rustup-init.exe` 双击运行。安装过程会提示你需要 **"Visual Studio C++ Build Tools"**（Rust 在 Windows 上依赖 MSVC 链接器），按提示装上即可。

装完后**新开一个终端**（让环境变量生效），验证一下：

```bash
rustc --version
cargo --version
```

看到类似下面的输出（版本号可能比这新），就说明装好了：

```text
rustc 1.97.1 (8bab26f4f 2026-07-14)
cargo 1.97.1 (c980f4866 2026-06-30)
```

> 💡 **提示**：以后想升级 Rust，一句 `rustup update` 就行；想看装在哪、有哪些版本，用 `rustup show`。`rustup` 管版本，`cargo` 管项目，`rustc` 是底层编译器——平时你几乎只跟 `cargo` 打交道。

## 第二步：配好编辑器（VS Code + rust-analyzer）

理论上记事本也能写 Rust，但**没有智能提示的 Rust 是很痛苦的**（它的类型系统信息量很大，靠人脑记不过来）。最省心的组合是：

1. 装 [VS Code](https://code.visualstudio.com/)（免费、跨平台）。
2. 在 VS Code 的扩展商店里搜 **`rust-analyzer`** 并安装（认准作者是 "rust-lang"）。

`rust-analyzer` 是官方的语言服务器，会给你：**自动补全、鼠标悬停看类型、错误实时下划线、跳转定义、把推断出来的类型以灰字提示（inlay hints）**。写起来行云流水。

> ⚠️ **注意**：别装那个老旧的、名叫 "Rust" 的扩展了，它已经被 `rust-analyzer` 取代。装错了会互相打架。

## 第三步：用 cargo 新建项目

`cargo` 是 Rust 的"项目管家"——建项目、加依赖、编译、运行、测试，全归它管。新建一个项目：

```bash
cargo new hello_cv
cd hello_cv
```

`cargo` 会给你生成这样一个最小项目骨架：

```text
hello_cv/
├── Cargo.toml      # 项目配置文件（名字、版本、依赖都写这里）
└── src/
    └── main.rs     # 程序入口，里面已有一个 Hello world
```

打开 `Cargo.toml` 看看，它长这样：

```toml
[package]
name = "hello_cv"      # 项目名
version = "0.1.0"      # 版本号
edition = "2024"       # Rust 版本代（本书用 2024）

[dependencies]
# 第三方库（crate）写在这里，现在还是空的
```

- **`[package]`**：项目的身份信息。`edition` 是 Rust 的"语言版本代"，2021/2024 都行，本书用 2024。
- **`[dependencies]`**：你要用到的第三方库清单。在 Rust 里，第三方库叫 **crate**（读作"克瑞特"，可以想成一个个"板条箱"），都发布在中央仓库 [crates.io](https://crates.io) 上。

先跑一下自带的 Hello world，确认链路通畅：

```bash
cargo run
```

第一次会编译一小会儿，然后打印 `Hello, world!`。链路通了，接下来换成我们自己的视觉程序。

## 第四步：加依赖，写出第一张图

我们要用到本书的主力图像库 **`image`**。在项目目录下执行：

```bash
cargo add image
```

`cargo` 会自动把最新的 `image` 版本写进 `Cargo.toml`，你会看到 `[dependencies]` 下多了一行（版本号可能略有不同，本书统一用 `0.25` 系列）：

```toml
[dependencies]
image = "0.25.10"
```

现在，用下面的内容**完整替换** `src/main.rs`。这段程序不读任何外部图片，纯靠代码"算"出每个像素的颜色——非常适合当第一个程序，因为它自包含、跑起来一定有结果：

```rust
// src/main.rs
use image::{ImageBuffer, Rgb, RgbImage};

fn main() {
    // 图像尺寸：宽 256、高 256（正好让每个方向的渐变从 0 铺到 255）
    let width: u32 = 256;
    let height: u32 = 256;

    // from_fn 会对每一个像素坐标 (x, y) 调用一次闭包，返回该像素的颜色。
    // Rgb([r, g, b])：红、绿、蓝三个通道，每个取值 0..=255。
    let img: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
        let r = x as u8; // 红色随 x 从左到右由暗变亮
        let g = y as u8; // 绿色随 y 从上到下由暗变亮
        let b = 128u8;   // 蓝色固定，让整体偏紫
        Rgb([r, g, b])
    });

    // 按扩展名自动选择 PNG 编码写盘；出错就直接 panic（第一个程序，简单点）。
    img.save("gradient.png").expect("保存图片失败");

    println!("已生成 gradient.png（{} x {}）", width, height);
}
```

运行它：

```bash
cargo run
```

第一次运行，`cargo` 会去 crates.io 下载 `image` 及其一大堆依赖（各种图片格式的编解码器）并全部编译，**耗时从十几秒到一两分钟都正常**。编译完成后，你会在终端看到：

```text
已生成 gradient.png（256 x 256）
```

同时，项目根目录下多出一个 `gradient.png`。用图片查看器打开它，你会看到一张平滑的彩色渐变：

```text
   左上：暗蓝 ────────▶ 右上：品红
      │  (r=0,g=0)        (r=255,g=0)   │
      │                                 │  ← 向右红色渐强
      ▼  向下绿色渐强                     ▼
   左下：青绿 ────────▶ 右下：浅黄
     (r=0,g=255)        (r=255,g=255)
```

四个角的颜色由 `(r, g, b)` 的取值决定，中间平滑过渡。**恭喜——你已经用 Rust 生成了第一张图像！** 别小看它：`ImageBuffer::from_fn` 这种"给定坐标算出颜色"的思路，正是后面画框、合成测试图、做几何变换的基础。

> 💡 **提示**：本书的配套代码工程在仓库 `code/` 目录（crate 名 `cv_examples`）。上面这段程序对应 `code/examples/ch03_first_image.rs`，你也可以进到 `code/` 目录用 `cargo run --example ch03_first_image` 直接跑。

## 常见坑与提示

> 🕳️ **坑：crates.io 拉取巨慢，卡在 `Updating crates.io index`。** 国内直连官方源经常龟速甚至超时。解决办法是换国内镜像源：编辑（没有就新建）`~/.cargo/config.toml`（Windows 在 `%USERPROFILE%\.cargo\config.toml`），写入：
>
> ```toml
> [source.crates-io]
> replace-with = 'rsproxy-sparse'
>
> [source.rsproxy-sparse]
> registry = "sparse+https://rsproxy.cn/index/"
> ```
>
> 保存后重新 `cargo run` 即可。镜像地址偶尔会变，以镜像站官方说明为准（常见的还有中科大、清华 tuna 源）。

> ⚠️ **注意：第一次编译特别慢，别以为卡死了。** `image` 会拉进 PNG / JPEG / WebP 等一堆解码器，首次 `cargo build` 要编译上百个 crate，慢是正常的。**之后是增量编译，改一行代码重编常常一两秒**。编译产物都堆在 `target/` 目录里，它会越长越大，纳入 Git 时记得在 `.gitignore` 里忽略掉 `target/`。

> 🕳️ **坑：rust-analyzer 没提示、没高亮，像个死的。** 十有八九是因为你**用 VS Code 直接打开了单个 `.rs` 文件**，而不是打开**包含 `Cargo.toml` 的整个项目文件夹**。rust-analyzer 靠 `Cargo.toml` 才能认出这是个 Rust 项目。正确做法：`File → Open Folder` 选中 `hello_cv` 目录。另外首次打开会在后台索引（状态栏能看到转圈的进度），**等它转完**补全才会生效。

> 💡 **提示：编译报错看不懂？** Rust 的编译器错误信息是出了名的友好，通常会直接告诉你哪一行、为什么、甚至怎么改。先耐心读英文提示；实在卡住，本书 [附录 D](../appendix/d-troubleshooting.md) 收录了新手最常见的编译错误与排查办法。

## 本章小结

- 用 **`rustup`** 一键安装 Rust（含 `rustc` 与 `cargo`），装完用 `rustc --version` / `cargo --version` 验证。
- 编辑器首选 **VS Code + `rust-analyzer`**：自动补全、类型提示、实时报错，写 Rust 的舒适度全靠它。
- **`cargo`** 是项目管家：`cargo new` 建项目、`cargo add` 加依赖、`cargo run` 编译运行；`Cargo.toml` 里 `[package]` 记身份、`[dependencies]` 列第三方 crate。
- 用 `image` 的 `ImageBuffer::from_fn` **纯靠代码合成**了一张彩色渐变图并 `save` 成 PNG——这是自包含、零外部素材的第一个视觉程序。
- 常见坑：crates.io 慢（换国内镜像）、首次编译慢（正常，之后增量很快）、rust-analyzer 不生效（要打开含 `Cargo.toml` 的文件夹）。

## 动手练习

1. **改颜色玩起来**：把 `b` 从固定的 `128` 改成 `255 - x as u8`，或者干脆写成 `(x ^ y) as u8`，重新 `cargo run`，看看渐变图变成什么样。体会"坐标 → 颜色"这个映射的威力。
2. **改尺寸**：把 `width`、`height` 改成 `512`，观察生成的图和文件大小的变化。想一想：为什么现在红色是从 `x=0` 到 `x=511`，但像素值只到 255 就"到顶"了？（提示：`x as u8` 对超过 255 的值会发生什么？）
3. **加一道竖线**：在保存前，用 `img.put_pixel(128, y, Rgb([255, 255, 255]))` 配合一个 `for y in 0..height` 循环，在图中间画一条白色竖线。（`put_pixel` 需要 `img` 是 `mut` 的，编译器会提醒你。）
4. **验证环境**：进入仓库 `code/` 目录，跑一次 `cargo run --example ch03_first_image`，确认配套工程在你的机器上也能正常编译运行。

环境备齐、第一张图也跑出来了。下一章，我们用一章"速成营"补齐后面要用到的 Rust 语法——够用就走，不求全。👉 [第 4 章](ch04-rust-crash-course.md)
