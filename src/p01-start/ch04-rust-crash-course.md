# 第 4 章 Rust 速成营（够用就行）

> **本章导读**
> - 用一章补齐后面机器视觉会**反复用到**的 Rust 知识：类型、集合、控制流、所有权、迭代器。
> - 全程围绕一个贯穿例子——**处理一个像素数组 `Vec<u8>`**，边学语法边培养"图像即数字"的手感。
> - 目标是"够用就行"：不追求覆盖整门语言，遇到没讲到的语法，查一下就能补上。
> - 有编程经验的读者可快速略读；零基础的读者请务必把代码敲一遍、跑起来。

## 一排像素：本章的贯穿例子

我们后面处理的所有图像，说白了都是**一大堆 0–255 的整数**（下一章 [第 5 章](ch05-what-is-an-image.md) 会正式拆开看）。为了让这一章不空洞，我们抓住一个最小的例子：一张 1 行、5 列的灰度小图，可以用一串数字表示：

```text
索引:    0     1     2     3     4
亮度: [  10,  250,  128,   0,  255 ]
        暗    亮    灰    黑    白
```

本章要学的每个 Rust 语法，我们都拿它来"折腾"——调亮、筛选、求平均、找最大值。学完你会发现：图像处理的代码，不过是把这些基本操作放大到几百万个像素上而已。

> 💡 **提示**：这是"速成营"，不是 Rust 大全。我只讲机器视觉路上高频出现的部分，讲到"能看懂、能改、能跑"为止。想系统学，官方免费教材《Rust 程序设计语言》（The Rust Programming Language，简称 "the book"）是最佳选择。遇到本章没覆盖的语法，随时去查，别卡住。

## 变量、可变性与基本数值类型

Rust 的变量**默认不可变**——这一点和很多语言不同，但对图像处理特别友好：

```rust
let width = 5;        // 不可变，之后不能再改
let mut sum = 0u32;   // 加 mut 才能改
sum += width;         // OK，因为 sum 是 mut 的
// width = 6;         // 编译报错：width 不可变
```

为什么默认只读是好事？因为图像处理里我们常要**区分"只读的原图"和"可改的输出缓冲"**。让编译器帮你盯住"谁能改、谁不能改"，能挡掉大量 bug。

数值类型里，你要先认识这几个"熟脸"：

| 类型 | 范围 | 在本书里干嘛 |
|------|------|-------------|
| `u8` | 0–255 | **像素值的天然容器**（正好一个字节） |
| `u32` / `usize` | 大非负整数 | 图像宽高、数组索引、长度 |
| `i16` / `i32` | 带符号整数 | 梯度、差值（可能为负） |
| `f32` / `f64` | 小数 | 权重、归一化像素（0.0–1.0）、坐标 |

**为什么像素是 `u8`？** 一个字节 8 位，能表示 \\( 2^8 = 256 \\) 个值，即 0–255，正好够表达"从全黑到全白"的 256 级灰度。这就是位深（bit depth）为 8 的含义，下一章细说。

`u8` 有个新手必踩的坑——**溢出**：

```rust
let p: u8 = 200;
// let bad = p + 100;          // 300 > 255，debug 模式直接 panic！
let good = p.saturating_add(100); // 饱和加法：超了就停在 255
```

调亮图像时，几乎每个像素都可能"加过头"，所以 `saturating_add` / `saturating_sub` 会是你的老朋友。需要精细计算时，还常先 `as f32` 转成小数算完再转回来。

> 🕳️ **坑**：`u8` 直接相加溢出，在 debug 编译下会 panic，在 release 下会**悄悄回绕**（300 变成 44），后者更可怕——图像会出现莫名其妙的亮点。处理像素做加减，优先用 `saturating_*` 或先转 `i32`/`f32`。

## Vec、数组与切片：像素装在哪

三种"装一排数据"的容器，用途各不同：

```rust
let arr: [u8; 3] = [255, 128, 0];      // 数组：长度固定，编译期已知
let pixels: Vec<u8> = vec![10, 250, 128, 0, 255]; // Vec：可增长，长度运行时才定
let region: &[u8] = &pixels[1..3];     // 切片：借 pixels 的一段 [250, 128]，不复制
```

- **`Vec<u8>`** 是图像数据的主力：一张图有多少像素运行时才知道，必须用可增长的 `Vec`。`vec![]` 是创建它的宏，`vec![0u8; 100]` 能一次造 100 个 0。
- **切片 `&[u8]`** 是"借一段看看"，不拷贝数据。写函数时，**参数尽量用 `&[u8]` 而不是 `Vec<u8>`**——这样传数组、传 Vec、传其中一段都行，更通用也更省。

## struct 与 enum：给数据起个名字

`struct`（结构体）把几个字段打包成一个有名字的类型。一张灰度图，本质就是"宽 + 高 + 一维像素数组"：

```rust
struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<u8>,   // 长度 = width * height，行优先铺开
}
```

`enum`（枚举）表示"从几个可能里选一个"，特别适合描述"通道布局""颜色类型"这类有限选项：

```rust
enum Layout {
    Gray,   // 单通道
    Rgb,    // 三通道
}
```

## 函数、方法与 impl

函数用 `fn`，参数和返回值都要写类型。给 `struct` 加方法，用 `impl` 块：

```rust
impl GrayImage {
    fn new(width: usize, height: usize) -> Self {
        // 一次性造出 width*height 个 0
        GrayImage { width, height, data: vec![0u8; width * height] }
    }
}
```

`&self` 表示"借用自己（只读）"，`&mut self` 表示"可改自己"，这和下面要讲的借用是一回事。

## Option 与 Result：优雅地处理"没有"和"出错"

Rust 没有 `null`，也不喜欢用异常。它用两个枚举把"可能没有"和"可能出错"摆到明面上：

- **`Option<T>`**：要么 `Some(值)`，要么 `None`。适合"可能取不到"。
- **`Result<T, E>`**：要么 `Ok(值)`，要么 `Err(错误)`。适合"可能失败"。

取像素时越界怎么办？返回 `Option` 而不是崩溃：

```rust
impl GrayImage {
    fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x]) // 行优先索引：y*宽 + x
        } else {
            None // 越界，明明白白告诉你"没有"
        }
    }
}
```

算平均亮度时，空数组算是一种错误，用 `Result`：

```rust
fn average(pixels: &[u8]) -> Result<f32, String> {
    if pixels.is_empty() {
        return Err("像素数组为空".to_string());
    }
    let sum: u32 = pixels.iter().map(|&p| p as u32).sum();
    Ok(sum as f32 / pixels.len() as f32)
}
```

**`?` 运算符**是处理 `Option`/`Result` 的神器：成功就取出里面的值继续走，失败就**立刻把错误返回给调用者**，省去一堆 `match`：

```rust
fn report(pixels: &[u8]) -> Result<(), String> {
    let avg = average(pixels)?; // 出错就直接 return Err，不用手写 if
    println!("平均亮度 = {avg:.1}");
    Ok(())
}
```

后面读图片 `image::open("a.png")?`、跑模型，几乎每一步都可能失败，`?` 会无处不在。

## match：模式匹配

`match` 像"超级 switch"，还能匹配范围、解构枚举，且**必须覆盖所有情况**（编译器帮你查漏）：

```rust
let level = match p {
    0..=63    => "暗",
    64..=191  => "中",
    _         => "亮",   // _ 兜底，缺了它编译不过
};

match img.get(2, 2) {
    Some(v) => println!("像素值 = {v}"),
    None    => println!("越界了"),
}
```

## 闭包与迭代器：像素处理的主力

**闭包**是"随手写的匿名小函数"，用 `|参数| 表达式` 表示，比如 `|&v| v.saturating_add(20)`。

**迭代器**则是逐个处理元素的流水线。这套组合拳是图像处理里出现频率最高的写法，务必熟练。四个最常用的：

```rust
let pixels: Vec<u8> = vec![10, 250, 128, 0, 255];

// map：每个元素做变换（这里给每个像素调亮 20）
let brightened: Vec<u8> = pixels.iter().map(|&v| v.saturating_add(20)).collect();

// filter：留下满足条件的（挑出"亮"像素）
let bright_count = pixels.iter().filter(|&&v| v > 127).count();

// enumerate：连同下标一起遍历
for (i, v) in pixels.iter().enumerate() {
    print!("[{i}]={v} ");
}
```

- `.iter()` 拿到迭代器，`.map(...)` / `.filter(...)` 定义"每步做什么"；
- **迭代器是惰性的**——不 `.collect()`（收集成 `Vec`）或 `.count()`/`.sum()`（汇总）就不会真正执行；
- `|&v|` 里的 `&` 是在解构：`iter()` 给出的是 `&u8`，用 `&v` 把里面的 `u8` 取出来。

这套写法既好读，编译器又能优化得和手写循环一样快，还能配合并行库 rayon 一键多核加速（[第 108 章](../p21-mastery/ch108-performance.md)）。

## 所有权与借用：图像处理里为什么到处是 `&`

这是 Rust 最有特色、也最让新手头疼的部分。但对我们来说，**记住几条直觉就够用了**，深入原理遇到再补。

一句话：**每份数据都有唯一的"主人"（owner），主人离开作用域，数据就被自动释放**——不用手动 free，也不会有垃圾回收停顿。这正是 Rust 又快又稳的秘密。

由此引出三种把数据"给"函数用的方式，理解它们就理解了满屏的 `&`：

| 写法 | 含义 | 生活类比 | 图像处理场景 |
|------|------|---------|-------------|
| `&pixels` | **不可变借用**：借来看，不能改 | 借书看，不能涂改 | 传"只读原图"给分析函数 |
| `&mut pixels` | **可变借用**：借来改 | 借你的笔在你本子上写 | 在"输出缓冲"上原地画框/调色 |
| `pixels.clone()` | **克隆**：复制一整份 | 复印一份带走，各改各的 | 想留住原图、又要改副本时 |

对应到代码：

```rust
fn print_len(p: &[u8]) {            // 只读借用
    println!("长度 = {}", p.len());
}
fn add_one(p: &mut [u8]) {          // 可变借用，原地修改
    for v in p.iter_mut() {
        *v = v.saturating_add(1);   // *v 解引用后赋值
    }
}

let pixels = vec![10, 250, 128, 0, 255];
print_len(&pixels);        // 借出去看一眼，pixels 还在
let mut buf = pixels.clone(); // 复制一份，避免动到原图
add_one(&mut buf);         // 在副本上改
// pixels 原封不动
```

规则也很朴素：**同一时刻，要么有任意多个只读借用，要么只有一个可变借用**（"共享不可变，可变不共享"）。这条规则从根上杜绝了"一边读一边被别人改"导致的诡异 bug。

> 💡 **提示**：新手常被借用检查器（borrow checker）"教育"到怀疑人生。别硬扛——**`clone()` 是完全合法的逃生舱**。图像不大时，先 `clone()` 一份让程序跑起来，等理解深了再优化掉多余的拷贝。先能跑，再变好。

> ⚠️ **注意**：`clone()` 是**深拷贝**，一张 1080p 彩图 clone 一次要复制约 600 万字节。在逐帧处理视频的热路径里滥用 `clone()` 会明显拖慢速度。到 [第 108 章](../p21-mastery/ch108-performance.md) 我们会讲怎么用借用消除不必要的拷贝。

## trait 与泛型：浅尝，够用就行

**trait（特征）**类似其他语言的"接口"，定义"能做什么"；**泛型**让一段代码适配多种类型。你现在只需能**看懂**，不用会写：

```rust
// 泛型函数：对任何"能比较大小、能复制"的类型都管用
fn max_of<T: PartialOrd + Copy>(items: &[T]) -> Option<T> {
    let mut iter = items.iter();
    let mut m = *iter.next()?; // 空切片时 next() 是 None，? 直接返回 None
    for &x in iter {
        if x > m { m = x; }
    }
    Some(m)
}
```

`<T: PartialOrd + Copy>` 读作"对于任意实现了 `PartialOrd`（可比较）和 `Copy`（可复制）的类型 T"。`image` 库就大量用泛型来同时支持灰度、RGB 等不同像素类型（如 `ImageBuffer<P, C>`）。**这些我们即用即学**，看到 `<...>` 别慌，它只是在说"这段代码通用"。

## 动手写代码：把它们串成一个小程序

把上面所有零件拼起来，就是一个"折腾像素数组"的完整程序。新建 `code/examples/ch04_rust_crash.rs`（本章不需要任何第三方依赖，纯标准库）：

```rust
fn brighten(pixel: u8, delta: u8) -> u8 {
    pixel.saturating_add(delta) // u8 饱和加法，超 255 就停在 255
}

fn average(pixels: &[u8]) -> Result<f32, String> {
    if pixels.is_empty() {
        return Err("像素数组为空".to_string());
    }
    let sum: u32 = pixels.iter().map(|&p| p as u32).sum();
    Ok(sum as f32 / pixels.len() as f32)
}

fn main() {
    // 变量、类型
    let mut count: u32 = 0;
    count += 1;
    let p: u8 = 200;
    println!("调亮：{p} -> {}", brighten(p, 100)); // -> 255

    // Vec / 迭代器 / 闭包
    let pixels: Vec<u8> = vec![10, 250, 128, 0, 255];
    let brightened: Vec<u8> = pixels.iter().map(|&v| v.saturating_add(20)).collect();
    let bright_count = pixels.iter().filter(|&&v| v > 127).count();
    println!("调亮后 = {brightened:?}，亮像素个数 = {bright_count}");

    // match
    let level = match p {
        0..=63 => "暗",
        64..=191 => "中",
        _ => "亮",
    };
    println!("亮度档位 = {level}");

    // Result + ?（用 if let 处理返回值）
    match average(&pixels) {
        Ok(avg) => println!("平均亮度 = {avg:.1}"),
        Err(e) => println!("出错：{e}"),
    }
}
```

运行：

```bash
cargo run --example ch04_rust_crash
```

你会看到类似输出：

```text
调亮：200 -> 255
调亮后 = [30, 255, 148, 20, 255]，亮像素个数 = 3
亮度档位 = 亮
平均亮度 = 128.6
```

短短几十行，就用上了变量、类型、Vec、闭包、迭代器、`match`、`Result`。这套"手感"，后面处理真实图像时会一模一样地复用。

## 常见坑与提示

> 🕳️ **坑**：忘了 `mut` 就想改变量，或者忘了 `&`/`&mut` 导致"值被移动（moved）"报错，是新手最常见的两类编译错误。**Rust 的编译错误信息非常详细**，还常直接给出修改建议——请务必读它，它是最好的老师，不是拦路虎。

> 💡 **提示**：`{:?}` 是"调试打印"，几乎能打印任何类型（`Vec`、`struct`、`enum`……），调试时比 `{}` 好用。`println!("{x:?}")` 这种把变量名直接写进花括号的写法（Rust 2021 起支持）也很方便。

> ⚠️ **注意**：整数除法会截断（`5 / 2 == 2`）。算平均、比例这类需要小数的地方，记得先 `as f32` 再除，否则结果会莫名其妙地取整。

## 本章小结

- Rust 变量**默认不可变**，要改加 `mut`；这让"只读原图 vs 可改缓冲"泾渭分明。
- **`u8`（0–255）是像素的天然容器**；做像素加减警惕溢出，多用 `saturating_add/sub` 或转 `f32` 计算。
- **`Vec<u8>` 装图像数据，`&[u8]` 切片用于借用一段**；函数参数优先用切片。
- `struct` 打包字段，`enum` 表示有限选项；`Option`/`Result` + `?` 优雅处理"没有"与"出错"。
- **闭包 + 迭代器（map/filter/enumerate/collect）是像素处理的主力写法**，好读又高效。
- 所有权三件套：`&`（只读借）、`&mut`（可变借）、`clone()`（复制一份）——这就是图像代码里满屏 `&` 的由来。卡住时 `clone()` 是合法逃生舱。
- trait 与泛型**看懂即可，后面即用即学**。

## 动手练习

1. **调暗与反相**：给上面的程序加一个函数 `invert(pixels: &[u8]) -> Vec<u8>`，把每个像素变成 `255 - v`（图像"反相"）。用迭代器 `map` 实现，并打印结果。
2. **找极值**：用本章的 `max_of` 思路，写一个同时返回最小值和最大值的函数 `min_max(&[u8]) -> Option<(u8, u8)>`，对像素数组求出最暗和最亮的值。
3. **阈值二值化**：写函数 `binarize(pixels: &[u8], t: u8) -> Vec<u8>`，用 `map` + `if`，把 `>t` 的像素变成 255、其余变成 0。这正是 [第 16 章](../p03-classic/ch16-threshold.md) 二值化的雏形。
4. **进阶**：把 5 个像素扩展成一个 `GrayImage { width, height, data }`（比如 3×2），实现 `get(x, y) -> Option<u8>` 并测试越界返回 `None`。想想 `data[y * width + x]` 这个"行优先"索引公式——下一章你会看到真实图像正是这么存的。

下一章，我们正式把镜头对准图像本身：**一张图，在计算机里到底是一堆什么样的数字？** 👉 [第 5 章](ch05-what-is-an-image.md)
