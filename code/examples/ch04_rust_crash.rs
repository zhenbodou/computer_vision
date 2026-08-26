//! 第 4 章 Rust 速成营：以「处理一个像素数组 Vec<u8>」贯穿。
//! 验证：cargo check --example ch04_rust_crash / cargo run --example ch04_rust_crash
#![allow(unused)]

// 一个函数：把某个像素调亮，u8 用饱和加法避免溢出回绕
fn brighten(pixel: u8, delta: u8) -> u8 {
    pixel.saturating_add(delta)
}

// struct：一张灰度小图 = 宽 + 高 + 一维像素数组
struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl GrayImage {
    fn new(width: usize, height: usize) -> Self {
        GrayImage { width, height, data: vec![0u8; width * height] }
    }

    // Option：越界返回 None，而不是崩溃
    fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }
}

// enum：图像的通道布局
enum Layout {
    Gray,
    Rgb,
}

// Result + ?：计算平均亮度，空数组视为错误
fn average(pixels: &[u8]) -> Result<f32, String> {
    if pixels.is_empty() {
        return Err("像素数组为空".to_string());
    }
    let sum: u32 = pixels.iter().map(|&p| p as u32).sum();
    Ok(sum as f32 / pixels.len() as f32)
}

fn report(pixels: &[u8]) -> Result<(), String> {
    let avg = average(pixels)?; // ? 把 Err 直接向上传播
    println!("平均亮度 = {avg:.1}");
    Ok(())
}

// 泛型 + trait 约束（浅尝）：求切片里的最大值
fn max_of<T: PartialOrd + Copy>(items: &[T]) -> Option<T> {
    let mut iter = items.iter();
    let mut m = *iter.next()?; // 空切片时 next() 为 None，? 直接返回 None
    for &x in iter {
        if x > m {
            m = x;
        }
    }
    Some(m)
}

// 借用：只读
fn print_len(p: &[u8]) {
    println!("长度 = {}", p.len());
}

// 可变借用：原地修改
fn add_one(p: &mut [u8]) {
    for v in p.iter_mut() {
        *v = v.saturating_add(1);
    }
}

fn main() {
    // 变量与可变性
    let title = "像素";
    let mut count: u32 = 0;
    count += 1;

    // 基本数值类型：u8 是像素的天然容器（0–255）
    let p: u8 = 200;
    let ratio: f32 = 0.5;
    let brighter = brighten(p, 100); // 300 会溢出，saturating 后停在 255
    println!("{title}: {p} -> {brighter}");

    // Vec / 数组 / 切片
    let pixels: Vec<u8> = vec![10, 250, 128, 0, 255];
    let arr: [u8; 3] = [1, 2, 3];
    let slice: &[u8] = &pixels[1..3]; // 借一段，不复制数据

    // match：按亮度分档
    let level = match p {
        0..=63 => "暗",
        64..=191 => "中",
        _ => "亮",
    };
    println!("亮度档位 = {level}");

    // 闭包 + 迭代器：map / filter / enumerate / collect
    let brightened: Vec<u8> = pixels.iter().map(|&v| v.saturating_add(20)).collect();
    let bright_count = pixels.iter().filter(|&&v| v > 127).count();
    for (i, v) in pixels.iter().enumerate() {
        print!("[{i}]={v} ");
    }
    println!();
    println!("调亮后 = {brightened:?}，亮像素个数 = {bright_count}");

    // Option
    let img = GrayImage::new(4, 4);
    match img.get(2, 2) {
        Some(v) => println!("(2,2) = {v}"),
        None => println!("越界"),
    }
    let v = img.get(10, 10).unwrap_or(0); // 越界就当作 0

    // Result
    let _ = report(&pixels);

    // 泛型
    let m = max_of(&pixels);
    println!("最大值 = {m:?}");

    // 所有权 / 借用 / clone
    let owned = pixels.clone(); // 深拷贝一份
    print_len(&pixels); // 只借读，pixels 还能继续用
    let mut buf = pixels.clone();
    add_one(&mut buf); // 可变借用，原地 +1
    println!("原数组仍在：{pixels:?}");

    let layout = Layout::Rgb;
    let channels = match layout {
        Layout::Gray => 1,
        Layout::Rgb => 3,
    };
    println!("通道数 = {channels}");
}
