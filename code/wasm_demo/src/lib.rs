// code/wasm_demo/src/lib.rs
// 把 Rust 视觉算法导出给浏览器：接收 Canvas 的 RGBA 像素缓冲，处理后返回。
// 这里刻意只依赖 wasm-bindgen、用纯手写循环，让编出来的 .wasm 尽量小。
use wasm_bindgen::prelude::*;

/// 原地把 RGBA 缓冲转灰度，返回处理后的缓冲。
/// `data` 长度应为 width * height * 4（正是 Canvas `ImageData.data` 的布局）。
#[wasm_bindgen]
pub fn grayscale(mut data: Vec<u8>) -> Vec<u8> {
    // chunks_exact_mut(4) 按“一像素 4 字节（R,G,B,A）”切片，零额外分配
    for px in data.chunks_exact_mut(4) {
        // Rec.601 亮度加权，用整数运算避免浮点（wasm 上更快更小）
        let y = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
        let y = y as u8;
        px[0] = y; // R
        px[1] = y; // G
        px[2] = y; // B
        // px[3] 是 alpha 透明度，保持不变
    }
    data
}

/// 灰度 + 二值化：亮度 >= `t` 的像素置白(255)，否则置黑(0)。
#[wasm_bindgen]
pub fn threshold(mut data: Vec<u8>, t: u8) -> Vec<u8> {
    for px in data.chunks_exact_mut(4) {
        let y = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
        let v: u8 = if y as u8 >= t { 255 } else { 0 };
        px[0] = v;
        px[1] = v;
        px[2] = v;
    }
    data
}
