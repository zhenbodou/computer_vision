# 第 110 章 WebAssembly 视觉：浏览器边界与可测性能

> **本章导读**
> - 搞懂一件很酷的事：把 Rust 写的视觉算法编成 **WebAssembly（简称 WASM）**，直接塞进浏览器里跑——用户什么都不用装，打开网页就能用。
> - 学会一条完整链路：`#[wasm_bindgen]` 导出函数 → `wasm-pack` 打包 → JS 把 canvas 像素传进来、Rust 处理完传回去。
> - **诚实**认清浏览器这堵墙：纯计算能力很强，但线程、文件、包体积都受限，什么能做、什么别硬来，本章一次说清。
> - 动手跑通一个"浏览器端灰度 / 阈值"demo（工程在 `code/wasm_demo/`，自带构建步骤）。

## 场景引入：一张不能上传的照片

产品经理找上门："我们要做个在线证件照工具，用户上传照片、转灰度、二值化预览。但有个硬要求——**照片绝对不能传到服务器**，合规要求，数据必须留在用户设备上。"

按传统思路，图像处理在后端做，可这条路被"数据不出端"堵死了。用 JavaScript 在前端写逐像素循环？能写，但性能一般，而且你那套辛苦调好的 Rust 视觉代码要用 JS 重写一遍。

WASM 给了第三条路：**把 Rust 算法编译成浏览器能直接执行的字节码**，在用户的浏览器里以接近原生的速度跑。照片从摄像头或文件进到 canvas，像素喂给 WASM 处理，结果画回 canvas——**全程不联网**。隐私友好、无需服务器、还省了重写。这正是 [第 2 章](../p01-start/ch02-why-rust.md) 说的"Rust 是 WASM 一等公民"的兑现现场。

## WASM 是什么，为什么 Rust 特别适合

**WebAssembly** 是一种可以在浏览器（以及 Node、边缘运行时）里执行的低级字节码格式。你可以把它理解成"浏览器里的一颗虚拟 CPU"：它不认识 Rust，也不认识 C，只认识编译好的 `.wasm` 字节码。Rust 只要把目标平台换成 `wasm32-unknown-unknown`，`cargo` 就能把源码编成这种字节码。

> 📖 **术语**：**`wasm32-unknown-unknown`** 是 Rust 的一个编译目标（target）。`wasm32` 指 32 位 WASM，第一个 `unknown` 是厂商、第二个是操作系统——两个"unknown"意味着**没有底层操作系统**。这就是浏览器 WASM 受限的根源：没有 OS，就没有现成的文件系统、没有现成的线程。

Rust 编 WASM 之所以顺，是因为它**没有垃圾回收器、没有庞大运行时**。像 Go、Java 编 WASM 得把整个 GC 运行时一起塞进去，动辄几 MB；Rust 编出来的 `.wasm` 可以小到几十 KB。对"打开网页要立刻能用"的场景，体积就是体验。

中间的胶水由两个工具搞定：

- **`wasm-bindgen`**：在 Rust 和 JavaScript 之间架桥。给函数加 `#[wasm_bindgen]`，它就自动生成 JS 能调用的接口，还负责把 JS 的 `Uint8Array`、字符串等类型和 Rust 的 `Vec<u8>`、`String` 对应起来。
- **`wasm-pack`**：一键把 crate 编成 `.wasm` + 配套 JS 胶水，打包进一个 `pkg/` 目录，网页直接 `import` 就能用。

## 数据怎么在 JS 和 Rust 之间流动

理解这条数据流，是写好浏览器视觉的关键。图像在浏览器里以 canvas 的 `ImageData` 存在，本质是一段 **RGBA 字节**（每个像素 4 字节：红、绿、蓝、透明度）。流程如下：

```text
浏览器 (JavaScript)                      WASM 模块 (Rust 编译而成)
┌──────────────────┐                     ┌───────────────────────┐
│ <canvas>         │                     │  linear memory 线性内存 │
│   │ getImageData │  Uint8Array 拷入     │  ┌─────────────────┐  │
│ ImageData.data ──┼────────────────────▶│  │ Vec<u8> (RGBA)  │  │
│  (RGBA 字节流)   │                     │  │  grayscale()    │  │
│                  │◀────────────────────┼──│  threshold()    │  │
│   putImageData   │  Uint8Array 拷回     │  └─────────────────┘  │
│   │              │                     └───────────────────────┘
└──────────────────┘
```

关键点：WASM 有一块自己的**线性内存（linear memory）**。JS 的 `Uint8Array` 传进函数时，`wasm-bindgen` 会把这段字节**拷贝**进线性内存；返回时再拷回一个新的 `Uint8Array`。这个"拷贝"是有成本的（后面"坑"里细说），但对一次处理一整帧来说完全可接受。

## 动手：浏览器端的灰度 / 阈值 demo

我们把一个灰度和一个阈值函数编成 WASM。工程放在 `code/wasm_demo/`，**独立于主 `code/` 工程**（自带 `Cargo.toml` 和 `[workspace]` 声明，不会被主工程构建）。

**① `code/wasm_demo/Cargo.toml`**（关键是 `crate-type = ["cdylib"]`）：

```toml
[package]
name = "wasm_demo"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]   # 编成 .wasm 模块

[dependencies]
wasm-bindgen = "0.2"      # 唯一依赖，保证 .wasm 尽量小

[workspace]               # 声明为独立 workspace，与主 code/ 隔离

[profile.release]         # 面向浏览器的体积优化
opt-level = "z"           # 体积优先
lto = true
codegen-units = 1
panic = "abort"
strip = true
```

**② `code/wasm_demo/src/lib.rs`**：算法就是纯手写循环，**刻意不依赖 image/imageproc**，让 `.wasm` 保持极小：

```rust
use wasm_bindgen::prelude::*;

/// 原地把 RGBA 缓冲转灰度。data 长度 = width * height * 4。
#[wasm_bindgen]
pub fn grayscale(mut data: Vec<u8>) -> Vec<u8> {
    for px in data.chunks_exact_mut(4) {         // 每像素 4 字节 (R,G,B,A)
        // Rec.601 亮度加权，用整数运算避免浮点
        let y = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
        let y = y as u8;
        px[0] = y; px[1] = y; px[2] = y;         // A(px[3]) 保持不变
    }
    data
}

/// 灰度 + 二值化：亮度 >= t 置白，否则置黑。
#[wasm_bindgen]
pub fn threshold(mut data: Vec<u8>, t: u8) -> Vec<u8> {
    for px in data.chunks_exact_mut(4) {
        let y = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
        let v: u8 = if y as u8 >= t { 255 } else { 0 };
        px[0] = v; px[1] = v; px[2] = v;
    }
    data
}
```

**③ 安装工具并构建**（`wasm-pack` 会自动帮你跑 `wasm-opt` 做体积优化）：

```bash
# 装目标平台与打包工具（一次性）
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# 在 code/wasm_demo/ 目录下打包成浏览器可用的 ES 模块
wasm-pack build --target web --release
# 产物在 ./pkg/ ：wasm_demo.js（胶水）+ wasm_demo_bg.wasm（字节码）
```

**④ 极简 `index.html`**（取 canvas 像素 → 调 wasm → 写回）：

```html
<input type="file" id="file" accept="image/*" />
<button id="gray">灰度</button>
<canvas id="cv"></canvas>
<script type="module">
  import init, { grayscale, threshold } from "./pkg/wasm_demo.js";
  await init();                              // 加载并实例化 .wasm

  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d", { willReadFrequently: true });

  document.getElementById("file").onchange = (e) => {
    const img = new Image();
    img.onload = () => {
      cv.width = img.width; cv.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(e.target.files[0]);
  };

  document.getElementById("gray").onclick = () => {
    const src = ctx.getImageData(0, 0, cv.width, cv.height);
    const out = grayscale(src.data);         // src.data 是 Uint8ClampedArray(RGBA)
    const result = new ImageData(new Uint8ClampedArray(out), cv.width, cv.height);
    ctx.putImageData(result, 0, 0);          // 处理结果画回画布
  };
</script>
```

**⑤ 起个本地服务器打开**（**不能直接双击 `file://` 打开**，见下方坑）：

```bash
python3 -m http.server 8080   # 然后浏览器访问 http://localhost:8080
```

选一张图、点"灰度"，你就看到 Rust 算法在浏览器里当场跑出了结果——**一个字节都没上传服务器**。`code/wasm_demo/` 里附了更完整的 `index.html`（带阈值滑块和"还原"按钮）。

> 💡 **提示**：本章的 `grayscale`/`threshold` 已在本机用 `cargo check --target wasm32-unknown-unknown` 编译通过。若你需要现成的滤波、边缘、形态学算法，`image` 0.25 与 `imageproc` 0.27 **确实也能编到 `wasm32-unknown-unknown`**（本机实测通过）——只是要付出体积代价，见下。

## 常见坑与提示

> 🕳️ **坑：包体积会爆**。纯手写循环的 demo 只依赖 `wasm-bindgen`，本机实测依赖树仅 **13 个 crate**，`.wasm` 通常几十 KB。可一旦引入 `image` 默认特性，它会拖进 png / gif / tiff / webp / avif / exr 等一大堆**编解码器 crate**——本机实测依赖树暴涨到 **149 个 crate**，`.wasm` 随之膨胀到 MB 级。对策：浏览器里我们从 canvas 拿到的已经是**解码后的 RGBA**，根本不需要这些文件编解码器。要用 `image`/`imageproc` 的算法时，加 `default-features = false` 只保留必要功能；能用手写循环解决的，就别引库。

> ⚠️ **注意：多线程在浏览器里不是免费的**。`wasm32-unknown-unknown` 默认**没有线程**。`imageproc` 依赖的 `rayon` 能编译通过，但在默认目标上不会真正并行——等于串行跑，别指望 `par_iter` 加速。要真多线程，需要开启 `atomics` 目标特性 + 共享内存，还要服务器发送 **COOP/COEP 跨源隔离响应头**，再配合 `wasm-bindgen-rayon`——这是一整套进阶配置，普通页面往往不值得。默认就当**单线程**来设计算法。

> 🕳️ **坑：文件系统和随机数**。没有 OS 就没有 `std::fs`——想读写文件得走浏览器的 File API 再把字节传进来。另外 `rand` 依赖的 `getrandom` 在 `wasm32-unknown-unknown` 上需要显式配置浏览器后端（否则运行时取不到随机数），本书 demo 不用随机数所以无感，但你若引入随机算法要留意。

> 💡 **提示：JS↔WASM 拷贝开销**。每次 `getImageData`/传参/返回都在拷字节。一帧 1080p 是 \\( 1920 \times 1080 \times 4 \approx 8.3 \\) MB，逐帧处理视频时这几次拷贝会累积成开销。优化方向：让数据**尽量驻留在 WASM 线性内存**、批量传元数据、**绝不逐像素回调 JS**（那是性能杀手）。一次传一整帧、在 Rust 里跑完再传回，是正确姿势。

> ⚠️ **注意：必须用 HTTP 服务打开**。ES 模块 `import` 和 `.wasm` 的 `fetch` 在 `file://` 协议下会被浏览器安全策略拦截。务必用 `python3 -m http.server` 之类起个本地服务器访问，别直接双击 HTML。

> 🕳️ **坑：大模型别硬塞进浏览器**。WASM 跑得动轻量算法和小模型（几 MB 参数量），但把几百 MB 的检测/分割大模型塞进网页，会让用户等着下载、吃满内存、手机直接崩。浏览器适合**实时滤镜、轻量检测、隐私敏感的本地预处理**；重活该留给服务器或 [第 111 章](ch111-embedded.md) 的边缘设备。想再快一点，可开启 **wasm SIMD**（`RUSTFLAGS='-C target-feature=+simd128'`）让逐像素运算向量化。

## 本章小结

- WASM 让 Rust 视觉算法**直接在浏览器跑**：无需服务器、数据不出端、隐私友好、接近原生速度——这是 Rust "一等公民"地位的兑现。
- 链路是：`#[wasm_bindgen]` 导出 → `wasm-pack build --target web` 打包 → JS 用 `getImageData` 取像素、调 wasm、`putImageData` 写回。
- 编译目标 `wasm32-unknown-unknown` **没有 OS**：文件系统、线程都受限；`image`/`imageproc` 能编过但会显著增大体积（实测 13 vs 149 crate）。
- 务实边界：浏览器适合轻量实时处理，大模型不适合；注意包体积（`opt-level='z'` + `wasm-opt`）、拷贝开销、`file://` 限制、SIMD 需手动开启。

## 动手练习

1. **跑通 demo**：按本章步骤构建 `code/wasm_demo/`，在浏览器里对一张照片做灰度和阈值。拖动阈值滑块，观察二值化结果如何随阈值变化。
2. **加个算子**：给 `lib.rs` 再导出一个 `invert`（反色：`255 - v`）函数，并在 `index.html` 加个按钮调用它。体会"改 Rust → 重新 `wasm-pack build` → 刷新网页"的完整回路。
3. **量体积**：分别用 `opt-level = "z"` 和默认 release 配置各构建一次，比较 `pkg/wasm_demo_bg.wasm` 的大小；再试着（另起工程）引入 `image` 默认特性，感受 `.wasm` 体积的量级差异。
4. **想一想**：如果要处理 30 FPS 的摄像头视频流，逐帧 `getImageData` + 拷贝进 WASM 的开销会不会成为瓶颈？你会怎么减少拷贝次数？（提示：数据驻留线性内存、复用缓冲。）

> 下一章我们走出浏览器，去到**摄像头盒子、树莓派、Jetson** 这些资源受限的边缘设备——那里同样是 Rust 的主场。👉 [第 111 章 边缘与嵌入式部署](ch111-embedded.md)
