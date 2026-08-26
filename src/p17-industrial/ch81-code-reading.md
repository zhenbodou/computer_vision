# 第 81 章 工业读码：一维码 / 二维码 / DataMatrix 与字符 OCR

> **本章导读**
> - 搞懂三类"给机器看的字"——一维码、二维码（QR）、DataMatrix——各自长什么样、凭什么能被读出来。
> - 用**纯 Rust** 亲手跑通一个 **QR 生成 → 解码** 的完整闭环：编码一段文本成图，再从图里原样读回来。
> - 认清纯 Rust 在一维码解码、DataMatrix、字符 OCR 上的**真实生态版图**——哪些现成能用、哪些要绕路，绝不糊弄你。
> - 学完你能给工厂流水线做一个"扫码追溯"的原型，并知道每一步该用哪个 crate。

## 场景引入：流水线上的"身份证"

走进任何一家现代化工厂，你会发现几乎每个零件、每箱货、每块电路板上都印着一小块黑白图案。产线上的相机"扫一眼"，就知道这是哪个批次、哪条线、什么时候生产的——这就是**工业读码**。它撑起了三件大事：

- **追溯**：一瓶药、一颗芯片出了问题，扫码就能倒查到原料批次和生产参数。
- **分拣**：快递流水线上，相机读包裹上的码，机械臂据此把它推向对应格口。
- **防错**：装配时扫一下零件码，系统核对"这个型号该不该装在这台机器上"，装错立即报警。

这些码本质上都是**把一串字符（数字、字母、URL）用黑白图案编码**，好让机器一眼认出。我们这一章就从"它怎么编、怎么读"讲起，然后动手写代码。

## 原理：黑白格子里藏着的字

### 一维码：靠"条宽"说话

超市商品上那种平行竖条，就是**一维码（1D barcode）**，也叫条形码。它的信息**只沿水平方向编码**：用黑条、白空的**宽窄组合**表示一个个字符，竖直方向只是把图案拉高、方便扫描枪对准。

```text
 静区 │ 起始符 │      数据条（宽/窄条组合编码字符）      │ 校验 │ 终止符 │ 静区
 (白) │▐█ ▐█▐█│ █ ▐▐█ ▐█ ▐█▐ █▐ ▐█▐█ █ ▐█ ▐█▐ █▐█ │ ▐█▐█ │▐█ █▐█ │ (白)
      └────────── 每个字符 = 一组固定的黑条+白空，靠"宽/窄"区分 ──────────┘
```

常见的一维码有：**EAN-13**（商品条码，就是超市那种 13 位数字）、**Code128**（能编数字+字母，物流常用，密度高）、**Code39**（老牌工业码，简单可靠）。它们的共同软肋是：**信息量小**（十几到几十个字符）、**怕污损**（一道划痕横穿就可能读不出）、而且左右**必须留白**（叫**静区（quiet zone）**，没有它扫描枪找不到边界）。

### 二维码 QR：横竖都存信息

**QR 码（Quick Response code）** 把信息**同时铺在横竖两个方向**，容量一下子上到几千字符。它最显眼的特征是三个角上的"回"字形方块：

```text
   ┌───────┐                     ┌───────┐
   │ █████ │  ← 定时图案(黑白相间)→ │ █████ │
   │ █   █ │                     │ █   █ │   三个角的"回"字 = 定位图案
   │ █ █ █ │      数据 + 纠错       │ █ █ █ │   (Finder Pattern)：
   │ █   █ │      码字填充区         │ █   █ │   解码器靠它找到码、
   │ █████ │                     │ █████ │   摆正朝向、算出模块栅格
   └───────┘                     └───────┘
   ┌───────┐
   │ █████ │     中间大片区域 = 数据码字 + Reed-Solomon 纠错码字
   │ █   █ │     每个最小黑白方块叫一个"模块(module)"
   │ █ █ █ │
   │ █   █ │     纠错让 QR 即使被遮挡/弄脏一部分，仍能读出
   │ █████ │
   └───────┘
```

三个要点：**定位图案（finder pattern）** 让解码器无论码怎么歪、怎么转都能找到并摆正；**定时图案（timing pattern）** 是连接定位图案的黑白相间线，用来数清有多少行多少列模块；**纠错（error correction）** 用 Reed-Solomon 编码加入冗余，分 L/M/Q/H 四档，最高档 H 即使 30% 面积损坏也能恢复——这正是它抗污损的底气。

### DataMatrix：芯片上的"小个子"

**DataMatrix** 是另一种二维码，特点是**极小也能可靠读**，所以电子元件、芯片、手术器械上的激光打标几乎都用它。它没有 QR 那种大"回"字，而是靠两条实边 + 两条虚边定位：

```text
   █ ▀ █ ▀ █ ▀ █    ← 上边 & 右边：虚线"时钟轨"，交替黑白，定模块行列
   █ ░░░░░░░░░ ▀
   █ ░ 数据区  ░ █    左边 & 下边：实线"L 形寻边图案"，定位 + 定向
   █ ░ ECC200 ░ ▀
   █ ░ +纠错   ░ █    数据用 ECC200（Reed-Solomon）纠错，
   █████████████    缺一角照样读得出——适合小面积、易磨损场景
```

### 读码的通用流程

不管哪种码，读码器内部都走同一条流水线，记住它，写代码/查故障都不慌：

```text
采集图像 → ①定位      → ②校正            → ③采样解码           → ④纠错          → 文本
          找到码在哪   透视掰正+二值化    读每个模块黑白→码字   Reed-Solomon 修复
```

## 动手写代码：QR 生成 → 解码闭环（纯 Rust）

纸上谈兵不如跑一遍。我们用两个**纯 Rust** 库做一个完整闭环：`qrcode` 负责把文本**编码**成 QR 图，`rqrr` 负责从图里**定位并解码**回文本，最后断言"读回来的 == 原文"。

> ⚠️ **注意**：读码相关的 crate 建议放在**独立的小工程**里演示（例如新建 `code_lab`），不要塞进本书主工程 `cv_examples`。原因下一节会讲——某些读码库会锁定特定的 `image` 补丁版本，和主工程冲突。

新建工程并配置依赖（版本为本机 `cargo` 实测、`cargo run` 跑通）：

```toml
# Cargo.toml
[dependencies]
qrcode = "0.14.1"   # 纯 Rust，QR 生成
rqrr   = "0.10.1"   # 纯 Rust，QR 定位 + 解码
image  = "0.25"     # 读写图片；与上面两者兼容
```

`src/main.rs`：

```rust
use image::Luma;
use qrcode::QrCode;

fn main() {
    // 原始要编码的文本（这里模拟一个产品追溯 URL）
    let text = "https://example.com/trace?sn=SN-20260827-000123";

    // 1) 生成：把文本编码成 QR，渲染成灰度图
    let code = QrCode::new(text.as_bytes()).expect("生成 QR 失败");
    let img = code
        .render::<Luma<u8>>()
        .min_dimensions(240, 240) // 放大，保证每个模块有足够像素
        .build();
    img.save("qr.png").expect("保存失败");
    println!("已生成 qr.png，尺寸 = {:?}", img.dimensions());

    // 2) 解码：rqrr 从灰度图里定位并解码
    let mut prepared = rqrr::PreparedImage::prepare(img);
    let grids = prepared.detect_grids();     // 找出图中所有 QR 栅格
    println!("检测到 {} 个 QR 码", grids.len());

    let (meta, decoded) = grids[0].decode().expect("解码失败");
    println!("版本={:?} 纠错等级={:?}", meta.version, meta.ecc_level);
    println!("原始文本 : {text}");
    println!("解码文本 : {decoded}");
    println!("闭环成功 : {}", decoded == text);
    assert_eq!(decoded, text); // 读回来的必须和原文一模一样
}
```

运行：

```bash
cargo run
```

实测输出：

```text
已生成 qr.png，尺寸 = (246, 246)
检测到 1 个 QR 码
版本=Version(4) 纠错等级=0
原始文本 : https://example.com/trace?sn=SN-20260827-000123
解码文本 : https://example.com/trace?sn=SN-20260827-000123
闭环成功 : true
```

`decoded == text` 为 `true`——你刚刚用纯 Rust 把一段文本"写"成 QR 又原样"读"了回来。`rqrr` 的 `PreparedImage::prepare` 会做灰度/二值化预处理，`detect_grids` 完成**定位 + 校正**，`decode` 完成**采样 + 纠错**，正好对应上面那条流水线。它还能顺带告诉你 QR 的版本号和纠错等级。

## 一维码与 DataMatrix：纯 Rust 的真实版图

到这里必须**说点实话**，别让你踩坑。

**一维码：生成容易，"专用"解码库薄。** 纯 Rust 的 `barcoders` crate 能生成 EAN/Code128/Code39 等一维码图像，但它**只管生成、不管解码**。想解码一维码，要么自己实现"条宽解析"（把一行像素二值化 → 量出每段黑白的宽度 → 按符号表翻译，原理不难但细节琐碎），要么用下面这个"通吃"方案。

**`rxing`：一个 crate 通吃 1D / QR / DataMatrix。** `rxing` 是大名鼎鼎的 ZXing（Java 读码库）的**纯 Rust 移植**，能从图像里解码一维码、QR、DataMatrix 等几乎所有主流码制。下面用 `barcoders` 生成一张 Code39、再用 `rxing` 解码，跑通一维码闭环（本机实测通过）：

```toml
# 注意：rxing 0.9.2 把 image 锁死在 =0.25.8
[dependencies]
barcoders  = { version = "2.0", features = ["image"] }
rxing      = "0.9.2"
image      = "=0.25.8"
```

```rust
use barcoders::generators::image::{Color, Image, Rotation};
use barcoders::sym::code39::Code39;
use image::{DynamicImage, GenericImage, Rgba, RgbaImage};

fn main() {
    let text = "RUST-CV-2026";
    let encoded = Code39::new(text).unwrap().encode(); // 0/1 序列

    // 生成条码图（黑条白底）
    let gen = Image::ImageBuffer {
        height: 80, xdim: 2, rotation: Rotation::Zero,
        foreground: Color::new([0, 0, 0, 255]),
        background: Color::new([255, 255, 255, 255]),
    };
    let bars: RgbaImage = gen.generate_buffer(&encoded).unwrap();

    // 一维码解码必须有静区！barcoders 不自动加，手动补 40px 白边
    let pad = 40;
    let mut canvas = RgbaImage::from_pixel(
        bars.width() + pad * 2, bars.height() + pad * 2, Rgba([255, 255, 255, 255]));
    canvas.copy_from(&bars, pad, pad).unwrap();

    // rxing 解码
    let res = rxing::helpers::detect_in_image(
        DynamicImage::ImageRgba8(canvas),
        Some(rxing::BarcodeFormat::CODE_39),
    ).unwrap();
    println!("解码={} 格式={:?} 闭环成功={}",
             res.getText(), res.getBarcodeFormat(), res.getText() == text);
}
```

实测输出 `解码=RUST-CV-2026 格式=CODE_39 闭环成功=true`。

> 🕳️ **坑**：`rxing` 功能全，代价是**依赖重**——它会拉进 `imageproc`、`nalgebra`、`chrono`、`regex` 等一大堆库，而且把 `image` **锁死在 `=0.25.8`**（本书主工程用的是 `0.25.10`）。所以务必单开一个工程放它，否则会和主工程报 "failed to select a version for image" 的冲突。轻量场景只需读 QR，就用上一节的 `rqrr`，别背上 `rxing` 这个大包袱。

**DataMatrix：`datamatrix` crate 能编能解。** 纯 Rust 的 `datamatrix` crate（0.3.3）同时支持编码与解码。注意它的 `decode` 吃的是**已经定位、二值化好的模块矩阵**（`&[bool]` + 宽度），并不直接吃相机原图——也就是说"从照片里找到 DataMatrix 并掰正"这步得你自己做（或者干脆交给 `rxing`）。下面是纯编解码闭环（实测通过）：

```rust
use datamatrix::{DataMatrix, SymbolList};

fn main() {
    let text = "DMX-ABC-123";
    let code = DataMatrix::encode(text.as_bytes(), SymbolList::default()).unwrap();
    let bm = code.bitmap();                    // 得到模块位图
    let bits: Vec<bool> = bm.bits().to_vec();  // 行主序，true=黑模块
    let bytes = DataMatrix::decode(&bits, bm.width()).unwrap();
    println!("闭环成功={}", String::from_utf8_lossy(&bytes) == text); // true
}
```

## 字符 OCR：这是深度学习的活儿

产线上还有一类需求：直接**认印在零件上的字符**（喷码的批号、激光刻的序列号）——这就是 **OCR（Optical Character Recognition，光学字符识别）**。必须坦白：**现代 OCR 是深度学习任务**，不是几行传统图像处理能搞定的，通常分**两段**：

1. **文本检测**：先找出"字在哪"，框出一个个文字区域（常用 DBNet、EAST 等分割/检测网络）。
2. **文本识别**：再把每个框里的像素"读"成字符串（常用 CRNN、SVTR 等序列识别网络）。

纯 Rust 现状（务必知道取舍）：

- **`ocrs`（0.12.2）+ `rten`（0.25.0）**：`ocrs` 是**纯 Rust** 的 OCR 引擎，底层跑在 `rten`——一个**纯 Rust 的 ONNX 推理运行时**上。这是目前"想纯 Rust 做 OCR"最现成的路子，需要下载它提供的检测/识别 ONNX 模型。
- **Tesseract（`tesseract` / `leptess` crate）**：老牌开源 OCR，但它是 **C++ 库 Tesseract + Leptonica 的 Rust 绑定**，**不是纯 Rust**，还要装系统依赖，和本书"纯 Rust"的底线相悖，仅作了解。

因为要下模型、篇幅也长，本章不塞一段"能跑的" OCR 代码（那样容易变成伪代码骗你）。完整的"检测 → 识别"实战放在 [第 99 章 OCR 项目](../p20-projects/ch99-ocr.md)，其中的模型加载与推理会用到 [第 19 部分：用 ONNX 跨框架推理](../p19-deploy/ch94-onnx.md) 的知识（纯 Rust 首选 `tract`，要极致性能可选非纯的 `ort`）。

## 常见坑与提示

> 🕳️ **坑**：**分辨率不够，神仙也读不出。** 读码对"每个模块占多少像素"有硬要求——经验上一维码窄条至少 2–3 像素、QR/DataMatrix 每模块至少 3–4 像素。相机离得太远或镜头太糊，模块糊成一团就必然失败。这直接牵扯到选型，见 [第 82 章 成像方案](ch82-imaging-hardware.md)。

> ⚠️ **注意**：**印刷/打标质量与反光是头号杀手。** 喷码发虚、激光打标深浅不均、金属件镜面反光盖住码区，都会让二值化失败。工业上常靠**打光**（低角度、同轴光）把码"打清楚"，而不是一味调算法——同样见下一章。

> 💡 **提示**：**留够静区、别把码怼到边缘。** 一维码尤其依赖左右白边（quiet zone），上面的例子里我们特意补了 40 像素白边才解得出。设计标签时给码留足空白，读码率立竿见影。

> 💡 **提示**：**纠错等级不是越高越好。** QR 的 H 档能扛 30% 破损，但同样数据要占更多模块、码变大变密。工业上按"码会不会被磨损/遮挡"来选：干净场景用 M，易磨损场景才上 Q/H。

> 🕳️ **坑**：**DataMatrix 别指望 QR 库能读。** 它和 QR 是两套定位与编码机制，`rqrr` 只读 QR。要读 DataMatrix，用 `datamatrix` crate 或 `rxing`。

## 本章小结

- 一维码靠**条宽**编码、信息少怕污损；QR 用**定位图案 + 定时图案 + Reed-Solomon 纠错**在横竖两向存大容量、抗污损；DataMatrix 小巧抗损，适合芯片打标。
- 读码通用流程：**定位 → 校正 → 采样解码 → 纠错**。
- 纯 Rust 实测可用：**QR 生成 `qrcode` + 解码 `rqrr`**（轻量，与 `image 0.25` 兼容）已跑通生成→解码闭环；**`rxing`** 是 ZXing 的纯 Rust 移植，一个库通吃 1D/QR/DataMatrix，但依赖重、锁 `image=0.25.8`；一维码生成用 `barcoders`；DataMatrix 编解码用 `datamatrix`。
- **OCR 是深度学习任务**：纯 Rust 走 `ocrs` + `rten`（纯 Rust ONNX 运行时）；Tesseract 是 C++ 绑定、非纯 Rust。实战见第 99 章，推理基础见第 94 章。

## 动手练习

1. **换内容、换纠错等级**：把 QR 例子里的 `text` 换成一段更长的文本，观察 `meta.version`（QR 版本号）如何变大；查阅 `qrcode` 文档，尝试指定纠错等级，看图变密后是否还能被 `rqrr` 读出。
2. **加干扰考验纠错**：在生成的 `qr.png` 上用画图工具涂掉一小块（或用前面章节学的绘制 API 画个黑方块），再跑解码，试出"涂多大" `rqrr` 就开始读不出——直观感受 Reed-Solomon 纠错的边界。
3. **一维码闭环**：仿照 Code39 例子，改用 `barcoders` 的 `EAN13` 生成一张商品条码，再用 `rxing` 解码（格式传 `BarcodeFormat::EAN_13`）；试试**不加白边**会发生什么。
4. **想一想（进阶）**：产线上码有时会旋转、倾斜甚至部分反光。对照本章的"定位 → 校正"两步，说说 QR 的三个定位图案分别帮解码器解决了"找得到"和"摆得正"中的哪个问题？

> 下一章我们换个视角：为什么老手都说"工业视觉一半功夫在打光"？镜头怎么选、光怎么打，才能让这一章的码、以及测量/缺陷检测都拍得又清楚又稳。👉 [第 82 章 光源、镜头与成像方案](ch82-imaging-hardware.md)
