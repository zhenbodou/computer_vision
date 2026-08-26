# 第 46 章 用 YOLO 做实时目标检测：可靠的 Rust 推理管线

> **本章导读**
> - 学完你能：亲手搭一个**极小的单尺度网格检测器**（YOLO 的核心思路），跑通一整条**目标检测推理管线**——预处理 → 前向 → **把网格输出解码成绝对坐标框** → 置信度阈值 → **NMS** → 画框 → 报 IoU。
> - 你会真正理解 [第 45 章](ch45-detection-principles.md) 讲的**解码与抑制**：不是背公式，而是看着代码把 6×6 个格子的原始数字，一步步变成图上一个个框。
> - 我们用**小模型 + 合成图**，让你**此刻、在纯 CPU 上、不下载任何模型**就跑通全流程；正文每个数字都来自本机 `cargo run --release` 的真实输出。
> - 最后给出**诚实的"上真实 YOLOv8"路线**：用纯 Rust 的 `tract-onnx` 加载官方 `.onnx`，附可编译的代码骨架，并讲清哪些地方**绝不能照抄下标**。

## 从"分类一张图"到"框出图里的每个目标"

[第 42 章](../p08-classification/ch42-candle-classify.md) 的分类器回答的是"这张图**是什么**"——一张图一个标签。但现实里更常见的问题是"这张图里**有哪些东西、各在哪**"：监控画面里有几个人、分别在什么位置？这就是**目标检测（object detection）**，输出的不再是一个标签，而是一串**框（bounding box）+ 类别 + 置信度**。

问题来了：一张图里目标数量不定（可能 0 个，也可能 10 个），网络怎么可能"输出一个不定长的框列表"？YOLO（You Only Look Once）给出的答案朴素又天才：**把图切成网格，让每个格子负责预测"以我为中心的那个目标"**。格子数是固定的，于是"不定长输出"就变成了"固定长度的网格张量"，一次前向全部算出来——这正是 YOLO"看一眼"就检测的含义。

本章我们就亲手搭一个**麻雀虽小、五脏俱全**的 YOLO 式检测器，重点放在**推理管线**上（训练的门道留到 [第 47 章](ch47-train-detector.md) 深入）。

## 原理：网格 + 每格一个预测向量

**直觉**：想象把 48×48 的输入图，盖上一张 6×6 的网格纸（每格 8×8 像素）。规定：**哪个目标的中心落在哪个格子里，就由那个格子负责预测它**。每个格子吐出一个长度固定的**预测向量**：

```text
输入 48×48 ──三次卷积+池化(÷8)──▶ 6×6 特征图 ──1×1卷积头──▶ 6×6×7 网格输出
                                                              │
每个格子(i,j) 预测 7 个数：                                    ▼
┌───────────┬───────────────────────┬──────────────────┐
│ obj 置信度 │  tx, ty, tw, th (框)   │  cls0, cls1 (类别) │
│  1 个      │      4 个              │     2 个           │
└───────────┴───────────────────────┴──────────────────┘
     ↑              ↑                        ↑
 这格有目标吗?  框相对本格的位置/大小      是"方块"还是"横条"?
```

我们的迷你检测器：输入 **48×48** RGB，经三段"卷积 + 2×2 最大池化"把边长砍半三次（48→24→12→6），得到 **6×6** 的特征图——**恰好对应 6×6 网格**；再用一个 **1×1 卷积头**在每个格子上输出 `A = 1 + 4 + C = 7` 个数（`C = 2` 类：方块 / 横条）。整张图的输出就是一个 `[1, 7, 6, 6]` 的张量。

> 📖 **术语**：**objectness（目标置信度）**。每个格子里那个专门回答"我这儿到底有没有目标"的分数。它和"是哪一类"是两件事：前者管**有无**，后者管**种类**。最终一个框的得分通常是两者相乘 `score = obj × 类别概率`——既要"确实有个东西"，又要"确实是这一类"。

### 把"格子输出"解码成"图上的框"

网络吐的是**相对量**（相对格子的偏移、相对整图的比例），必须**解码**成绝对像素坐标。设格子在第 `i` 行、第 `j` 列，stride `s = 8`，整图边长 `W = H = 48`，网络原始输出经 sigmoid 压到 \\((0,1)\\)：

\\[ c_x = (j + \sigma(t_x))\cdot s, \quad c_y = (i + \sigma(t_y))\cdot s \\]

\\[ w = \sigma(t_w)\cdot W, \quad h = \sigma(t_h)\cdot H \\]

即：中心 = **格子左上角 + 格内偏移**（偏移用 sigmoid 锁在本格内），宽高 = **整图的一个比例**。这样每个数都有明确的几何含义，也天然被约束在合理范围。解码出中心和宽高后，转成左上-右下角点 `[x1,y1,x2,y2]` 就是最终的框。

> ⚠️ **注意**：**解码必须严格逆转训练时的编码**。训练时你怎么把真值框"编码"成 `(tx,ty,tw,th)` 的目标，推理时就得怎么"解码"回来——差一个 stride、忘了 sigmoid、宽高的基准搞错，框就会飞到九霄云外。这是 [第 45 章](ch45-detection-principles.md) 反复强调的"解码要逆转参数化"的具体落地。

**一个目标，多个格子抢答怎么办？** 一个稍大的目标会覆盖好几个格子，邻近格子也可能给出高 objectness，于是同一个目标被画了好几个框。这时就要 **NMS（非极大值抑制，non-maximum suppression）**：按得分从高到低，保留最高分的框，删掉与它**重叠过多（IoU 超阈值）**的同类框。IoU（交并比）和 NMS 的原理见 [第 45 章](ch45-detection-principles.md)，本章直接用代码实现它。

## 动手：搭一个能真检出框的小检测器（candle）

> ⚠️ **注意**：本章代码基于 **candle 0.11.0**，纯 CPU。完整独立工程见 `code/dl_labs/ch46_yolo_infer/`，**`cd` 进去 `cargo run --release`** 即可复现（首次编译 candle + imageproc 较久）。

`Cargo.toml`：

```toml
[package]
name = "ch46_yolo_infer"
version = "0.1.0"
edition = "2021"

[workspace] # 空表：让本工程独立，不被上层 code/Cargo.toml 收编

[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
image = "0.25"
imageproc = "0.27"    # 画框存 PNG
```

### 检测器结构：三段卷积 + 1×1 检测头

```rust
const IMG: usize = 48;      // 输入边长
const G: usize = 6;         // 网格边长
const STRIDE: usize = IMG / G;   // = 8
const C: usize = 2;         // 类别数
const A: usize = 1 + 4 + C; // 每格属性数 = obj + (tx,ty,tw,th) + 类别 = 7

struct Detector { conv1: Conv2d, conv2: Conv2d, conv3: Conv2d, head: Conv2d }
impl Detector {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };  // 3x3 保持边长
        let cfg0 = Conv2dConfig { padding: 0, ..Default::default() }; // 1x1 头
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            head:  conv2d(64, A, 1, cfg0, vb.pp("head"))?, // → [N,A,6,6]
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 48→24
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;// 24→12
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?;// 12→6
        self.head.forward(&x) // 每个空间位置就是一个网格格子
    }
}
```

这个"骨干抽特征 → 头输出预测"的结构，和 [第 42 章](../p08-classification/ch42-candle-classify.md) 的分类网一脉相承，区别只在**头**：分类头输出 `[N, 类别数]`，检测头输出 `[N, A, G, G]`——**每个格子一份预测**。

### 合成数据 + 快速自训练

我们合成 48×48 的图，每张随机放 1~3 个白色几何形状（方块 / 横条），并记下它们的真值框。检测的**损失三件套**（objectness BCE + 类别 BCE + 框回归 L1）如何加权求和、如何只在"负责格"上算框和类别损失——这些**训练细节是 [第 47 章](ch47-train-detector.md) 的主角**，本章先把它当黑盒快速训练两三百轮（纯 CPU 也就几秒），好拿到一份"真会检测"的权重。训练输出（真实运行结果，250 轮，学习率 2e-3 后段降到 4e-4）：

```text
训练集：384 张图，共 765 个目标（平均每图 1.99 个）
开始训练小检测器（AdamW, lr=2e-3→4e-4, λ_box=5, λ_cls=1）...
  epoch     total       obj       box       cls
      1    2.2515    0.8848    0.1353    0.6900
     50    0.8456    0.4441    0.0624    0.0897
    150    0.3388    0.1407    0.0362    0.0172
    250    0.2372    0.1207    0.0212    0.0104
```

三项损失一起稳稳下降：objectness 从 0.88 降到 0.12，类别几乎清零，框回归 L1 降到 0.021（归一化坐标下，约合每个坐标 1 像素的平均误差）。

### 存 safetensors → 加载回来（真实推理入口）

训练完把权重存成 `.safetensors`，再**像加载任何真实模型一样加载回来**——这套 API 和 [第 42 章](../p08-classification/ch42-candle-classify.md) 加载 HF 预训练模型**完全一致**：

```rust
vm.save("detector.safetensors")?; // 存
// ⚠️ from_mmaped_safetensors 是 unsafe（内存映射文件）
let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["detector.safetensors"], DType::F32, &dev)? };
let model = Detector::load(vb2)?; // 权重来自文件，而非随机初始化
```

文件里的张量（真实输出）——注意检测头 `head.weight` 形状是 `[7, 64, 1, 1]`，即"每格 7 个输出、64 维输入、1×1 卷积"：

```text
  conv1.weight  形状 [16, 3, 3, 3]      conv2.weight  形状 [32, 16, 3, 3]
  conv3.weight  形状 [64, 32, 3, 3]      head.weight   形状 [7, 64, 1, 1]
```

### 完整推理管线：解码 → 阈值 → NMS → 画框 → IoU

**解码**：把网络输出 `[1,A,G,G]` 逐格还原成绝对坐标框（对照上面的解码公式，逐行对应）：

```rust
fn sigmoidf(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn decode(out: &Tensor, conf_thr: f32) -> Result<Vec<Det>> {
    let v = out.i(0)?.to_vec3::<f32>()?; // [A][G][G]
    let mut dets = Vec::new();
    for i in 0..G { for j in 0..G {
        let obj = sigmoidf(v[0][i][j]);                 // 该格“有目标”的置信度
        let (mut best_c, mut best_logit) = (0usize, f32::MIN);
        for c in 0..C { if v[5 + c][i][j] > best_logit { best_logit = v[5 + c][i][j]; best_c = c; } }
        let score = obj * sigmoidf(best_logit);         // obj × 类别置信度
        if score < conf_thr { continue; }               // 置信度阈值过滤
        let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32; // 解码中心
        let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
        let w = sigmoidf(v[3][i][j]) * IMG as f32;        // 解码宽高
        let h = sigmoidf(v[4][i][j]) * IMG as f32;
        dets.push(Det { cls: best_c, score,
            xyxy: [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0] });
    }}
    Ok(dets)
}
```

**IoU 与类别感知 NMS**（呼应 [第 45 章](ch45-detection-principles.md)，用半开区间 `xyxy` 算交并比）：

```rust
fn iou(a: &[f32;4], b: &[f32;4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
              * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ua = (a[2]-a[0]).max(0.0)*(a[3]-a[1]).max(0.0);
    let ub = (b[2]-b[0]).max(0.0)*(b[3]-b[1]).max(0.0);
    let u = ua + ub - inter;
    if u <= 0.0 { 0.0 } else { inter / u }
}
fn nms(mut dets: Vec<Det>, iou_thr: f32) -> Vec<Det> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap()); // 按分数降序
    let mut removed = vec![false; dets.len()];
    let mut keep = Vec::new();
    for i in 0..dets.len() {
        if removed[i] { continue; }
        keep.push(dets[i].clone());
        for j in (i+1)..dets.len() {   // 删掉与它同类且重叠过多的框
            if !removed[j] && dets[j].cls == dets[i].cls
               && iou(&dets[i].xyxy, &dets[j].xyxy) > iou_thr { removed[j] = true; }
        }
    }
    keep
}
```

**串起来**：对每张测试图，预处理成 `[1,3,48,48]` → 前向 → 解码（阈值 0.5）→ NMS（IoU 0.45）→ 把每个检出框贪心匹配到同类、IoU 最大的真值框并打印，最后 `imageproc` 画框存 PNG。真实运行输出（节选）：

```text
测试图 #2：1 个真值目标，检出 1 个框
   ✓ 横条  score=0.93  框=[  16,  16,  39,  22]  IoU=0.825
测试图 #4：2 个真值目标，检出 2 个框
   ✓ 方块  score=0.95  框=[  25,  22,  39,  34]  IoU=0.840
   ✓ 横条  score=0.76  框=[   7,  20,  31,  27]  IoU=0.681

===== 汇总 =====
真值目标共 10 个，正确检出 10 个（检出率 100.0%）
命中框的平均 IoU = 0.775
可视化结果已存为 det_0.png .. det_4.png（绿=真值，红=检出）
```

**5 张测试图共 10 个目标，全部检出，命中框平均 IoU 0.775**——这个"从一堆网格数字到图上一个个框"的完整链路，就是所有单阶段检测器的通用骨架。生成的 PNG 里，绿框（真值）和红框（检出）大体重合，肉眼可见检得挺准。

> 💡 **提示**：**留意那两个"多出来的框"。** 5 张图里，除了 10 个正确框，还冒出 2 个来自**相邻格子**的重复框（同类、但和主框形状差异大，IoU 没到 NMS 阈值，于是没被抑制掉）。这不是 bug，而是单尺度网格检测器的真实脾气——它恰好说明：**光有模型不够，后处理（阈值、NMS）和它的参数，是把"原始输出"变成"可用结果"的关键一环**（[第 45 章](ch45-detection-principles.md) 的观点在此落地）。把 NMS 阈值、置信度阈值调一调，结果就会变，这正是 [第 47 章](ch47-train-detector.md)、[第 48 章](ch48-metrics-annotation.md) 要系统讨论的。

## 诚实的"上真实 YOLOv8"路线（tract-onnx，纯 Rust）

上面那个是**我们自己训的迷你检测器**。你八成想问：**能不能在纯 Rust 里跑官方那个又快又准的 YOLOv8？** 能，但要说清边界：

> ⚠️ **注意（诚实边界）**：**纯 Rust 生态里没有"内置权重、开箱即跑"的 YOLO。** 官方 YOLOv8 的权重是 Ultralytics 用 PyTorch 训练的，你需要**自己把它导出成 `.onnx`**（`yolo export model=yolov8n.pt format=onnx`，或从其发布页/模型仓库下载现成的 `yolov8n.onnx`），再用纯 Rust 的推理引擎加载。因为要联网下载模型，这条路线**不纳入本章的一键 demo**，但代码是**真实、正确、可编译**的。

纯 Rust 跑 ONNX 首选 **`tract-onnx`**（[第 94 章](../p19-deploy/ch94-onnx.md) 详述，零 C/C++ 依赖）。下面是加载真实 `yolov8n.onnx` 的推理骨架。**关键的两个正确性要点**：① **letterbox 预处理**（保长宽比缩放 + 灰边填充，并记住逆变换参数）；② **按模型实际输出布局解码**。

**letterbox 预处理**（保长宽比，把框映射回原图靠这里记下的 `scale / pad`）：

```rust
const INPUT: usize = 640; // YOLOv8 常用输入边长
struct LetterMeta { scale: f32, pad_x: f32, pad_y: f32 }

fn letterbox(img: &RgbImage) -> (Vec<f32>, LetterMeta) {
    let (w0, h0) = (img.width() as f32, img.height() as f32);
    let scale = (INPUT as f32 / w0).min(INPUT as f32 / h0);       // 按长边缩放
    let (nw, nh) = ((w0*scale).round() as u32, (h0*scale).round() as u32);
    let resized = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
    let (pad_x, pad_y) = ((INPUT as u32 - nw) as f32/2.0, (INPUT as u32 - nh) as f32/2.0);
    let mut canvas = RgbImage::from_pixel(INPUT as u32, INPUT as u32, image::Rgb([114,114,114]));
    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);
    let mut data = vec![0f32; 3*INPUT*INPUT];                     // HWC→CHW + /255
    for y in 0..INPUT { for x in 0..INPUT {
        let p = canvas.get_pixel(x as u32, y as u32).0;
        for c in 0..3 { data[c*INPUT*INPUT + y*INPUT + x] = p[c] as f32 / 255.0; }
    }}
    (data, LetterMeta { scale, pad_x, pad_y })
}
```

**加载、推理、解码**（tract API 见 [第 94 章](../p19-deploy/ch94-onnx.md)）：

```rust
let model = tract_onnx::onnx()
    .model_for_path("yolov8n.onnx")?
    .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
    .into_optimized()?.into_runnable()?;

let img = image::open("test.jpg")?.to_rgb8();
let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
let (data, meta) = letterbox(&img);
let input = Tensor::from_shape(&[1, 3, INPUT, INPUT], &data)?;
let result = model.run(tvec!(input.into()))?;
let out = result[0].to_plain_array_view::<f32>()?; // 期望 [1,84,8400]
let shape = out.shape().to_vec();
println!("原始输出形状 = {:?}（务必用模型检查器确认，别照抄下标！）", shape);

// YOLOv8 常见布局 [1, 84, 8400]：84 = 4(cx,cy,w,h, 输入像素) + 80 类别分数（无独立 objectness）
let (n_ch, n_anchor) = (shape[1], shape[2]);
let n_class = n_ch - 4;
let s = out.as_slice().unwrap();               // C 序连续：元素 (0,k,a) 在 s[k*n_anchor + a]
let at = |k: usize, a: usize| s[k * n_anchor + a];
let mut dets: Vec<Det> = Vec::new();
for a in 0..n_anchor {
    let (mut best_c, mut best_s) = (0usize, 0f32);
    for c in 0..n_class { let v = at(4 + c, a); if v > best_s { best_s = v; best_c = c; } }
    if best_s < 0.25 { continue; }
    let (cx, cy, w, h) = (at(0,a), at(1,a), at(2,a), at(3,a));
    // letterbox 逆变换：减 padding，除以 scale，裁剪到原图边界
    let x1 = (((cx - w/2.0) - meta.pad_x)/meta.scale).clamp(0.0, orig_w);
    let y1 = (((cy - h/2.0) - meta.pad_y)/meta.scale).clamp(0.0, orig_h);
    let x2 = (((cx + w/2.0) - meta.pad_x)/meta.scale).clamp(0.0, orig_w);
    let y2 = (((cy + h/2.0) - meta.pad_y)/meta.scale).clamp(0.0, orig_h);
    dets.push(Det { cls: best_c, score: best_s, xyxy: [x1, y1, x2, y2] });
}
let dets = nms(dets, 0.45); // 复用上面的 iou/nms（Det 用同样的 xyxy 字段）
```

> 🕳️ **坑（本章最想让你记住的一条）**：**别把某个数组下标当"通用 YOLO"写死。** 不同型号、不同导出参数，输出布局差别很大：可能是 `[1,84,8400]`，也可能是 `[1,8400,84]`（把上面的 `at(k,a)` 索引换一换）；YOLOv5/v7 带**独立 objectness**（那一列要单独乘进去），YOLOv8 则**没有**；有的导出**已内置 NMS**（输出直接是框列表，你就别再 NMS 了）；类别分数有的已过 sigmoid、有的是原始 logit。**正确做法**：导出后先用模型检查器（如 Netron）看清输入输出的名字、形状、语义，用 20 个黄金样本和原框架对齐结果，再把预处理/解码/NMS 固化下来。上面代码里那句 `println!("原始输出形状 = ...")` 就是在提醒你：**先看形状，再谈解码**。

> 💡 **提示（验证到什么程度）**：本骨架已在独立工程用 **`tract-onnx 0.23.5` `cargo check` 编译通过**（Rust 1.97.1），但**没有跑真实模型**——因为那需要你自备 `yolov8n.onnx`。所以这里**不会**给你"检出 N 个框、mAP 多少"之类的数字：**没跑过的模型，本书绝不编数字**。等你下载好模型、按上面骨架接上，就能得到真实结果；跑不通时，十有八九是**预处理或解码没对齐**（回去查 letterbox 的 scale/pad、输出布局、是否要 sigmoid）。

## 常见坑与提示

> 🕳️ **坑**：**解码和编码必须严丝合缝地互逆。** 训练时中心用 `(格号 + sigmoid偏移) / 网格数` 编码、宽高用"整图占比"编码，推理解码就得原样逆回来（本章 `decode` 里每一行都对应训练时的一步）。任何一处基准不一致（stride、是否 sigmoid、宽高相对格子还是相对整图），框都会系统性地偏。这类 bug **不报错**，只是框全歪——最难查。

> ⚠️ **注意**：**NMS 是"按类"还是"跨类"，是个要想清楚的选择。** 本章用**类别感知 NMS**（只抑制同类框），好处是不会把挨得近的不同类目标误删；代价是同一个目标若被预测成两个不同类，两个框都会留下。生产里常见的还有**跨类 NMS**、Soft-NMS、DIoU-NMS 等变体，各有适用场景（[第 45 章](ch45-detection-principles.md)）——**按你的目标是否会紧贴、是否允许重叠来选**，别无脑套默认。

> ⚠️ **注意**：**letterbox 的逆变换别偷懒。** 预处理时为了保长宽比做了缩放 + 填充，后处理就必须**减 padding、除 scale**才能把框映射回原图；两侧 padding 若因奇偶不同差一个像素，也要如实记录（[第 45 章](ch45-detection-principles.md)）。直接把 640 坐标系的框画到原图上，是新手最常见的"框整体偏移/缩放"故障。

> 🕳️ **坑**：**candle 0.11 的 CPU 后端不能 `set_seed`。** 和 [第 42 章](../p08-classification/ch42-candle-classify.md) 一样，本章用**自造的确定性初始化**（固定种子的 LCG 写权重）来保证每次 `cargo run` 的 loss 和检出数字逐字可复现，而不是依赖框架的随机初始化。

> 💡 **提示**：**"benchmark FPS" 不等于"系统 FPS"。** 真要做实时检测，模型前向只是流水线的一环；采集、解码、letterbox、内存拷贝、NMS、绘制都吃时间。评估实时性要**分段计时、预热后报告 P50/P95**，还要看**丢帧率与帧龄**——能处理 30 FPS 但延迟越积越多，并不叫实时。

## 本章小结

- **目标检测**把"一张图一个标签"升级成"一串框 + 类别 + 置信度"。YOLO 的核心思路是**把图切成网格、每格预测一个固定长度的向量**，从而把"不定长输出"变成"固定形状的网格张量"，一次前向全部算出。
- 一条完整的**检测推理管线**是：预处理 → 前向 → **解码**（把相对量还原成绝对坐标框）→ **置信度阈值** → **NMS** → 映射回原图 → 输出。解码必须严格逆转训练时的编码。
- 我们实测：一个 6×6 网格、每格 7 维输出的迷你检测器，训练后在合成测试图上 **10 个目标全部检出、命中框平均 IoU 0.775**；`VarMap::save` 存 safetensors、`unsafe from_mmaped_safetensors` 加载回来，与加载真实预训练模型是同一套 API。
- **诚实边界**：纯 Rust 没有内置权重的开箱 YOLO；要跑官方 YOLOv8，用 `tract-onnx`（纯 Rust）加载你自己导出的 `.onnx`，配 letterbox 预处理与**按实际布局**的解码。**不同型号输出布局不同，绝不能照抄某个下标**——先看形状、对齐黄金样本，再固化后处理。

## 动手练习

1. **调阈值看变化**：把 `decode` 的置信度阈值从 0.5 改成 0.3 和 0.7，把 NMS 的 IoU 阈值从 0.45 改成 0.3 和 0.6，各跑一遍，观察"检出数、误检数、漏检数"怎么此消彼长——亲手体会后处理参数的分量。
2. **故意写错解码**：把 `decode` 里宽高的 `* IMG` 误改成 `* STRIDE`，重跑，看看框会崩成什么样——记住这种"不报错但全歪"的故障长相。
3. **加一个类别**：把合成数据加上第三类（比如"竖条"），`C` 改成 3、`A` 变成 8，重新训练+推理，验证整条管线对类别数是通用的。
4. **接真实模型（进阶）**：`cargo add tract-onnx image`，下载一个 `yolov8n.onnx`，用本章的 tract 骨架跑一张真实照片；先 `println!` 出输出形状，确认布局后再解码。跑通后对照官方 Python 结果，验证你的 letterbox 逆变换和解码是否对齐。

下一章，我们把镜头对准**训练**本身：检测的**损失三件套**到底怎么加权反向、怎么用**数据闭环与受控实验**（划分泄漏、单变量消融）把一个检测器**真正调好**。👉 [第 47 章](ch47-train-detector.md)
