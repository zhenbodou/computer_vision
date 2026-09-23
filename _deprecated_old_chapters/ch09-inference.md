# 第 9 章 · 模型推理：用 Rust 把模型跑起来

> **本章导读**：前八章所有的铺垫——像素、预处理、模型文件、四要素——都是为了本章这一刻：把一张图喂给模型，拿到输出张量。我们会用两个引擎完成这件事：纯 Rust 的 tract（简单好部署）和微软 ONNX Runtime 的 Rust 绑定 ort（性能标杆）。跑通之后，你会拥有一个完整可运行的图像分类程序——它正是第 11 章目标检测项目的骨架。

**学习目标**：
- 理解推理引擎存在的意义（图优化、内存规划、硬件后端）
- 会用 tract-onnx 加载模型、喂张量、取输出
- 会用 ort 做同样的事，知道两者的取舍
- 掌握推理程序的性能常识（预热、计时、Session 复用）
- 独立完成端到端图像分类程序（读图→预处理→推理→softmax→top-5）

## 9.1 推理引擎是什么

模型文件不能直接"运行"——它只是一份"菜谱 + 食材清单"（7.5 节）。需要一个程序把菜谱翻译成机器指令并执行，这个程序就是**推理引擎**（inference engine）。

类比：模型文件是 MP3，推理引擎是播放器。播放器默默做了很多事：解码、均衡器、调用声卡。推理引擎也做了四件你不写代码就享受不到的事：

1. **读图**：解析 onnx 文件，重建计算图和权重
2. **图优化**：常量折叠（能提前算掉的先算掉）、算子融合（Conv+ReLU 合并成一个 kernel，省一次内存往返）、布局变换——同一张图，优化后快 20~50% 是常态
3. **内存规划**：规划所有中间张量的存放位置，能复用的内存块复用（4K 特征图级别的内存省下来不是小数）
4. **后端调度**：决定每个算子跑在哪——CPU 用 SIMD 指令，有 GPU/NPU 就调专用 kernel

所以别手写推理（教学 toy 除外），引擎比你快得多、稳得多。

## 9.2 推理引擎怎么选

| 引擎 | crate | 特点 | 适合 |
| --- | --- | --- | --- |
| ONNXRuntime | `ort` (2.x) | 性能标杆，支持 TensorRT/OpenVINO 等后端 | x86 服务器、要性能 |
| tract | `tract-onnx` (0.21) | **纯 Rust 零 C 依赖**，交叉编译友好 | 嵌入式、Rust 洁癖、快速上手 |
| OpenCV DNN | `opencv` | 轻量、顺便有全套图像算法 | 本来就依赖 opencv 的项目 |
| TFLite | `tflite-rs` | 移动端血统 | 安卓/嵌入式 |
| Candle | `candle` | HuggingFace 出品，纯 Rust 含 GPU | LLM 向，视觉也在扩展 |
| 厂商 SDK | FFI 自绑 | RKNN/NCNN/TensorRT 等 | 特定硬件榨性能 |

**选择建议**：

- 刚入门/要快速跑通 → **tract**（本章先用它）
- 生产环境要性能 → **ort**（9.5 节）
- 目标设备有 NPU → 厂商 SDK（第 11.6 节思路）

## 9.3 tract 跑通第一个模型

先来个最小程序，把"加载→喂→取"的骨架立起来：

```bash
cargo new ch09_infer && cd ch09_infer
cargo add tract-onnx ndarray anyhow
```

```rust
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    // ① 加载模型（三部曲：解析 → 图优化 → 变成可执行的）
    let model = tract_onnx::onnx()
        .model_for_path("yolov8n_sim.onnx")?
        .into_optimized()?
        .into_runnable()?;

    // ② 造一个模型期望的输入（四要素：images, [1,3,640,640], f32）
    let input = tract_ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

    // ③ 推理：输入张量列表 → 输出张量列表
    let mut outputs = model.run(tvec!(input.into()))?;

    // ④ 拿第一个输出，看它的形状
    let output = outputs.remove(0);
    let view = output.to_array_view::<f32>()?;
    println!("输出形状: {:?}", view.shape());   // [1, 84, 8400]
    Ok(())
}
```

四步对应四件事，一次记住：

```text
model_for_path  → 读文件、建图
into_optimized  → 图优化（9.1 的第 2 件事）
into_runnable   → 编译成可执行计划
run             → 真跑，喂入 tvec![]（一个或多个张量），拿回 outputs
```

注意输入是 `tract_ndarray::Array4`——tract 自带的 ndarray 版本。它和 crates.io 的 ndarray 同源但版本可能错开，工程上要么统一用 `tract_ndarray`，要么类型转换时小心版本匹配。**这就是 tract 唯一的槽点**。

## 9.4 完整图像分类实战

光打印形状没意思，做个完整的：**用 SqueezeNet 对图片做 ImageNet 分类**。

模型获取（选一个有公开地址的）：onnx 官方模型仓库的 squeezenet：

```bash
curl -L -o squeezenet.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet-1.2-7.onnx
```

类别名文件（1000 个 ImageNet 类别，很多仓库都有，搜索 "imagenet_classes.txt" 下载一份，放 `data/imagenet_classes.txt`，一行一个类别名）。

程序（接第 3、5 章的所有知识）：

```rust
use tract_onnx::prelude::*;
use image::open;

fn main() -> TractResult<()> {
    // 预处理（第 5 章的知识：resize → 归一化 → CHW）
    let img = open("test.jpg").unwrap().to_rgb8();
    let resized = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Triangle);
    let raw: Vec<f32> = resized.into_raw()
        .iter().map(|&v| v as f32 / 255.0).collect();
    // ImageNet 标准化
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let hwc = tract_ndarray::Array3::from_shape_vec((224, 224, 3), raw).unwrap();
    let mut chw = hwc.permuted_axes([2, 0, 1]).to_owned();
    for c in 0..3 {
        chw.index_axis_mut(tract_ndarray::Axis(0), c)
            .mapv_inplace(|v| (v - mean[c]) / std[c]);
    }
    let input = chw.insert_axis(tract_ndarray::Axis(0));  // [1,3,224,224]

    // 推理
    let model = tract_onnx::onnx()
        .model_for_path("squeezenet.onnx")?
        .into_optimized()?
        .into_runnable()?;
    let mut outputs = model.run(tvec!(input.into()))?;
    let view = outputs.remove(0).to_array_view::<f32>()?.to_owned();

    // 后处理（第 10.2 节详讲）：softmax → top-5
    let logits = view.index_axis(tract_ndarray::Axis(0), 0);  // [1000]
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<(usize, f32)> = exps.iter().enumerate()
        .map(|(i, e)| (i, e / sum)).collect();

    // 读类别名
    let names: Vec<String> = std::fs::read_to_string("data/imagenet_classes.txt")
        .unwrap().lines().map(|s| s.to_string()).collect();

    // 取 top-5
    let mut sorted = probs.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (idx, p) in sorted.iter().take(5) {
        println!("{:>6.2}%  {}", p * 100.0, names.get(*idx).map(|s| s.as_str()).unwrap_or("?"));
    }
    Ok(())
}
```

预期输出长这样：

```text
 78.31%  tabby, tabby cat
 12.11%  tiger cat
  3.55%  Egyptian cat
  ...
```

恭喜——到这一行代码为止，你已经走完了机器视觉流水线的**完整一圈**：解码（image crate）→ 预处理（resize/归一化/CHW）→ 推理（tract）→ 后处理雏形（softmax/top-k）。剩下两章就是把每个环节做深。

## 9.5 ort：ONNX Runtime 的 Rust 绑定

**ort** 把微软的 ONNX Runtime（工业界使用最广的推理引擎）包成了 Rust API。相比 tract：性能更强（可用 GPU、TensorRT 后端）、算子覆盖更全；代价是要带 C 动态库（或编译时链接），部署重一些。

两种引入方式：

```toml
[dependencies]
# 方式一：下载官方预编译动态库（最省心，默认行为）
ort = "2"
# 方式二：链接系统库（服务器上已装 onnxruntime 时）
# ort = { version = "2", features = ["load-dynamic"] }
```

用法和 tract 几乎一一对应：

```rust
use ort::{Session, Value};
use ndarray::Array4;

fn main() -> anyhow::Result<()> {
    // 加载
    let mut session = Session::builder()?
        .with_model_from_file("yolov8n_sim.onnx")?;

    // 按名字喂输入（"images" 来自四要素！）
    let array = Array4::<f32>::zeros((1, 3, 640, 640));
    let outputs = session.run(ort::inputs!["images" => array]?)?;

    // 取输出，转 ndarray 视图
    let output = &outputs[0];
    let view = output.try_extract::<f32>()?.view().to_owned();
    println!("输出形状: {:?}", view.shape());
    Ok(())
}
```

`ort::inputs!["images" => array]` 这个宏：按**输入名**组装输入。名字打错会报"找不到输入"——再次印证四要素的重要性。**ort 用的是 crates.io 标准 ndarray**，和 image crate 互转时不用像 tract 那样担心版本错开。

## 9.6 张量操作细节清单

实战中会反复用到的细节，一次备齐：

**dtype 对应表**（ONNX 类型 ↔ Rust 类型）：

| ONNX | Rust | 说明 |
| --- | --- | --- |
| `tensor(float)` | `f32` | 最常见 |
| `tensor(uint8)` | `u8` | INT8 量化模型 |
| `tensor(int64)` | `i64` | 常见于索引类输入 |
| `tensor(float16)` | `f16`（half crate） | 半精度 |

**Session 只能这么用**：

```rust
// ❌ 错误：循环里反复加载模型 —— 每次加载要几秒钟 + 几个 GB 内存
for img in images { let m = load_model(); ... }

// ✅ 正确：加载一次，循环复用
let model = load_model();
for img in images { model.run(...); }
```

**输出转 ndarray**：tract 的 `to_array_view::<f32>()` 和 ort 的 `try_extract::<f32>()` 都是零拷贝视图；要长期持有就 `.to_owned()`。

**动态维度**：tract 可以在加载时用 `.with_input_fact(0, f32::fact([4, 3, 640, 640]).into())?` 把动态 batch 固定成 4；ort 直接喂 `[4,3,640,640]` 的数组即可，多数维度自适应。

## 9.7 性能意识：预热与计时

**预热（warmup）**：引擎首次推理往往慢 2~10 倍——内存首次分配、kernel 首次编译、CPU 缓存冷启动。所以 benchmark 必须**先空跑 3~10 次再计时**，否则数据毫无意义。

```rust
use std::time::Instant;

// 预热
for _ in 0..10 { let _ = model.run(tvec!(input.clone().into()))?; }

// 正式计时：跑 100 次取平均
let start = Instant::now();
let n = 100;
for _ in 0..n { let _ = model.run(tvec!(input.clone().into()))?; }
println!("平均单次推理: {:?}", start.elapsed() / n);
```

其他性能杠杆：

- **线程**：ONNXRuntime 默认用满 CPU 核；ort 可通过环境变量 `OMP_NUM_THREADS` 或 API 控制
- **图优化级别**：ort 有 `Session.builder()?.with_optimization_level(...)`，生产用 `Level3`
- **并行粒度**：单帧延迟优先（摄像头）→ 单线程 + 大开销 kernel；吞吐量优先（离线批处理）→ 多实例并行

## 9.8 常见报错与排查

| 报错 | 病因 | 对策 |
| --- | --- | --- |
| `expected shape [...] got [...]` | 输入 shape 和四要素不符 | 打印双方 shape 对比 |
| `found input type f32 expected u8` | dtype 喂错 | INT8 模型要喂 u8 张量 |
| `no input named 'xxx'` | inputs! 名字打错 | 以 8.6 打印为准 |
| 跑起来但输出全 NaN | 预处理错（如除零、忘归一化） | 把输入张量存成图肉眼看 |
| 内存占用持续上涨 | Session 在循环里重复创建 | 挪出循环（9.6） |
| tract 编译报 ndarray 版本冲突 | tract_ndarray 和 ndarray 混用 | 统一用 tract_ndarray 或显式转换 |

## 9.9 小结

- 推理引擎替你做了读图、图优化、内存规划、硬件调度四件事，不要手写推理。
- tract 四步：`model_for_path → into_optimized → into_runnable → run`；ort 对应 `Session::builder → with_model_from_file → run(inputs![])`。
- 分类实战串起了全书：解码→预处理→推理→softmax→top-5。
- Session/Model 必须加载一次复用无数次；benchmark 先预热。
- 排障三板斧：打印 shape、打印 dtype、把输入存成图看一眼。

## 9.10 练习

1. 把 9.4 的分类程序换一张完全不同的图（比如食物、汽车），验证 top-5 合理性。
2. 用 ort 重写 9.4（模型和预处理不变），对比两个引擎的输出是否一致（应该几乎逐位相同或误差 <1e-4）。
3. 给推理加预热和 100 次计时，报告你的机器上单张 YOLOv8n 的平均耗时。
4. 故意把输入归一化步骤删掉再跑分类，观察 top-1 变成什么——体会预处理的重要性。
