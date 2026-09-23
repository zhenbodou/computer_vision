# 第 10 章 · 后处理：把一堆数字变成看得懂的结果

> **本章导读**：上一章模型吐出的 `[1, 84, 8400]` 是什么？它既不是框也不是类别，而是 8400 个"候选答案"的压缩包：每个候选带着 4 个位置数字和 80 个类别分数。本章就是"解包"的全过程：置信度过滤、坐标还原（letterbox 的逆变换）、NMS 去重。学完本章，你手里就握着检测任务的全部零件——第 11 章把它们组装成完整的程序。

**学习目标**：
- 看懂 YOLOv8 输出张量 `[1, 84, 8400]` 的每一维、每一行
- 会算 softmax/sigmoid，知道 YOLOv8 的分数为什么要过 sigmoid
- 完成置信度过滤 + 坐标逆变换的完整代码
- 理解 IoU 和 NMS 的每一步，能独立写出 NMS
- 了解分割任务后处理的基本流程

## 10.1 后处理是什么

回顾第 1 章那张漂流图：模型输出的**不是**画好框的图片，而是一堆数字。从"数字"到"3 只猫在哪些位置"，中间的翻译工作就是**后处理（post-processing）**。

类比：模型是一位只说术语的鉴定专家，输出的是一行行实验数据；后处理是站在你跟前的翻译官，把数据整理成"这里、这个、这么大概率"。

后处理按任务分三兄弟，本章主线是检测，分类已在 9.4 节见过，分割 10.7 节简介：

```text
分类后处理：softmax → argmax/top-k               （简单）
检测后处理：置信度过滤 → 坐标还原 → NMS           （本章主角，最复杂）
分割后处理：框解码 → 掩码系数 × 原型 → 阈值        （进阶）
```

## 10.2 分类后处理：softmax 与 top-k

（9.4 实战过，这里把原理补齐。）

模型最后一层输出的原始分数叫 **logits**——可以是任意实数，包括负数。要变成"概率"需要 softmax：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**手算一遍**（附录 A 有更详细的版本）。假设 logits 是 `[2.0, 1.0, 0.1]`：

```text
e^2.0 ≈ 7.39,  e^1.0 ≈ 2.72,  e^0.1 ≈ 1.11
分母 = 7.39 + 2.72 + 1.11 = 11.22
概率 = [7.39/11.22, 2.72/11.22, 1.11/11.22] ≈ [0.659, 0.242, 0.099]   ✔ 和为 1
```

工程细节：先减最大值再取 e（代码里的 `v - max`），防止 e 的大数次方溢出。`e^200` 是天文数字，但 softmax 的分子分母同乘同除，先减 max 不改变结果——这是数值稳定性标准操作。

**top-k**：按概率排序取前 k 个。第 9.4 的 top-5 就是它。

## 10.3 看懂 YOLOv8 的输出：[1, 84, 8400]

### 从 8400 说起

YOLO 把 640×640 的输入图在三个尺度上打网格： stride 8（80×80）、stride 16（40×40）、stride 32（20×20）。每个格子是一个"候选位置"：

```text
80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400 个候选
```

"stride 8"意思是每 8 像素一个候选格——格子越密，小目标越不容易漏（为什么小目标检测难？因为小目标在高 stride 的粗网格里可能只占半个格）。

### 张量解剖

```text
形状：[1, 84, 8400]
       │   │    └─ 8400 个候选框，每列是一个候选
       │   └─ 84 行 = 4 个位置数 + 80 个 COCO 类别分数
       └─ batch = 1（一次一张图）

84 行的含义（按行看）：
行 0: cx   候选框中心 x（0~640 画布坐标）
行 1: cy   候选框中心 y
行 2: w    框宽
行 3: h    框高
行 4~83:   80 个类别的分数（未归一化的 logits）
```

用 ASCII 画出来（数值虚构）：

```text
          候选0   候选1   候选2  ...  候选8399
  cx  [  320.5,  12.0,  500.3, ... ]
  cy  [  210.2,  33.1,  480.0, ... ]
  w   [   85.0,   9.2,  200.0, ... ]
  h   [  120.4,   8.8,  180.5, ... ]
  人  [   0.02,  0.61,   0.01, ... ]   ← 候选1 可能是个人
  猫  [   6.20, -1.10,   0.05, ... ]   ← 候选0 的猫 logit 很高！
  狗  [   1.10,  0.20,   4.50, ... ]   ← 候选8399 可能是条狗
  ...
```

### 与 YOLOv5 的关键区别（读老代码必踩）

```text
YOLOv5：85 行 = 4 位置 + 1 objectness + 80 类
        置信度 = sigmoid(objectness) × sigmoid(类别分)
YOLOv8：84 行 = 4 位置 + 80 类（没有 objectness！）
        置信度 = sigmoid(类别分) 取各类中的最大值
```

v8 把"是不是物体"的判断隐式地融入了类别分数。所以网上抄来的 v5 后处理代码不能直接用在 v8 上——少乘了一个 objectness，分数体系完全不同。看输出是 84 还是 85 行，立刻知道该用哪套解码逻辑。

## 10.4 置信度过滤：把 8400 筛到几十

8400 个候选里，绝大多数是"什么都不是"的背景格。第一道工序：每列取 80 个类别分数里最大的，过 sigmoid，低于阈值（常用 0.25）整列丢弃。

```rust
use ndarray::{ArrayView3, Axis};

/// 一个候选检测
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32, pub y: f32,     // 中心坐标（640 画布上）
    pub w: f32, pub h: f32,     // 宽高
    pub score: f32,             // 置信度 0~1
    pub class_id: usize,        // 类别
}

pub fn filter_candidates(
    output: &ArrayView3<f32>,   // [1, 84, 8400]
    conf_thresh: f32,
) -> Vec<Detection> {
    let view = output.index_axis(Axis(0), 0);        // [84, 8400]
    let boxes = view.slice(s![..4, ..]);              // [4, 8400] 位置
    let scores = view.slice(s![4.., ..]);             // [80, 8400] 类别 logits

    let mut dets = Vec::new();
    for i in 0..view.dim().1 {                        // 遍历 8400 列
        // 找该列分数最高的类别
        let (class_id, &max_logit) = scores
            .column(i).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let score = sigmoid(max_logit);               // sigmoid 压到 0~1
        if score < conf_thresh { continue; }

        dets.push(Detection {
            x: boxes[[0, i]], y: boxes[[1, i]],
            w: boxes[[2, i]], h: boxes[[3, i]],
            score, class_id,
        });
    }
    dets
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
```

通常 8400 个候选过滤后只剩 20~200 个，进入下一道工序。

## 10.5 坐标还原：从 640 画布回到原图

模型看到的图是 letterbox 缩放过、垫了灰边的 640×640（第 5.3 节）。所以输出的 (x, y, w, h) 全部基于这张画布——要画回原图，必须逆变换：

$$\text{原图坐标} = \frac{\text{画布坐标} - \text{灰边偏移}}{\text{缩放比 } r}$$

```rust
/// letterbox 逆变换 + xywh → xyxy（左上角/右下角，方便画框）
pub fn restore_and_convert(
    d: &Detection, r: f32, pad: (f32, f32), img_w: u32, img_h: u32,
) -> (f32, f32, f32, f32, f32, usize) {
    // ① 减灰边、除缩放比 → 回到原图坐标系
    let x = (d.x - pad.0) / r;
    let y = (d.y - pad.1) / r;
    let w = d.w / r;
    let h = d.h / r;
    // ② xywh（中心+宽高）→ xyxy（左上+右下）
    let x1 = x - w / 2.0;
    let y1 = y - h / 2.0;
    let x2 = x + w / 2.0;
    let y2 = y + h / 2.0;
    // ③ 裁到图内，防止画框画出界
    (
        x1.max(0.0).min(img_w as f32),
        y1.max(0.0).min(img_h as f32),
        x2.max(0.0).min(img_w as f32),
        y2.max(0.0).min(img_h as f32),
        d.score, d.class_id,
    )
}
```

如果你忘了在第 5 章记录 r 和 pad——这就是全书一直警告的那件事：现在你没法定回原图坐标，所有框都画歪。预处理和后处理是**记账与销账**的关系。

## 10.6 NMS：非极大值抑制

### 为什么需要

过滤后的候选里，同一只猫可能站着 50 个高度重叠的框——网格上相邻的格子都觉得"我这里有猫"。全画出来就是灾难。**NMS 的思想：每堆重叠框里，只留最自信（分数最高）的那个，把重叠的兄弟都删掉。**

类比：一群人同时举手喊"这里有猫"。NMS 就是主持人：分数最高的先发言，凡和他位置重叠的（说的是同一只猫）都坐下；然后剩下的接着来。

### IoU：交并比——"重叠程度"的度量

$$IoU = \frac{\text{两个框的交集面积}}{\text{两个框的并集面积}}$$

手算一遍：

```text
框A: (0,0)-(4,4)，面积 16      框B: (2,2)-(6,6)，面积 16
交集: (2,2)-(4,4)，面积 4
并集 = 16 + 16 - 4 = 28
IoU = 4/28 ≈ 0.143
```

IoU=0 表示完全不沾边，IoU=1 表示完全重合。NMS 用 IoU 判断"是不是同一只猫"：IoU 超过阈值（常用 0.45）就认为是重复。

```rust
fn iou(a: &(f32,f32,f32,f32), b: &(f32,f32,f32,f32)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = a.2.min(b.2);
    let y2 = a.3.min(b.3);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);   // 无交集时为 0
    let area_a = (a.2 - a.0) * (a.3 - a.1);
    let area_b = (b.2 - b.0) * (b.3 - b.1);
    inter / (area_a + area_b - inter)
}
```

### NMS 完整实现

```rust
/// 按分数降序，贪心保留；同类别的框 IoU 超阈值则抑制
pub fn nms(
    mut dets: Vec<(f32,f32,f32,f32, f32, usize)>,   // (x1,y1,x2,y2,score,class)
    iou_thresh: f32,
) -> Vec<(f32,f32,f32,f32, f32, usize)> {
    // ① 按分数从高到低排序
    dets.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    let mut keep: Vec<(f32,f32,f32,f32, f32, usize)> = Vec::new();
    while let Some(best) = dets.first().cloned() {
        keep.push(best);
        // ② 删除与 best 同类且 IoU 超阈值的其余框
        dets.retain(|d| {
            d.5 != best.5                                    // 不同类，不抑制
                || iou(&(d.0,d.1,d.2,d.3), &(best.0,best.1,best.2,best.3)) < iou_thresh
        });
        dets.remove(0);   // 删掉已处理的 best（retain 后它在首位）
    }
    keep
}
```

五步对应五个注释：排序 → 取当前最高 → 删它的重叠同类兄弟 → 重复 → 收工。

两个工程变体，知道名字即可：

- **类别感知 NMS（class-aware）**：就是上面实现的——只有同类才互相抑制（猫和狗挨着坐不该互相压制）。有些教程实现的是类别无关版，精度会差一点。
- **Soft NMS**：不直接删除，而是给重叠框的分数打折扣。精度略升，速度略降，竞赛用得多。

## 10.7 分割后处理简述

YOLOv8-seg 的输出比检测多一个：

```text
output0: [1, 116, 8400]      ← 84 个检测值 + 32 个掩码系数
output1: [1, 32, 160, 160]   ← 32 张原型掩码（低分辨率）
```

流程（有直觉即可）：

1. 前两步和检测完全一样：过滤 + NMS 得到最终框
2. 每个幸存框带着 32 个系数，和 32 张原型掩码做**加权求和**（外积 + reshape）→ 得到一张 160×160 的原始掩码
3. sigmoid + 阈值（0.5）→ 二值掩码
4. 掩码放大到原图，用框裁掉框外部分 → 该物体的精确轮廓

分割的后处理代码量是检测的两三倍，但每一步都能对应上本章的概念（过滤、坐标还原、阈值），不至于陌生。

## 10.8 实战：YOLOv8n 完整后处理函数

把 10.4~10.6 封装起来——**这个函数会原样进入第 11 章**：

```rust
use ndarray::ArrayView3;

pub struct Bbox { pub x1: f32, pub y1: f32, pub x2: f32, pub y2: f32,
                  pub score: f32, pub class_id: usize }

pub fn postprocess(
    output: &ArrayView3<f32>,        // tract 输出 [1, 84, 8400]
    r: f32, pad: (f32, f32),
    img_w: u32, img_h: u32,
    conf_thresh: f32, iou_thresh: f32,
) -> Vec<Bbox> {
    // 工序一：置信度过滤
    let dets = filter_candidates(output, conf_thresh);
    // 工序二：坐标还原 + xyxy 化 + 裁剪到图内
    let restored: Vec<_> = dets.iter()
        .map(|d| restore_and_convert(d, r, pad, img_w, img_h))
        .collect();
    // 工序三：NMS
    nms(restored, iou_thresh)
        .into_iter()
        .map(|(x1,y1,x2,y2,score,class_id)| Bbox { x1,y1,x2,y2,score,class_id })
        .collect()
}
```

建议给它写个单元测试：手工构造两个重叠框（同类、IoU=0.8，分数 0.9/0.8）+ 一个不重叠框，验证 NMS 后只剩两个框：

```rust
#[test]
fn test_nms() {
    let dets = vec![
        (0.,0.,10.,10., 0.9, 0),
        (1.,1.,11.,11., 0.8, 0),   // 与上面 IoU≈0.68，应被抑制
        (50.,50.,60.,60., 0.7, 0), // 远，应保留
    ];
    let keep = nms(dets, 0.45);
    assert_eq!(keep.len(), 2);
    assert!(keep.iter().all(|d| d.4 != 0.8));   // 0.8 的被删了
}
```

## 10.9 常见坑清单

| 症状 | 病因 | 对策 |
| --- | --- | --- |
| 分数都是负数或巨大值 | 忘了 sigmoid | v8 的输出是 logits，必须过 sigmoid |
| 框位置整体偏移 | 忘了减 pad / 除 r | 检查 letterbox 逆变换 |
| 框尺寸和原图对不上 | 还原的是 xywh 却当 xyxy 画 | 统一坐标格式（建议全程 xyxy） |
| 一堆重叠框 | 没做 NMS 或 IoU 阈值太高 | 阈值试 0.45~0.65 |
| 漏检多 | 置信度阈值太高 | 降到 0.25 甚至 0.1 观察 |
| 类别名对不上 | coco.names 编号从 1 开始而代码从 0 | 检查类别表是否带 background 行 |

## 10.10 小结

- `[1, 84, 8400]`：84 = 4 位置 + 80 类（无 objectness，v8 与 v5 的分水岭），8400 = 三种网格密度之和。
- 检测后处理三道工序：sigmoid + 阈值过滤 → letterbox 逆变换还原坐标 → NMS 去重。
- NMS 的数学工具是 IoU（交并比），流程是"排序→留最高→删重叠同类→循环"。
- 预处理的 r/pad 和后处理的逆变换是记账关系，忘了记账画框必歪。

## 10.11 练习

1. 手算 IoU：框 A (0,0)-(2,2)，框 B (1,1)-(3,3)。再算框 C (5,5)-(6,6) 与 A 的 IoU。
2. 给 `filter_candidates` 加统计：打印 8400 个候选里各类别各剩几个，观察一张图的"候选分布"。
3. 把 NMS 改成类别无关版（去掉 `d.5 != best.5` 条件），找一张猫狗同框的图，对比两种版本的检测结果差异。
4. 挑战：实现 Soft NMS（把删除改成分数 ×(1-IoU) 折扣再重新排序），对比硬 NMS 的召回差异。
