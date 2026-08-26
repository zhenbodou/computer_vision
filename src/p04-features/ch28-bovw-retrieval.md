# 第 28 章 词袋模型与图像检索基础（BoVW）

> **本章导读**
> - 学完你能：搭一个最小的**以图搜图**（图像检索）流程——给一张查询图，从图库里找出最像的那几张。
> - 掌握 **BoVW（Bag of Visual Words，视觉词袋）** 范式：把海量局部描述子用聚类归成"视觉单词"，每张图表示成"单词出现频次直方图"。
> - **手写一个精简版 K-means**（约 20 行），亲手建一个小词典；顺带认识给单词加权的 **TF-IDF**。
> - 诚实地知道它的历史定位：这是深度学习之前图像检索的主流范式，后面用**深度特征**替换手工特征会强得多。

## 场景：图库里几万张图，怎么"以图搜图"

[第 24 章](ch24-descriptors.md) 我们学会了从一张图里提出几百个局部描述子（ORB/BRIEF）。两张图要比"像不像"，可以把它们的描述子两两匹配——可这只适合**一对一**比较。

现在换个规模：图库里有**几万张**图，用户丢进来一张查询图，要求秒回"最像的 10 张"。如果对每一张库图都做一次描述子匹配，慢得没法用。我们需要把每张图**压缩成一个定长向量**（像上一章 HOG/LBP 那样），这样检索就变成了"在向量空间里找最近邻"，又快又能建索引。

问题是：一张图有几百个局部描述子，数量还不固定，怎么变成**一个**定长向量？答案就是本章的主角——**词袋模型**。

## 从"文本词袋"到"视觉词袋"

先看它的灵感来源——**文本检索**里的词袋（Bag of Words）。要判断两篇文章讲的是不是一个主题，一个糙但有效的办法是：**只统计每个词出现了多少次，完全不管词序**。"机器"出现 20 次、"视觉"15 次、"美食"0 次——这张"词频表"就是文章的指纹。两篇文章的词频表接近，主题多半接近。

**视觉词袋（BoVW）把这套原封不动搬到图像上**，只需回答一个问题：图像的"单词"是什么？图像里没有现成的词，于是我们**自己造**：

- 把图库里**所有**图的局部描述子（几十万个）倒进一个大池子；
- 用**聚类**把它们归成 \\( K \\) 堆，每一堆的中心就是一个**视觉单词（visual word）**，这 \\( K \\) 个单词合起来叫**视觉词典（codebook）**。一个单词代表一类长得像的局部图案（比如"一个横着的边角""一小片斜纹"）；
- 于是任意一个描述子，都能被"量化"成离它最近的那个单词。一张图的几百个描述子，就变成了几百次"投票"，统计成一个 \\( K \\) 维的**词频直方图**——这就是这张图的定长指纹。

整个流程串起来是这样：

```text
海量局部描述子          K-means 聚类           视觉词典 (K 个单词)
 (每图几百个) ───────▶  (归成 K 堆) ────────▶  w0 w1 w2 ... w(K-1)
                                                    │
 某张图的描述子 ── 每个描述子投给"最近的单词" ───────┘
                            │
                            ▼
                  词频直方图 (K 维) = 这张图的指纹
                            │
             查询图算同样的直方图，比直方图距离找最像的库图
```

### K-means：把描述子归堆的聚类算法

上面"归成 K 堆"用的是最经典的聚类算法 **K-means（K 均值）**。它的循环极简单，就两步反复迭代：

- **分配**：把每个点归给离它最近（欧氏距离 \\( \lVert x - c \rVert^2 \\)）的那个中心；
- **更新**：把每个中心挪到它那一堆点的**平均位置**。

反复几十轮就收敛。这一章我们**手写**一个二十来行的精简版，先建立肌肉记忆；到 [第 33 章](../p06-ml/ch33-linfa.md) 会用 `linfa` 这个纯 Rust 机器学习库正式地讲聚类（含更好的初始化、评估等）。

### TF-IDF：给单词加权（简介）

直接用词频还有个毛病：有些视觉单词（比如对应"平坦区域"的那种）**几乎每张图都有一大堆**，它们对区分图像毫无帮助，却在直方图里占了大头。文本检索里用 **TF-IDF** 解决：**越是"哪儿都有"的词，权重越低**。

\\[ \text{idf}(w) = \ln\frac{N}{n_w}, \qquad \text{weight}(w) = \text{tf}(w)\cdot \text{idf}(w) \\]

其中 \\( N \\) 是图库总图数、\\( n_w \\) 是含单词 \\( w \\) 的图数。某单词出现的图越多（\\( n_w \\) 越大），idf 越小，权重被压得越低。本章代码用最朴素的归一化词频演示主干，把加 TF-IDF 留作练习。

### 为什么 BoVW 能撑起百万级图库

把图变成词频直方图，最大的红利是**能像搜索引擎搜网页一样搜图**。文本搜索靠**倒排索引（inverted index）**：为每个词建一张"含这个词的文档清单"，查询时只需把清单取交/并，无需扫全库。BoVW 原样照搬——为每个视觉单词建一张"含这个单词的图片清单"。查询图先量化出它包含哪些单词，然后**只去比对那些和它共享单词的库图**，绝大多数毫不相干的图根本不参与计算。正因如此，BoVW 才能在百万级图库上做到毫秒级检索，这也是它当年能落地成产品的关键。

## 动手写代码：搭一个最小以图搜图

流程：合成 3 类图案（方块 / 菱形 / 圆点，每类 2 张入库）→ 每张图用 FAST 角点 + 梯度方向直方图提局部描述子 → 手写 K-means 建词典 → 每图转词频向量 → 给查询图找最相似的库图。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
imageproc = "0.27"
```

完整代码（`examples/ch28_bovw_retrieval.rs`）：

```rust
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_filled_circle_mut;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};

/// 极简确定性随机数（线性同余），避免引入 rand 依赖，保证结果可复现
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u32() % (hi - lo) as u32) as i32
    }
}

fn canvas() -> GrayImage { ImageBuffer::from_pixel(128, 128, Luma([230u8])) }

/// 类别一：散布的小圆点（边缘朝向各异 → 梯度方向弥散）
fn dots(seed: u64) -> GrayImage {
    let mut img = canvas();
    let mut r = Lcg::new(seed);
    for _ in 0..40 {
        draw_filled_circle_mut(&mut img, (r.range(8, 120), r.range(8, 120)), 5, Luma([30]));
    }
    img
}

/// 类别二：轴对齐小方块（边缘只有水平/垂直 → 梯度集中在 0°/90°）
fn squares(seed: u64) -> GrayImage {
    let mut img = canvas();
    let mut r = Lcg::new(seed);
    for _ in 0..40 {
        let (x, y) = (r.range(6, 112), r.range(6, 112));
        for dy in 0..10 {
            for dx in 0..10 {
                img.put_pixel((x + dx) as u32, (y + dy) as u32, Luma([30]));
            }
        }
    }
    img
}

/// 类别三：45° 菱形（边缘为对角线 → 梯度集中在 45°/135°）
fn diamonds(seed: u64) -> GrayImage {
    let mut img = canvas();
    let mut r = Lcg::new(seed);
    for _ in 0..40 {
        let (cx, cy) = (r.range(10, 118), r.range(10, 118));
        let rad = 7i32;
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                if dx.abs() + dy.abs() <= rad {
                    img.put_pixel((cx + dx) as u32, (cy + dy) as u32, Luma([30]));
                }
            }
        }
    }
    img
}

/// 局部描述子：每个 FAST 角点周围 9x9 邻域，统计 8 方向（无符号）梯度直方图，再 L2 归一化
fn descriptors(img: &GrayImage) -> Vec<Vec<f32>> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let gx = horizontal_sobel(img);
    let gy = vertical_sobel(img);
    let radius = 4i32;
    let bin_w = std::f32::consts::PI / 8.0; // 无符号方向 0..π 均分 8 份
    let mut out = Vec::new();
    for c in corners_fast9(img, 40) {
        let (cx, cy) = (c.x as i32, c.y as i32);
        if cx - radius < 0 || cy - radius < 0 || cx + radius >= w || cy + radius >= h {
            continue; // 邻域越界的关键点跳过
        }
        let mut hist = [0f32; 8];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let (px, py) = ((cx + dx) as u32, (cy + dy) as u32);
                let ex = gx.get_pixel(px, py)[0] as f32;
                let ey = gy.get_pixel(px, py)[0] as f32;
                let mag = (ex * ex + ey * ey).sqrt();
                if mag < 1.0 { continue; }
                let mut ang = ey.atan2(ex); // -π..π
                if ang < 0.0 { ang += std::f32::consts::PI; } // 折叠到 0..π
                let bin = ((ang / bin_w) as usize).min(7);
                hist[bin] += mag; // 用梯度强度加权投票
            }
        }
        let norm = hist.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            out.push(hist.iter().map(|v| v / norm).collect());
        }
    }
    out
}

/// 欧氏距离下，找 x 最近的中心（视觉单词）编号
fn nearest(x: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d: f32 = x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best_d { best_d = d; best = i; }
    }
    best
}

/// 手写精简版 K-means：把描述子聚成 k 个"视觉单词"，返回 k 个中心
fn kmeans(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let dim = data[0].len();
    let mut rng = Lcg::new(seed);
    // 初始化：随机挑 k 个描述子当初始中心（K-means 对初始化敏感，实践中要多跑几次）
    let mut centroids: Vec<Vec<f32>> =
        (0..k).map(|_| data[rng.next_u32() as usize % data.len()].clone()).collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0u32; k];
        for x in data {
            let c = nearest(x, &centroids); // 分配
            counts[c] += 1;
            for d in 0..dim { sums[c][d] += x[d]; }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim { centroids[j][d] = sums[j][d] / counts[j] as f32; } // 更新=均值
            }
        }
    }
    centroids
}

/// 把一张图的描述子量化成"词频直方图"（BoVW 向量），并 L1 归一化
fn bovw_vector(descs: &[Vec<f32>], dict: &[Vec<f32>]) -> Vec<f32> {
    let mut hist = vec![0f32; dict.len()];
    for d in descs { hist[nearest(d, dict)] += 1.0; }
    let s = hist.iter().sum::<f32>().max(1.0);
    for v in &mut hist { *v /= s; }
    hist
}

/// 两个词频直方图的 L1 距离（越小越相似）
fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1) 建立图库：3 个类别，每类 2 张（不同随机排布）
    let db: Vec<(&str, GrayImage)> = vec![
        ("方块#1", squares(1)), ("方块#2", squares(2)),
        ("菱形#1", diamonds(1)), ("菱形#2", diamonds(2)),
        ("圆点#1", dots(1)), ("圆点#2", dots(2)),
    ];

    // 2) 提取每张库图的描述子，并全部汇入"训练池"用于建词典
    let db_descs: Vec<Vec<Vec<f32>>> = db.iter().map(|(_, im)| descriptors(im)).collect();
    let pool: Vec<Vec<f32>> = db_descs.iter().flatten().cloned().collect();
    println!("库图数 = {}, 汇总描述子总数 = {}", db.len(), pool.len());

    // 3) 手写 K-means 建立视觉词典
    let k = 12;
    let dict = kmeans(&pool, k, 20, 42);
    println!("视觉词典大小 K = {}\n", k);

    // 4) 每张库图 → 词频向量
    let db_vecs: Vec<Vec<f32>> = db_descs.iter().map(|d| bovw_vector(d, &dict)).collect();

    // 5) 查询：一张全新的菱形图，找库里最像的
    let query = diamonds(99);
    let q_vec = bovw_vector(&descriptors(&query), &dict);
    println!("查询图 = 菱形#99（此前从未见过），按相似度排序：");
    let mut ranked: Vec<(usize, f32)> =
        db_vecs.iter().enumerate().map(|(i, v)| (i, l1(&q_vec, v))).collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (rank, (i, dist)) in ranked.iter().enumerate() {
        println!("  第{}名  {}  L1距离={:.3}", rank + 1, db[*i].0, dist);
    }

    // 再换一张查询图验证
    let q2 = bovw_vector(&descriptors(&squares(88)), &dict);
    let best = db_vecs.iter().enumerate()
        .min_by(|a, b| l1(&q2, a.1).partial_cmp(&l1(&q2, b.1)).unwrap()).unwrap().0;
    println!("\n查询图 = 方块#88，最相似库图 = {}", db[best].0);
    Ok(())
}
```

运行 `cargo run --example ch28_bovw_retrieval`，输出（可复现）：

```text
库图数 = 6, 汇总描述子总数 = 3585
视觉词典大小 K = 12

查询图 = 菱形#99（此前从未见过），按相似度排序：
  第1名  菱形#2  L1距离=0.181
  第2名  菱形#1  L1距离=0.221
  第3名  圆点#1  L1距离=1.866
  第4名  圆点#2  L1距离=1.944
  第5名  方块#2  L1距离=1.990
  第6名  方块#1  L1距离=1.997

查询图 = 方块#88，最相似库图 = 方块#1
```

结果很干净：一张**从没见过**的菱形查询图，排在最前的两张恰恰是库里的两张菱形（L1 距离 `0.18`、`0.22`），而到其他类别（圆点、方块）距离陡增到 `1.8` 以上——**同类挨得近、异类离得远**，检索成立。换成方块查询图，最相似的也确实是方块。这背后没有任何"标签"参与，纯靠视觉单词的分布把三类图案自动区分开了。

## 常见坑与提示

> ⚠️ **注意**：**词典大小 K 要选得当。** K 太小，不同图案被塞进同一个单词，"分辨率"不够（欠拟合）；K 太大，同一种图案被拆成好几个单词，直方图变稀疏、还容易过拟合噪声。真实图库的 K 常取几百到几万，需要在验证集上调。

> 🕳️ **坑**：**K-means 对初始化很敏感，一次不一定靠谱。** 随机挑的初始中心不好，可能收敛到很差的局部解（甚至出现没有任何点的"空簇"）。工程上要**多跑几次取最好**（用簇内平方和衡量），或用 k-means++ 这类更聪明的初始化。本章为可复现固定了随机种子，实际项目别只跑一次就信。

> 🕳️ **坑**：**量化误差与"同义词/歧义词"。** 把一个连续描述子硬塞给最近的单词，边界附近的描述子会被"四舍五入"得很粗暴（量化误差）；而一个视觉单词可能同时对应几种不同图案（歧义），几个单词也可能其实是"同义词"。软分配（soft assignment）、VLAD、Fisher Vector 等是缓解手段。

> 💡 **提示**：**高维空间里"距离"会变钝。** 描述子维度一高，最近和最远的距离差距会缩小（维度灾难）。这也是为什么 BoVW 直方图之间常用 L1、卡方（chi-square）距离而非普通欧氏距离，检索时还要配 TF-IDF 加权和 L2 归一化。

## 本章小结

- **以图搜图**的关键，是把每张图压成一个**定长向量**，让检索变成向量最近邻。
- **BoVW** 借用文本词袋思想：局部描述子 → K-means 聚成**视觉词典** → 每图表示成 \\( K \\) 维**词频直方图** → 比直方图距离即可检索。
- 我们**手写了精简 K-means**（分配 + 更新两步迭代），并在合成数据上验证了"同类近、异类远"；正式的聚类工具见 [第 33 章 linfa](../p06-ml/ch33-linfa.md)。
- **TF-IDF** 压低"哪儿都有"的单词权重，是提升检索质量的常用一招。
- 要点提醒：词典大小 K、K-means 初始化敏感、量化误差与高维距离，都是落地时要拿捏的地方。

## 动手练习

1. **调 K**：把 `k` 依次设成 3、12、50，观察菱形查询图的排名和距离怎么变，体会"太小/太大"的两难。
2. **换距离**：把 `l1` 换成卡方距离 \\( \sum (a_i-b_i)^2/(a_i+b_i+\varepsilon) \\)，比较检索结果是否更稳。
3. **加 TF-IDF**：统计每个单词出现在多少张库图里，算出 idf 权重，给词频向量逐维加权后再检索，看排名有没有变化。
4. **进阶**：把 K-means 跑 5 次不同种子，用"簇内平方和"挑最好的那次做词典——这就是对抗初始化敏感的标准做法。

## 诚实的历史定位与展望

BoVW 是 **2003–2012 年前后图像检索和分类的绝对主流**（配合 SIFT + SVM 拿过很多比赛冠军）。但它有天花板：视觉单词是"硬"聚出来的、描述子是手工设计的，语义表达能力有限——它能认出"图案纹理像不像"，却很难懂"这是猫还是狗"。

深度学习到来后，人们改用 **CNN 提取的深度特征**（甚至端到端学一个检索嵌入向量）替换手工描述子，检索质量和语义理解都大幅超越 BoVW。这条线我们会在 [第八部分（分类）](../p08-classification/ch42-candle-classify.md) 和 [第 73 章 CLIP 图文检索](../p15-generative/ch73-clip-vlm.md) 展开；到 [第 100 章](../p20-projects/ch100-image-search.md) 会用深度特征 + 向量检索做一个真正能用的以图搜图引擎。但 BoVW 的"把局部特征聚合成全局向量"这个思想，至今仍活在 VLAD、NetVLAD 等现代方法里，值得你先在这一章把它吃透。

至此，第四部分（特征工程）就全部走完了。你已经掌握了从角点、局部描述子、匹配对齐，到全局手工特征、图像检索的整条经典链路。下一部分，我们补齐**数值计算的基石**（ndarray 与 nalgebra），为进入机器学习和深度学习做好准备。👉 [第 29 章 ndarray：多维数组与张量运算](../p05-numeric/ch29-ndarray.md)

## 第四部分阶段验收：鲁棒几何与检索评测

构造含旋转、尺度、模糊、重复纹理、动态物体和无重叠图像的配对集。报告关键点重复率、
匹配 precision/recall、RANSAC 内点率与单应重投影误差，并显式检测共线、纯重复纹理等退化。
检索部分使用独立 query/gallery，报告 Recall@K、mAP、查询 P95 和索引内存；与颜色直方图、
BoVW 和深度 embedding 做配对比较。只展示一对成功匹配不构成验收证据。
