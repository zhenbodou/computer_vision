# 第 100 章 项目：以图搜图引擎（特征 + 向量检索）

> **本章导读**
> - 学完你能：用 **candle**（0.11.0）从零搭一个**以图搜图（CBIR，Content-Based Image Retrieval，基于内容的图像检索）** 引擎——给一张查询图，从图库里把「长得最像的」前 K 张捞出来。核心三步：**图库图 → 编码成向量 → L2 归一化建库**；**查询图 → 同一编码器 → 与全库暴力算余弦相似度 → 排序取 top-K**。
> - 你会看到一组**最有说服力的真数字**（本机 `cargo run --release` 实测、确定性可复现）：同一批图库（72 张查询 × 270 张图库），**32 维的学习嵌入** precision@5 = **98.6%**、top-1 = **98.6%**；而**1728 维的原始像素** precision@5 只有 **45.3%**、top-1 = **84.7%**。一个小 54 倍的向量，检索却准了一倍多——**「学到的特征」碾压「原始像素」**。
> - 一条**诚实边界**：本章用**暴力余弦**（拿查询和图库里每一个向量都比一遍）把道理讲透；真实百万级图库要换 **ANN 近似最近邻**索引（HNSW / IVF，纯 Rust 有 `hnsw_rs`、`instant-distance`），且检索嵌入通常在业务数据上专门微调。
> - 这是[第 73 章](../p15-generative/ch73-clip-vlm.md) CLIP 图像塔、[第 69 章](../p15-generative/ch69-autoencoder.md) 潜在表示、[第 57 章](../p12-face/ch57-face-recognition.md) ArcFace「归一化 + 余弦」三条线的**汇合与落地**——把「嵌入 + 相似度」变成一个能跑的检索系统。

## 场景：给一张图，帮我把「像的」都找出来

你手里有个几万张图片的图库：可能是电商的商品图、相册、监控截图。用户丢来一张图说：「帮我找长得像这张的。」——注意，他没给关键词，就给了**一张图**。这就是**以图搜图**。淘宝的「拍立淘」、Google 的「以图搜图」、相册的「找相似照片」，本质都是同一件事：

> 把「图片」变成一个能**比远近**的**向量**，然后在库里找**离查询图最近**的那些。

难点不在「怎么排序」——排序是小学生都会的活。难点在**怎么把图变成向量**：这个向量要满足「**内容相似 → 向量相近**」。最朴素的想法是「直接把像素拉平当向量」，我们这一章就要**亲手证明这个朴素想法为什么不行**，以及**学出来的特征凭什么行**。

## 原理：把图变「指纹」，比指纹而不是比像素

### 直觉：检索 = 建库 + 比距离

以图搜图分两个阶段，一定要分清：

```text
【建库（离线，做一次）】
  图库每张图 ─▶ 图像编码器 ─▶ 32维向量 ─L2─▶ 单位向量 ┐
                                                      ├─▶ 向量索引（270 个向量）
【检索（在线，每次查询都做）】                          │
  查询图 ─▶ 同一个编码器 ─▶ 32维 ─L2─▶ q̂ ──余弦──▶ 和索引里每个向量比 ─▶ 排序取 top-K
```

「编码器」就是把图变成向量的那个「翻译官」，向量叫**嵌入（embedding）**。只要两张图内容像、它们的嵌入就该靠得近，检索就成了「在向量空间里找最近邻」。

### 关键：为什么不能直接拿原始像素当向量？

一张 24×24 的 RGB 图，拉平就是 \\( 3\times24\times24 = 1728 \\) 维向量，维度比我们的 32 维嵌入高 54 倍，信息「更全」——那直接拿它比不就行了？

**不行，而且错得离谱。** 原始像素向量的相似度，衡量的是「**两张图有多少像素在同一位置、同一颜色地重叠**」。它对**位置**极其敏感：同一个红色方块，挪到画面另一角，像素向量就几乎完全变了；反过来，两个**不同类**的物体只要**恰好压在同一片位置**，像素相似度反而很高。原始像素只会按「**谁和我在同一位置糊了一片**」乱配，根本读不懂「这是个红方块」这种**语义**。

**学出来的嵌入**不一样。我们让编码器去做一个「代理任务」——**给图分类**——并用**位置随机**的数据训练它。为了在任意位置都答对类别，网络被逼着学会**对平移不敏感**的、抓「颜色 + 形状」本质的特征。训练完把分类头一扔，编码器倒数第二层那个 32 维输出，就是一个「读得懂内容」的检索嵌入。这正是[第 69 章](../p15-generative/ch69-autoencoder.md)线性探针的老结论（8 维潜在码 85% > 256 维原始像素 75.8%）——**维度不等于信息质量，语义才是**。

> 📖 **术语**：**CBIR（基于内容的图像检索）** 指「用图的**视觉内容**（而非文件名/标签）来检索」。**嵌入（embedding）**：把图映射成的低维向量。**最近邻检索**：在向量空间里找离查询最近的若干个。

### 相似度与评价指标

两个向量的「像不像」，用**余弦相似度**——只看方向、不看长短。先各自做 **L2 归一化**（除以自己的长度，落到单位球面），归一化后**点积就等于余弦**（这套路你在[第 57 章](../p12-face/ch57-face-recognition.md) ArcFace、[第 73 章](../p15-generative/ch73-clip-vlm.md) CLIP 里都见过）：

\\[ \hat{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{\sum_i x_i^2 + \varepsilon}}, \qquad \cos(\mathbf{q}, \mathbf{g}) = \frac{\mathbf{q}\cdot\mathbf{g}}{\lVert\mathbf{q}\rVert\lVert\mathbf{g}\rVert} = \hat{\mathbf{q}}\cdot\hat{\mathbf{g}} \\]

（eps 要加在 **sqrt 之前**防零向量除零出 NaN。）检索质量用两个指标：**top-1 命中率**（排第 1 的结果类别对不对）和 **precision@K**（前 K 个里同类的比例，本章 K=5）：

\\[ \text{precision@}K = \frac{1}{Q}\sum_{q=1}^{Q}\frac{r_q}{K}, \qquad \text{top-1} = \frac{1}{Q}\sum_{q=1}^{Q}\mathbf{1}[\text{第 1 名与查询 } q \text{ 同类}] \\]

其中 \\( r_q \\) 是查询 \\( q \\) 的 top-K 里与它同类的张数，\\( Q \\) 是查询总数。

## 动手写代码：搭一个能跑的以图搜图引擎

工程独立于主工程，`Cargo.toml`（含空 `[workspace]`，不被上层收编）：

```toml
[package]
name = "ch100_image_search"
version = "0.1.0"
edition = "2021"

[workspace]           # 空 workspace：本工程独立构建

[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
image = "0.25"
```

### 第一步：合成一个「类内多样、类间可分」的图库

不下载任何数据，用代码造 9 类（3 颜色 × 3 形状）的小图。**成败全在这里**：要让检索有得可比，必须让「位置」成为主要干扰项，还要**堵掉原始像素的两个后门**——

1. **颜色不走正交通道**：三种颜色都点亮全部三通道（主色 1.0、陪衬 0.35），这样红/绿/蓝在像素空间里**不正交**，原始像素余弦没法「免费」区分颜色。
2. **物体位置随机、紧凑**：竖条 3×11、横条 11×3、方块 7×7，在 24×24 画面里随处落——同一类的图散在各处，像素重叠很低。

随机数用固定种子的 LCG，**离散选择务必取高位**（`below(n)` 走 `unit()`，别用 `% n`，见后文的坑）：

```rust
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self { Self(s) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 { (self.next() >> 11) as f32 / (1u64 << 53) as f32 } // 取高位
    fn below(&mut self, n: usize) -> usize { (self.unit() as f64 * n as f64) as usize } // 别用 % n
}

const IMG: usize = 24;
const RAW: usize = 3 * IMG * IMG;                 // 原始像素维度 = 1728
const NCLASS: usize = 9;                           // 3 颜色 × 3 形状
// 三色都点亮三通道 → 像素空间里不正交，原始像素没法免费分颜色
const COLORS: [[f32; 3]; 3] = [[1.0, 0.35, 0.35], [0.35, 1.0, 0.35], [0.35, 0.35, 1.0]];

fn draw(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let (color, shape) = (class / 3, class % 3);
    let col = COLORS[color];
    let bright = 0.85 + rng.unit() * 0.15;         // 轻微亮度抖动（外观扰动）
    let mut g = vec![0f32; RAW];
    let mut fill = |x0: usize, y0: usize, w: usize, h: usize| {
        for y in y0..y0 + h { for x in x0..x0 + w {
            for c in 0..3 { g[c * IMG * IMG + y * IMG + x] = col[c] * bright; }
        }}
    };
    match shape {                                   // 位置随机 → 类内多样、平移干扰
        0 => fill(rng.below(IMG - 3), rng.below(IMG - 11), 3, 11),  // 竖条
        1 => fill(rng.below(IMG - 11), rng.below(IMG - 3), 11, 3),  // 横条
        _ => fill(rng.below(IMG - 7), rng.below(IMG - 7), 7, 7),    // 方块
    }
    g
}

fn make_set(per_class: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let (mut imgs, mut labels) = (Vec::new(), Vec::new());
    for c in 0..NCLASS { for _ in 0..per_class { imgs.extend(draw(c, &mut rng)); labels.push(c); } }
    (imgs, labels)
}
```

我们造**三份实例互不相同**的数据（不同种子、不同随机位置）：训练集 360 张（训编码器）、图库 270 张（被搜）、查询集 72 张（去搜）。训练集和图库/查询**不共享任何一张图**——这样测出来的检索效果才是真「泛化」，也顺带证明编码器学到了**位置无关**的特征。

### 第二步：图像编码器（复用第 73 章的图像塔）

编码器就是[第 73 章](../p15-generative/ch73-clip-vlm.md) CLIP 那个图像塔：两层卷积 + 池化 + 一个线性投影，吐 32 维。24×24 经两次 `avg_pool2d(2)` 变 6×6：

```rust
const D: usize = 32;                    // 嵌入维度
const POOLED: usize = IMG / 4;          // 两次池化：24 → 12 → 6

struct ImageEncoder { c1: Conv2d, c2: Conv2d, fc: Linear }
impl ImageEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: conv2d(3, 8, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(8, 16, 3, cfg, vb.pp("c2"))?,
            fc: linear(16 * POOLED * POOLED, D, vb.pp("fc"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?.avg_pool2d(2)?;
        let h = self.c2.forward(&h)?.relu()?.avg_pool2d(2)?;
        self.fc.forward(&h.flatten_from(1)?)         // [N, 32] —— 这就是检索嵌入
    }
}

// L2 归一化（eps 加在 sqrt 之前防除零）：归一化后点积 = 余弦相似度
fn l2n(x: &Tensor) -> Result<Tensor> {
    let denom = x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, 1e-12)?.sqrt()?;
    x.broadcast_div(&denom)
}
fn embed(enc: &ImageEncoder, imgs: &[f32], n: usize, dev: &Device) -> Result<Tensor> {
    let x = Tensor::from_vec(imgs.to_vec(), (n, 3, IMG, IMG), dev)?;
    l2n(&enc.forward(&x)?)
}
```

### 第三步：训练（代理任务=分类）+ 检索评测

关键思路：**用分类当代理任务训练编码器**——编码器接一个线性分类头，用交叉熵在 9 类上训到收敛；训完**把分类头丢掉**，只留编码器的 32 维输出当检索嵌入。检索时对图库和查询各编码一遍、L2 归一化，然后 `query @ gallery^T` 一把矩阵乘就是**全部余弦相似度**，纯 Rust 排序取 top-K：

```rust
// 检索评测：sims[q][g] = 查询 q 对图库 g 的余弦；算 precision@K 与 top-1
fn eval_retrieval(sims: &[Vec<f32>], qlab: &[usize], glab: &[usize], k: usize) -> (f32, f32) {
    let (mut prec, mut top1) = (0.0f32, 0usize);
    for (qi, row) in sims.iter().enumerate() {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap()); // 相似度降序
        let hit = idx[..k].iter().filter(|&&gi| glab[gi] == qlab[qi]).count();
        prec += hit as f32 / k as f32;
        if glab[idx[0]] == qlab[qi] { top1 += 1; }
    }
    (prec / sims.len() as f32, top1 as f32 / sims.len() as f32)
}
```

`main` 里把三件事串起来——造数据（并打印图库直方图自查无空类）、训编码器、然后**分别用「学习嵌入」和「原始像素」建库检索**做对照：

```rust
fn main() -> Result<()> {
    let dev = Device::Cpu;
    let (train_imgs, train_lab) = make_set(40, 1); // 训练 360
    let (gal_imgs, gal_lab) = make_set(30, 2);     // 图库 270
    let (qry_imgs, qry_lab) = make_set(8, 3);      // 查询 72
    let (ntr, ng, nq) = (train_lab.len(), gal_lab.len(), qry_lab.len());
    // …此处打印图库每类计数直方图，确认无空类（见 §18 自查）…

    // 训练编码器（分类代理任务，400 步 AdamW lr=3e-3）
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let enc = ImageEncoder::load(vb.pp("enc"))?;
    let head = linear(D, NCLASS, vb.pp("head"))?;  // 只训练时用，检索时丢弃
    det_init(&vm, 42)?;                            // 固定种子初始化（可复现）
    let x_tr = Tensor::from_vec(train_imgs.clone(), (ntr, 3, IMG, IMG), &dev)?;
    let y_tr = Tensor::from_vec(train_lab.iter().map(|&c| c as u32).collect::<Vec<_>>(), (ntr,), &dev)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
    for _ in 1..=400 {
        let loss = cross_entropy(&head.forward(&enc.forward(&x_tr)?)?, &y_tr)?;
        opt.backward_step(&loss)?;
    }

    // 建库 + 检索：① 学习嵌入   ② 原始像素（拉平后同样 L2 归一化）
    let g_learned = embed(&enc, &gal_imgs, ng, &dev)?;
    let q_learned = embed(&enc, &qry_imgs, nq, &dev)?;
    let g_raw = l2n(&Tensor::from_vec(gal_imgs.clone(), (ng, RAW), &dev)?)?;
    let q_raw = l2n(&Tensor::from_vec(qry_imgs.clone(), (nq, RAW), &dev)?)?;
    let learned_sims = q_learned.matmul(&g_learned.t()?)?.to_vec2::<f32>()?; // 暴力余弦
    let raw_sims = q_raw.matmul(&g_raw.t()?)?.to_vec2::<f32>()?;

    let k = 5;
    let (pl, tl) = eval_retrieval(&learned_sims, &qry_lab, &gal_lab, k);
    let (pr, tr) = eval_retrieval(&raw_sims, &qry_lab, &gal_lab, k);
    println!("学习嵌入 precision@{k}={:.1}% top1={:.1}%", 100.0*pl, 100.0*tl);
    println!("原始像素 precision@{k}={:.1}% top1={:.1}%", 100.0*pr, 100.0*tr);
    Ok(())
}
```

> 💡 **提示**：`det_init` 用固定种子的 LCG 逐个给参数赋值（`Var::set`），代替 `Device::set_seed`（candle CPU 会报 "cannot seed the CPU rng"）；把工程里的 LCG/初始化/图像塔与第 72、73 章共用即可。完整可运行代码（含 `det_init`、把 top-5 结果拼成对照图的 `save_montage`，约 40 行像素排版，本章从略）在仓库 `code/projects/ch100_image_search/src/main.rs`。

### 运行与真实输出

```bash
cd code/projects/ch100_image_search
cargo run --release
```

本机实测输出（确定性，多次运行完全一致）：

```text
图库类别直方图：红竖条=30 红横条=30 红方块=30 绿竖条=30 绿横条=30 绿方块=30 蓝竖条=30 蓝横条=30 蓝方块=30 （共 270 张，无空类）

==== 训练图像编码器（分类代理任务，400 步）====
  step   1  分类损失 = 2.1970  训练准确率 = 13.1%
  step 100  分类损失 = 0.0013  训练准确率 = 100.0%
  step 200  分类损失 = 0.0004  训练准确率 = 100.0%
  step 300  分类损失 = 0.0002  训练准确率 = 100.0%
  step 400  分类损失 = 0.0001  训练准确率 = 100.0%

==== 检索效果（72 张查询 × 270 张图库，暴力余弦，K=5）====
  学习嵌入（32 维）    precision@5 = 98.6%   top-1 命中 = 98.6%
  原始像素（1728 维）  precision@5 = 45.3%   top-1 命中 = 84.7%
  → 学习嵌入把'同色同形'的图拉近；原始像素只按'谁和我重叠在同一位置'乱配。

对照图查询：红竖条 绿横条 蓝方块 红方块
已存 out_ch100_retrieval.png（每行：查询图 | 学习嵌入 top-5 | 原始像素 top-5）
```

### 怎么读这组数字（本章最值钱的部分）

- **32 维学习嵌入 precision@5 = 98.6%，原始像素只有 45.3%。** 一个小 54 倍的向量，检索准了一倍多。这就是「**学到的特征 > 原始像素**」的铁证，和[第 69 章](../p15-generative/ch69-autoencoder.md)线性探针（潜在 85% > 像素 75.8%）一脉相承。
- **看那个耐人寻味的裂口：原始像素 top-1 有 84.7%，但 precision@5 掉到 45.3%。** 说明原始像素**偶尔能撞对第 1 名**（恰好有张同类图压在相近位置、重叠最多），但放宽到前 5，**一多半是错的**——它在按「位置重叠」乱配，同类图散在各处根本聚不拢。学习嵌入则从 top-1 到 precision@5 都稳在 98.6%，说明同类图在向量空间里**真的挤成了一簇**。
- **对照图 `out_ch100_retrieval.png`（每行：查询 | 学习 top-5 | 原始像素 top-5）看得一清二楚**：查「蓝方块」，左边学习嵌入返回的清一色是蓝方块；右边原始像素却混进了绿方块、绿竖条、蓝横条——**只要压在相近位置、颜色蹭得上，就被当成「像」**。唯独「红方块」那行原始像素也基本对，因为方块够大，位置重叠恰好和类别高度相关，这个「例外」反而印证了原始像素靠的是位置而非语义。

## 常见坑与提示

> 🕳️ **坑**：**自造随机数取离散值，务必用高位，别 `% n` / `& 1`。** LCG 的**最低位周期极短**，`next() % 9` 选类别、`next() % IMG` 选位置会**退化**——可能整整一类一张都不生成、位置也非随机。正解是先 `unit()`（内部 `state >> 11` 取高位转 `[0,1)`），再 `below(n) = (unit()*n as f64) as usize`。**造完先打印每类直方图自查**（本章确认 9 类各 30 张、无空类）再开训，比训练后才发现缺类省心得多。

> 🕳️ **坑**：**「维度高」不等于「信息好」，别被 1728 维唬住。** 原始像素维度是学习嵌入的 54 倍，检索却差一大截，因为它编码的是「像素在哪、多亮」这类**表层外观**，对平移极敏感、读不懂语义。检索/识别看的是**嵌入的质量**，不是长度。这也是为什么造对照数据时要**故意让位置成为主要干扰**——否则（比如物体都居中）原始像素也能刷到高分，对照就显不出来了。

> 💡 **提示**：**L2 归一化 + 余弦 = 全书通用的「比远近」套路。** [第 57 章](../p12-face/ch57-face-recognition.md) ArcFace 比人脸、[第 73 章](../p15-generative/ch73-clip-vlm.md) CLIP 比图文、本章比图库，都是「嵌入归一化到单位球面、点积当相似度」。归一化后 `query @ gallery^T` 一次矩阵乘就是**整张相似度矩阵**，无比顺手；记得 eps 加在 sqrt 之前防除零 NaN。

> ⚠️ **注意**：**暴力检索 O(N·D) 不扩展到百万级。** 本章 270 张全比一遍毫无压力，但图库上百万时，每次查询都和全库比就太慢了。真实系统用 **ANN（近似最近邻）** 索引——**HNSW**（分层小世界图）或 **IVF**（倒排 + 量化），纯 Rust 有 `hnsw_rs`、`instant-distance` 可直接用，牺牲一点点召回换几个数量级的加速。**先建对嵌入，再谈索引**：索引只负责「快」，「准」还得靠嵌入质量。

> 💡 **提示**：**检索嵌入从哪来，决定了检索有多准。** 本章用「分类代理任务」的倒数第二层，简单有效。要更强的检索嵌入，可上**度量学习**（[第 57 章](../p12-face/ch57-face-recognition.md) ArcFace/三元组损失，直接优化「同类近、异类远」）、**自监督**（无标签也能学）、或 **CLIP**（[第 73 章](../p15-generative/ch73-clip-vlm.md)，还能顺带支持「用文字搜图」）。工业界还会在**自己的业务图**上微调嵌入，因为「相似」的定义各行各业都不同。

## 本章小结

- **以图搜图（CBIR）= 建库 + 比距离**：图库图 → 编码器 → 嵌入 → L2 归一化建索引；查询图同样编码，与库里向量算**余弦相似度**、排序取 **top-K**。
- **核心结论**：**学到的特征碾压原始像素**。同一图库，32 维学习嵌入 precision@5 = **98.6%**，1728 维原始像素只有 **45.3%**——维度不代表信息质量，**语义**才是。原始像素 top-1 尚可（84.7%）但 precision@5 崩盘，暴露了它按「位置重叠」乱配的本质。
- **检索嵌入怎么来**：本章用「分类代理任务」的倒数第二层；数据要**类内多样（位置随机）、类间可分**，并堵掉原始像素的「颜色正交」后门，对照才显形。
- **两个通用套路**：L2 归一化 + 余弦（`query @ gallery^T` 一把出全矩阵）；自造 RNG 取离散值走高位、造完打直方图自查。
- **诚实边界**：暴力余弦 O(N·D) 只适合中小库；百万级换 **ANN**（HNSW/IVF，纯 Rust `hnsw_rs`/`instant-distance`）。想更准就换更强的嵌入（度量学习 / 自监督 / CLIP）并在业务数据上微调。

## 动手练习

1. **改 K 看曲线**：把 `k` 从 5 改成 1、3、10，分别打印学习嵌入与原始像素的 precision@K，画一条「K–精度」曲线，体会两者随 K 增大的不同走势（谁掉得快？）。
2. **验证「位置是关键干扰」**：把 `draw` 里的物体**固定画在画面正中**（位置不再随机），重跑，看原始像素的 precision@5 会不会被「送」上去——从而反推：正是「位置多样」让原始像素露了怯。
3. **换更强的嵌入**：把「分类代理任务」换成[第 57 章](../p12-face/ch57-face-recognition.md)的**度量学习损失**（同类拉近、异类推开），或直接用[第 73 章](../p15-generative/ch73-clip-vlm.md)训好的 CLIP 图像塔当编码器，比较检索精度。
4. **（挑战）接一个 ANN 索引**：把图库扩到几万张（多造几类、每类几千张），引入 `instant-distance` 或 `hnsw_rs` 建 HNSW 索引，测「暴力 vs ANN」的**查询耗时**与**召回率**差异，亲手感受「用一点召回换几个数量级速度」。

## 下一章预告

以图搜图把「嵌入 + 相似度检索」跑成了系统。下一章我们做一个更贴近落地的项目——**人脸考勤**：把[第 56、57 章](../p12-face/ch57-face-recognition.md)的人脸检测与识别串成一条「注册底库 → 实时比对 → 打卡记录」的完整流水线，你会发现它的内核和本章如出一辙：**把脸编码成嵌入，在底库里找最近邻**。👉 [第 101 章 项目：人脸考勤系统](ch101-face-attendance.md)
