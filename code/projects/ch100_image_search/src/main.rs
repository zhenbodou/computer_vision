// 第 100 章 项目：以图搜图引擎（CBIR，基于内容的图像检索）
// candle 0.11.0，纯 CPU，无需下载任何模型/数据，一键 `cargo run --release`。
//
// 思路（复用第 73 章的图像编码器 / 嵌入 / L2 归一化 / 余弦相似度）：
//   1) 用一个"代理任务"（这里是分类）训练小 CNN 图像编码器 → D 维嵌入
//   2) 图库每张图 → 编码 → L2 归一化 → 建库（一堆向量）
//   3) 查询图同样编码 → 与全库暴力算余弦相似度 → 排序取 top-K
//   4) 对照：学习到的嵌入 vs 原始像素，看谁检索得准（precision@K / top-1）
//
// 造数据的关键（决定成败）：让"位置"成为主要干扰项——同一类的物体出现在画面
// 各处（类内多样），且颜色不再各占一个正交通道（三通道都点亮、只是主色更亮），
// 于是"按原始像素比相似度"只会按'谁和我重叠在同一位置'乱配；而被监督训练过的
// 编码器学到了对平移不敏感的类别特征。这正是"学习嵌入 > 原始像素"的来由。
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

const IMG: usize = 24; // 图像边长（够大，物体能出现在很多位置）
const POOLED: usize = IMG / 4; // 两次 avg_pool2d(2)：24 → 12 → 6
const D: usize = 32; // 嵌入维度
const NCLASS: usize = 9; // 3 颜色 × 3 形状
const RAW: usize = 3 * IMG * IMG; // 原始像素维度 = 1728

// 三种颜色都点亮全部三通道（主色 1.0、陪衬 0.35），故它们在像素空间"不正交"，
// 原始像素余弦无法免费区分颜色——必须靠编码器去学。
const COLORS: [[f32; 3]; 3] = [[1.0, 0.35, 0.35], [0.35, 1.0, 0.35], [0.35, 0.35, 1.0]];
const COLOR_NAME: [&str; 3] = ["红", "绿", "蓝"];
const SHAPE_NAME: [&str; 3] = ["竖条", "横条", "方块"];
fn class_name(c: usize) -> String { format!("{}{}", COLOR_NAME[c / 3], SHAPE_NAME[c % 3]) }

// ---------- 极简随机数发生器（固定种子、可复现；离散随机务必走高位，见 API_NOTES §18） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self { Self(s) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 { (self.next() >> 11) as f32 / (1u64 << 53) as f32 } // 取高位
    fn below(&mut self, n: usize) -> usize { (self.unit() as f64 * n as f64) as usize } // 离散选择走高位，别用 % n
}
fn det_init(vm: &VarMap, seed: u64) -> Result<()> {
    let mut rng = Lcg::new(seed);
    let data = vm.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    for name in &names {
        let var = &data[name];
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let dev = var.device();
        if name.ends_with("bias") {
            var.set(&Tensor::zeros(dims, DType::F32, dev)?)?;
        } else {
            let fan_in = (n / dims[0].max(1)).max(1);
            let lim = (1.0 / fan_in as f32).sqrt();
            let vals: Vec<f32> = (0..n).map(|_| (rng.unit() * 2.0 - 1.0) * lim).collect();
            var.set(&Tensor::from_vec(vals, dims, dev)?)?;
        }
    }
    Ok(())
}

// ---------- 合成图库：24x24 RGB，黑底上画一个紧凑物体，位置随机（制造类内多样、平移干扰） ----------
fn draw(class: usize, rng: &mut Lcg) -> Vec<f32> {
    let (color, shape) = (class / 3, class % 3);
    let col = COLORS[color];
    let bright = 0.85 + rng.unit() * 0.15; // 轻微亮度抖动（外观扰动）
    let mut g = vec![0f32; RAW];
    let mut fill = |x0: usize, y0: usize, w: usize, h: usize| {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                for c in 0..3 {
                    g[c * IMG * IMG + y * IMG + x] = col[c] * bright;
                }
            }
        }
    };
    match shape {
        0 => fill(rng.below(IMG - 3), rng.below(IMG - 11), 3, 11), // 竖条 3x11
        1 => fill(rng.below(IMG - 11), rng.below(IMG - 3), 11, 3), // 横条 11x3
        _ => fill(rng.below(IMG - 7), rng.below(IMG - 7), 7, 7), // 方块 7x7
    }
    g
}
// 造一个数据集：每类 per_class 张，返回（拼接的像素, 每张的类别）
fn make_set(per_class: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let (mut imgs, mut labels) = (Vec::new(), Vec::new());
    for c in 0..NCLASS {
        for _ in 0..per_class {
            imgs.extend(draw(c, &mut rng));
            labels.push(c);
        }
    }
    (imgs, labels)
}

// ---------- 图像编码器：小 CNN → D 维（与第 73 章同款图像塔） ----------
struct ImageEncoder { c1: Conv2d, c2: Conv2d, fc: Linear }
impl ImageEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: conv2d(3, 8, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(8, 16, 3, cfg, vb.pp("c2"))?,
            fc: linear(16 * POOLED * POOLED, D, vb.pp("fc"))?, // 24→池化→12→池化→6
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?.avg_pool2d(2)?;
        let h = self.c2.forward(&h)?.relu()?.avg_pool2d(2)?;
        self.fc.forward(&h.flatten_from(1)?) // [N, D] —— 这就是"检索嵌入"
    }
}

// L2 归一化（eps 加在 sqrt 之前，防除零 NaN）：归一化后点积 = 余弦相似度
fn l2n(x: &Tensor) -> Result<Tensor> {
    let denom = x.sqr()?.sum_keepdim(D::Minus1)?.affine(1.0, 1e-12)?.sqrt()?;
    x.broadcast_div(&denom)
}

// 把一批图（拼接像素）编码成 L2 归一化的嵌入 [n, D]
fn embed(enc: &ImageEncoder, imgs: &[f32], n: usize, dev: &Device) -> Result<Tensor> {
    let x = Tensor::from_vec(imgs.to_vec(), (n, 3, IMG, IMG), dev)?;
    l2n(&enc.forward(&x)?)
}

// 检索评测：sims[q][g] = 查询 q 对图库 g 的余弦相似度；算 precision@K 与 top-1 命中率
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

// ---------- 可视化：几行"查询图 | 学习嵌入 top-5 | 原始像素 top-5"对照图 ----------
fn chw_to_rgb(g: &[f32]) -> RgbImage {
    let mut raw = vec![0u8; RAW];
    for y in 0..IMG {
        for x in 0..IMG {
            for c in 0..3 {
                raw[(y * IMG + x) * 3 + c] = (g[c * IMG * IMG + y * IMG + x].clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    RgbImage::from_raw(IMG as u32, IMG as u32, raw).unwrap()
}
fn topk_idx(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    idx[..k].to_vec()
}
fn save_montage(
    path: &str, queries: &[usize], q_imgs: &[f32],
    learned_sims: &[Vec<f32>], raw_sims: &[Vec<f32>], g_imgs: &[f32], k: usize,
) {
    let (disp, sg, bg, rg) = (48u32, 3u32, 16u32, 6u32); // 单元/小间隔/大间隔/行间隔
    let block = k as u32 * disp + (k as u32 - 1) * sg;
    let w = disp + bg + block + bg + block;
    let h = queries.len() as u32 * disp + (queries.len() as u32 - 1) * rg;
    let mut canvas = RgbImage::from_pixel(w, h, Rgb([240, 240, 240]));
    let tile = |canvas: &mut RgbImage, g: &[f32], x: u32, y: u32| {
        let big = image::imageops::resize(&chw_to_rgb(g), disp, disp, FilterType::Nearest);
        image::imageops::overlay(canvas, &big, x as i64, y as i64);
    };
    let l_x0 = disp + bg;
    let r_x0 = l_x0 + block + bg;
    for (row, &q) in queries.iter().enumerate() {
        let y = row as u32 * (disp + rg);
        tile(&mut canvas, &q_imgs[q * RAW..(q + 1) * RAW], 0, y); // 查询图
        for (j, &gi) in topk_idx(&learned_sims[q], k).iter().enumerate() {
            tile(&mut canvas, &g_imgs[gi * RAW..(gi + 1) * RAW], l_x0 + j as u32 * (disp + sg), y);
        }
        for (j, &gi) in topk_idx(&raw_sims[q], k).iter().enumerate() {
            tile(&mut canvas, &g_imgs[gi * RAW..(gi + 1) * RAW], r_x0 + j as u32 * (disp + sg), y);
        }
    }
    canvas.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 1) 造数据：训练集（训编码器）+ 图库（被搜）+ 查询集（去搜），三者实例互不相同 =====
    let (train_imgs, train_lab) = make_set(40, 1); // 40×9 = 360 张训练
    let (gal_imgs, gal_lab) = make_set(30, 2); //     30×9 = 270 张图库
    let (qry_imgs, qry_lab) = make_set(8, 3); //       8×9 = 72  张查询
    let (ntr, ng, nq) = (train_lab.len(), gal_lab.len(), qry_lab.len());
    // 自查：打印图库每类计数（确认无空类，见 API_NOTES §18）
    let mut hist = [0usize; NCLASS];
    for &c in &gal_lab { hist[c] += 1; }
    print!("图库类别直方图：");
    for c in 0..NCLASS { print!("{}={} ", class_name(c), hist[c]); }
    println!("（共 {} 张，无空类）", ng);

    // ===== 2) 训练图像编码器（代理任务=分类；取编码器 D 维输出当检索嵌入） =====
    println!("\n==== 训练图像编码器（分类代理任务，400 步）====");
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let enc = ImageEncoder::load(vb.pp("enc"))?;
    let head = linear(D, NCLASS, vb.pp("head"))?; // 仅训练用的分类头，检索时丢弃
    det_init(&vm, 42)?;
    let x_tr = Tensor::from_vec(train_imgs.clone(), (ntr, 3, IMG, IMG), &dev)?;
    let y_tr = Tensor::from_vec(train_lab.iter().map(|&c| c as u32).collect::<Vec<_>>(), (ntr,), &dev)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
    for step in 1..=400 {
        let logits = head.forward(&enc.forward(&x_tr)?)?;
        let loss = cross_entropy(&logits, &y_tr)?;
        opt.backward_step(&loss)?;
        if step % 100 == 0 || step == 1 {
            let pred = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let acc = pred.iter().zip(&train_lab).filter(|(p, l)| **p as usize == **l).count();
            println!("  step {step:>3}  分类损失 = {:.4}  训练准确率 = {:.1}%",
                loss.to_scalar::<f32>()?, 100.0 * acc as f32 / ntr as f32);
        }
    }

    // ===== 3) 建库 + 检索：学习嵌入 vs 原始像素 =====
    let g_learned = embed(&enc, &gal_imgs, ng, &dev)?; // [ng, D]
    let q_learned = embed(&enc, &qry_imgs, nq, &dev)?; // [nq, D]
    let g_raw = l2n(&Tensor::from_vec(gal_imgs.clone(), (ng, RAW), &dev)?)?; // [ng, 1728]
    let q_raw = l2n(&Tensor::from_vec(qry_imgs.clone(), (nq, RAW), &dev)?)?; // [nq, 1728]
    // 暴力余弦相似度：查询 × 图库^T
    let learned_sims: Vec<Vec<f32>> = q_learned.matmul(&g_learned.t()?)?.to_vec2::<f32>()?;
    let raw_sims: Vec<Vec<f32>> = q_raw.matmul(&g_raw.t()?)?.to_vec2::<f32>()?;

    let k = 5;
    let (pl, tl) = eval_retrieval(&learned_sims, &qry_lab, &gal_lab, k);
    let (pr, tr) = eval_retrieval(&raw_sims, &qry_lab, &gal_lab, k);
    println!("\n==== 检索效果（{} 张查询 × {} 张图库，暴力余弦，K={}）====", nq, ng, k);
    println!("  学习嵌入（{} 维）    precision@{} = {:.1}%   top-1 命中 = {:.1}%", D, k, 100.0 * pl, 100.0 * tl);
    println!("  原始像素（{} 维）  precision@{} = {:.1}%   top-1 命中 = {:.1}%", RAW, k, 100.0 * pr, 100.0 * tr);
    println!("  → 学习嵌入把'同色同形'的图拉近；原始像素只按'谁和我重叠在同一位置'乱配。");

    // ===== 4) 可视化几个查询的 top-5 对照 =====
    let queries = [0usize, 32, 64, 16]; // 各类第 1 张（每类 8 张）：红竖条 / 绿横条 / 蓝方块 / 红方块
    print!("\n对照图查询：");
    for &q in &queries { print!("{} ", class_name(qry_lab[q])); }
    println!();
    save_montage("out_ch100_retrieval.png", &queries, &qry_imgs, &learned_sims, &raw_sims, &gal_imgs, k);
    println!("已存 out_ch100_retrieval.png（每行：查询图 | 学习嵌入 top-5 | 原始像素 top-5）");
    Ok(())
}
