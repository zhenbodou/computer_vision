// 第 99 章 项目：受限字符集 OCR（candle 0.11.0，纯 Rust，纯 CPU，无需下载任何模型；结果可复现）
// 运行：cd code/projects/ch99_ocr && cargo run --release
//
// 纯 Rust 没有成熟的“通用 OCR 引擎”（Tesseract=C++、PaddleOCR=Python/C++，都违反纯 Rust 约束）。
// 本章走【诚实可行】的受限字符集路线——读“序列号/仪表/编码”这类固定字形、干净背景的短串：
//     ① 合成短字符串图（数字 0-9 + 易混字母 O/I/S）
//     ② 8 邻域 BFS 连通域把一行切成单字，按外接框 x 左到右排序（复用第 102 章那套）
//     ③ candle 小 CNN 逐字分类
//     ④ 左到右拼成字符串，与真值比对：逐字准确率 + 整串精确匹配率 + 易混/粘连观察
//
// 本章头号教训（第 20 部分级联域偏移 §20 在 OCR 上的落地）：
//     字符分类器【必须】在“和推理同一条分割管线产出的字形”上训练。
//     用“居中干净字形”训练、却拿“分割出来被拉伸的紧贴外接框字形”推理 → 准确率崩，且不报错。
//     本程序同时训两个分类器（NAIVE 干净居中 vs PIPELINE 分割产出），端到端对拍，让域偏移显形。

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::imageops::FilterType;
use image::{GrayImage, Luma, Rgb, RgbImage};
use std::collections::VecDeque;

const CLS_SZ: usize = 24; // 分类器输入边长（正方形灰度）
const SCALE: usize = 3; // 位图字体放大倍数：5x7 → 15x21
const STR_H: usize = 28; // 字符串图高度
const THR: f32 = 0.35; // 分割二值化阈值：亮字>THR 为前景（取 0.35 让轻模糊后的笔画边缘仍连得住）
const MIN_AREA: usize = 15; // 连通域最小面积（滤掉噪点）
const CHARSET: &str = "0123456789OIS"; // 受限字符集：10 数字 + 易混字母 O/I/S（构造 0/O、1/I、5/S 三对形近字，观察谁真会混）

// ---------- 确定性随机数（candle CPU 不能 set_seed；离散采样一律走高位，避免 LCG 低位规律，§18） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // 取高位做 [0,1)
    }
    // [lo,hi) 内整数——用高位缩放，绝不用 self.next() % k（LCG 低位质量差，§18）
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.unit() * (hi - lo) as f32) as i32
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// ---------- 5x7 位图字体：每个字形 7 行，每行低 5 位（最高位=第 0 列）。都是【单个 8-连通块】 ----------
fn glyph(c: char) -> [u8; 7] {
    match c {
        // 0 与 O 只差 1 个像素（0 中心多一点），专门制造 0/O 混淆
        '0' => [0b01110, 0b10001, 0b10001, 0b10101, 0b10001, 0b10001, 0b01110],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        // 1 与 I 只差顶部一横：I 顶上有横杠、1 没有（其余竖杠+底座相同）——顶横是【边界】多像素特征，模糊抹不掉
        '1' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        // 5 与 S 同为 Z 形折线、共享上半段，仅下半略异——差异同样在【边界】拐角上，模糊抹不掉
        '5' => [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b10001, 0b01110],
        'S' => [0b11111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        _ => [0; 7],
    }
}
fn char_at(i: usize) -> char {
    CHARSET.chars().nth(i).unwrap()
}
fn idx_of(c: char) -> usize {
    CHARSET.chars().position(|x| x == c).unwrap()
}

// 把一个字形以左上角 (ox,oy)、放大 s 倍画进灰度画布（text=亮 val，配合“亮字暗底”，同第 42/102 章约定）
fn paint_glyph(buf: &mut [f32], w: usize, h: usize, g: &[u8; 7], ox: i32, oy: i32, s: usize, val: f32) {
    for (r, row) in g.iter().enumerate() {
        for c in 0..5 {
            if (row >> (4 - c)) & 1 == 1 {
                for dy in 0..s {
                    for dx in 0..s {
                        let x = ox + (c * s + dx) as i32;
                        let y = oy + (r * s + dy) as i32;
                        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                            buf[y as usize * w + x as usize] = val;
                        }
                    }
                }
            }
        }
    }
}

// 分离式 3 抽头模糊 [1,2,1]/4（边界复制）：模拟光学/低质量成像，软化字形细节。
// 关键效应：被【暗背景包围】的内部小特征（0 中心那一点）会被抹淡 → 0/O 变混；
// 而落在【字形边界】上的多像素特征（I 顶横、5/S 拐角）模糊抹不掉 → 这两对几乎不混（见正文混淆分析）。
fn blur3(buf: &mut [f32], w: usize, h: usize) {
    let mut tmp = buf.to_vec();
    for y in 0..h {
        for x in 0..w {
            let l = buf[y * w + x.saturating_sub(1)];
            let c = buf[y * w + x];
            let r = buf[y * w + (x + 1).min(w - 1)];
            tmp[y * w + x] = (l + 2.0 * c + r) / 4.0;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let u = tmp[y.saturating_sub(1) * w + x];
            let c = tmp[y * w + x];
            let d = tmp[(y + 1).min(h - 1) * w + x];
            buf[y * w + x] = (u + 2.0 * c + d) / 4.0;
        }
    }
}

// 合成一张短字符串图：亮字（0.9）暗底（0.1）+ 轻模糊 + 高斯噪声 + 位置抖动 + 字间距变化。
// touching=true 时字间距取负（字符粘连）——演示连通域分割的已知难点。
// 返回：灰度缓冲、宽、高、真值串。
fn make_string(rng: &mut Lcg, touching: bool) -> (Vec<f32>, usize, usize, String) {
    let len = rng.range(4, 7) as usize; // 4..=6 个字符
    let gw = 5 * SCALE; // 单字形宽 15
    let gh = 7 * SCALE; // 单字形高 21
    let gap = if touching {
        rng.range(-(SCALE as i32), 1) // 负/零间距 → 相邻字符可能粘连（有的碰上、有的没碰上，更真实）
    } else {
        rng.range(2 * SCALE as i32, 4 * SCALE as i32) // 清晰间隔
    };
    let margin = 2 * SCALE as i32;
    let w = (margin as usize * 2 + len * gw + (len - 1) * gap.max(0) as usize).max(len * gw + 8);
    let h = STR_H;
    let mut buf = vec![0.10f32; w * h]; // 干净暗底
    let mut gt = String::new();
    let mut x = margin;
    let oy0 = ((h - gh) / 2) as i32;
    for _ in 0..len {
        let ci = rng.range(0, CHARSET.len() as i32) as usize;
        let c = char_at(ci);
        gt.push(c);
        let jx = rng.range(-1, 2); // ±1 抖动
        let jy = rng.range(-2, 3); // ±2 抖动
        paint_glyph(&mut buf, w, h, &glyph(c), x + jx, oy0 + jy, SCALE, 0.9);
        x += gw as i32 + gap;
    }
    blur3(&mut buf, w, h); // 一次轻模糊 → 模拟成像退化：把 0 中心那一个像素点抹淡（0/O 变混）
    for v in buf.iter_mut() {
        *v = (*v + rng.gauss() * 0.12).clamp(0.0, 1.0); // 高斯噪声
    }
    (buf, w, h, gt)
}

// ---------- 分割：8 邻域 BFS 连通域（复用第 102 章那套），返回按外接框 x0 排序的 bbox ----------
fn segment(gray: &[f32], w: usize, h: usize, thr: f32, min_area: usize) -> Vec<[usize; 4]> {
    let fg: Vec<bool> = gray.iter().map(|&v| v > thr).collect(); // 亮字为前景
    let mut visited = vec![false; w * h];
    let mut boxes = Vec::new();
    for start in 0..w * h {
        if !fg[start] || visited[start] {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        visited[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        let mut area = 0usize;
        while let Some(idx) = q.pop_front() {
            area += 1;
            let (x, y) = (idx % w, idx / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = ny as usize * w + nx as usize;
                    if fg[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        q.push_back(nidx);
                    }
                }
            }
        }
        if area >= min_area {
            boxes.push([x0, y0, x1, y1]);
        }
    }
    boxes.sort_by_key(|b| b[0]); // 左到右
    boxes
}

// 把一个外接框内的灰度抠出来，拉伸到 CLS_SZ×CLS_SZ（不保持长宽比——这正是 §20 里“紧贴框+拉伸”的域特征）
fn crop_resize(gray: &[f32], w: usize, _h: usize, b: [usize; 4]) -> Vec<f32> {
    let (cw, ch) = (b[2] - b[0] + 1, b[3] - b[1] + 1);
    let mut sub = GrayImage::new(cw as u32, ch as u32);
    for yy in 0..ch {
        for xx in 0..cw {
            let v = (gray[(b[1] + yy) * w + (b[0] + xx)].clamp(0.0, 1.0) * 255.0) as u8;
            sub.put_pixel(xx as u32, yy as u32, Luma([v]));
        }
    }
    let r = image::imageops::resize(&sub, CLS_SZ as u32, CLS_SZ as u32, FilterType::Triangle);
    r.pixels().map(|p| p[0] as f32 / 255.0).collect()
}

// NAIVE 分类器的“干净居中”训练字形：保持长宽比、居中画进 CLS_SZ 画布、只加轻噪声。
// —— 这是初学者会想当然生成的训练数据，和分割器真实产出的“拉伸紧贴框”字形分布【不一致】。
fn render_centered(c: char, rng: &mut Lcg) -> Vec<f32> {
    let mut buf = vec![0.10f32; CLS_SZ * CLS_SZ]; // 暗底，与分割 crop 的底色一致
    let (gw, gh) = (5 * SCALE, 7 * SCALE); // 15x21，保持字体原长宽比
    let ox = ((CLS_SZ - gw) / 2) as i32 + rng.range(-1, 2);
    let oy = ((CLS_SZ - gh) / 2) as i32 + rng.range(-1, 2);
    paint_glyph(&mut buf, CLS_SZ, CLS_SZ, &glyph(c), ox, oy, SCALE, 0.9);
    blur3(&mut buf, CLS_SZ, CLS_SZ); // 同样一次轻模糊 → 与分割产出的差异【只在几何】（居中保长宽比 vs 拉伸紧贴框）
    for v in buf.iter_mut() {
        *v = (*v + rng.gauss() * 0.12).clamp(0.0, 1.0);
    }
    buf
}

// ---------- 小 CNN 逐字分类器（1 通道 24×24 → 13 类），复用第 42 章骨架 ----------
struct CharCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    head: Linear,
}
impl CharCnn {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() }; // pad=1 → 卷积不改变边长
        Ok(Self {
            conv1: conv2d(1, 8, 3, cfg, vb.pp("conv1"))?, // 1→8
            conv2: conv2d(8, 16, 3, cfg, vb.pp("conv2"))?, // 8→16
            fc1: linear(16 * (CLS_SZ / 4) * (CLS_SZ / 4), 64, vb.pp("fc1"))?, // 16*6*6=576 → 64
            head: linear(64, CHARSET.len(), vb.pp("head"))?, // 64 → 13 类
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 24→12
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 12→6
        let x = x.flatten_from(1)?; // → [N,576]
        let x = self.fc1.forward(&x)?.relu()?; // → [N,64]
        self.head.forward(&x) // → [N,13] logits
    }
}

// 确定性初始化：bias=0，weight 取 [-lim,lim] 均匀，lim=1/sqrt(fan_in)（PyTorch 默认风格）
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

// 一批字符样本（每个 CLS_SZ*CLS_SZ）+ 标签 → 训练一个 CharCnn，返回训练好的模型和它的 VarMap
fn train_classifier(name: &str, xs: &[Vec<f32>], ys: &[u32], seed: u64, epochs: usize, dev: &Device) -> Result<(CharCnn, VarMap)> {
    let n = xs.len();
    let mut flat = Vec::with_capacity(n * CLS_SZ * CLS_SZ);
    for s in xs {
        flat.extend_from_slice(s);
    }
    let x = Tensor::from_vec(flat, (n, 1, CLS_SZ, CLS_SZ), dev)?;
    let y = Tensor::from_vec(ys.to_vec(), (n,), dev)?; // 标签 U32（cross_entropy 要求）
    let vm = VarMap::new();
    let net = CharCnn::load(VarBuilder::from_varmap(&vm, DType::F32, dev))?;
    det_init(&vm, seed)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("  训练 {name} 分类器（{n} 个字符样本，AdamW lr=1e-3，{epochs} epoch）...");
    for epoch in 1..=epochs {
        if epoch == epochs * 3 / 4 {
            opt.set_learning_rate(3e-4);
        }
        let loss = cross_entropy(&net.forward(&x)?, &y)?;
        opt.backward_step(&loss)?;
        if epoch % (epochs / 4) == 0 || epoch == 1 {
            println!("    epoch {epoch:>3}  交叉熵 loss = {:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok((net, vm))
}

// 对一张字符串图跑完整 OCR：分割 → 逐字分类 → 左到右拼串
fn ocr(net: &CharCnn, gray: &[f32], w: usize, h: usize, dev: &Device) -> Result<(String, Vec<[usize; 4]>)> {
    let boxes = segment(gray, w, h, THR, MIN_AREA);
    let mut s = String::new();
    for &b in &boxes {
        let crop = crop_resize(gray, w, h, b);
        let x = Tensor::from_vec(crop, (1, 1, CLS_SZ, CLS_SZ), dev)?;
        let pred = net.forward(&x)?.argmax(D::Minus1)?.to_vec1::<u32>()?[0] as usize;
        s.push(char_at(pred));
    }
    Ok((s, boxes))
}

// 存一张“字符串图 + 红色分割框”PNG（放大便于看），手写画框，不引入 imageproc
fn save_vis(gray: &[f32], w: usize, h: usize, boxes: &[[usize; 4]], scale: u32, path: &str) {
    let mut img = RgbImage::new(w as u32 * scale, h as u32 * scale);
    for y in 0..h as u32 * scale {
        for x in 0..w as u32 * scale {
            let v = (gray[(y / scale) as usize * w + (x / scale) as usize].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    let red = Rgb([255u8, 60, 60]);
    for b in boxes {
        let (x0, y0, x1, y1) = (b[0] as u32 * scale, b[1] as u32 * scale, (b[2] as u32 + 1) * scale - 1, (b[3] as u32 + 1) * scale - 1);
        for x in x0..=x1.min(w as u32 * scale - 1) {
            img.put_pixel(x, y0, red);
            img.put_pixel(x, y1.min(h as u32 * scale - 1), red);
        }
        for y in y0..=y1.min(h as u32 * scale - 1) {
            img.put_pixel(x0, y, red);
            img.put_pixel(x1.min(w as u32 * scale - 1), y, red);
        }
    }
    img.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    println!("受限字符集 OCR：字符集 = \"{CHARSET}\"（{} 类），亮字暗底短串，纯 Rust 纯 CPU。\n", CHARSET.len());

    // ================= 造训练数据 =================
    // PIPELINE 训练集：走真实分割管线——生成串→分割→抠框拉伸，只保留“分割数=真值长度”的串并按序打标签。
    let mut rng = Lcg::new(20);
    let (mut px, mut py) = (Vec::new(), Vec::new());
    let mut pipe_strings = 0usize;
    let mut pipe_kept = 0usize;
    while pipe_kept < 260 {
        let (g, w, h, gt) = make_string(&mut rng, false);
        pipe_strings += 1;
        let boxes = segment(&g, w, h, THR, MIN_AREA);
        if boxes.len() != gt.len() {
            continue; // 分割数对不上 → 无法可靠打标签，丢弃（训练只用分割正确的样本）
        }
        for (b, c) in boxes.iter().zip(gt.chars()) {
            px.push(crop_resize(&g, w, h, *b));
            py.push(idx_of(c) as u32);
        }
        pipe_kept += 1;
    }
    println!("PIPELINE 训练：扫过 {pipe_strings} 串、保留分割正确的 {pipe_kept} 串 → 共 {} 个字符样本（走真实分割分布）。", px.len());

    // NAIVE 训练集：每类若干“干净居中”字形（保持长宽比、居中、轻噪声）——分布和分割产出【不一致】。
    let (mut nx, mut ny) = (Vec::new(), Vec::new());
    let per_class = px.len() / CHARSET.len(); // 与 PIPELINE 样本量对齐，公平对比
    for ci in 0..CHARSET.len() {
        for _ in 0..per_class {
            nx.push(render_centered(char_at(ci), &mut rng));
            ny.push(ci as u32);
        }
    }
    println!("NAIVE  训练：每类 {per_class} 个“干净居中”字形 → 共 {} 个字符样本（分布与分割产出不一致，§20 对照）。\n", nx.len());

    // ================= 训练两个分类器 =================
    println!("========== 训练 ==========");
    let (naive, _vm_n) = train_classifier("NAIVE ", &nx, &ny, 42, 60, &dev)?;
    let (pipe, _vm_p) = train_classifier("PIPELINE", &px, &py, 42, 60, &dev)?;

    // ================= 干净测试集：逐字准确率 + 整串精确匹配率 + 混淆 =================
    println!("\n========== 评测（干净测试集，字间距清晰）==========");
    let n_test = 200usize;
    let mut test = Vec::new();
    let mut rng_t = Lcg::new(999);
    for _ in 0..n_test {
        let (g, w, h, gt) = make_string(&mut rng_t, false);
        test.push((g, w, h, gt));
    }

    // 分割成功率（分割框数 == 真值长度）
    let mut seg_ok = 0usize;
    for (g, w, h, gt) in &test {
        if segment(g, *w, *h, THR, MIN_AREA).len() == gt.len() {
            seg_ok += 1;
        }
    }
    println!("分割成功率（连通域数=真值长度）：{:.1}%（{}/{}）\n", 100.0 * seg_ok as f32 / n_test as f32, seg_ok, n_test);

    // 对两个分类器分别端到端评测
    let mut confusion = vec![vec![0u32; CHARSET.len()]; CHARSET.len()]; // 只统 PIPELINE 的混淆
    for (tag, net, collect_conf) in [("NAIVE ", &naive, false), ("PIPELINE", &pipe, true)] {
        let (mut char_ok, mut char_total, mut exact_ok) = (0usize, 0usize, 0usize);
        for (g, w, h, gt) in &test {
            let (pred, _) = ocr(net, g, *w, *h, &dev)?;
            char_total += gt.len();
            if pred.len() == gt.len() {
                let mut all = true;
                for (pc, gc) in pred.chars().zip(gt.chars()) {
                    if pc == gc {
                        char_ok += 1;
                    } else {
                        all = false;
                    }
                    if collect_conf {
                        confusion[idx_of(gc)][idx_of(pc)] += 1;
                    }
                }
                if all {
                    exact_ok += 1;
                }
            }
            // 分割数对不上：该串所有字符按错处理（同时 exact 记失败），如实惩罚
        }
        println!(
            "{tag}：逐字准确率 = {:.1}%（{}/{}），整串精确匹配率 = {:.1}%（{}/{}）",
            100.0 * char_ok as f32 / char_total as f32,
            char_ok,
            char_total,
            100.0 * exact_ok as f32 / n_test as f32,
            exact_ok,
            n_test
        );
    }

    // ---------- 混淆观察（PIPELINE）：打印最主要的错分对，并专门看 0/O、1/I、5/S ----------
    println!("\n========== 混淆观察（PIPELINE 分类器）==========");
    let mut pairs: Vec<(usize, usize, u32)> = Vec::new();
    for gi in 0..CHARSET.len() {
        for pi in 0..CHARSET.len() {
            if gi != pi && confusion[gi][pi] > 0 {
                pairs.push((gi, pi, confusion[gi][pi]));
            }
        }
    }
    pairs.sort_by(|a, b| b.2.cmp(&a.2));
    println!("最主要的错分对（真值→预测：次数）：");
    for (gi, pi, cnt) in pairs.iter().take(8) {
        println!("  {} → {} ：{}", char_at(*gi), char_at(*pi), cnt);
    }
    let show = |a: char, b: char| {
        let (ia, ib) = (idx_of(a), idx_of(b));
        println!("  {a}/{b}: {a}→{b} {} 次，{b}→{a} {} 次", confusion[ia][ib], confusion[ib][ia]);
    };
    println!("重点易混对：");
    show('0', 'O');
    show('1', 'I');
    show('5', 'S');

    // ================= 粘连测试集：连通域分割的已知难点 =================
    println!("\n========== 粘连测试集（字间距为负，字符相互粘连）==========");
    let mut rng_c = Lcg::new(555);
    let (n_touch, mut merged, mut touch_exact) = (100usize, 0usize, 0usize);
    for k in 0..n_touch {
        let (g, w, h, gt) = make_string(&mut rng_c, true);
        let (pred, boxes) = ocr(&pipe, &g, w, h, &dev)?;
        if boxes.len() < gt.len() {
            merged += 1; // 连通域数 < 真值长度 = 有字符被粘连成一块
        }
        if pred == gt {
            touch_exact += 1;
        }
        if k == 0 {
            save_vis(&g, w, h, &boxes, 4, "ocr_touch.png");
            println!("  例：真值\"{gt}\"（{} 字）→ 分割出 {} 块 → 识别\"{pred}\"", gt.len(), boxes.len());
        }
    }
    println!(
        "粘连集：{:.1}% 的串出现“分割块数<真值长度”（{}/{}），整串精确匹配率骤降到 {:.1}%（{}/{}）",
        100.0 * merged as f32 / n_touch as f32,
        merged,
        n_touch,
        100.0 * touch_exact as f32 / n_touch as f32,
        touch_exact,
        n_touch
    );

    // 存一张干净串的分割可视化
    save_vis(&test[0].0, test[0].1, test[0].2, &segment(&test[0].0, test[0].1, test[0].2, THR, MIN_AREA), 4, "ocr_clean.png");
    println!("\n已存 ocr_clean.png（干净串+红色分割框，分割正确）、ocr_touch.png（粘连串，分割框跨了多个字符=失败）。");
    Ok(())
}
