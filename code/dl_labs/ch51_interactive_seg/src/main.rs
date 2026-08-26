// 第 51 章：交互式分割与 SAM（candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现）
// 运行：cd code/dl_labs/ch51_interactive_seg && cargo run --release
//
// 两部分：
//   A. 经典交互式分割——种子区域生长（seeded region growing）：给一个「点提示」，
//      按颜色相似度 + 连通性把目标从背景里“长”出来；演示点提示、框提示、阈值敏感性，
//      报真实数字（分割像素数、与 GT 的 IoU）。纯算法、瞬时。
//   B. 迷你 SAM——「点条件」分割小网络（candle）：输入 = RGB + 点提示热力图，
//      输出 = 该点所指物体的掩码。这正是 SAM“可提示分割（promptable）”范式的玩具版：
//      同一张图、点不同物体 → 得到不同掩码。报测试集掩码 IoU。

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use std::collections::VecDeque;

const SZ: usize = 28; // 图像边长（正方形，需被 4 整除）
// 背景基色 + 4 种物体基色（红 / 绿 / 蓝 / 黄）
const BG: [f32; 3] = [26.0, 30.0, 38.0];
const OBJ_COLORS: [[f32; 3]; 4] = [
    [220.0, 80.0, 70.0],
    [80.0, 200.0, 110.0],
    [90.0, 120.0, 230.0],
    [230.0, 200.0, 70.0],
];

// ---------- 确定性随机数发生器（candle CPU 不能 set_seed，随机性我们自己掌控） ----------
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }
}

// 一个物体：它占据的像素掩码 + 质心（作为点提示的落点）+ 外接框
#[derive(Clone)]
struct Obj {
    mask: Vec<bool>, // 长度 SZ*SZ
    cx: i32,
    cy: i32,
    bbox: [i32; 4], // x0,y0,x1,y1（含）
}

// 合成一张场景：dark 背景 + 2~4 个纯色矩形物体（各叠噪声）。
// 返回 (图, 物体列表)。允许物体偶尔同色——这样“点条件”网络必须靠点提示区分它们。
fn make_scene(rng: &mut Lcg) -> (RgbImage, Vec<Obj>) {
    let mut img = RgbImage::new(SZ as u32, SZ as u32);
    // 先铺背景（低噪声，保证区域生长时背景与物体差距稳定）
    let mut base = vec![[0f32; 3]; SZ * SZ];
    for p in base.iter_mut() {
        *p = BG;
    }
    let n = rng.range(2, 5); // 2..=4 个物体
    let mut objs: Vec<Obj> = Vec::new();
    let mut tries = 0;
    while (objs.len() as i32) < n && tries < 60 {
        tries += 1;
        let w = rng.range(6, 12);
        let h = rng.range(6, 12);
        let x0 = rng.range(1, SZ as i32 - w - 1);
        let y0 = rng.range(1, SZ as i32 - h - 1);
        let (x1, y1) = (x0 + w - 1, y0 + h - 1);
        // 与已有物体保持 2px 间隔（用背景隔开，保证连通性能区分不同物体）
        let overlap = objs.iter().any(|o| {
            x0 <= o.bbox[2] + 2 && x1 >= o.bbox[0] - 2 && y0 <= o.bbox[3] + 2 && y1 >= o.bbox[1] - 2
        });
        if overlap {
            continue;
        }
        let color = OBJ_COLORS[rng.range(0, OBJ_COLORS.len() as i32) as usize];
        let mut mask = vec![false; SZ * SZ];
        for y in y0..=y1 {
            for x in x0..=x1 {
                mask[y as usize * SZ + x as usize] = true;
                base[y as usize * SZ + x as usize] = color;
            }
        }
        objs.push(Obj { mask, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, bbox: [x0, y0, x1, y1] });
    }
    // 叠噪声写进真正的图
    for y in 0..SZ {
        for x in 0..SZ {
            let c = base[y * SZ + x];
            let mut px = [0u8; 3];
            for k in 0..3 {
                px[k] = (c[k] + rng.gauss() * 12.0).round().clamp(0.0, 255.0) as u8;
            }
            img.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    (img, objs)
}

// 颜色的平方欧氏距离
fn color_dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}
fn pixel_rgb(img: &RgbImage, x: i32, y: i32) -> [f32; 3] {
    let p = img.get_pixel(x as u32, y as u32).0;
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

// 种子区域生长：从 (seed) 出发，BFS 把与「种子颜色」相似（平方距离 < thr2）且连通的像素纳入。
// 可选 bbox 约束（框提示）：只在框内生长。返回布尔掩码。
fn region_grow(img: &RgbImage, seed: (i32, i32), thr2: f32, bbox: Option<[i32; 4]>) -> Vec<bool> {
    let mut mask = vec![false; SZ * SZ];
    let ref_color = pixel_rgb(img, seed.0, seed.1); // 参考色 = 种子像素颜色
    let inside = |x: i32, y: i32| -> bool {
        if x < 0 || y < 0 || x >= SZ as i32 || y >= SZ as i32 {
            return false;
        }
        match bbox {
            Some(b) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3],
            None => true,
        }
    };
    let mut q = VecDeque::new();
    if inside(seed.0, seed.1) {
        mask[seed.1 as usize * SZ + seed.0 as usize] = true;
        q.push_back(seed);
    }
    while let Some((x, y)) = q.pop_front() {
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if !inside(nx, ny) {
                continue;
            }
            let idx = ny as usize * SZ + nx as usize;
            if mask[idx] {
                continue;
            }
            if color_dist2(pixel_rgb(img, nx, ny), ref_color) < thr2 {
                mask[idx] = true;
                q.push_back((nx, ny));
            }
        }
    }
    mask
}

// 两个布尔掩码的 IoU
fn iou_mask(a: &[bool], b: &[bool]) -> f32 {
    let mut inter = 0usize;
    let mut uni = 0usize;
    for i in 0..a.len() {
        if a[i] && b[i] {
            inter += 1;
        }
        if a[i] || b[i] {
            uni += 1;
        }
    }
    if uni == 0 {
        0.0
    } else {
        inter as f32 / uni as f32
    }
}
fn count(mask: &[bool]) -> usize {
    mask.iter().filter(|&&b| b).count()
}

// 把 (输入图+点标记 | 分割掩码 | GT 掩码) 拼成三联图存 PNG
fn save_triptych(img: &RgbImage, click: (i32, i32), pred: &[bool], gt: &[bool], path: &str) {
    const S: u32 = 8;
    const GAP: u32 = 4;
    let cell = SZ as u32 * S;
    let mut canvas = RgbImage::from_pixel(cell * 3 + GAP * 2, cell, Rgb([255, 255, 255]));
    // ① 输入 + 十字点标记
    let mut marked = img.clone();
    for d in -2..=2 {
        put(&mut marked, click.0 + d, click.1, Rgb([255, 255, 0]));
        put(&mut marked, click.0, click.1 + d, Rgb([255, 255, 0]));
    }
    let big = image::imageops::resize(&marked, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &big, 0, 0);
    // ② 预测掩码（白=前景，灰=背景）
    blit_mask(&mut canvas, pred, (cell + GAP) as i64);
    // ③ GT 掩码
    blit_mask(&mut canvas, gt, (cell * 2 + GAP * 2) as i64);
    canvas.save(path).unwrap();
}
fn put(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < SZ as i32 && y < SZ as i32 {
        img.put_pixel(x as u32, y as u32, c);
    }
}
fn blit_mask(canvas: &mut RgbImage, mask: &[bool], ox: i64) {
    const S: u32 = 8;
    let cell = SZ as u32 * S;
    let mut m = RgbImage::new(SZ as u32, SZ as u32);
    for y in 0..SZ {
        for x in 0..SZ {
            let c = if mask[y * SZ + x] { [245, 245, 250] } else { [55, 58, 66] };
            m.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
    let big = image::imageops::resize(&m, cell, cell, image::imageops::FilterType::Nearest);
    image::imageops::overlay(canvas, &big, ox, 0);
}

// ============================ 第 B 部分：迷你 SAM（点条件分割网络） ============================

// 点提示热力图：以 (cx,cy) 为中心的高斯，值域 [0,1]，作为网络的第 4 个输入通道
fn point_heat(cx: i32, cy: i32) -> Vec<f32> {
    let sigma = SZ as f32 / 6.0;
    let mut h = vec![0f32; SZ * SZ];
    for y in 0..SZ {
        for x in 0..SZ {
            let d2 = ((x as i32 - cx).pow(2) + (y as i32 - cy).pow(2)) as f32;
            h[y * SZ + x] = (-d2 / (2.0 * sigma * sigma)).exp();
        }
    }
    h
}

// 把 (图, 点提示) 打包成 [4,SZ,SZ]：通道 0-2 = RGB/255，通道 3 = 点热力图
fn make_input(img: &RgbImage, cx: i32, cy: i32) -> Vec<f32> {
    let mut data = vec![0f32; 4 * SZ * SZ];
    for y in 0..SZ {
        for x in 0..SZ {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * SZ * SZ + y * SZ + x] = p[c] as f32 / 255.0;
            }
        }
    }
    let heat = point_heat(cx, cy);
    data[3 * SZ * SZ..].copy_from_slice(&heat);
    data
}

// 迷你 SAM 网络：与第 49 章 U-Net 同构，但输入 4 通道（含点提示）、输出 2 类（背景/前景）
struct PromptSeg {
    enc1: Conv2d,
    enc2: Conv2d,
    bott: Conv2d,
    up2: Conv2d,
    dec2: Conv2d,
    up1: Conv2d,
    dec1: Conv2d,
    head: Conv2d,
}
impl PromptSeg {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        let c1 = Conv2dConfig { padding: 0, ..Default::default() };
        Ok(Self {
            enc1: conv2d(4, 16, 3, c, vb.pp("enc1"))?, // 4 通道输入（RGB + 点提示）
            enc2: conv2d(16, 32, 3, c, vb.pp("enc2"))?,
            bott: conv2d(32, 64, 3, c, vb.pp("bott"))?,
            up2: conv2d(64, 32, 3, c, vb.pp("up2"))?,
            dec2: conv2d(64, 32, 3, c, vb.pp("dec2"))?,
            up1: conv2d(32, 16, 3, c, vb.pp("up1"))?,
            dec1: conv2d(32, 16, 3, c, vb.pp("dec1"))?,
            head: conv2d(16, 2, 1, c1, vb.pp("head"))?, // 2 类：背景 / 前景
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let e1 = self.enc1.forward(x)?.relu()?;
        let p1 = e1.max_pool2d(2)?;
        let e2 = self.enc2.forward(&p1)?.relu()?;
        let p2 = e2.max_pool2d(2)?;
        let b = self.bott.forward(&p2)?.relu()?;
        let u2 = b.upsample_nearest2d(SZ / 2, SZ / 2)?;
        let u2 = self.up2.forward(&u2)?.relu()?;
        let c2 = Tensor::cat(&[&u2, &e2], 1)?; // 跳跃连接
        let d2 = self.dec2.forward(&c2)?.relu()?;
        let u1 = d2.upsample_nearest2d(SZ, SZ)?;
        let u1 = self.up1.forward(&u1)?.relu()?;
        let c1 = Tensor::cat(&[&u1, &e1], 1)?; // 跳跃连接
        let d1 = self.dec1.forward(&c1)?.relu()?;
        self.head.forward(&d1) // [N,2,SZ,SZ]
    }
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

// 逐像素 2 类交叉熵（同第 49 章：摊平成 [N*H*W, 2] + U32 目标）
fn pixel_ce(logits: &Tensor, target_u32: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = logits.dims4()?;
    let flat = logits.permute((0, 2, 3, 1))?.contiguous()?.reshape((n * h * w, c))?;
    let tgt = target_u32.reshape((n * h * w,))?;
    candle_nn::loss::cross_entropy(&flat, &tgt)
}

// 造点条件训练/测试数据：每个样本 = 一张场景图 + 随机选一个物体做点提示 → 目标是该物体的掩码。
// 返回 (输入张量 [N,4,SZ,SZ], 目标张量 U32 [N,SZ,SZ])
fn make_prompt_dataset(rng: &mut Lcg, n: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let mut xs = Vec::with_capacity(n * 4 * SZ * SZ);
    let mut ys = Vec::with_capacity(n * SZ * SZ);
    for _ in 0..n {
        let (img, objs) = make_scene(rng);
        let k = rng.range(0, objs.len() as i32) as usize; // 随机选一个物体
        let o = &objs[k];
        // 点提示落在该物体质心附近（加少许抖动，模拟用户点击不精确）
        let jx = (o.cx + rng.range(-1, 2)).clamp(0, SZ as i32 - 1);
        let jy = (o.cy + rng.range(-1, 2)).clamp(0, SZ as i32 - 1);
        xs.extend_from_slice(&make_input(&img, jx, jy));
        ys.extend(o.mask.iter().map(|&b| if b { 1u32 } else { 0u32 }));
    }
    let x = Tensor::from_vec(xs, (n, 4, SZ, SZ), dev)?;
    let y = Tensor::from_vec(ys, (n, SZ, SZ), dev)?;
    Ok((x, y))
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ========================= 第 A 部分：经典种子区域生长 =========================
    println!("========== A. 经典交互式分割：种子区域生长（点提示 / 框提示）==========");
    let mut rng = Lcg::new(100);
    let (img, objs) = make_scene(&mut rng);
    println!("合成场景：{}x{} 图，{} 个物体。", SZ, SZ, objs.len());

    // (1) 点提示：分别点在每个物体质心上 → 得到各自的掩码，报像素数与 IoU（阈值固定 70^2）
    let thr2 = 70.0f32 * 70.0;
    println!("\n[点提示] 阈值 = 70（颜色平方距离 < 70² 视为同物体），逐个物体点击：");
    let mut iou_sum = 0f32;
    for (k, o) in objs.iter().enumerate() {
        let mask = region_grow(&img, (o.cx, o.cy), thr2, None);
        let iou = iou_mask(&mask, &o.mask);
        iou_sum += iou;
        println!(
            "  物体#{k} 点({:>2},{:>2})：分割 {:>3} 像素（GT {:>3}），IoU = {:.3}",
            o.cx,
            o.cy,
            count(&mask),
            count(&o.mask),
            iou
        );
        if k < 3 {
            save_triptych(&img, (o.cx, o.cy), &mask, &o.mask, &format!("grow_obj{k}.png"));
        }
    }
    println!("  → 平均 IoU = {:.3}（点在哪个物体上，就分割出哪个 → 这就是“点提示→掩码”）", iou_sum / objs.len() as f32);

    // (2) 阈值敏感性：同一个点，阈值太紧→欠分割，太松→漏进背景（真实数字）
    let o0 = &objs[0];
    println!("\n[阈值敏感性] 固定点在物体#0 的质心({},{})，改变阈值：", o0.cx, o0.cy);
    for &t in &[15.0f32, 40.0, 70.0, 150.0, 300.0] {
        let mask = region_grow(&img, (o0.cx, o0.cy), t * t, None);
        let iou = iou_mask(&mask, &o0.mask);
        let tag = if count(&mask) > count(&o0.mask) * 3 { "（漏进背景/相邻物体）" } else if iou < 0.7 { "（欠分割）" } else { "（刚好）" };
        println!("  阈值={:>4.0}：分割 {:>4} 像素，IoU={:.3} {tag}", t, count(&mask), iou);
    }
    println!("  → 交互式分割的“提示工程”：阈值这个旋钮太紧漏掉目标、太松粘连背景，得调。");

    // (3) 框提示：给一个把物体#0 圈住的框，即使阈值很松也不会漏出去（框约束了生长范围）
    let b = o0.bbox;
    let mask_loose = region_grow(&img, (o0.cx, o0.cy), 300.0 * 300.0, None); // 无框：松阈值漏了
    let mask_box = region_grow(&img, (o0.cx, o0.cy), 300.0 * 300.0, Some(b)); // 有框：被框住
    println!("\n[框提示] 同样的松阈值(300)，对比“无框” vs “框住物体#0”：");
    println!("  无框：分割 {:>4} 像素，IoU={:.3}", count(&mask_loose), iou_mask(&mask_loose, &o0.mask));
    println!("  框内：分割 {:>4} 像素，IoU={:.3}", count(&mask_box), iou_mask(&mask_box, &o0.mask));
    println!("  → 框提示（box prompt）用空间约束兜住了区域生长，比裸点提示更稳——这正是 SAM 支持点/框多种提示的动机。");

    // ========================= 第 B 部分：迷你 SAM（点条件分割网络） =========================
    println!("\n========== B. 迷你 SAM：用 candle 训一个“点条件”分割小网络 ==========");
    let mut drng = Lcg::new(2024);
    let n_train = 128;
    let n_test = 48;
    let (xtr, ytr) = make_prompt_dataset(&mut drng, n_train, &dev)?;
    let (xte, yte) = make_prompt_dataset(&mut drng, n_test, &dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = PromptSeg::load(vb)?;
    det_init(&vm, 77)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    let epochs = 120;
    println!("训练点条件分割网络（输入=RGB+点热力图，输出=该点物体掩码，AdamW lr=2e-3→5e-4）...");
    for epoch in 1..=epochs {
        if epoch == epochs * 3 / 4 {
            opt.set_learning_rate(5e-4);
        }
        let logits = net.forward(&xtr)?;
        let loss = pixel_ce(&logits, &ytr)?;
        opt.backward_step(&loss)?;
        if epoch % 20 == 0 || epoch == 1 {
            println!("  epoch {epoch:>3}  loss = {:.4}", loss.to_scalar::<f32>()?);
        }
    }

    // 存 + 加载（与加载真实 SAM safetensors 是同一套 API，只是模型大小天差地别）
    vm.save("promptseg.safetensors")?;
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["promptseg.safetensors"], DType::F32, &dev)? };
    let model = PromptSeg::load(vb2)?;

    // 测试集评估：前景（掩码）IoU + 逐像素准确率
    let logits = model.forward(&xte)?;
    let gt = yte.to_vec3::<u32>()?; // [N][SZ][SZ]
    let (mut inter, mut uni, mut correct, mut total) = (0usize, 0usize, 0usize, 0usize);
    for k in 0..n_test {
        let pred = logits.i(k)?.argmax(0)?.to_vec2::<u32>()?; // [SZ][SZ]，0/1
        for y in 0..SZ {
            for x in 0..SZ {
                let p = pred[y][x] == 1;
                let g = gt[k][y][x] == 1;
                if p && g {
                    inter += 1;
                }
                if p || g {
                    uni += 1;
                }
                if (pred[y][x]) == gt[k][y][x] {
                    correct += 1;
                }
                total += 1;
            }
        }
    }
    println!("\n===== 测试集评估（{} 个点提示样本）=====", n_test);
    println!("逐像素准确率 = {:.2}%", 100.0 * correct as f32 / total as f32);
    println!("前景掩码 IoU = {:.3}", inter as f32 / uni as f32);

    // 关键演示：同一张图、点不同物体 → 网络给出不同掩码（真正的“可提示”行为）
    println!("\n[可提示演示] 同一张图，把点提示放到不同物体上，看网络输出的掩码怎么变：");
    let mut srng = Lcg::new(555);
    let (scene, sobjs) = make_scene(&mut srng);
    for (k, o) in sobjs.iter().enumerate().take(3) {
        let input = Tensor::from_vec(make_input(&scene, o.cx, o.cy), (1, 4, SZ, SZ), &dev)?;
        let pred = model.forward(&input)?.i(0)?.argmax(0)?.to_vec2::<u32>()?;
        let pred_flat: Vec<bool> = pred.iter().flatten().map(|&v| v == 1).collect();
        let iou = iou_mask(&pred_flat, &o.mask);
        println!(
            "  点在物体#{k}({:>2},{:>2})：网络分割 {:>3} 像素（GT {:>3}），IoU = {:.3}",
            o.cx,
            o.cy,
            count(&pred_flat),
            count(&o.mask),
            iou
        );
        save_triptych(&scene, (o.cx, o.cy), &pred_flat, &o.mask, &format!("sam_obj{k}.png"));
    }
    println!("  → 同图不同点得到不同掩码：点提示真正“条件”住了输出。这就是 SAM 可提示分割的核心直觉。");
    println!("\n已存 PNG：grow_obj*.png（区域生长）、sam_obj*.png（迷你 SAM），左=输入+点，中=分割，右=GT。");
    Ok(())
}
