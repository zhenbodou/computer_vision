// 第 101 章 项目：人脸考勤 / 门禁 Demo（端到端集成）
// candle 0.11.0，纯 CPU，无需下载任何模型；结果可复现。
// 运行：cd code/projects/ch101_face_attendance && cargo run --release
//
// 把前面三章串成一条落地流水线：
//   摄像头帧 → ①人脸检测(ch56, 框+5关键点) → ②五点对齐(裁剪归一化)
//            → ③活体闸门(ch58, 假体直接拒) → ④识别比对(ch57 ArcFace, 与注册库余弦相似度+阈值)
//            → ⑤决策：放行打卡 / 陌生人拒绝 / 活体拦截
// 报真实数字：注册数、检测质量、识别准确率、陌生人拒绝率、假体(打印/翻拍)拒绝率、决策示例。

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::loss::cross_entropy;
use candle_nn::ops::sigmoid;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{conv2d, linear, linear_no_bias, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use image::{Rgb, RgbImage};

const IMG_F: usize = 48; // 摄像头帧边长
const G: usize = 6; // 检测网格
const STRIDE: usize = IMG_F / G; // 8
const NL: usize = 5; // 关键点数
const A: usize = 1 + 4 + 2 * NL; // 检测头每格属性数 = 15
const IMG_C: usize = 32; // 对齐后人脸裁剪边长
const EMB: usize = 32; // 人脸 embedding 维度

// ================= 确定性 RNG（关键坑：LCG 低位退化，离散选择必须用高位） =================
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
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 // 取高位，低位质量差
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
    fn boolean(&mut self) -> bool {
        self.unit() < 0.5 // 不要用 next()&1（LCG bit0 会严格交替）
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.unit() * n as f32) as u32).min(n - 1) // 不要用 next()%n（低位退化）
    }
}

fn clampf(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

// ================= 身份与人脸渲染 =================
#[derive(Clone, Copy)]
struct Id {
    r: f32,       // 肤色三通道（直接给 RGB，制造明显“色相”差异——对齐/尺度归一化后仍保留，是强身份线索）
    g: f32,
    b: f32,
    aspect: f32,  // 脸高/宽
    eye_dx: f32,  // 双眼间距（占半宽）
    eye_dy: f32,  // 眼在中心上方多少（占半高）
    eye_r: f32,   // 眼半径
    mouth_w: f32, // 嘴半宽
    brow: f32,    // 眉毛粗细（0=无眉，最多约 2.5px）
}
fn sample_id(rng: &mut Lcg) -> Id {
    let base = rng.uniform(150.0, 232.0);
    Id {
        r: base,
        g: base - rng.uniform(12.0, 72.0),  // 不同暖度
        b: base - rng.uniform(28.0, 118.0), // 更大范围 → 色相各异
        aspect: rng.uniform(1.08, 1.52),
        eye_dx: rng.uniform(0.32, 0.46),
        eye_dy: rng.uniform(0.14, 0.32),
        eye_r: rng.uniform(0.09, 0.20),
        mouth_w: rng.uniform(0.18, 0.40),
        brow: rng.uniform(0.0, 2.5),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Live {
    Real,
    Print,
    Screen,
}

fn set_px(img: &mut RgbImage, x: i32, y: i32, c: Rgb<u8>) {
    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, c);
    }
}

// 在 img 上以 (cx,cy) 为中心、半宽 a 画一张脸，返回 5 关键点(交错 x,y 绝对坐标) 与紧致框
fn draw_face(img: &mut RgbImage, cx: f32, cy: f32, a: f32, id: &Id, live: Live, rng: &mut Lcg) -> ([f32; 2 * NL], [f32; 4]) {
    let b = a * id.aspect;
    let ramp = live != Live::Print; // 打印照片扁平无 3D 明暗
    let (lx, ly) = (rng.uniform(-1.0, 1.0), rng.uniform(-0.6, 1.0));
    let shade_k = rng.uniform(28.0, 50.0);
    let bright = rng.uniform(-7.0, 7.0); // 类内全局亮度扰动（各通道同量 → 不改色相）
    // 椭圆脸（每通道用身份肤色 + 明暗 + 亮度扰动）
    let x0 = (cx - a).floor() as i32;
    let x1 = (cx + a).ceil() as i32;
    let y0 = (cy - b).floor() as i32;
    let y1 = (cy + b).ceil() as i32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (dx, dy) = ((x as f32 - cx) / a, (y as f32 - cy) / b);
            if dx * dx + dy * dy <= 1.0 {
                let off = bright + if ramp { shade_k * (dx * lx + dy * ly) } else { 0.0 };
                set_px(img, x, y, Rgb([clampf(id.r + off), clampf(id.g + off), clampf(id.b + off)]));
            }
        }
    }
    // 关键点：左眼/右眼/鼻/左嘴角/右嘴角
    let le = (cx - id.eye_dx * a, cy - id.eye_dy * b);
    let re = (cx + id.eye_dx * a, cy - id.eye_dy * b);
    let no = (cx, cy + 0.06 * b);
    let lm = (cx - id.mouth_w * a, cy + 0.46 * b);
    let rm = (cx + id.mouth_w * a, cy + 0.46 * b);
    let dark = Rgb([28u8, 26, 30]);
    let disk = |img: &mut RgbImage, c: (f32, f32), r: f32| {
        let rr = r.max(0.8);
        for y in (c.1 - rr).floor() as i32..=(c.1 + rr).ceil() as i32 {
            for x in (c.0 - rr).floor() as i32..=(c.0 + rr).ceil() as i32 {
                if (x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2) <= rr * rr {
                    set_px(img, x, y, dark);
                }
            }
        }
    };
    disk(img, le, id.eye_r * a);
    disk(img, re, id.eye_r * a);
    disk(img, no, 0.08 * a);
    // 眉毛：每只眼上方一条深色横条，粗细由身份决定（对齐不变的强身份线索）
    if id.brow > 0.4 {
        let bw = id.eye_r * a + 1.0; // 眉宽略大于眼
        let by = le.1 - id.eye_r * a - 1.5; // 眼上方
        for (ex, _) in [le, re] {
            for t in 0..id.brow.round() as i32 {
                for x in (ex - bw).round() as i32..=(ex + bw).round() as i32 {
                    set_px(img, x, by.round() as i32 - t, dark);
                }
            }
        }
    }
    // 嘴：一条线
    let steps = ((rm.0 - lm.0).abs()).round() as i32;
    for s in 0..=steps.max(1) {
        let t = s as f32 / steps.max(1) as f32;
        let x = lm.0 + t * (rm.0 - lm.0);
        set_px(img, x.round() as i32, lm.1.round() as i32, dark);
        set_px(img, x.round() as i32, lm.1.round() as i32 + 1, dark);
    }
    // 打印：压对比（扁平）；屏幕：叠摩尔纹
    let bb = [
        (cx - a).max(0.0) as i32,
        (cy - b).max(0.0) as i32,
        (cx + a).min(img.width() as f32) as i32,
        (cy + b).min(img.height() as f32) as i32,
    ];
    if live == Live::Print {
        for y in bb[1]..bb[3] {
            for x in bb[0]..bb[2] {
                let p = img.get_pixel(x as u32, y as u32).0;
                let f = |c: u8| clampf(128.0 + (c as f32 - 128.0) * 0.55);
                img.put_pixel(x as u32, y as u32, Rgb([f(p[0]), f(p[1]), f(p[2])]));
            }
        }
    } else if live == Live::Screen {
        let (fx, fy, ph, amp) = (rng.uniform(5.0, 10.0), rng.uniform(5.0, 10.0), rng.uniform(0.0, 6.28), rng.uniform(0.10, 0.17));
        for y in bb[1]..bb[3] {
            for x in bb[0]..bb[2] {
                let m = 1.0 + amp * (2.0 * std::f32::consts::PI * (fx * x as f32 + fy * y as f32) / IMG_C as f32 + ph).sin();
                let p = img.get_pixel(x as u32, y as u32).0;
                img.put_pixel(x as u32, y as u32, Rgb([clampf(p[0] as f32 * m), clampf(p[1] as f32 * m), clampf(p[2] as f32 * m + 5.0)]));
            }
        }
    }
    // 细纹理噪声
    let tex = match live {
        Live::Real => 6.0,
        Live::Print => 3.0,
        Live::Screen => 5.0,
    };
    for y in bb[1]..bb[3] {
        for x in bb[0]..bb[2] {
            let n = rng.uniform(-tex, tex);
            let p = img.get_pixel(x as u32, y as u32).0;
            img.put_pixel(x as u32, y as u32, Rgb([clampf(p[0] as f32 + n), clampf(p[1] as f32 + n), clampf(p[2] as f32 + n)]));
        }
    }
    let lmk = [le.0, le.1, re.0, re.1, no.0, no.1, lm.0, lm.1, rm.0, rm.1];
    let boxf = [cx - a, cy - b, cx + a, cy + b];
    (lmk, boxf)
}

// 渲染一整帧 48x48：背景噪声 + 一张随机位置/尺度的脸
fn render_frame(id: &Id, live: Live, rng: &mut Lcg) -> (RgbImage, [f32; 2 * NL], [f32; 4]) {
    let mut img = RgbImage::new(IMG_F as u32, IMG_F as u32);
    let bg = rng.uniform(40.0, 92.0);
    for y in 0..IMG_F as u32 {
        for x in 0..IMG_F as u32 {
            let v = clampf(bg + rng.uniform(-8.0, 8.0));
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    let a = rng.uniform(8.0, 11.0);
    let b = a * id.aspect;
    let (cx, cy) = (rng.uniform(a + 3.0, IMG_F as f32 - a - 3.0), rng.uniform(b + 3.0, IMG_F as f32 - b - 3.0));
    let (lmk, boxf) = draw_face(&mut img, cx, cy, a, id, live, rng);
    (img, lmk, boxf)
}

// ================= 五点对齐：用双眼关键点归一化位置与尺度，裁 32x32 =================
fn sample_clamped(img: &RgbImage, sx: f32, sy: f32) -> Rgb<u8> {
    let x = (sx.round() as i32).clamp(0, img.width() as i32 - 1) as u32;
    let y = (sy.round() as i32).clamp(0, img.height() as i32 - 1) as u32;
    *img.get_pixel(x, y)
}
// 5 点相似变换对齐（ch54 的做法）：把检测到的 5 关键点最小二乘对齐到 32x32 的一组
// 标准位置，得到 scale+旋转+平移。用全部 5 点 → 平均掉单点噪声，比只用双眼瞳距稳得多。
const CANON: [(f32, f32); NL] = [(11.0, 12.5), (21.0, 12.5), (16.0, 18.0), (12.5, 23.0), (19.5, 23.0)];
fn align_crop(frame: &RgbImage, lmk: &[f32; 2 * NL]) -> RgbImage {
    // 源点(检测关键点) src → 目标点(标准位置) dst 的相似变换：dst = M(src-ms)+md
    let (mut msx, mut msy, mut mdx, mut mdy) = (0f32, 0f32, 0f32, 0f32);
    for k in 0..NL {
        msx += lmk[2 * k];
        msy += lmk[2 * k + 1];
        mdx += CANON[k].0;
        mdy += CANON[k].1;
    }
    let nf = NL as f32;
    let (msx, msy, mdx, mdy) = (msx / nf, msy / nf, mdx / nf, mdy / nf);
    let (mut a, mut bb, mut ss) = (0f32, 0f32, 0f32);
    for k in 0..NL {
        let (sx, sy) = (lmk[2 * k] - msx, lmk[2 * k + 1] - msy);
        let (dx, dy) = (CANON[k].0 - mdx, CANON[k].1 - mdy);
        a += sx * dx + sy * dy; // 点积
        bb += sx * dy - sy * dx; // 叉积（估计旋转）
        ss += sx * sx + sy * sy;
    }
    let ss = ss.max(1e-6);
    let (p, q) = (a / ss, bb / ss); // p=scale·cosθ, q=scale·sinθ
    let det = (p * p + q * q).max(1e-6);
    // 逆变换 (crop 像素 → frame 坐标)：src = M^{-1}(dst-md)+ms，M^{-1}=1/det[[p,q],[-q,p]]
    let mut crop = RgbImage::new(IMG_C as u32, IMG_C as u32);
    for j in 0..IMG_C {
        for i in 0..IMG_C {
            let (ux, uy) = (i as f32 - mdx, j as f32 - mdy);
            let fx = (p * ux + q * uy) / det + msx;
            let fy = (-q * ux + p * uy) / det + msy;
            crop.put_pixel(i as u32, j as u32, sample_clamped(frame, fx, fy));
        }
    }
    crop
}

fn chw(img: &RgbImage) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * h * w + y * w + x] = p[c] as f32 / 255.0;
            }
        }
    }
    data
}
fn batch(imgs: &[RgbImage], sz: usize, dev: &Device) -> Result<Tensor> {
    let mut xs = Vec::with_capacity(imgs.len() * 3 * sz * sz);
    for im in imgs {
        xs.extend_from_slice(&chw(im));
    }
    Tensor::from_vec(xs, (imgs.len(), 3, sz, sz), dev)
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
fn bce_elem(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let relu_x = logits.relu()?;
    let xy = logits.broadcast_mul(targets)?;
    let sp = logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    relu_x.broadcast_sub(&xy)?.broadcast_add(&sp)
}
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ================= ① 人脸检测器（ch56 网格检测器：obj + 框 + 5 关键点） =================
struct Detector {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    head: Conv2d,
}
impl Detector {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let cfg0 = Conv2dConfig { padding: 0, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
            head: conv2d(64, A, 1, cfg0, vb.pp("head"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?;
        self.head.forward(&x)
    }
}
#[derive(Clone)]
struct Det {
    score: f32,
    xyxy: [f32; 4],
    lmk: [f32; 2 * NL],
}
fn decode(out: &Tensor, thr: f32) -> Result<Vec<Det>> {
    let v = out.i(0)?.to_vec3::<f32>()?; // [A][G][G]
    let mut dets = Vec::new();
    for i in 0..G {
        for j in 0..G {
            let score = sigmoidf(v[0][i][j]);
            if score < thr {
                continue;
            }
            let cx = (j as f32 + sigmoidf(v[1][i][j])) * STRIDE as f32;
            let cy = (i as f32 + sigmoidf(v[2][i][j])) * STRIDE as f32;
            let w = sigmoidf(v[3][i][j]) * IMG_F as f32;
            let h = sigmoidf(v[4][i][j]) * IMG_F as f32;
            let mut lmk = [0f32; 2 * NL];
            for k in 0..NL {
                lmk[2 * k] = ((j as f32 + 0.5) / G as f32 + v[5 + k][i][j]) * IMG_F as f32;
                lmk[2 * k + 1] = ((i as f32 + 0.5) / G as f32 + v[5 + NL + k][i][j]) * IMG_F as f32;
            }
            dets.push(Det { score, xyxy: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], lmk });
        }
    }
    Ok(dets)
}
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ua = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let ub = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let u = ua + ub - inter;
    if u <= 0.0 { 0.0 } else { inter / u }
}
// 门禁场景每帧一张脸：取 objectness 最高的检出
fn detect_best(model: &Detector, frame: &RgbImage, dev: &Device, thr: f32) -> Result<Option<Det>> {
    let x = Tensor::from_vec(chw(frame), (1, 3, IMG_F, IMG_F), dev)?;
    let out = model.forward(&x)?;
    let dets = decode(&out, thr)?;
    Ok(dets.into_iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()))
}

// 关键工程点：识别/活体消费的是“检测→对齐”的真实产物，训练也必须走这条链，
// 否则用真值关键点裁剪训练、却用检测关键点裁剪推理 → 级联域偏移，识别相似度崩塌。
fn crop_via_detect(det: &Detector, frame: &RgbImage, dev: &Device) -> Result<Option<RgbImage>> {
    Ok(detect_best(det, frame, dev, 0.5)?.map(|d| align_crop(frame, &d.lmk)))
}
// 反复渲染 + 检测对齐，收集 per 张成功裁剪的人脸 crop（检测偶有失败则跳过）
fn collect(det: &Detector, id: &Id, live: Live, per: usize, rng: &mut Lcg, dev: &Device) -> Result<Vec<RgbImage>> {
    let mut v = Vec::new();
    let mut tries = 0;
    while v.len() < per && tries < per * 3 {
        tries += 1;
        let (f, _, _) = render_frame(id, live, rng);
        if let Some(c) = crop_via_detect(det, &f, dev)? {
            v.push(c);
        }
    }
    Ok(v)
}

// 训练检测器（合成真人帧）
fn train_detector(dev: &Device) -> Result<Detector> {
    let (pos_w, l_box, l_lmk) = (8.0f32, 5.0f64, 4.0f64);
    let mut rng = Lcg::new(7);
    let n = 280;
    let mut frames = Vec::new();
    let mut boxes = Vec::new();
    let mut lmks = Vec::new();
    for _ in 0..n {
        let id = sample_id(&mut rng);
        let (f, l, bx) = render_frame(&id, Live::Real, &mut rng);
        frames.push(f);
        boxes.push(bx);
        lmks.push(l);
    }
    let xtr = batch(&frames, IMG_F, dev)?;
    // 目标张量
    let (mut obj, mut wgt) = (vec![0f32; n * G * G], vec![1f32; n * G * G]);
    let mut boxn = vec![0f32; n * 4 * G * G];
    let mut lmkn = vec![0f32; n * 2 * NL * G * G];
    let mut num_pos = 0f32;
    for k in 0..n {
        let bx = boxes[k];
        let (cx, cy) = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0);
        let (w, h) = (bx[2] - bx[0], bx[3] - bx[1]);
        let (ci, cj) = (((cy / STRIDE as f32) as usize).min(G - 1), ((cx / STRIDE as f32) as usize).min(G - 1));
        let cell = ci * G + cj;
        obj[k * G * G + cell] = 1.0;
        wgt[k * G * G + cell] = pos_w;
        num_pos += 1.0;
        let comps = [cx / IMG_F as f32, cy / IMG_F as f32, w / IMG_F as f32, h / IMG_F as f32];
        for c in 0..4 {
            boxn[(k * 4 + c) * G * G + cell] = comps[c];
        }
        for p in 0..NL {
            lmkn[(k * 2 * NL + p) * G * G + cell] = lmks[k][2 * p] / IMG_F as f32;
            lmkn[(k * 2 * NL + NL + p) * G * G + cell] = lmks[k][2 * p + 1] / IMG_F as f32;
        }
    }
    let obj_t = Tensor::from_vec(obj, (n, 1, G, G), dev)?;
    let wgt_t = Tensor::from_vec(wgt, (n, 1, G, G), dev)?;
    let boxn_t = Tensor::from_vec(boxn, (n, 4, G, G), dev)?;
    let lmkn_t = Tensor::from_vec(lmkn, (n, 2 * NL, G, G), dev)?;
    let (mut cxv, mut cyv) = (vec![0f32; G * G], vec![0f32; G * G]);
    for i in 0..G {
        for j in 0..G {
            cxv[i * G + j] = j as f32;
            cyv[i * G + j] = i as f32;
        }
    }
    let cell_x = Tensor::from_vec(cxv, (1, 1, G, G), dev)?;
    let cell_y = Tensor::from_vec(cyv, (1, 1, G, G), dev)?;

    let vm = VarMap::new();
    let net = Detector::load(VarBuilder::from_varmap(&vm, DType::F32, dev))?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    println!("① 训练人脸检测器（{n} 帧, {} 轮）...", 160);
    for epoch in 1..=160 {
        if epoch == 120 {
            opt.set_learning_rate(4e-4);
        }
        let out = net.forward(&xtr)?;
        let obj_logit = out.narrow(1, 0, 1)?;
        let box_raw = out.narrow(1, 1, 4)?;
        let lmk_x = out.narrow(1, 5, NL)?;
        let lmk_y = out.narrow(1, 5 + NL, NL)?;
        let lo = bce_elem(&obj_logit, &obj_t)?.broadcast_mul(&wgt_t)?.sum_all()?.affine(1.0 / (n * G * G) as f64, 0.0)?;
        let sig = sigmoid(&box_raw)?;
        let cxp = sig.narrow(1, 0, 1)?.broadcast_add(&cell_x)?.affine(1.0 / G as f64, 0.0)?;
        let cyp = sig.narrow(1, 1, 1)?.broadcast_add(&cell_y)?.affine(1.0 / G as f64, 0.0)?;
        let pred_box = Tensor::cat(&[&cxp, &cyp, &sig.narrow(1, 2, 1)?, &sig.narrow(1, 3, 1)?], 1)?;
        let lb = pred_box.broadcast_sub(&boxn_t)?.abs()?.broadcast_mul(&obj_t)?.sum_all()?.affine(1.0 / (num_pos * 4.0) as f64, 0.0)?;
        let ccx = cell_x.affine(1.0 / G as f64, 0.5 / G as f64)?;
        let ccy = cell_y.affine(1.0 / G as f64, 0.5 / G as f64)?;
        let pred_lmk = Tensor::cat(&[&lmk_x.broadcast_add(&ccx)?, &lmk_y.broadcast_add(&ccy)?], 1)?;
        let ll = pred_lmk.broadcast_sub(&lmkn_t)?.abs()?.broadcast_mul(&obj_t)?.sum_all()?.affine(1.0 / (num_pos * 2.0 * NL as f32) as f64, 0.0)?;
        let loss = lo.broadcast_add(&lb.affine(l_box, 0.0)?)?.broadcast_add(&ll.affine(l_lmk, 0.0)?)?;
        opt.backward_step(&loss)?;
        if epoch % 65 == 0 {
            println!("   epoch {epoch:>3}  loss={:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok(net)
}

// ================= 32x32 骨干（识别 / 活体共用同一结构，各自独立实例） =================
struct Trunk {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
}
impl Trunk {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, cfg, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, cfg, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, cfg, vb.pp("conv3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?;
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?;
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?;
        x.flatten_from(1) // [N, 64*4*4]
    }
}
fn l2norm(t: &Tensor) -> Result<Tensor> {
    let n = t.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-12)?.sqrt()?;
    t.broadcast_div(&n)
}

// ③ 识别：ArcFace embedding
struct EmbNet {
    trunk: Trunk,
    fc: Linear,
}
impl EmbNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self { trunk: Trunk::load(vb.pp("trunk"))?, fc: linear(64 * 4 * 4, EMB, vb.pp("fc"))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc.forward(&self.trunk.forward(x)?)
    }
}
fn arcface_logits(cos: &Tensor, one_hot: &Tensor, s: f64, m: f64) -> Result<Tensor> {
    let (cm, sm) = (m.cos(), m.sin());
    let sin = cos.sqr()?.affine(-1.0, 1.0)?.relu()?.sqrt()?;
    let phi = cos.affine(cm, 0.0)?.broadcast_sub(&sin.affine(sm, 0.0)?)?;
    let diff = phi.broadcast_sub(cos)?;
    cos.broadcast_add(&one_hot.broadcast_mul(&diff)?)?.affine(s, 0.0)
}
fn embed(model: &EmbNet, crops: &[RgbImage], dev: &Device) -> Result<Vec<Vec<f32>>> {
    let x = batch(crops, IMG_C, dev)?;
    l2norm(&model.forward(&x)?)?.to_vec2::<f32>()
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// 训练 embedding（ArcFace，训练身份用“检测→对齐”得到的真人 crop，与推理同分布）
fn train_embed(detector: &Detector, train_ids: &[Id], dev: &Device) -> Result<EmbNet> {
    let per = 24;
    let n_id = train_ids.len();
    let mut rng = Lcg::new(100);
    let mut crops = Vec::new();
    let mut ys = Vec::new();
    for (label, id) in train_ids.iter().enumerate() {
        for c in collect(detector, id, Live::Real, per, &mut rng, dev)? {
            crops.push(c);
            ys.push(label as u32);
        }
    }
    let n = ys.len();
    let x = batch(&crops, IMG_C, dev)?;
    let yt = Tensor::from_vec(ys.clone(), (n,), dev)?;
    let mut oh = vec![0f32; n * n_id];
    for (i, &l) in ys.iter().enumerate() {
        oh[i * n_id + l as usize] = 1.0;
    }
    let one_hot = Tensor::from_vec(oh, (n, n_id), dev)?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let net = EmbNet::load(vb.pp("emb"))?;
    let arc = linear_no_bias(EMB, n_id, vb.pp("arc"))?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("③ 训练 ArcFace 识别 embedding（{n_id} 训练身份, 共 {n} 张检测对齐 crop, 150 轮）...");
    for epoch in 1..=150 {
        let f = l2norm(&net.forward(&x)?)?;
        let w = l2norm(arc.weight())?;
        let cos = f.matmul(&w.t()?)?;
        let loss = cross_entropy(&arcface_logits(&cos, &one_hot, 10.0, 0.50)?, &yt)?;
        opt.backward_step(&loss)?;
        if epoch % 75 == 0 {
            println!("   epoch {epoch:>3}  loss={:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok(net)
}

// ② 活体：真/假二分类
struct LiveNet {
    trunk: Trunk,
    fc: Linear,
    head: Linear,
}
impl LiveNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            trunk: Trunk::load(vb.pp("trunk"))?,
            fc: linear(64 * 4 * 4, 64, vb.pp("fc"))?,
            head: linear(64, 2, vb.pp("head"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.head.forward(&self.fc.forward(&self.trunk.forward(x)?)?.relu()?)
    }
}
// 返回 true=判为真人
fn is_live(model: &LiveNet, crop: &RgbImage, dev: &Device) -> Result<bool> {
    let x = Tensor::from_vec(chw(crop), (1, 3, IMG_C, IMG_C), dev)?;
    let pred = model.forward(&x)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
    Ok(pred[0] == 1) // 标签 1=真人
}
fn train_live(detector: &Detector, pool_ids: &[Id], dev: &Device) -> Result<LiveNet> {
    let target = 480;
    let mut rng = Lcg::new(211);
    let mut crops = Vec::new();
    let mut ys = Vec::new();
    let mut tries = 0;
    while crops.len() < target && tries < target * 3 {
        tries += 1;
        let id = pool_ids[rng.below(pool_ids.len() as u32) as usize];
        let r = rng.below(4);
        let live = if r < 2 { Live::Real } else if r == 2 { Live::Print } else { Live::Screen };
        let (f, _, _) = render_frame(&id, live, &mut rng);
        if let Some(c) = crop_via_detect(detector, &f, dev)? {
            crops.push(c);
            ys.push(if live == Live::Real { 1u32 } else { 0 });
        }
    }
    let x = batch(&crops, IMG_C, dev)?;
    let yt = Tensor::from_vec(ys.clone(), (ys.len(),), dev)?;
    let vm = VarMap::new();
    let net = LiveNet::load(VarBuilder::from_varmap(&vm, DType::F32, dev).pp("live"))?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 1e-3, ..Default::default() })?;
    println!("② 训练活体检测器（{} 张检测对齐 crop, 150 轮）...", crops.len());
    for epoch in 1..=150 {
        let loss = cross_entropy(&net.forward(&x)?, &yt)?;
        opt.backward_step(&loss)?;
        if epoch % 75 == 0 {
            println!("   epoch {epoch:>3}  loss={:.4}", loss.to_scalar::<f32>()?);
        }
    }
    Ok(net)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    println!("===== 人脸考勤 / 门禁 Demo：检测 → 对齐 → 活体 → 识别 → 决策 =====\n");

    // ---- 训练三个模块（自包含、可复现）----
    let detector = train_detector(&dev)?;
    // 身份宇宙：前 16 个训练 embedding；后 12 个是“门禁世界”里的新面孔（embedding 从没见过）
    let mut irng = Lcg::new(2024);
    let all_ids: Vec<Id> = (0..28).map(|_| sample_id(&mut irng)).collect();
    let emb_train = &all_ids[..16];
    let world = &all_ids[16..]; // 12 个新身份：前 8 注册为员工，后 4 是陌生人
    let embnet = train_embed(&detector, emb_train, &dev)?;
    let livenet = train_live(&detector, emb_train, &dev)?;

    // ---- 检测阶段体检（几帧真人）----
    let mut drng = Lcg::new(555);
    let (mut det_ok, mut iou_sum, mut lmk_err, mut det_tot) = (0, 0f32, 0f32, 0);
    for _ in 0..40 {
        let id = world[drng.below(8) as usize];
        let (f, gl, gb) = render_frame(&id, Live::Real, &mut drng);
        det_tot += 1;
        if let Some(d) = detect_best(&detector, &f, &dev, 0.5)? {
            if iou(&d.xyxy, &gb) >= 0.5 {
                det_ok += 1;
                iou_sum += iou(&d.xyxy, &gb);
                let mut e = 0f32;
                for k in 0..NL {
                    e += ((d.lmk[2 * k] - gl[2 * k]).powi(2) + (d.lmk[2 * k + 1] - gl[2 * k + 1]).powi(2)).sqrt();
                }
                lmk_err += e / NL as f32;
            }
        }
    }
    println!("\n[检测体检] {det_ok}/{det_tot} 帧成功定位人脸，命中框平均 IoU={:.3}，关键点平均误差={:.2}px", iou_sum / det_ok as f32, lmk_err / det_ok as f32);

    // ---- 注册：8 名员工，每人 5 张登记照（走同一条检测→对齐链），取平均 embedding 作模板 ----
    const N_EMP: usize = 8;
    let mut grng = Lcg::new(4242);
    let mut gallery: Vec<Vec<f32>> = Vec::new(); // 员工 idx → 模板向量
    for e in 0..N_EMP {
        let shots = collect(&detector, &world[e], Live::Real, 5, &mut grng, &dev)?;
        let embs = embed(&embnet, &shots, &dev)?;
        let mut mean = vec![0f32; EMB];
        for v in &embs {
            for k in 0..EMB {
                mean[k] += v[k] / embs.len() as f32;
            }
        }
        let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for k in 0..EMB {
            mean[k] /= norm;
        }
        gallery.push(mean);
    }
    println!("[注册] 已登记 {N_EMP} 名员工（每人 5 张登记照取平均模板）；另有 4 个陌生人身份未登记");

    // ---- 定阈值：验证集同样走“检测→对齐”，选让(员工通过率+陌生人拒绝率)最高的阈值 ----
    let mut vrng = Lcg::new(818);
    let mut genuine_sim = Vec::new();
    let mut impostor_sim = Vec::new();
    for _ in 0..120 {
        // 员工真人 → 与本人模板的相似度（genuine）
        let e = vrng.below(N_EMP as u32) as usize;
        let (f, _, _) = render_frame(&world[e], Live::Real, &mut vrng);
        if let Some(c) = crop_via_detect(&detector, &f, &dev)? {
            genuine_sim.push(cosine(&embed(&embnet, &[c], &dev)?[0], &gallery[e]));
        }
        // 陌生人真人 → 与最相似模板的相似度（impostor，1:N 取最大）
        let s = 8 + vrng.below(4) as usize;
        let (f2, _, _) = render_frame(&world[s], Live::Real, &mut vrng);
        if let Some(c2) = crop_via_detect(&detector, &f2, &dev)? {
            let emb2 = &embed(&embnet, &[c2], &dev)?[0];
            impostor_sim.push((0..N_EMP).map(|k| cosine(emb2, &gallery[k])).fold(f32::MIN, f32::max));
        }
    }
    // 选让 (员工通过率 + 陌生人拒绝率) 最大的阈值（自然最佳工作点；实战应再往安全侧上移）
    let (mut best_thr, mut best_score) = (0.5f32, -1.0f32);
    let mut thr = 0.20f32;
    while thr <= 0.95 {
        let ta = genuine_sim.iter().filter(|&&v| v >= thr).count() as f32 / genuine_sim.len() as f32;
        let tr = impostor_sim.iter().filter(|&&v| v < thr).count() as f32 / impostor_sim.len() as f32;
        if ta + tr > best_score {
            best_score = ta + tr;
            best_thr = thr;
        }
        thr += 0.01;
    }
    let g_mean = genuine_sim.iter().sum::<f32>() / genuine_sim.len() as f32;
    let i_mean = impostor_sim.iter().sum::<f32>() / impostor_sim.len() as f32;
    let v_ta = genuine_sim.iter().filter(|&&v| v >= best_thr).count() as f32 / genuine_sim.len() as f32;
    let v_tr = impostor_sim.iter().filter(|&&v| v < best_thr).count() as f32 / impostor_sim.len() as f32;
    println!("[定阈值] 验证集 genuine 均值={g_mean:.3}、impostor(1:N最大) 均值={i_mean:.3}");
    println!("         选识别阈值 {best_thr:.2}：验证集员工通过率 {:.0}%、陌生人拒绝率 {:.0}%", 100.0 * v_ta, 100.0 * v_tr);

    // ---- 处理“摄像头帧流”：60 帧（30 员工真人 / 15 陌生人真人 / 15 员工假体）----
    println!("\n[实时决策] 处理 60 帧：30 员工真人、15 陌生人、15 假体(打印/翻拍)\n");
    let mut trng = Lcg::new(31337);
    #[derive(Clone, Copy, PartialEq)]
    enum Cat {
        Genuine,
        Stranger,
        Spoof,
    }
    let mut plan = Vec::new();
    for _ in 0..30 {
        plan.push((Cat::Genuine, trng.below(N_EMP as u32) as usize, Live::Real));
    }
    for _ in 0..15 {
        plan.push((Cat::Stranger, 8 + trng.below(4) as usize, Live::Real));
    }
    for _ in 0..15 {
        let sp = if trng.boolean() { Live::Print } else { Live::Screen };
        plan.push((Cat::Spoof, trng.below(N_EMP as u32) as usize, sp));
    }

    let (mut genuine_ok, mut stranger_ok, mut spoof_ok) = (0, 0, 0);
    let (mut genuine_n, mut stranger_n, mut spoof_n) = (0, 0, 0);
    let (mut spoof_live, mut spoof_nodet) = (0, 0); // 假体被拦：活体闸门 / 检测阶段
    let mut breach = 0; // 假体被放行 = 安全事故
    let mut shown = 0;
    for (cat, who, live) in &plan {
        // 每帧都要有决策，先计入该类别分母
        match cat {
            Cat::Genuine => genuine_n += 1,
            Cat::Stranger => stranger_n += 1,
            Cat::Spoof => spoof_n += 1,
        }
        let (frame, _gl, _gb) = render_frame(&world[*who], *live, &mut trng);
        let decision;
        let mut ok = false;
        let live_disp;
        match detect_best(&detector, &frame, &dev, 0.5)? {
            None => {
                // ①检测失败：没检出人脸 → 不放行（对陌生人/假体是安全的，对员工是漏检=误拒）
                decision = "未检出人脸(拒绝)".to_string();
                live_disp = "-";
                if *cat != Cat::Genuine {
                    ok = true;
                }
                if *cat == Cat::Spoof {
                    spoof_nodet += 1;
                }
            }
            Some(det) => {
                let crop = align_crop(&frame, &det.lmk); // ②五点对齐
                let live_ok = is_live(&livenet, &crop, &dev)?; // ③活体闸门
                live_disp = if live_ok { "真" } else { "假" };
                let emb = &embed(&embnet, &[crop], &dev)?[0]; // ④识别
                let (mut best_k, mut best_s) = (0usize, f32::MIN);
                for k in 0..N_EMP {
                    let s = cosine(emb, &gallery[k]);
                    if s > best_s {
                        best_s = s;
                        best_k = k;
                    }
                }
                // ⑤决策
                if !live_ok {
                    decision = "活体拦截(拒绝)".to_string();
                    if *cat == Cat::Spoof {
                        ok = true;
                        spoof_live += 1;
                    }
                } else if best_s >= best_thr {
                    decision = format!("放行·员工#{best_k}(sim={best_s:.2})");
                    if *cat == Cat::Genuine && best_k == *who {
                        ok = true;
                    }
                    if *cat == Cat::Spoof {
                        breach += 1; // 假体骗过活体又被识别为员工 → 放行 = 安全事故
                    }
                } else {
                    decision = format!("陌生人(拒绝, best_sim={best_s:.2})");
                    if *cat == Cat::Stranger {
                        ok = true;
                    }
                }
            }
        }
        match cat {
            Cat::Genuine => {
                if ok {
                    genuine_ok += 1;
                }
            }
            Cat::Stranger => {
                if ok {
                    stranger_ok += 1;
                }
            }
            Cat::Spoof => {
                if ok {
                    spoof_ok += 1;
                }
            }
        }
        if shown < 9 {
            let truth = match cat {
                Cat::Genuine => format!("员工#{who}真人"),
                Cat::Stranger => "陌生人真人".to_string(),
                Cat::Spoof => format!("员工#{who}的{}", if *live == Live::Print { "打印照" } else { "屏幕翻拍" }),
            };
            let mark = if ok { "✓" } else { "✗" };
            println!("  {mark} 真况={truth:<16} 活体={live_disp} → 决策：{decision}");
            shown += 1;
        }
    }

    println!("\n===== 汇总（真实数字，共 60 帧每帧一个门禁决策）=====");
    println!("注册员工数：{N_EMP}（另 4 个未注册身份充当陌生人）");
    println!("员工真人  正确放行且认对本人：{genuine_ok}/{genuine_n}（{:.1}%）", 100.0 * genuine_ok as f32 / genuine_n as f32);
    println!("陌生人    正确拒止(判陌生人/未检出)：{stranger_ok}/{stranger_n}（{:.1}%）", 100.0 * stranger_ok as f32 / stranger_n as f32);
    println!("假体攻击  正确拒止：{spoof_ok}/{spoof_n}（{:.1}%）— 其中活体拦截 {spoof_live}、检测阶段挡掉 {spoof_nodet}", 100.0 * spoof_ok as f32 / spoof_n as f32);
    println!("安全事故（假体骗过活体又被认成员工而放行）：{breach} 起");
    Ok(())
}
