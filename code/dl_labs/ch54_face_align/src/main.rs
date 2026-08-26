// 第 54 章：合成人脸上的 5 点关键点检测（直接坐标回归）+ 相似变换人脸对齐
// candle 0.11.0 + nalgebra 0.35，纯 CPU，可复现。
// 运行：cd code/dl_labs/ch54_face_align && cargo run --release
//
// 管线：合成人脸（椭圆脸+双眼+鼻+双嘴角，随机旋转/缩放/位置/噪声）
//       → 卷积骨干 + 全连接头 直接回归 5 点归一化坐标（MSE）
//       → 报定位误差 → 用 nalgebra 求相似变换（Umeyama）把 5 点对齐到标准模板
//       → 手写反向映射把人脸摆正 → 存“对齐前/后”PNG + 报旋转角/缩放/残差。

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, Optimizer, VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_cross_mut;
use nalgebra::{Matrix2, Vector2};

const IMG: usize = 48; // 输入边长
const K: usize = 5; // 关键点数：左眼 右眼 鼻 左嘴角 右嘴角
const LM_NAMES: [&str; K] = ["左眼", "右眼", "鼻", "左嘴角", "右嘴角"];
// 人脸局部坐标系下的规范 5 点（中心在原点，-y 朝上）
const CANON: [[f32; 2]; K] = [
    [-6.0, -5.0], // 左眼
    [6.0, -5.0],  // 右眼
    [0.0, 2.0],   // 鼻
    [-4.0, 8.0],  // 左嘴角
    [4.0, 8.0],   // 右嘴角
];
// 标准模板点（48x48 正脸框里的规范布局），对齐的目标
const TEMPLATE: [[f32; 2]; K] = [
    [18.0, 19.0],
    [30.0, 19.0],
    [24.0, 27.0],
    [20.0, 35.0],
    [28.0, 35.0],
];

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
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
    fn rangef(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
}

type Lms = [[f32; 2]; K];

// 合成一张人脸：随机相似位姿（旋转+缩放+平移）+ 关节小抖动 + 像素噪声。返回图 + 5 点真值。
fn make_face(rng: &mut Lcg) -> (RgbImage, Lms) {
    let s = rng.rangef(0.95, 1.4); // 缩放
    let theta = rng.rangef(-0.5, 0.5); // 旋转弧度（约 ±28°）
    let cx = rng.rangef(21.0, 27.0);
    let cy = rng.rangef(21.0, 27.0);
    let (ct, st) = (theta.cos(), theta.sin());
    let tf = |lx: f32, ly: f32| [cx + s * (ct * lx - st * ly), cy + s * (st * lx + ct * ly)];
    // 5 点（带 ±1px 形状抖动）
    let mut lms: Lms = [[0.0; 2]; K];
    for k in 0..K {
        lms[k] = tf(CANON[k][0] + rng.rangef(-1.0, 1.0), CANON[k][1] + rng.rangef(-1.0, 1.0));
    }
    // 画脸：把每个像素逆变换回局部坐标，判断是否在椭圆内 → 填肤色
    let mut img = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    let (a, b) = (12.0 * s, 15.0 * s); // 椭圆半轴（图像坐标下）
    for y in 0..IMG {
        for x in 0..IMG {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            // 逆旋转到局部
            let lx = ct * dx + st * dy;
            let ly = -st * dx + ct * dy;
            if (lx / a).powi(2) + (ly / b).powi(2) <= 1.0 {
                img.put_pixel(x as u32, y as u32, Rgb([200, 200, 200]));
            }
        }
    }
    // 画眼（深）、鼻（中灰）、嘴（深线）在对应关键点上
    let stamp = |img: &mut RgbImage, p: [f32; 2], r: i32, col: Rgb<u8>| {
        let (xi, yi) = (p[0].round() as i32, p[1].round() as i32);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let (x, y) = (xi + dx, yi + dy);
                if x >= 0 && x < IMG as i32 && y >= 0 && y < IMG as i32 {
                    img.put_pixel(x as u32, y as u32, col);
                }
            }
        }
    };
    stamp(&mut img, lms[0], 2, Rgb([25, 25, 25])); // 左眼
    stamp(&mut img, lms[1], 2, Rgb([25, 25, 25])); // 右眼
    stamp(&mut img, lms[2], 1, Rgb([95, 95, 95])); // 鼻
    // 嘴：两嘴角之间连一条深色线
    let steps = 20;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let mx = lms[3][0] + (lms[4][0] - lms[3][0]) * t;
        let my = lms[3][1] + (lms[4][1] - lms[3][1]) * t;
        stamp(&mut img, [mx, my], 1, Rgb([40, 40, 40]));
    }
    // 像素噪声
    for _ in 0..12 {
        let x = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let y = (rng.unit() * IMG as f32) as u32 % IMG as u32;
        let v = (rng.unit() * 255.0) as u8;
        img.put_pixel(x, y, Rgb([v, v, v]));
    }
    (img, lms)
}

fn img_to_chw(img: &RgbImage) -> Vec<f32> {
    let mut data = vec![0f32; 3 * IMG * IMG];
    for y in 0..IMG {
        for x in 0..IMG {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * IMG * IMG + y * IMG + x] = p[c] as f32 / 255.0;
            }
        }
    }
    data
}

// 造数据：返回 (输入 [N,3,IMG,IMG], 归一化坐标目标 [N,2K], 原始像素坐标真值)
fn make_dataset(rng: &mut Lcg, n: usize, dev: &Device) -> Result<(Tensor, Tensor, Vec<Lms>)> {
    let mut xs = Vec::with_capacity(n * 3 * IMG * IMG);
    let mut ys = Vec::with_capacity(n * 2 * K);
    let mut lms_all = Vec::with_capacity(n);
    for _ in 0..n {
        let (img, lms) = make_face(rng);
        xs.extend_from_slice(&img_to_chw(&img));
        for k in 0..K {
            ys.push(lms[k][0] / IMG as f32); // 归一化到 [0,1]
            ys.push(lms[k][1] / IMG as f32);
        }
        lms_all.push(lms);
    }
    let x = Tensor::from_vec(xs, (n, 3, IMG, IMG), dev)?;
    let y = Tensor::from_vec(ys, (n, 2 * K), dev)?;
    Ok((x, y, lms_all))
}

// ---------- 直接坐标回归网络：卷积骨干 + 全连接头 ----------
// 关键：用“全连接头”而非“1×1 卷积头”。卷积平移等变，1×1 头回归绝对坐标会学不动（损失死卡）；
// 全连接层的权重与展平后的空间位置一一对应，才能把“眼睛在图哪个位置”映射成绝对坐标。
struct FaceNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    fc1: Linear,
    out: Linear,
}
impl FaceNet {
    fn load(vb: VarBuilder) -> Result<Self> {
        let c = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: conv2d(3, 16, 3, c, vb.pp("conv1"))?,
            conv2: conv2d(16, 32, 3, c, vb.pp("conv2"))?,
            conv3: conv2d(32, 64, 3, c, vb.pp("conv3"))?,
            fc1: linear(64 * 6 * 6, 128, vb.pp("fc1"))?,
            out: linear(128, 2 * K, vb.pp("out"))?, // 输出 2K 个归一化坐标
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?.max_pool2d(2)?; // 48 -> 24
        let x = self.conv2.forward(&x)?.relu()?.max_pool2d(2)?; // 24 -> 12
        let x = self.conv3.forward(&x)?.relu()?.max_pool2d(2)?; // 12 -> 6
        let x = x.flatten_from(1)?; // [N, 64*6*6]
        let x = self.fc1.forward(&x)?.relu()?;
        self.out.forward(&x) // [N, 2K]
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

// 预测一张图的 5 点像素坐标
fn predict(model: &FaceNet, chw: &[f32], dev: &Device) -> Result<Lms> {
    let x = Tensor::from_vec(chw.to_vec(), (1, 3, IMG, IMG), dev)?;
    let v = model.forward(&x)?.to_vec2::<f32>()?[0].clone();
    let mut lms = [[0.0; 2]; K];
    for k in 0..K {
        lms[k] = [v[2 * k] * IMG as f32, v[2 * k + 1] * IMG as f32];
    }
    Ok(lms)
}

// ---------- Umeyama 相似变换：求 (scale, R, t) 使 Σ||scale·R·src + t − dst||² 最小 ----------
fn umeyama(src: &Lms, dst: &Lms) -> (f32, Matrix2<f32>, Vector2<f32>) {
    let n = K as f32;
    let mut mu_s = Vector2::zeros();
    let mut mu_d = Vector2::zeros();
    for k in 0..K {
        mu_s += Vector2::new(src[k][0], src[k][1]);
        mu_d += Vector2::new(dst[k][0], dst[k][1]);
    }
    mu_s /= n;
    mu_d /= n;
    let mut sigma = Matrix2::zeros(); // (1/n) Σ (dst−μd)(src−μs)^T
    let mut var_s = 0.0f32;
    for k in 0..K {
        let ds = Vector2::new(src[k][0], src[k][1]) - mu_s;
        let dd = Vector2::new(dst[k][0], dst[k][1]) - mu_d;
        sigma += dd * ds.transpose();
        var_s += ds.norm_squared();
    }
    sigma /= n;
    var_s /= n;
    let svd = sigma.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    // 保证是纯旋转（不含镜像）：若 det(U·V^T)<0，翻转最后一个奇异方向
    let d = (u * v_t).determinant();
    let s_fix = Matrix2::new(1.0, 0.0, 0.0, if d < 0.0 { -1.0 } else { 1.0 });
    let r = u * s_fix * v_t;
    let w = svd.singular_values;
    let scale = (w[0] * s_fix[(0, 0)] + w[1] * s_fix[(1, 1)]) / var_s;
    let t = mu_d - (r * mu_s) * scale;
    (scale, r, t)
}

// 双线性采样
fn sample(img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x >= (IMG - 1) as f32 || y >= (IMG - 1) as f32 {
        return Rgb([18, 18, 22]);
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let mut out = [0f32; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let p00 = img.get_pixel(x0, y0).0[i] as f32;
        let p10 = img.get_pixel(x0 + 1, y0).0[i] as f32;
        let p01 = img.get_pixel(x0, y0 + 1).0[i] as f32;
        let p11 = img.get_pixel(x0 + 1, y0 + 1).0[i] as f32;
        *o = p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy) + p01 * (1.0 - fx) * fy + p11 * fx * fy;
    }
    Rgb([out[0] as u8, out[1] as u8, out[2] as u8])
}

// 按相似变换把人脸摆正：输出像素 q → 源像素 p = (1/scale)·R^T·(q − t)
fn warp_align(img: &RgbImage, scale: f32, r: &Matrix2<f32>, t: &Vector2<f32>) -> RgbImage {
    let rt = r.transpose();
    let mut out = RgbImage::from_pixel(IMG as u32, IMG as u32, Rgb([18, 18, 22]));
    for y in 0..IMG {
        for x in 0..IMG {
            let q = Vector2::new(x as f32 + 0.5, y as f32 + 0.5);
            let p = rt * (q - t) / scale;
            out.put_pixel(x as u32, y as u32, sample(img, p[0] - 0.5, p[1] - 0.5));
        }
    }
    out
}

// 放大存图并在指定点画十字
fn save_marked(img: &RgbImage, pts: &Lms, color: Rgb<u8>, path: &str) {
    const S: u32 = 6;
    let mut c = image::imageops::resize(img, IMG as u32 * S, IMG as u32 * S, image::imageops::FilterType::Nearest);
    for p in pts.iter() {
        draw_cross_mut(&mut c, color, (p[0] * S as f32) as i32, (p[1] * S as f32) as i32);
    }
    c.save(path).unwrap();
}

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // ===== 第 1 步：合成训练集（归一化坐标目标）=====
    let mut rng = Lcg::new(7);
    let n_train = 256;
    let (xtr, ytr, _) = make_dataset(&mut rng, n_train, &dev)?;
    println!("训练集：{n_train} 张 48x48 合成人脸，每张 {K} 个关键点（左右眼/鼻/左右嘴角）");
    println!("网络：卷积骨干 + 全连接头，直接回归 {} 个归一化坐标（MSE）\n", 2 * K);

    // ===== 第 2 步：训练直接坐标回归网络 =====
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = FaceNet::load(vb)?;
    det_init(&vm, 42)?;
    let mut opt = AdamW::new(vm.all_vars(), ParamsAdamW { lr: 2e-3, ..Default::default() })?;
    println!("开始训练（AdamW lr=2e-3→3e-4，MSE 坐标回归）...");
    println!("  {:>5}  {:>10}", "epoch", "MSE");
    let epochs = 300;
    for epoch in 1..=epochs {
        if epoch == 200 {
            opt.set_learning_rate(3e-4);
        }
        let pred = net.forward(&xtr)?;
        let loss = pred.broadcast_sub(&ytr)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        if epoch % 30 == 0 || epoch == 1 {
            println!("  {epoch:>5}  {:>10.6}", loss.to_scalar::<f32>()?);
        }
    }

    // 存 / 加载（真实推理入口）
    vm.save("face.safetensors")?;
    let vb2 = unsafe { VarBuilder::from_mmaped_safetensors(&["face.safetensors"], DType::F32, &dev)? };
    let model = FaceNet::load(vb2)?;

    // ===== 第 3 步：在全新测试集上报定位误差 =====
    let mut test_rng = Lcg::new(2025);
    let n_test = 64;
    let (mut sum_e, mut per) = (0f32, [0f32; K]);
    let mut faces: Vec<(RgbImage, Lms)> = Vec::new();
    for _ in 0..n_test {
        let (img, lms) = make_face(&mut test_rng);
        faces.push((img, lms));
    }
    for (img, lms) in &faces {
        let pred = predict(&model, &img_to_chw(img), &dev)?;
        for k in 0..K {
            let e = ((pred[k][0] - lms[k][0]).powi(2) + (pred[k][1] - lms[k][1]).powi(2)).sqrt();
            sum_e += e;
            per[k] += e;
        }
    }
    println!("\n===== 定位误差（{n_test} 张全新测试脸，共 {} 个关键点）=====", n_test * K);
    println!("  平均定位误差 = {:.3} 像素", sum_e / (n_test * K) as f32);
    print!("  分关键点(像素)：");
    for k in 0..K {
        print!("{}={:.2}  ", LM_NAMES[k], per[k] / n_test as f32);
    }
    println!();

    // ===== 第 4 步：人脸对齐——用预测的 5 点求相似变换对齐到标准模板 =====
    println!("\n===== 人脸对齐（相似变换 Umeyama，对齐到 48x48 标准模板）=====");
    for (i, (img, _)) in faces.iter().enumerate().take(3) {
        let pred = predict(&model, &img_to_chw(img), &dev)?;
        let (scale, r, t) = umeyama(&pred, &TEMPLATE);
        let angle = r[(1, 0)].atan2(r[(0, 0)]).to_degrees();
        // 残差：把预测点用求得的相似变换映射后，与模板点的平均距离
        let mut resid = 0f32;
        for k in 0..K {
            let p = Vector2::new(pred[k][0], pred[k][1]);
            let q = r * p * scale + t;
            resid += ((q[0] - TEMPLATE[k][0]).powi(2) + (q[1] - TEMPLATE[k][1]).powi(2)).sqrt();
        }
        resid /= K as f32;
        println!("  脸 #{i}: 旋转 {angle:>6.1}°  缩放 {scale:.3}  对齐后残差 {resid:.3} 像素");
        // 存对齐前（画预测点）/ 后（画模板点）
        let aligned = warp_align(img, scale, &r, &t);
        save_marked(img, &pred, Rgb([250, 220, 40]), &format!("face_before_{i}.png"));
        save_marked(&aligned, &TEMPLATE, Rgb([40, 230, 90]), &format!("face_after_{i}.png"));
    }
    // 全测试集平均残差
    let mut all_resid = 0f32;
    for (img, _) in &faces {
        let pred = predict(&model, &img_to_chw(img), &dev)?;
        let (scale, r, t) = umeyama(&pred, &TEMPLATE);
        for k in 0..K {
            let p = Vector2::new(pred[k][0], pred[k][1]);
            let q = r * p * scale + t;
            all_resid += ((q[0] - TEMPLATE[k][0]).powi(2) + (q[1] - TEMPLATE[k][1]).powi(2)).sqrt();
        }
    }
    println!("  全测试集对齐后平均残差 = {:.3} 像素（{} 张脸）", all_resid / (n_test * K) as f32, n_test);
    println!("已存 face_before_0..2.png（黄叉=预测点）与 face_after_0..2.png（绿叉=模板点，脸已摆正）");
    Ok(())
}
