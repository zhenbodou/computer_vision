// 第 28 章：词袋模型 BoVW 与以图搜图
// 运行：cargo run --example ch28_bovw_retrieval
// 依赖：image、imageproc（K-means 手写，不引入额外依赖）
use image::{GrayImage, ImageBuffer, Luma};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_filled_circle_mut;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};

/// 一个极简的确定性随机数发生器（线性同余），避免引入 rand 依赖，保证结果可复现
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u32() % (hi - lo) as u32) as i32
    }
}

fn canvas() -> GrayImage {
    ImageBuffer::from_pixel(128, 128, Luma([230u8]))
}

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

/// 局部描述子：在每个 FAST 角点周围 9x9 邻域，统计 8 方向（无符号）梯度直方图，再 L2 归一化。
/// 返回该图所有关键点的 8 维描述子列表。
fn descriptors(img: &GrayImage) -> Vec<Vec<f32>> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let gx = horizontal_sobel(img); // 水平梯度 Image<Luma<i16>>
    let gy = vertical_sobel(img); // 垂直梯度
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
                if mag < 1.0 {
                    continue;
                }
                let mut ang = ey.atan2(ex); // -π..π
                if ang < 0.0 {
                    ang += std::f32::consts::PI; // 折叠到 0..π（不区分正反向）
                }
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
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// 手写精简版 K-means：把海量描述子聚成 k 个"视觉单词"，返回 k 个中心
fn kmeans(data: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let dim = data[0].len();
    let mut rng = Lcg::new(seed);
    // 初始化：随机挑 k 个描述子当初始中心（K-means 对初始化敏感，实践中要多跑几次）
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|_| data[rng.next_u32() as usize % data.len()].clone())
        .collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0u32; k];
        for x in data {
            let c = nearest(x, &centroids); // 分配：每个点归到最近中心
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += x[d];
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    centroids[j][d] = sums[j][d] / counts[j] as f32; // 更新：中心=成员均值
                }
            }
        }
    }
    centroids
}

/// 把一张图的描述子量化成"词频直方图"（BoVW 向量），并 L1 归一化
fn bovw_vector(descs: &[Vec<f32>], dict: &[Vec<f32>]) -> Vec<f32> {
    let mut hist = vec![0f32; dict.len()];
    for d in descs {
        hist[nearest(d, dict)] += 1.0;
    }
    let s = hist.iter().sum::<f32>().max(1.0);
    for v in &mut hist {
        *v /= s;
    }
    hist
}

/// 两个词频直方图的 L1 距离（越小越相似）
fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1) 建立图库：3 个类别，每类 2 张（不同随机排布）
    let db: Vec<(&str, GrayImage)> = vec![
        ("方块#1", squares(1)),
        ("方块#2", squares(2)),
        ("菱形#1", diamonds(1)),
        ("菱形#2", diamonds(2)),
        ("圆点#1", dots(1)),
        ("圆点#2", dots(2)),
    ];

    // 2) 对每张库图提取局部描述子；并把它们全部汇入"训练池"用于建词典
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
    let mut ranked: Vec<(usize, f32)> = db_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l1(&q_vec, v)))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (rank, (i, dist)) in ranked.iter().enumerate() {
        println!("  第{}名  {}  L1距离={:.3}", rank + 1, db[*i].0, dist);
    }

    // 再验证：换一张查询图看看
    let query2 = squares(88);
    let q2 = bovw_vector(&descriptors(&query2), &dict);
    let best = db_vecs
        .iter()
        .enumerate()
        .min_by(|a, b| l1(&q2, a.1).partial_cmp(&l1(&q2, b.1)).unwrap())
        .unwrap()
        .0;
    println!("\n查询图 = 方块#88，最相似库图 = {}", db[best].0);

    Ok(())
}
