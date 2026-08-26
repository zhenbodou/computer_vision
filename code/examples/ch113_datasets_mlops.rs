//! 第 113 章 数据与 MLOps：解析 COCO 风格标注 + 类别统计 + 防泄漏划分
//! 运行：cargo run --example ch113_datasets_mlops
//! 依赖：serde（derive）、serde_json —— 均为纯 Rust。
//!
//! 全自包含：内嵌一段合成的 COCO 风格标注 JSON（8 张图、4 类、若干标注框），
//! 用 serde 解析后：① 统计每类标注数、检测类别不平衡；② 演示 bbox 的
//! xywh↔xyxy 坐标约定；③ 按 image_id 分组做 train/val 划分（同一张图不跨集）；
//! ④ 对照“按标注随机划分”会造成同图跨集的泄漏。所有数字均为程序真实输出。

use serde::Deserialize;
use std::collections::BTreeMap;

// ============ COCO 风格标注的最小数据模型（只取本例要用的字段）============
// COCO 的 bbox 约定是 [x, y, width, height]，原点在左上角（xywh）。
#[derive(Debug, Deserialize)]
struct Coco {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnn>,
    categories: Vec<CocoCat>,
}

#[derive(Debug, Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

#[derive(Debug, Deserialize)]
struct CocoAnn {
    #[allow(dead_code)]
    id: u64,
    image_id: u64,
    category_id: u64,
    bbox: [f64; 4], // [x, y, w, h]
}

#[derive(Debug, Deserialize)]
struct CocoCat {
    id: u64,
    name: String,
}

// 极简可复现伪随机（LCG）——固定种子，不引 rand，保证每次划分完全一致。
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        // 数值来自 Numerical Recipes 的经典 LCG 常数
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    // Fisher–Yates 洗牌
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn main() {
    // ---------- 0. 解析 JSON ----------
    let ds: Coco = serde_json::from_str(COCO_JSON).expect("COCO JSON 解析失败");
    let id2name: BTreeMap<u64, &str> =
        ds.categories.iter().map(|c| (c.id, c.name.as_str())).collect();
    println!(
        "解析成功：{} 张图，{} 条标注，{} 个类别\n",
        ds.images.len(),
        ds.annotations.len(),
        ds.categories.len()
    );

    // ---------- 1. 类别分布与不平衡检测 ----------
    let mut per_cat: BTreeMap<&str, usize> = BTreeMap::new();
    for a in &ds.annotations {
        *per_cat.entry(id2name[&a.category_id]).or_insert(0) += 1;
    }
    println!("== 类别分布（按标注框计）==");
    let total = ds.annotations.len();
    for (name, cnt) in &per_cat {
        let bar = "#".repeat(*cnt);
        println!("  {name:<8} {cnt:>3} ({:>5.1}%) {bar}", 100.0 * *cnt as f64 / total as f64);
    }
    let max = *per_cat.values().max().unwrap();
    let min = *per_cat.values().min().unwrap();
    println!("  不平衡比 max/min = {}/{} = {:.1}x", max, min, max as f64 / min as f64);
    if max as f64 / min as f64 >= 10.0 {
        println!("  ⚠️ 类别严重不平衡（≥10x）：只看 accuracy 会被多数类主导，须报 macro-F1/每类召回");
    }
    println!();

    // ---------- 2. bbox 坐标约定：xywh -> xyxy ----------
    let a0 = &ds.annotations[0];
    let [x, y, w, h] = a0.bbox;
    println!("== bbox 约定演示（COCO 用 xywh）==");
    println!("  第一条标注 xywh = [{x}, {y}, {w}, {h}]");
    println!("  转成 xyxy       = [{x}, {y}, {}, {}]（喂检测框/IoU 前务必对齐约定）\n", x + w, y + h);

    // ---------- 3. 按 image_id 分组做 train/val 划分（防泄漏）----------
    // 关键：划分的“单位”是图片，不是单个标注框。同一张图的所有框只能整体进一个集合。
    let mut img_ids: Vec<u64> = ds.images.iter().map(|i| i.id).collect();
    Lcg::new(42).shuffle(&mut img_ids); // 固定种子 → 可复现
    let val_ratio = 0.35;
    let n_val = ((img_ids.len() as f64) * val_ratio).round() as usize;
    let val_set: std::collections::HashSet<u64> = img_ids[..n_val].iter().copied().collect();

    let (mut tr_imgs, mut va_imgs) = (0usize, 0usize);
    let (mut tr_anns, mut va_anns) = (0usize, 0usize);
    let mut tr_cat: BTreeMap<&str, usize> = BTreeMap::new();
    let mut va_cat: BTreeMap<&str, usize> = BTreeMap::new();
    let mut va_files: Vec<&str> = Vec::new();
    for img in &ds.images {
        if val_set.contains(&img.id) {
            va_imgs += 1;
            va_files.push(&img.file_name);
        } else {
            tr_imgs += 1;
        }
    }
    for a in &ds.annotations {
        let cat = id2name[&a.category_id];
        if val_set.contains(&a.image_id) {
            va_anns += 1;
            *va_cat.entry(cat).or_insert(0) += 1;
        } else {
            tr_anns += 1;
            *tr_cat.entry(cat).or_insert(0) += 1;
        }
    }
    println!("== 按 image_id 分组划分（val_ratio={val_ratio}, seed=42）==");
    println!("  train: {tr_imgs} 张图 / {tr_anns} 标注   {tr_cat:?}");
    println!("  val  : {va_imgs} 张图 / {va_anns} 标注   {va_cat:?}");
    println!("  val 图片清单：{va_files:?}");
    if !va_cat.contains_key("dog") {
        println!("  ⚠️ 稀有类 dog 全落在 train、val 里一个都没有——随机分组对长尾类不友好，");
        println!("     真实项目要做分组+分层（group-stratified）划分，保证每类在各集都有样本。");
    }

    // 校验：没有任何 image_id 同时出现在两个集合里
    let train_img_set: std::collections::HashSet<u64> =
        ds.images.iter().map(|i| i.id).filter(|id| !val_set.contains(id)).collect();
    let overlap = train_img_set.intersection(&val_set).count();
    println!("  同图跨集检查：{overlap} 张图跨集（应为 0 → 无泄漏）\n");

    // ---------- 4. 反面对照：按“单个标注”随机划分会泄漏 ----------
    // 把 23 条标注直接随机分到 train/val，同一张图的多个框会被拆到不同集合。
    let mut ann_idx: Vec<usize> = (0..ds.annotations.len()).collect();
    Lcg::new(7).shuffle(&mut ann_idx);
    let cut = ((ann_idx.len() as f64) * (1.0 - val_ratio)).round() as usize;
    let mut where_of: BTreeMap<u64, (bool, bool)> = BTreeMap::new(); // image_id -> (进过train, 进过val)
    for (rank, &ai) in ann_idx.iter().enumerate() {
        let img = ds.annotations[ai].image_id;
        let e = where_of.entry(img).or_insert((false, false));
        if rank < cut { e.0 = true } else { e.1 = true }
    }
    let leaked: Vec<u64> =
        where_of.iter().filter(|(_, v)| v.0 && v.1).map(|(k, _)| *k).collect();
    println!("== 反面对照：按单个标注随机划分 ==");
    println!("  有标注同时落入 train 和 val 的图片：{} 张 → {:?}", leaked.len(), leaked);
    println!("  这些图既在训练里“见过”、又在验证里被考——验证分数虚高，就是数据泄漏。");
}

// ================= 合成的 COCO 风格标注（8 图 / 4 类 / 23 框，故意做出类别不平衡）=================
const COCO_JSON: &str = r#"
{
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "dog"}
  ],
  "images": [
    {"id": 100, "file_name": "street_01.jpg", "width": 640,  "height": 480},
    {"id": 101, "file_name": "street_02.jpg", "width": 640,  "height": 480},
    {"id": 102, "file_name": "park_01.jpg",   "width": 800,  "height": 600},
    {"id": 103, "file_name": "park_02.jpg",   "width": 800,  "height": 600},
    {"id": 104, "file_name": "lot_01.jpg",    "width": 1024, "height": 768},
    {"id": 105, "file_name": "lot_02.jpg",    "width": 1024, "height": 768},
    {"id": 106, "file_name": "alley_01.jpg",  "width": 640,  "height": 480},
    {"id": 107, "file_name": "alley_02.jpg",  "width": 640,  "height": 480}
  ],
  "annotations": [
    {"id": 1,  "image_id": 100, "category_id": 1, "bbox": [ 34,  50,  40, 120]},
    {"id": 2,  "image_id": 100, "category_id": 1, "bbox": [120,  60,  38, 118]},
    {"id": 3,  "image_id": 100, "category_id": 2, "bbox": [300, 200, 180,  90]},
    {"id": 4,  "image_id": 100, "category_id": 3, "bbox": [220, 260,  70,  60]},
    {"id": 5,  "image_id": 101, "category_id": 1, "bbox": [ 60,  40,  44, 130]},
    {"id": 6,  "image_id": 101, "category_id": 2, "bbox": [260, 180, 200, 100]},
    {"id": 7,  "image_id": 102, "category_id": 1, "bbox": [ 80,  90,  50, 150]},
    {"id": 8,  "image_id": 102, "category_id": 1, "bbox": [200, 100,  48, 148]},
    {"id": 9,  "image_id": 102, "category_id": 4, "bbox": [420, 380,  90,  70]},
    {"id": 10, "image_id": 102, "category_id": 3, "bbox": [520, 300,  80,  70]},
    {"id": 11, "image_id": 103, "category_id": 1, "bbox": [110, 120,  52, 160]},
    {"id": 12, "image_id": 103, "category_id": 1, "bbox": [300, 130,  50, 158]},
    {"id": 13, "image_id": 104, "category_id": 2, "bbox": [ 40, 300, 220, 120]},
    {"id": 14, "image_id": 104, "category_id": 2, "bbox": [320, 320, 210, 118]},
    {"id": 15, "image_id": 104, "category_id": 2, "bbox": [600, 310, 230, 122]},
    {"id": 16, "image_id": 104, "category_id": 1, "bbox": [880, 200,  46, 140]},
    {"id": 17, "image_id": 105, "category_id": 2, "bbox": [ 60, 330, 215, 120]},
    {"id": 18, "image_id": 105, "category_id": 2, "bbox": [400, 340, 205, 116]},
    {"id": 19, "image_id": 105, "category_id": 1, "bbox": [820, 210,  44, 138]},
    {"id": 20, "image_id": 106, "category_id": 1, "bbox": [ 90,  70,  42, 128]},
    {"id": 21, "image_id": 106, "category_id": 1, "bbox": [260,  80,  40, 126]},
    {"id": 22, "image_id": 107, "category_id": 1, "bbox": [130, 100,  46, 132]},
    {"id": 23, "image_id": 107, "category_id": 3, "bbox": [330, 250,  75,  65]}
  ]
}
"#;
