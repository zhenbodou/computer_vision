//! 第 84 章 ROI 区域过滤与多边形内外判断
//! 运行：cargo run --example ch84_roi_filter
//! 依赖：无（只用 std）——合成 Vec<Detection> 当输入，自包含、可复现。
//!
//! BBox / Detection 就是第 83 章给出的共享数据模型，这里为了能独立运行再抄一份。
//! 核心新内容：射线法（point-in-polygon）判断"脚点是否落在 ROI 多边形内"。

#![allow(dead_code)]

/// 轴对齐包围盒（同第 83 章）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        BBox { x, y, w, h }
    }
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
    /// 底边中点（脚点）——ROI / 过线判定用它而非中心。
    pub fn bottom_center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h)
    }
    pub fn area(&self) -> f32 {
        (self.w * self.h).max(0.0)
    }
}

/// 检测框（同第 83 章）。
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: BBox,
    pub class_id: u32,
    pub label: String,
    pub score: f32,
}

/// 射线法（ray casting / even-odd rule）判断点 (px, py) 是否在多边形 poly 内。
///
/// poly 是按顺序给出的顶点（顺时针或逆时针都行），首尾自动相连成闭合多边形。
/// 想象从该点向右射出一条水平射线，数它穿过了多边形几条边：
///   穿过奇数次 -> 在内部；穿过偶数次 -> 在外部。
/// 用半开区间约定 (yi > py) != (yj > py) 来判断某条边是否"纵向跨过" py，
/// 既能避开正好扫到顶点被数两次的麻烦，又天然适用于凹多边形。
fn point_in_polygon(px: f32, py: f32, poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false; // 少于 3 个点构不成多边形
    }
    let mut inside = false;
    let mut j = n - 1; // 上一个顶点：先指向最后一个，和第 0 个连成收尾那条边
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        // 这条边 (j -> i) 的两个端点，是否一个在射线上方、一个在下方？
        let crosses = (yi > py) != (yj > py);
        if crosses {
            // 求边与水平线 y = py 的交点横坐标；交点在点的右边（px < x_at）才算被射线穿过
            let x_at = xi + (py - yi) / (yj - yi) * (xj - xi);
            if px < x_at {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// 把矩形 (x, y, w, h) 表达成 4 个顶点的多边形——矩形只是多边形的特例。
fn rect_roi(x: f32, y: f32, w: f32, h: f32) -> Vec<(f32, f32)> {
    vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
}

fn yn(b: bool) -> &'static str {
    if b { "是" } else { "否" }
}

/// 造一个"站立目标"：给定脚点 (cx_foot, y_foot) 和宽高，反推出左上角。
fn subject(label: &str, cx_foot: f32, y_foot: f32, w: f32, h: f32) -> Detection {
    Detection {
        bbox: BBox::new(cx_foot - w / 2.0, y_foot - h, w, h),
        class_id: 0,
        label: label.to_string(),
        score: 0.9,
    }
}

fn make_dets() -> Vec<Detection> {
    vec![
        subject("person", 300.0, 300.0, 40.0, 120.0), // 站在车道中间：脚点在内
        subject("person", 60.0, 300.0, 40.0, 120.0),  // 车道左外侧：脚点在外
        subject("person", 320.0, 60.0, 40.0, 40.0),   // 画面上方远处：脚点在外
        subject("person", 300.0, 340.0, 40.0, 120.0), // 贴车道下边、身子探进来：脚点在外但中心在内
        subject("person", 250.0, 200.0, 36.0, 100.0), // 车道内：脚点在内
        subject("car", 470.0, 310.0, 90.0, 50.0),     // 车道内的车：脚点在内
    ]
}

fn main() {
    // ① 定义一个梯形 ROI：透视下的一条"车道 / 通道"，图像 y 向下，所以上窄下宽。
    let roi: Vec<(f32, f32)> = vec![
        (200.0, 100.0), // 左上
        (440.0, 100.0), // 右上
        (520.0, 320.0), // 右下
        (120.0, 320.0), // 左下
    ];

    // ② 逐个判定：用脚点判断在不在 ROI 内，同时也算一遍"中心点"作对比。
    let dets = make_dets();
    println!("== 逐个判定（脚点 bottom_center 是否落在 ROI 内）==");
    let mut hit = 0;
    for d in &dets {
        let (fx, fy) = d.bbox.bottom_center();
        let (cx, cy) = d.bbox.center();
        let in_foot = point_in_polygon(fx, fy, &roi);
        let in_center = point_in_polygon(cx, cy, &roi);
        if in_foot {
            hit += 1;
        }
        println!(
            "[{}] {:6} 脚点=({:>3.0},{:>3.0}) 在内(脚点)={} 在内(中心)={}",
            if in_foot { "保留" } else { "剔除" },
            d.label,
            fx,
            fy,
            yn(in_foot),
            yn(in_center),
        );
    }
    println!("\nROI 内命中（按脚点）：{} / {} 个检测", hit, dets.len());

    // ③ 矩形 ROI 是多边形的特例：同一个 point_in_polygon 就能用。
    let rect = rect_roi(300.0, 150.0, 120.0, 120.0);
    let p = (350.0, 200.0);
    println!(
        "\n矩形 ROI 特例：点 ({:.0},{:.0}) 在矩形内 = {}",
        p.0,
        p.1,
        yn(point_in_polygon(p.0, p.1, &rect))
    );

    // ④ 多个 ROI 取并集：落在任意一块里就算命中。
    let roi_b = rect_roi(540.0, 60.0, 80.0, 100.0); // 右上角另开一块矩形区
    let rois = [roi.clone(), roi_b];
    let q = (580.0, 110.0); // 在第二块里、不在梯形车道里
    let in_any = rois.iter().any(|poly| point_in_polygon(q.0, q.1, poly));
    println!(
        "多 ROI 并集：点 ({:.0},{:.0}) 命中任一区域 = {}",
        q.0,
        q.1,
        yn(in_any)
    );

    // ⑤ 边界情形：点正好落在边上。半开区间约定下，上边算"内"、下边算"外"（避免相邻区共享边被数两次）。
    println!(
        "\n边界点 (300,320) 落在下边上，在内 = {}",
        yn(point_in_polygon(300.0, 320.0, &roi))
    );
    println!(
        "边界点 (300,100) 落在上边上，在内 = {}",
        yn(point_in_polygon(300.0, 100.0, &roi))
    );
}
