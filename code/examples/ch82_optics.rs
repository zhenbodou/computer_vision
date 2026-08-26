//! 第 82 章 光学计算小工具：由「传感器尺寸 + 焦距 + 工作距离」算出
//! 视场(FOV)与每像素对应的物理尺寸(GSD)。纯计算，无需图像库。
//! 验证：cargo run --example ch82_optics
#![allow(unused)]

/// 相机在某一方向（水平或竖直）的光学参数
struct Optics {
    sensor_mm: f64, // 传感器该方向的物理尺寸(mm)
    focal_mm: f64,  // 镜头焦距(mm)
    wd_mm: f64,     // 工作距离：镜头到被测物的距离(mm)
    pixels: u32,    // 该方向的像素数
}

impl Optics {
    /// 视场 FOV ≈ 传感器尺寸 × 工作距离 / 焦距
    /// （放大率 m = f/(WD−f)，当 WD ≫ f 时 ≈ f/WD，故 FOV ≈ sensor·WD/f）
    fn fov_mm(&self) -> f64 {
        self.sensor_mm * self.wd_mm / self.focal_mm
    }
    /// 每像素对应的物理尺寸(mm/像素)，即 GSD——测量精度的物理下限
    fn mm_per_pixel(&self) -> f64 {
        self.fov_mm() / self.pixels as f64
    }
}

fn report(name: &str, o: &Optics) {
    let gsd = o.mm_per_pixel();
    println!(
        "[{name}] f={}mm  WD={}mm  传感器={}mm  {}px",
        o.focal_mm, o.wd_mm, o.sensor_mm, o.pixels
    );
    println!(
        "      视场 FOV ≈ {:.1} mm ；每像素 ≈ {:.4} mm（{:.1} µm）",
        o.fov_mm(),
        gsd,
        gsd * 1000.0
    );
}

fn main() {
    // 场景 A：整数例子，便于手算核对
    let a = Optics { sensor_mm: 8.0, focal_mm: 16.0, wd_mm: 400.0, pixels: 2000 };
    report("A", &a);
    assert!((a.fov_mm() - 200.0).abs() < 1e-9); // 8×400/16 = 200 mm
    assert!((a.mm_per_pixel() - 0.1).abs() < 1e-9); // 200/2000 = 0.1 mm/px

    // 场景 B：只把镜头换成长焦(35mm)，其余不变——看视场如何变窄、精度如何变高
    let b = Optics { sensor_mm: 8.0, focal_mm: 35.0, wd_mm: 400.0, pixels: 2000 };
    report("B", &b);

    // 精度直觉：一个像素对应的物理尺寸，就是能测的最小刻度（还需亚像素才更细）
    println!(
        "\n结论：换长焦后视场从 {:.0}mm 缩到 {:.0}mm，每像素从 {:.1}µm 降到 {:.1}µm——",
        a.fov_mm(),
        b.fov_mm(),
        a.mm_per_pixel() * 1000.0,
        b.mm_per_pixel() * 1000.0
    );
    println!("视场越小、精度越高，但能看的范围也越小，这就是选型要权衡的核心。");
    println!("与手算一致 ✓");
}
