//! 第 90 章 布防排班 + 相机遮挡(tamper)检测
//! 运行：cargo run --example ch90_arming_tamper
//! 依赖：核心逻辑仅用 std；image 0.25 只用来造/存合成灰度帧（主工程已有依赖）。
//!
//! 两部分：
//!  (A) 布防排班：给定 ts_ms(毫秒时间戳，合成、不用系统时钟)，判断当前是否布防、哪些区域生效。
//!  (B) tamper 检测：对“正常/遮挡/失焦/移位/光照突变”五种合成帧算统计量，阈值判定设备异常。

use image::{GrayImage, Luma};

// ============ 第 83 章定义的共享事件类型（此处沿用 Event）============
#[derive(Clone, Debug)]
struct Event {
    kind: String,
    ts_ms: u64,
    message: String,
}

// ==================== (A) 布防排班 ====================

/// 一天内的时段（分钟计，[start,end)）。若 start>end 表示跨午夜，如 20:00→06:00。
#[derive(Clone, Copy)]
struct TimeRange {
    start_min: u32,
    end_min: u32,
}
impl TimeRange {
    fn contains(&self, minute_of_day: u32) -> bool {
        if self.start_min <= self.end_min {
            minute_of_day >= self.start_min && minute_of_day < self.end_min
        } else {
            // 跨午夜：如 1200..360 -> [20:00,24:00) ∪ [00:00,06:00)
            minute_of_day >= self.start_min || minute_of_day < self.end_min
        }
    }
}

/// 一条布防规则：某个区域、在哪几个星期几、哪些时段布防。
struct ArmingRule {
    zone: String,
    days: [bool; 7],       // 周一..周日
    ranges: Vec<TimeRange>,
    enabled: bool,         // 区域布防总开关（可手动关掉）
}

struct Schedule {
    tz_offset_ms: i64, // 本地时区相对 UTC 的偏移，如东八区 = +8h
    rules: Vec<ArmingRule>,
}

/// 把 UTC 毫秒时间戳按时区换算成 (星期几0..6, 一天内分钟, 时, 分)
fn decode_local(ts_ms: u64, tz_offset_ms: i64) -> (usize, u32, u32, u32) {
    let local = ts_ms as i64 + tz_offset_ms;
    let days = local.div_euclid(86_400_000);
    let ms_in_day = local.rem_euclid(86_400_000);
    let sec = (ms_in_day / 1000) as u32;
    let (hh, mm) = (sec / 3600, (sec % 3600) / 60);
    // 1970-01-01 是星期四；令周一=0，则 weekday = (days + 3) mod 7
    let weekday = ((days + 3).rem_euclid(7)) as usize;
    (weekday, hh * 60 + mm, hh, mm)
}

impl Schedule {
    /// 返回此刻生效（布防）的区域名列表
    fn active_zones(&self, ts_ms: u64) -> Vec<&str> {
        let (wd, minute, _, _) = decode_local(ts_ms, self.tz_offset_ms);
        self.rules
            .iter()
            .filter(|r| r.enabled && r.days[wd] && r.ranges.iter().any(|t| t.contains(minute)))
            .map(|r| r.zone.as_str())
            .collect()
    }
}

const WD_NAME: [&str; 7] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];

/// 便于构造测试时间戳：给定“本地”星期对应的天序号 + 时:分，反算 UTC 毫秒
fn make_ts(day_index: i64, hour: u32, minute: u32, tz_offset_ms: i64) -> u64 {
    let local = day_index * 86_400_000 + (hour as i64) * 3_600_000 + (minute as i64) * 60_000;
    (local - tz_offset_ms) as u64
}

// ==================== (B) tamper 检测 ====================

/// std 内无随机数：确定性 LCG，给帧加可复现的轻微噪声
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self { Lcg(s) }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((self.0 >> 40) as i32 % (hi - lo + 1)).abs()
    }
}

/// 正常场景：低频大幅背景(平滑起伏，模糊也抹不掉) + 高频细棋盘(每4px翻转，一模糊就没)。
/// 这样“失焦”只掉高频(梯度)、方差还在；“遮挡”则高低频全没(方差骤降)——两者可区分。
fn make_normal(w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let bg = 110.0 + 60.0 * (x as f32 / 40.0).sin() * (y as f32 / 28.0).cos(); // 低频背景
        let checker = if ((x / 4) + (y / 4)) % 2 == 0 { -30.0 } else { 30.0 };     // 高频细纹
        Luma([(bg + checker).clamp(0.0, 255.0) as u8])
    })
}

/// 遮挡：镜头被喷漆/糊住 —— 大片近似同色，方差骤降
fn make_occluded(w: u32, h: u32) -> GrayImage {
    let mut r = Lcg::new(42);
    GrayImage::from_fn(w, h, |_, _| Luma([(30 + r.range(0, 4)) as u8]))
}

/// 3x3 盒式模糊一次
fn box_blur(img: &GrayImage) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    GrayImage::from_fn(w, h, |x, y| {
        let mut s = 0u32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                s += img.get_pixel(nx, ny).0[0] as u32;
            }
        }
        Luma([(s / 9) as u8])
    })
}

/// 失焦：正常帧连续模糊多次 —— 高频(细棋盘/边缘)能量骤降，低频背景仍在
fn make_defocus(base: &GrayImage) -> GrayImage {
    let mut b = base.clone();
    for _ in 0..8 {
        b = box_blur(&b);
    }
    b
}

/// 移位：镜头被挪动 —— 画面整体平移(边缘复制)，结构与参考帧对不上
fn make_moved(base: &GrayImage, dx: i32, dy: i32) -> GrayImage {
    let (w, h) = (base.width(), base.height());
    GrayImage::from_fn(w, h, |x, y| {
        let sx = (x as i32 - dx).clamp(0, w as i32 - 1) as u32;
        let sy = (y as i32 - dy).clamp(0, h as i32 - 1) as u32;
        *base.get_pixel(sx, sy)
    })
}

/// 光照突变：整帧加亮 —— 结构不变，只有亮度整体抬升（不该判成 tamper！）
fn make_brighter(base: &GrayImage, delta: i32) -> GrayImage {
    let (w, h) = (base.width(), base.height());
    GrayImage::from_fn(w, h, |x, y| {
        Luma([(base.get_pixel(x, y).0[0] as i32 + delta).clamp(0, 255) as u8])
    })
}

/// 全图均值
fn mean(img: &GrayImage) -> f32 {
    let s: u64 = img.as_raw().iter().map(|&p| p as u64).sum();
    s as f32 / img.as_raw().len() as f32
}

/// 全图方差
fn variance(img: &GrayImage) -> f32 {
    let m = mean(img);
    let s: f32 = img.as_raw().iter().map(|&p| (p as f32 - m).powi(2)).sum();
    s / img.as_raw().len() as f32
}

/// 平均 Sobel 梯度幅值（复用第 15 章 Sobel 思路），衡量高频/清晰度
fn mean_gradient(img: &GrayImage) -> f32 {
    let (w, h) = (img.width(), img.height());
    let g = |x: u32, y: u32| img.get_pixel(x, y).0[0] as i32;
    let mut sum = 0f64;
    let mut cnt = 0u32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = -g(x - 1, y - 1) - 2 * g(x - 1, y) - g(x - 1, y + 1)
                + g(x + 1, y - 1) + 2 * g(x + 1, y) + g(x + 1, y + 1);
            let gy = -g(x - 1, y - 1) - 2 * g(x, y - 1) - g(x + 1, y - 1)
                + g(x - 1, y + 1) + 2 * g(x, y + 1) + g(x + 1, y + 1);
            sum += ((gx * gx + gy * gy) as f64).sqrt();
            cnt += 1;
        }
    }
    (sum / cnt as f64) as f32
}

/// 原始平均绝对差（会被光照突变骗到）
fn mean_abs_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    let s: u32 = a.as_raw().iter().zip(b.as_raw())
        .map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs())
        .sum();
    s as f32 / a.as_raw().len() as f32
}

/// 结构差异：各自减去自身均值后再比 —— 抹掉整体亮度差，只看“图案对不对得上”
fn structure_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    let (ma, mb) = (mean(a), mean(b));
    let s: f32 = a.as_raw().iter().zip(b.as_raw())
        .map(|(&p, &q)| ((p as f32 - ma) - (q as f32 - mb)).abs())
        .sum();
    s / a.as_raw().len() as f32
}

fn main() {
    // ---------- (A) 布防排班 ----------
    let tz = 8 * 3_600_000i64; // 东八区
    let sched = Schedule {
        tz_offset_ms: tz,
        rules: vec![
            ArmingRule {
                zone: "仓库(夜间)".into(),
                days: [true, true, true, true, true, false, false], // 周一~周五
                ranges: vec![TimeRange { start_min: 20 * 60, end_min: 6 * 60 }], // 20:00~次日06:00
                enabled: true,
            },
            ArmingRule {
                zone: "全区(周末)".into(),
                days: [false, false, false, false, false, true, true], // 周六~周日
                ranges: vec![TimeRange { start_min: 0, end_min: 24 * 60 }], // 全天
                enabled: true,
            },
        ],
    };

    // 造几个有代表性的“本地时刻”（day_index 选得让星期几符合预期）
    let cases = [
        ("周一 14:30 上班时间", make_ts(4, 14, 30, tz)),
        ("周一 19:59 布防前一分钟", make_ts(4, 19, 59, tz)),
        ("周一 20:00 夜间布防开始", make_ts(4, 20, 0, tz)),
        ("周一 05:30 凌晨(仍在夜间段)", make_ts(4, 5, 30, tz)),
        ("周六 12:00 周末白天", make_ts(2, 12, 0, tz)),
    ];

    println!("== (A) 布防排班 ==");
    let mut arming_events: Vec<Event> = Vec::new();
    for (desc, ts) in cases {
        let (wd, _min, hh, mm) = decode_local(ts, tz);
        let zones = sched.active_zones(ts);
        let armed = !zones.is_empty();
        println!(
            "{:<24} 本地 {} {:02}:{:02} => {} 生效区域: {:?}",
            desc, WD_NAME[wd], hh, mm, if armed { "布防" } else { "撤防" }, zones
        );
        if armed {
            arming_events.push(Event {
                kind: "armed".into(),
                ts_ms: ts,
                message: format!("{} 布防，生效区域 {:?}", desc, zones),
            });
        }
    }
    println!("布防时段共产生 {} 条布防态事件\n", arming_events.len());

    // ---------- (B) tamper 检测 ----------
    let (w, h) = (160u32, 120u32);
    let reference = make_normal(w, h); // 安装时拍下的“健康基准帧”
    let frames: [(&str, GrayImage); 5] = [
        ("正常", make_normal(w, h)),
        ("遮挡", make_occluded(w, h)),
        ("失焦", make_defocus(&reference)),
        ("移位", make_moved(&reference, 18, 10)),
        ("光照突变", make_brighter(&reference, 40)),
    ];

    // 以基准帧自身指标作为“健康值”，阈值按比例设——这就是最朴素的自适应标定
    let ref_var = variance(&reference);
    let ref_grad = mean_gradient(&reference);
    println!("== (B) tamper 检测（基准: 方差 {:.0}, 梯度 {:.1}）==", ref_var, ref_grad);
    println!("{:<10}{:>8}{:>8}{:>9}{:>10}{:>12}  判定", "帧", "均值", "方差", "梯度", "原始差", "结构差");

    let var_min = 0.35 * ref_var;  // 方差低于基准 35% -> 遮挡
    let grad_min = 0.5 * ref_grad; // 梯度低于基准 50% -> 失焦
    let struct_max = 20.0f32;      // 结构差超过它 -> 移位

    let mut tamper_events: Vec<Event> = Vec::new();
    let mut ts = 0u64;
    for (name, f) in &frames {
        let m = mean(f);
        let v = variance(f);
        let gr = mean_gradient(f);
        let raw = mean_abs_diff(f, &reference);
        let st = structure_diff(f, &reference);

        let verdict = if v < var_min {
            "遮挡!"
        } else if gr < grad_min {
            "失焦!"
        } else if st > struct_max {
            "移位!"
        } else {
            "正常"
        };
        println!("{:<10}{:>8.1}{:>8.0}{:>9.1}{:>10.1}{:>12.1}  {}", name, m, v, gr, raw, st, verdict);

        if verdict != "正常" {
            tamper_events.push(Event {
                kind: "tamper".into(),
                ts_ms: ts,
                message: format!("设备异常({}): {}", verdict.trim_end_matches('!'), name),
            });
        }
        ts += 1000;
    }

    // 存几张帧看看
    reference.save("out_ch90_normal.png").unwrap();
    frames[1].1.save("out_ch90_occluded.png").unwrap();
    frames[2].1.save("out_ch90_defocus.png").unwrap();

    println!("\n== tamper 事件 ({} 条) ==", tamper_events.len());
    for e in &tamper_events {
        println!("[{}] @{}ms {}", e.kind, e.ts_ms, e.message);
    }
    println!("\n注意“光照突变”原始差很大，但结构差≈0，正确地没被误报成移位。");
}
