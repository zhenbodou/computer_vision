# 第 90 章 布防排班、相机遮挡（tamper）与场景鲁棒性

> **本章导读**
> - 学完你能：给报警系统装上"作息表"——**按时段/星期/区域布防**，该睡觉的时候别乱叫；并让它能发现**摄像头被人遮挡、挪动、搞失焦**。
> - 亲手实现：一个排班表结构（含跨午夜、时区）、以及基于**画面统计量突变**（方差、梯度、结构差）的 tamper 检测。
> - 顺带聊透工程里最烦的**场景鲁棒性**：白天黑夜、光照突变、逆光，怎么不把"天亮了"误报成"镜头被动了"。
> - 全程只用 `std`（合成灰度帧、存 PNG 时用一下 `image`），时间戳自己造（不看系统时钟），可复现。

## 场景：两个"教科书不讲、现场天天挨骂"的问题

你把前面九章的规则都调好了：过滤、ROI、绊线、停留、轨迹，报警又准又稳。上线第一天，甲方打来两个电话：

**电话一**："你这系统白天上班时间也一直在报'有人进入'——废话，白天当然有人！我只想让它**晚上和周末**盯着仓库。" 这就是**布防排班（arming schedule）**：报警不能 24 小时无脑响，得按时段、按星期、按区域决定"现在到底管不管"。

**电话二**："昨晚有人拿喷漆把摄像头糊了，你的系统**啥都没报**！" ——当然没报，因为画面全黑，检测器"看不到人"自然不告警。可镜头被遮挡、被挪动、被搞失焦，本身就是最该报的**设备异常（tamper）**。这活儿检测器不管，得我们在事件层单独盯。

这两件事，学术书基本不提，却是安防项目验收的硬指标。这一章就把它们做出来。

## 布防排班：给系统排个班

排班的本质，是一张"什么时间 + 哪个区域 = 布防"的表。核心结构三层：

```text
Schedule（总表，带时区）
  └─ ArmingRule（一条规则：某区域）
       ├─ days[7]     哪几个星期几生效（周一..周日）
       └─ ranges[]    一天内的哪些时段生效（跨午夜也支持）
```

给一个时间戳 `ts_ms`，我们要回答："此刻哪些区域在布防？"两步走：

**第一步：把毫秒时间戳变成'本地星期几 + 几点几分'。** 这里藏着第一个大坑。

> 🕳️ **坑**：**时间戳是 UTC，布防按的是本地时间——中间差着时区，还可能差着夏令时（DST）。** `ts_ms`（Unix 毫秒）是相对 UTC 的，直接拆出来的"小时"是 UTC 的小时。东八区要先 `+8` 小时再拆。更阴险的是夏令时：有 DST 的地区一年里时区偏移会变（相差 1 小时），且存在"不存在的一小时"和"重复的一小时"。**认真的排班一定要用带时区库（如 `chrono-tz`）算本地时间**；本章为可复现只用一个固定偏移演示，但你要知道这个坑的存在——排班边界（如 06:00 撤防）恰好撞上 DST 切换时，最容易出事。

不引外部库、纯 `std` 把 UTC 毫秒拆成本地时间，就是几行整数运算（`1970-01-01` 是星期四，令周一=0，则 `weekday = (天数 + 3) mod 7`）：

```rust
/// 把 UTC 毫秒时间戳按时区换算成 (星期几0..6, 一天内分钟, 时, 分)
fn decode_local(ts_ms: u64, tz_offset_ms: i64) -> (usize, u32, u32, u32) {
    let local = ts_ms as i64 + tz_offset_ms;
    let days = local.div_euclid(86_400_000);
    let ms_in_day = local.rem_euclid(86_400_000);
    let sec = (ms_in_day / 1000) as u32;
    let (hh, mm) = (sec / 3600, (sec % 3600) / 60);
    let weekday = ((days + 3).rem_euclid(7)) as usize; // 1970-01-01 是周四
    (weekday, hh * 60 + mm, hh, mm)
}
```

**第二步：拿本地时间去比对每条规则。** 时段用"一天内的分钟数"表示，比如 `20:00` 就是 `1200`。这里第二个坑：

> ⚠️ **注意**：**跨午夜时段（如 20:00→次日 06:00）不能直接比大小。** 若 `start_min <= end_min`（如 09:00→18:00），判 `start <= t < end` 就行；若 `start > end`（跨午夜），得判 `t >= start 或 t < end`（分成"今晚 20:00 之后"和"明早 06:00 之前"两截）。写反了，凌晨 5 点的仓库就没人管了。

排班部分的完整代码（数据模型沿用 [第 83 章](ch83-filtering-basics.md) 的 `Event`）：

```rust
use image::{GrayImage, Luma};

#[derive(Clone, Debug)]
struct Event { kind: String, ts_ms: u64, message: String }

/// 一天内的时段（分钟计，[start,end)）。start>end 表示跨午夜。
#[derive(Clone, Copy)]
struct TimeRange { start_min: u32, end_min: u32 }
impl TimeRange {
    fn contains(&self, t: u32) -> bool {
        if self.start_min <= self.end_min {
            t >= self.start_min && t < self.end_min
        } else {
            t >= self.start_min || t < self.end_min // 跨午夜
        }
    }
}

/// 一条布防规则：某区域，哪几个星期几、哪些时段布防。
struct ArmingRule {
    zone: String,
    days: [bool; 7],        // 周一..周日
    ranges: Vec<TimeRange>,
    enabled: bool,          // 区域布防总开关（可手动关掉）
}

struct Schedule { tz_offset_ms: i64, rules: Vec<ArmingRule> }

impl Schedule {
    fn active_zones(&self, ts_ms: u64) -> Vec<&str> {
        let (wd, minute, _, _) = decode_local(ts_ms, self.tz_offset_ms);
        self.rules.iter()
            .filter(|r| r.enabled && r.days[wd] && r.ranges.iter().any(|t| t.contains(minute)))
            .map(|r| r.zone.as_str())
            .collect()
    }
}

const WD_NAME: [&str; 7] = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];

/// 便于造测试时间戳：给定本地天序号 + 时:分，反算 UTC 毫秒
fn make_ts(day_index: i64, hour: u32, minute: u32, tz_offset_ms: i64) -> u64 {
    ((day_index * 86_400_000 + hour as i64 * 3_600_000 + minute as i64 * 60_000) - tz_offset_ms) as u64
}
```

其中 `enabled` 就是"区域布防开关"：值班员想临时撤掉某个区域的布防，把它置 `false` 即可，排班照旧、这个区域先歇着。

## 相机遮挡（tamper）：盯的是"画面统计量的突变"

镜头被破坏时，检测器看不出异常，但**画面的整体统计特征会剧变**。三种典型破坏，对应三种突变：

| 破坏方式 | 画面变化 | 抓它的统计量 |
|---------|---------|-------------|
| **遮挡**（喷漆/糊住/挡布） | 大片近似同色 | **方差骤降**（像素都差不多，没起伏） |
| **失焦**（对焦被拧乱） | 边缘全糊、细节没了 | **平均梯度骤降**（高频能量没了） |
| **移位**（镜头被挪动） | 拍的还是清晰的，但拍的**不是原来那块地方** | **与参考帧的结构差变大** |

三个量的算法都很朴素。**方差**衡量"画面有没有起伏"：

\\[ \sigma^2 = \frac{1}{N}\sum_i (p_i - \mu)^2 \\]

**平均梯度**衡量"清不清晰"，直接复用 [第 15 章](../p03-classic/ch15-edges.md) 的 Sobel 算子，对每个像素算梯度幅值再取平均：

\\[ |\nabla p| = \sqrt{G_x^2 + G_y^2} \\]

失焦会把边缘抹平，高频能量塌陷，平均梯度随之骤降。

**移位**最微妙。最直接的想法是拿当前帧和安装时的"参考帧"逐像素比，算平均绝对差。但这里埋着 tamper 检测最大的坑：

> 🕳️ **坑**：**光照突变会让"原始帧差"暴涨，但它不是 tamper！** 天亮了、灯打开了，整幅画面一起变亮——原始帧差会很大，可镜头好好的，画面结构（哪里是门、哪里是墙）根本没变。若用原始帧差判"移位"，每天天亮都误报一次。**破解办法：比之前先各自减掉自己的均值**，抹掉"整体亮度"这个自由度，只看"图案对不对得上"。这个**结构差**对纯亮度变化免疫：

\\[ D_{\text{struct}} = \frac{1}{N}\sum_i \left| (a_i - \mu_a) - (b_i - \mu_b) \right| \\]

判定顺序也有讲究：先查方差（遮挡）→ 再查梯度（失焦）→ 最后查结构差（移位）。因为遮挡时方差、梯度会一起垮，先拦方差才不会把遮挡误判成失焦。

tamper 部分完整代码：

```rust
/// std 内无随机数：确定性 LCG，给帧加可复现的轻微噪声
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self { Lcg(s) }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + ((self.0 >> 40) as i32 % (hi - lo + 1)).abs()
    }
}

/// 正常场景：低频大幅背景(模糊也抹不掉) + 高频细棋盘(一模糊就没)
fn make_normal(w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let bg = 110.0 + 60.0 * (x as f32 / 40.0).sin() * (y as f32 / 28.0).cos();
        let checker = if ((x / 4) + (y / 4)) % 2 == 0 { -30.0 } else { 30.0 };
        Luma([(bg + checker).clamp(0.0, 255.0) as u8])
    })
}
fn make_occluded(w: u32, h: u32) -> GrayImage { // 大片近似同色
    let mut r = Lcg::new(42);
    GrayImage::from_fn(w, h, |_, _| Luma([(30 + r.range(0, 4)) as u8]))
}
fn box_blur(img: &GrayImage) -> GrayImage {
    let (w, h) = (img.width(), img.height());
    GrayImage::from_fn(w, h, |x, y| {
        let mut s = 0u32;
        for dy in -1i32..=1 { for dx in -1i32..=1 {
            let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
            let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
            s += img.get_pixel(nx, ny).0[0] as u32;
        }}
        Luma([(s / 9) as u8])
    })
}
fn make_defocus(base: &GrayImage) -> GrayImage { // 连续模糊 8 次
    let mut b = base.clone();
    for _ in 0..8 { b = box_blur(&b); }
    b
}
fn make_moved(base: &GrayImage, dx: i32, dy: i32) -> GrayImage { // 平移，边缘复制
    let (w, h) = (base.width(), base.height());
    GrayImage::from_fn(w, h, |x, y| {
        let sx = (x as i32 - dx).clamp(0, w as i32 - 1) as u32;
        let sy = (y as i32 - dy).clamp(0, h as i32 - 1) as u32;
        *base.get_pixel(sx, sy)
    })
}
fn make_brighter(base: &GrayImage, delta: i32) -> GrayImage { // 整帧加亮（结构不变）
    let (w, h) = (base.width(), base.height());
    GrayImage::from_fn(w, h, |x, y| {
        Luma([(base.get_pixel(x, y).0[0] as i32 + delta).clamp(0, 255) as u8])
    })
}

fn mean(img: &GrayImage) -> f32 {
    let s: u64 = img.as_raw().iter().map(|&p| p as u64).sum();
    s as f32 / img.as_raw().len() as f32
}
fn variance(img: &GrayImage) -> f32 {
    let m = mean(img);
    img.as_raw().iter().map(|&p| (p as f32 - m).powi(2)).sum::<f32>() / img.as_raw().len() as f32
}
/// 平均 Sobel 梯度幅值（复用第 15 章思路），衡量清晰度
fn mean_gradient(img: &GrayImage) -> f32 {
    let (w, h) = (img.width(), img.height());
    let g = |x: u32, y: u32| img.get_pixel(x, y).0[0] as i32;
    let (mut sum, mut cnt) = (0f64, 0u32);
    for y in 1..h - 1 { for x in 1..w - 1 {
        let gx = -g(x-1,y-1) - 2*g(x-1,y) - g(x-1,y+1) + g(x+1,y-1) + 2*g(x+1,y) + g(x+1,y+1);
        let gy = -g(x-1,y-1) - 2*g(x,y-1) - g(x+1,y-1) + g(x-1,y+1) + 2*g(x,y+1) + g(x+1,y+1);
        sum += ((gx*gx + gy*gy) as f64).sqrt();
        cnt += 1;
    }}
    (sum / cnt as f64) as f32
}
fn mean_abs_diff(a: &GrayImage, b: &GrayImage) -> f32 { // 会被光照突变骗到
    let s: u32 = a.as_raw().iter().zip(b.as_raw()).map(|(&p, &q)| (p as i32 - q as i32).unsigned_abs()).sum();
    s as f32 / a.as_raw().len() as f32
}
/// 结构差：各自减去自身均值后再比 —— 抹掉整体亮度差，只看图案对不对得上
fn structure_diff(a: &GrayImage, b: &GrayImage) -> f32 {
    let (ma, mb) = (mean(a), mean(b));
    a.as_raw().iter().zip(b.as_raw())
        .map(|(&p, &q)| ((p as f32 - ma) - (q as f32 - mb)).abs())
        .sum::<f32>() / a.as_raw().len() as f32
}
```

最后把两部分拼进 `main`（完整文件见 `examples/ch90_arming_tamper.rs`）：排班遍历几个代表性时刻；tamper 拿基准帧自身的方差/梯度当"健康值"，阈值按比例设——**这就是最朴素的自适应标定**：

```rust
fn main() {
    // (A) 布防排班：东八区，仓库工作日夜间布防、全区周末布防
    let tz = 8 * 3_600_000i64;
    let sched = Schedule { tz_offset_ms: tz, rules: vec![
        ArmingRule { zone: "仓库(夜间)".into(), days: [true,true,true,true,true,false,false],
            ranges: vec![TimeRange { start_min: 20*60, end_min: 6*60 }], enabled: true },
        ArmingRule { zone: "全区(周末)".into(), days: [false,false,false,false,false,true,true],
            ranges: vec![TimeRange { start_min: 0, end_min: 24*60 }], enabled: true },
    ]};
    let cases = [
        ("周一 14:30 上班时间", make_ts(4, 14, 30, tz)),
        ("周一 19:59 布防前一分钟", make_ts(4, 19, 59, tz)),
        ("周一 20:00 夜间布防开始", make_ts(4, 20, 0, tz)),
        ("周一 05:30 凌晨(仍在夜间段)", make_ts(4, 5, 30, tz)),
        ("周六 12:00 周末白天", make_ts(2, 12, 0, tz)),
    ];
    println!("== (A) 布防排班 ==");
    for (desc, ts) in cases {
        let (wd, _m, hh, mm) = decode_local(ts, tz);
        let zones = sched.active_zones(ts);
        println!("{:<24} 本地 {} {:02}:{:02} => {} 生效区域: {:?}",
            desc, WD_NAME[wd], hh, mm, if zones.is_empty() { "撤防" } else { "布防" }, zones);
    }

    // (B) tamper 检测
    let (w, h) = (160u32, 120u32);
    let reference = make_normal(w, h); // 安装时拍下的“健康基准帧”
    let frames: [(&str, GrayImage); 5] = [
        ("正常", make_normal(w, h)),
        ("遮挡", make_occluded(w, h)),
        ("失焦", make_defocus(&reference)),
        ("移位", make_moved(&reference, 18, 10)),
        ("光照突变", make_brighter(&reference, 40)),
    ];
    let (ref_var, ref_grad) = (variance(&reference), mean_gradient(&reference));
    let var_min = 0.35 * ref_var;   // 方差低于基准 35% -> 遮挡
    let grad_min = 0.5 * ref_grad;  // 梯度低于基准 50% -> 失焦
    let struct_max = 20.0f32;       // 结构差超过它 -> 移位

    println!("\n== (B) tamper 检测（基准: 方差 {:.0}, 梯度 {:.1}）==", ref_var, ref_grad);
    println!("{:<10}{:>8}{:>8}{:>9}{:>10}{:>12}  判定", "帧","均值","方差","梯度","原始差","结构差");
    for (name, f) in &frames {
        let (v, gr) = (variance(f), mean_gradient(f));
        let (raw, st) = (mean_abs_diff(f, &reference), structure_diff(f, &reference));
        let verdict = if v < var_min { "遮挡!" }
            else if gr < grad_min { "失焦!" }
            else if st > struct_max { "移位!" } else { "正常" };
        println!("{:<10}{:>8.1}{:>8.0}{:>9.1}{:>10.1}{:>12.1}  {}", name, mean(f), v, gr, raw, st, verdict);
    }
}
```

运行：

```bash
cargo run --example ch90_arming_tamper
```

终端输出（数字真实可复现）：

```text
== (A) 布防排班 ==
周一 14:30 上班时间            本地 周一 14:30 => 撤防 生效区域: []
周一 19:59 布防前一分钟          本地 周一 19:59 => 撤防 生效区域: []
周一 20:00 夜间布防开始          本地 周一 20:00 => 布防 生效区域: ["仓库(夜间)"]
周一 05:30 凌晨(仍在夜间段)       本地 周一 05:30 => 布防 生效区域: ["仓库(夜间)"]
周六 12:00 周末白天            本地 周六 12:00 => 布防 生效区域: ["全区(周末)"]

== (B) tamper 检测（基准: 方差 1733, 梯度 163.8）==
帧               均值      方差       梯度       原始差         结构差  判定
正常           104.4    1733    163.8       0.0         0.0  正常
遮挡            32.0       2      6.2      72.8        34.5  遮挡!
失焦           100.9     827     11.2      29.1        29.1  失焦!
移位           107.0    1755    155.6      37.4        37.2  移位!
光照突变         144.4    1733    163.8      40.0         0.0  正常
```

结论逐条对上：

1. **排班**：周一 `19:59` 还撤防、`20:00` 立刻布防——边界精确到分钟；凌晨 `05:30` 仍在"20:00→06:00"的跨午夜段里，正确布防；周六白天被"全区(周末)"规则覆盖。
2. **遮挡**：方差从 `1733` 崩到 `2`（远低于阈值 `606`）——大片同色，一眼识破。
3. **失焦**：方差还有 `827`（没垮），但梯度从 `163.8` 塌到 `11.2`（低于阈值 `81.9`）——清晰度没了，正是失焦特征。
4. **移位**：方差、梯度都正常（画面依旧清晰），但结构差 `37.2`（超过阈值 `20`）——拍的不是原来那块地方。
5. **光照突变**：均值从 `104.4` 抬到 `144.4`、原始差高达 `40.0`，可结构差是 `0.0`——**正确地判为"正常"，没被"天亮了"骗到误报移位**。这正是"减均值再比"的威力。

## 场景鲁棒性：别把"环境变化"当成"故障"

tamper 检测的死敌是**正常的环境变化**。白天黑夜交替、云飘过、车灯扫过、逆光……这些都会搅动画面统计量。几条实战心得：

- **区分渐变 vs 突变。** 日照是**缓慢渐变**（几十分钟），破坏是**瞬间突变**（一两秒）。所以别用固定阈值硬卡绝对值，而是看**短时间内的变化率**：几秒内方差/梯度"断崖式"掉才算 tamper，慢慢变的放行。
- **阈值要自适应、按场景标定。** 本章用"基准帧指标 × 比例"当阈值，就是最简单的自适应。更进一步可以**滚动更新基准**：用一个缓慢更新的背景/参考（[第 63 章](../p13-tracking/ch63-background-motion.md) 的背景建模思路），让系统慢慢适应光照漂移，只对突变敏感。
- **逆光、夜间**：可结合 [第 16 章](../p03-classic/ch16-threshold.md) 的自适应阈值、直方图均衡先把画面"拉正"，再算统计量；夜间红外/低照度画面本身梯度就低，失焦阈值得单独定，别套白天的。

> 💡 **提示**：**tamper 报警也要走前面几章的去抖。** 单帧统计量突变可能只是相机自动曝光在调整。把 tamper 判定接上 [第 85 章](ch85-temporal-window.md) 的时间窗投票（连续 N 秒都异常才报）和 [第 86 章](ch86-cooldown-dedup.md) 的 cooldown，能挡掉绝大多数瞬时抖动误报。

## 常见坑与提示

> 🕳️ **坑**：**排班的时区与夏令时是头号坑。** 跨时区部署、或用户所在地有 DST 时，务必用带时区数据库的库（`chrono` + `chrono-tz`）算本地时间，别自己 `+8`。排班边界（如凌晨撤防）撞上 DST 切换那一小时，最容易出现"该布防没布防"。

> ⚠️ **注意**：**tamper 阈值没有万能值，必须按现场标定。** 室内稳定光照和室外风吹草动的"正常波动"天差地别。上线前先录一段该相机的正常视频，统计方差/梯度的正常范围，再把阈值定在"正常波动之外、异常突变之内"。

> 🕳️ **坑**：**遮挡和失焦会同时压低方差和梯度，判定顺序不能错。** 必须**先判方差（遮挡）再判梯度（失焦）**；反过来会把"糊住镜头"误报成"失焦"。移位放最后，因为它是"清晰但拍错地方"，得在排除前两者后才谈结构差。

## 本章小结

- **布防排班**让报警按"时段 × 星期 × 区域"生效：核心是把 `ts_ms` 换算成本地时间（当心**时区/夏令时**），再比对规则的星期与时段（当心**跨午夜**时段的判定）；`enabled` 提供区域级布防开关。
- **相机遮挡（tamper）检测**盯的是**画面统计量突变**：**遮挡→方差骤降**、**失焦→平均梯度骤降**（复用 Sobel）、**移位→与参考帧的结构差变大**。
- **结构差 = 减掉各自均值再比**，对纯亮度变化免疫，能把"天亮了/开灯了"和"镜头被挪了"区分开——避免光照突变误报。
- 判定顺序 **方差 → 梯度 → 结构差**；阈值应**自适应 + 按场景标定**，并区分**渐变（环境）与突变（故障）**，再叠加时间窗/cooldown 去抖。

## 动手练习

1. **加一条排班规则**：给 `Schedule` 加一条"办公室工作日 09:00→18:00 撤防、其余布防"的规则（提示：可以拆成两个时段，或反过来判"不在工作时段"），验证周一 `14:30` 变成布防。
2. **失焦更细的判据**：目前用平均梯度判失焦。试试改用"梯度大于某阈值的像素占比"（清晰画面这个占比高，失焦后骤降），对比两种指标哪个更稳。
3. **抗光照突变的移位检测**：把 `make_moved` 的位移改小（如 `(4, 2)`），看结构差还能不能超过阈值；再给移位帧叠加 `make_brighter`，确认结构差仍能识破移位（亮度免疫）。
4. **进阶——渐变 vs 突变**：合成一个"方差在 10 帧里缓慢下降"和一个"方差在 1 帧里骤降"的序列，写个逻辑：只有**单帧变化率**超过阈值才报 tamper，缓慢下降的放行。

到这里，从检测框到业务事件的"判定"逻辑就基本齐了：过滤、区域、时序、去重、绊线、停留、轨迹行为、布防防遮挡。下一章我们让事件"动起来"——**抓拍、录像，以及通过 Webhook / MQTT / Kafka 把告警推出去**。👉 [第 91 章 事件联动：抓拍、录像、Webhook/MQTT/Kafka 推送](ch91-actions-push.md)

## 已完成事件章节的阶段验收：状态机与时间语义

用属性测试验证 ROI 边界、IoU 对称性、冷却幂等、轨迹顺序和状态机不变量。所有窗口以事件
时间还是处理时间计算必须显式规定；注入乱序、重复、迟到、时钟回拨、进程重启和配置热更。
对一段冻结录像生成黄金事件流，比较事件 precision/recall、重复率、告警延迟和最坏队列。
状态需可持久化或明确接受丢失语义，外部动作使用幂等键；“每帧判断正确”不代表事件系统
在故障后仍正确。
