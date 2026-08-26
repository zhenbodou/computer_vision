use image::{GrayImage, ImageBuffer, Luma};
use imageproc::contrast::{otsu_level, threshold, ThresholdType};
use imageproc::distance_transform::Norm;
use imageproc::morphology::{close, dilate, erode, open};

/// 统计一张二值图里「白色前景」像素的个数
fn count_white(img: &GrayImage) -> u32 {
    img.as_raw().iter().filter(|&&v| v > 0).count() as u32
}

/// 合成一张「带毛刺 + 小噪点 + 小洞」的二值图，前景为白 255
fn make_binary(w: u32, h: u32) -> GrayImage {
    let mut img: GrayImage = ImageBuffer::new(w, h);
    // 主体：一个大实心矩形（前景白）
    for y in 30..110 {
        for x in 40..160 {
            img.put_pixel(x, y, Luma([255]));
        }
    }
    // 毛刺：矩形顶部伸出的 1 像素宽尖刺（腐蚀/开运算应当把它啃掉）
    for y in 24..30 {
        img.put_pixel(81, y, Luma([255]));
    }
    // 小洞：主体内部挖掉一个 2x2 的小块（变回背景 0）——闭运算应当把它填上
    // （k=1 的结构元素只能填「细小」的洞；洞太大就得加大 k，见正文）
    for y in 63..65 {
        for x in 97..99 {
            img.put_pixel(x, y, Luma([0]));
        }
    }
    // 背景上的椒盐噪点（孤立小白点）——开运算应当把它们清掉
    let dots = [(15, 15), (25, 120), (180, 20), (170, 125), (100, 18)];
    for (x, y) in dots {
        img.put_pixel(x, y, Luma([255]));
    }
    // 一个 2x2 的稍大噪块
    for (x, y) in [(10, 70), (11, 70), (10, 71), (11, 71)] {
        img.put_pixel(x, y, Luma([255]));
    }
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin = make_binary(200, 140);

    // 用 L1 距离、k=1 的结构元素（十字形）做四种基本运算
    let n = Norm::L1;
    let eroded = erode(&bin, n, 1);
    let dilated = dilate(&bin, n, 1);
    let opened = open(&bin, n, 1);
    let closed = close(&bin, n, 1);

    println!("== 前景白像素个数（k=1, Norm::L1）==");
    println!("原图   : {}", count_white(&bin));
    println!("腐蚀   : {}  (前景被啃小)", count_white(&eroded));
    println!("膨胀   : {}  (前景长大)", count_white(&dilated));
    println!("开运算 : {}  (去噪点, 主体基本不变)", count_white(&opened));
    println!("闭运算 : {}  (填小洞, 主体基本不变)", count_white(&closed));

    // 验证「开运算去掉了背景噪点」：数左上角 (0,0)-(30,30) 这块的白像素
    let count_region = |img: &GrayImage| -> u32 {
        let mut c = 0;
        for y in 0..30 {
            for x in 0..30 {
                if img.get_pixel(x, y)[0] > 0 {
                    c += 1;
                }
            }
        }
        c
    };
    println!(
        "\n左上角噪点区白像素：开运算前 {} → 开运算后 {}",
        count_region(&bin),
        count_region(&opened)
    );

    // 验证「闭运算填上了小洞」：看洞里的一个点 (98,64)
    println!(
        "小洞内像素 (98,64)：原图 {} → 闭运算后 {}",
        bin.get_pixel(98, 64)[0],
        closed.get_pixel(98, 64)[0]
    );

    // ---- 实战：先 Otsu 二值化，再开运算去噪 ----
    // 合成灰度图：暗背景(30~70 渐变) + 亮物体(180) + 椒盐噪点(255)
    let (gw, gh) = (160u32, 120u32);
    let mut gray: GrayImage = ImageBuffer::from_fn(gw, gh, |x, _| {
        Luma([(30 + x * 40 / (gw - 1)) as u8]) // 背景从左到右 30 → 70
    });
    for y in 30..90 {
        for x in 40..120 {
            gray.put_pixel(x, y, Luma([180])); // 亮物体
        }
    }
    // 确定性地撒一些亮噪点（255）到背景，保证输出可复现
    let mut seed: u32 = 12345;
    let mut rng = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) & 0x7fff
    };
    for _ in 0..250 {
        gray.put_pixel(rng() % gw, rng() % gh, Luma([255]));
    }

    let level = otsu_level(&gray);
    // 亮物体 -> 白(前景)，暗背景 -> 黑；前景为白，符合形态学约定
    let bw = threshold(&gray, level, ThresholdType::Binary);
    let bw_open = open(&bw, Norm::LInf, 1); // LInf, k=1 => 3x3 方形结构元素

    println!("\n== Otsu + 开运算去噪 ==");
    println!("Otsu 阈值 = {}", level);
    println!("二值化后前景白像素 = {}", count_white(&bw));
    println!(
        "开运算后前景白像素 = {}  (噪点被清掉，物体主体保留)",
        count_white(&bw_open)
    );
    Ok(())
}
