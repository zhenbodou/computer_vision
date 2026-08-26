# 第 22 章 频域处理：傅里叶变换与频域滤波

> **本章导读**
> - 换一副眼镜看图像：不再盯着一个个像素，而是把图像看成**一堆不同频率"波纹"的叠加**。
> - 用**纯 Rust 的 `rustfft`** 亲手做 2D 傅里叶变换（image/imageproc 没有 FFT），看懂**幅度谱**。
> - 动手做**低通滤波**（去高频=模糊/去噪）和**高通滤波**（去低频=提边缘），并理解**周期性噪声在频谱里就是几个孤立亮点**。
> - 本章实测：FFT→逆变换往返最大误差 `0.000015`，正弦条纹在频谱里精准现形于第 12 个频率。

## 场景：一张布满规律纹路的图，怎么"手术式"去噪

前面几章去噪，我们用的都是**空间域（spatial domain）**的办法：中值滤波 [第 14 章](ch14-nonlinear-filter.md)、高斯模糊 [第 13 章](ch13-convolution.md)——本质都是"看某个像素和它邻居的关系"。

可有一类噪声，空间域很难对付：**周期性噪声**。比如老式扫描仪的横向条纹、传感器的固定花纹、显示器拍照时的摩尔纹。它们在整幅图上**有规律地重复**，你用模糊去压它，噪声没除干净，图倒先糊了。

这时就该**频域（frequency domain）**登场了。它的核心洞见是：**一张图像，可以等价地看成许许多多不同频率、不同方向的"波纹"叠加而成**。而那种"整幅图规律重复"的周期性噪声，在频域里会**集中成几个孤立的亮点**——你只要把那几个点抹掉、再变回来，就能**精准地**除掉噪声，几乎不伤原图。这种"定点手术"，正是频域分析的独门绝技。

## 直觉：图像 = 波纹的叠加

先建立最关键的直觉。想象一池水面：**平缓的大波浪**对应**低频**，**细密的小涟漪**对应**高频**。任何复杂的水面，都能拆成一堆不同频率波浪的叠加。图像也一样：

- **低频**：大块平滑、缓慢变化的区域（天空、墙面、渐变背景）——图像的"骨架"。
- **高频**：变化剧烈的地方（边缘、纹理、细节，以及噪声）——图像的"细节"。

**傅里叶变换（Fourier Transform）** 干的就是这件事：把图像从"每个位置的亮度是多少"（空间域），换算成"每种频率的波纹各占多少分量"（频域）。它完全可逆——逆变换能一分不差地变回来。

一维离散傅里叶变换（DFT）的定义是：

\\[ X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-2\pi i\, kn/N} \\]

别被 \\( e^{-2\pi i\,kn/N} \\) 吓到，它就是一个"频率为 \\( k \\) 的旋转波纹"探针。整个式子在问：信号 \\( x \\) 里含有多少成分和这个频率合拍？结果 \\( X[k] \\) 是**复数**，它的**模**代表这个频率的**强度**，**辐角**代表**相位**（波纹的位置偏移）。

二维图像只是把它做两遍。二维 DFT 是**可分离的**：

\\[ F[u,v] = \sum_{x}\sum_{y} f[x,y]\, e^{-2\pi i\,(ux/W + vy/H)} \\]

这正是 2D FFT 的实现窍门——**先对每一行做一维 FFT，再对每一列做一维 FFT**，两趟下来就得到完整的二维频谱。

> 📖 **术语**：**FFT（快速傅里叶变换）** 是 DFT 的快速算法，把复杂度从 \\( O(N^2) \\) 降到 \\( O(N\log N) \\)。二者结果完全一样，FFT 只是算得快。本章说 DFT 是讲原理，写代码一律用 FFT。

## 纯 Rust 怎么做 FFT：认识 rustfft

坦白说：**纯 Rust 生态目前没有"开箱即用的 2D 图像 FFT"封装**（`image`/`imageproc` 都不带 FFT）。但这不是问题——成熟的纯 Rust FFT 库 [`rustfft`](https://crates.io/crates/rustfft) 提供了高质量的**一维 FFT**，我们把它按行、按列各跑一遍，就能自己拼出 2D FFT。**这个"拼"的过程，恰恰是本章要教你的核心。**

> 💡 **提示**：本章代码基于 **`rustfft` 6.4.1**（`cargo add rustfft` 实测解析到的版本），它会顺带引入 **`num-complex` 0.4.6** 并以 `rustfft::num_complex` 再导出，复数类型直接用它即可，无需单独添加。API 若随版本变动，请以该版本文档为准。

`rustfft` 的用法就三步，非常直白：

```rust
use rustfft::{FftPlanner, num_complex::Complex};

let mut planner = FftPlanner::<f32>::new();   // 1. 建一个规划器
let fft = planner.plan_fft_forward(len);      // 2. 为长度 len 规划一次正变换
let mut buf: Vec<Complex<f32>> = /* ... */;   //    数据是复数数组
fft.process(&mut buf);                        // 3. 就地（in-place）变换
```

两个务必记住的点：

1. **数据是复数**。灰度像素是实数，放进复数的实部、虚部填 0 即可。
2. **`rustfft` 的逆变换不含 \\( 1/N \\) 归一化**。正变换后再逆变换，结果会被放大 \\( N \\) 倍（一维是 \\( N \\)，二维是 \\( W\times H \\)）。所以**逆变换后必须手动除以 \\( N \\)**，否则数值爆炸。

## 幅度谱与 fftshift

变换出来的 \\( F[u,v] \\) 是复数矩阵，没法直接看。我们通常看它的**幅度谱（magnitude spectrum）**——每个频率的模 \\( |F[u,v]| = \sqrt{\text{Re}^2 + \text{Im}^2} \\)。

但有两个显示上的讲究：

- **能量极度集中**：绝大部分能量堆在低频，直接显示只会看到中间一个亮点、其余全黑。所以要取**对数** \\( \log(1+|F|) \\) 压缩动态范围，细节才看得见。
- **DC 在角落**：FFT 输出里，**直流分量（DC，即 \\( u=v=0 \\)，代表整幅图的平均亮度）在左上角**，四个高频角对着中心。习惯上我们做一次 **fftshift**——把频谱的四个象限对调，**把 DC 挪到正中央**，看起来更符合"中间低频、四周高频"的直觉。

## 动手写代码：2D FFT、幅度谱、低通/高通滤波

例子做五件事：① 合成一张"平滑亮斑 + 竖向正弦条纹"的灰度图；② 2D FFT；③ 逆变换往返验证；④ 幅度谱（log + fftshift）并找出条纹对应的孤立亮点；⑤ 低通、高通滤波。

`Cargo.toml`：

```toml
[dependencies]
image = "0.25"
rustfft = "6.4.1"   # 纯 Rust FFT；会自动带来 num-complex，经 rustfft::num_complex 再导出
```

运行：`cargo run --example ch22_frequency`。完整代码：

```rust
use image::{GrayImage, ImageBuffer, Luma};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// 合成灰度图：平滑亮斑（低频）+ 一组竖向正弦条纹（一个明确的高频）
fn make_scene(w: u32, h: u32) -> GrayImage {
    ImageBuffer::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        let blob = (-((fx - 0.5).powi(2) + (fy - 0.5).powi(2)) * 8.0).exp() * 120.0; // 中心亮斑
        let stripes = (2.0 * std::f32::consts::PI * 12.0 * fx).cos() * 60.0; // 全图 12 个周期
        let v = 128.0 + blob * 0.5 + stripes;
        Luma([v.round().clamp(0.0, 255.0) as u8])
    })
}

/// 就地 2D FFT：先对每一行做 1D FFT，再对每一列做 1D FFT。
/// inverse=true 为逆变换（rustfft 逆变换不含 1/N 归一化）。
fn fft_2d(data: &mut [Complex<f32>], w: usize, h: usize,
          planner: &mut FftPlanner<f32>, inverse: bool) {
    // ① 每一行（长度 w）
    let fft_row = if inverse { planner.plan_fft_inverse(w) } else { planner.plan_fft_forward(w) };
    for row in data.chunks_mut(w) {
        fft_row.process(row);
    }
    // ② 每一列（长度 h）：抠出一列 -> 变换 -> 写回
    let fft_col = if inverse { planner.plan_fft_inverse(h) } else { planner.plan_fft_forward(h) };
    let mut col = vec![Complex::new(0.0f32, 0.0); h];
    for x in 0..w {
        for y in 0..h { col[y] = data[y * w + x]; }
        fft_col.process(&mut col);
        for y in 0..h { data[y * w + x] = col[y]; }
    }
}

/// 折叠频率下标：返回下标离 DC 的「周期性距离」。
/// DC 在角落 (0,0)，u 与 n-u 是同一频率的正负两支，最高频在正中间 n/2。
fn folded(u: usize, n: usize) -> f32 { u.min(n - u) as f32 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (128usize, 128usize); // FFT 建议用 2 的幂，最快
    let scene = make_scene(w as u32, h as u32);
    scene.save("out_ch22_input.png")?;

    let mut planner = FftPlanner::<f32>::new();
    // 灰度像素 -> 复数矩阵（实部=像素值，虚部=0）
    let mut freq: Vec<Complex<f32>> =
        scene.as_raw().iter().map(|&p| Complex::new(p as f32, 0.0)).collect();

    // 1. 2D 正变换
    fft_2d(&mut freq, w, h, &mut planner, false);

    // 2. 逆变换往返验证（应几乎无损）
    let mut rt = freq.clone();
    fft_2d(&mut rt, w, h, &mut planner, true);
    let n = (w * h) as f32;
    let mut max_err = 0f32;
    for (c, &orig) in rt.iter().zip(scene.as_raw().iter()) {
        max_err = max_err.max((c.re / n - orig as f32).abs()); // 逆变换后手动除以 N
    }
    println!("FFT->IFFT->/N 与原图最大误差 = {max_err:.6}");

    // 3. 幅度谱（log 尺度 + fftshift 居中）
    let mag: Vec<f32> = freq.iter().map(|c| (1.0 + c.norm()).ln()).collect();
    let maxmag = mag.iter().cloned().fold(0.0f32, f32::max);
    let mut spec: GrayImage = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = ((x + w / 2) % w, (y + h / 2) % h); // fftshift：DC 挪到中央
            spec.put_pixel(sx as u32, sy as u32, Luma([(mag[y * w + x] / maxmag * 255.0) as u8]));
        }
    }
    spec.save("out_ch22_spectrum.png")?;

    // 找出最强的「非 DC」频率——就是那组正弦条纹的孤立亮点
    let (mut peak, mut uv) = (0f32, (0usize, 0usize));
    for y in 0..h { for x in 0..w {
        if x == 0 && y == 0 { continue; } // DC 恒为最大，跳过
        let m = freq[y * w + x].norm();
        if m > peak { peak = m; uv = (x, y); }
    }}
    println!("最强非 DC 峰在 (u={}, v={})，横向折叠频率 = {} 个周期", uv.0, uv.1, folded(uv.0, w));

    // 4. 低通滤波：只保留中心低频，其余置零
    let radius = 8.0;
    let mut low = freq.clone();
    for y in 0..h { for x in 0..w {
        let (du, dv) = (folded(x, w), folded(y, h));
        if (du * du + dv * dv).sqrt() > radius { low[y * w + x] = Complex::new(0.0, 0.0); }
    }}
    fft_2d(&mut low, w, h, &mut planner, true);
    let lp: Vec<u8> = low.iter().map(|c| (c.re / n).round().clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, lp).unwrap()
        .save("out_ch22_lowpass.png")?;

    // 5. 高通滤波：去掉中心低频，只留边缘/细节（结果有正负，偏移 128 可视化）
    let mut high = freq.clone();
    for y in 0..h { for x in 0..w {
        let (du, dv) = (folded(x, w), folded(y, h));
        if (du * du + dv * dv).sqrt() <= radius { high[y * w + x] = Complex::new(0.0, 0.0); }
    }}
    fft_2d(&mut high, w, h, &mut planner, true);
    let hp: Vec<u8> = high.iter().map(|c| (c.re / n + 128.0).round().clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, hp).unwrap()
        .save("out_ch22_highpass.png")?;
    Ok(())
}
```

运行输出（真实可复现）：

```text
FFT->IFFT->/N 与原图最大误差 = 0.000015
最强非 DC 峰在 (u=12, v=0)，横向折叠频率 = 12 个周期
```

对照结果看几个关键点：

1. **往返几乎无损**：FFT 再逆变换、除以 \\( N \\)，与原图最大只差 `0.000015`（浮点舍入级）。这验证了我们的 2D FFT 拼装完全正确，也印证了傅里叶变换**可逆**。
2. **正弦条纹精准现形**：我们在图里放了"全图 12 个周期"的竖条纹，频谱里最强的非 DC 峰恰好落在 **横向第 12 个频率**（`u=12, v=0`）——理论与实测严丝合缝。这就是"**周期性花纹 = 频谱里的孤立亮点**"的铁证。打开 `out_ch22_spectrum.png`，你会看到正中央一团亮（低频亮斑）和它两侧对称的一对亮点（条纹）。
3. **低通 = 模糊/去噪**：`out_ch22_lowpass.png` 里，竖条纹**消失了**，只剩下平滑的亮斑——高频被砍掉，图像变模糊。若那对亮点是周期性噪声，这一刀就把它除了。
4. **高通 = 提边缘**：`out_ch22_highpass.png` 里，平滑区域变成中性灰（低频被去掉），只有**变化剧烈处**被凸显——这正是锐化和边缘检测的频域视角。

## 常见坑与提示

> 🕳️ **坑**：**逆变换后一定要除以 \\( N \\)，还要取实部。** `rustfft` 的逆变换不做归一化，正逆一趟下来数值会放大 \\( W\times H \\) 倍——本例是 16384 倍，不除以 \\( N \\) 直接存图就是一片纯白。另外，理论上逆变换回来的虚部应为 0，实际会有极小的浮点残留，**取实部 `c.re` 即可**，不要去算模（`norm()` 恒为正，会把负的像素值也变成正的，图就错了）。

> ⚠️ **注意**：**DC 在角落，别忘了 fftshift 只是"为了看"。** FFT 输出的 \\( u=v=0 \\)（DC，平均亮度）在左上角，频率沿两个方向"绕环"增长，最高频在中间。可视化时做 fftshift 把 DC 搬到中央只是**为了好看好懂**；真正做滤波时，我们用代码里的 `folded()`（取 `u.min(n-u)`）在**原始未平移的频谱**上直接算"离 DC 的距离"，省去来回搬移，也更不容易搞错。

> 💡 **提示**：**尺寸尽量用 2 的幂。** FFT 对 \\( 2^n \\) 尺寸最快（本例用 128×128）。`rustfft` 用混合基数算法，任意尺寸也能跑，但含大质因数的尺寸会明显变慢。工程上常把图像 padding 到 2 的幂再变换。

> ⚠️ **注意**：**理想"一刀切"滤波会带来振铃（ringing）。** 我们这里用"半径内保留、半径外清零"的**理想滤波器**，简单直观，但频域里的硬边缘会让空间域出现波纹状的振铃伪影。要更干净，可换成**高斯型**或**巴特沃斯（Butterworth）**滤波器，让过渡平滑——这留给你在练习里试。

## 本章小结

- **频域视角**：图像 = 不同频率波纹的叠加。**低频**=大块平滑区（骨架），**高频**=边缘/细节/噪声。
- **傅里叶变换**在空间域与频域间可逆换算；\\( F[u,v] \\) 是复数，**模=强度，辐角=相位**。**2D FFT = 先对每行、再对每列做 1D FFT**。
- 纯 Rust 无现成 2D 图像 FFT，用 **`rustfft` 6.4.1** 的一维 FFT 自行组合：数据填复数、逆变换后**除以 \\( N \\) 并取实部**。
- **幅度谱**取 \\( \log(1+|F|) \\) 压缩、用 fftshift 把 DC 挪到中央。**周期性花纹在频谱里是孤立亮点**（本章条纹精准落在第 12 个频率）。
- **低通**去高频=模糊/去噪，**高通**去低频=锐化/提边缘；周期性噪声可在频谱里**定点抹除**。坑：归一化、取实部、2 的幂尺寸、理想滤波的振铃。

## 动手练习

1. **定点除噪**：在 `make_scene` 里把条纹幅度调大当作"噪声"，在频谱里找到那对亮点的坐标，**只把这两个点（及其附近小邻域）置零**，逆变换后看条纹是否被干净除掉、而亮斑几乎不受影响——体会频域"手术式去噪"的威力。
2. **横纹变竖纹**：把 `stripes` 改成随 `fy`（纵向）变化的正弦，重新看幅度谱，确认孤立亮点从横轴挪到了**纵轴**——频谱的方向对应条纹的方向。
3. **改半径看模糊程度**：把低通的 `radius` 从 8 调到 4、16、32，观察 `out_ch22_lowpass.png` 越来越清晰（保留频率越多），理解"截止频率"就是模糊强弱的旋钮。
4. **进阶·高斯低通**：把"半径内保留、半径外清零"换成乘以高斯权重 \\( e^{-(d^2)/(2\sigma^2)} \\)（\\( d \\) 为到 DC 的折叠距离），对比理想低通，看振铃伪影是否减轻。

至此，第三部分（经典图像处理）就全部走完了——你已经掌握了卷积、边缘、阈值、形态学、轮廓、霍夫、模板匹配、金字塔、频域这一整套经典工具箱。下一部分，我们进入**特征工程**：让机器从图像里提炼出可比较、可匹配的"指纹"，第一站是**角点检测**。👉 [第 23 章 角点检测（Harris、FAST、Shi-Tomasi）](../p04-features/ch23-corners.md)

## 第三部分阶段验收：算法选择而不是滤镜堆叠

选择一个真实检测问题，先写噪声、尺度、光照、边界和速度假设，再建立最简单基线。对卷积
边界模式、Canny 阈值、结构元素、金字塔层数和频域截止频率做单因素实验，使用任务指标而
不是“看起来更清楚”评价。为每个算子保留朴素参考实现，与优化实现做随机差分测试；报告
复杂度、P95、峰值内存和最坏输入。最后给出失效样本及改用学习方法的判据。
