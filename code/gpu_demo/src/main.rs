//! 第 109 章 GPU 加速：wgpu compute shader 骨架（灰度 / 逐像素重计算）
//!
//! 演示纯 Rust 走 GPU 的完整往返：建 device/queue → 建 buffer → 写 WGSL kernel →
//! 建 pipeline → dispatch → 读回 → 与 CPU 参考对拍。两个 kernel 对比“算术强度”对
//! GPU 收益的决定性影响：
//!   - 轻算子（灰度）：每像素只算一次加权和 —— 计算少、搬运占比大；
//!   - 重算子（每像素 300 次三角迭代）：计算多、搬运占比小 —— GPU 才真正碾压。
//!
//! ⚠️ 需要一个可用 GPU 适配器；headless / 无卡环境会打印提示并优雅退出。
//! 本机（Apple M1 Pro，Metal，统一内存）实测输出见 README / 正文。

use std::time::Instant;
use wgpu::util::DeviceExt;

// 轻算子：RGBA(u32) -> 灰度。每像素一次加权和，几乎没有计算量。
const WGSL_LIGHT: &str = r#"
@group(0) @binding(0) var<storage, read>       inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) { return; }        // 尾部越界保护
    let px = inp[i];
    let r = f32(px & 0xffu);
    let g = f32((px >> 8u) & 0xffu);
    let b = f32((px >> 16u) & 0xffu);
    let y = u32(0.299 * r + 0.587 * g + 0.114 * b);
    outp[i] = y | (y << 8u) | (y << 16u) | (0xffu << 24u);
}
"#;

// 重算子：每像素做 300 次三角迭代，人为拉高算术强度，模拟“重预处理/滤波”。
const WGSL_HEAVY: &str = r#"
@group(0) @binding(0) var<storage, read>       inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) { return; }
    var v = f32(inp[i] & 0xffu);
    for (var k = 0u; k < 300u; k = k + 1u) {
        v = sin(v) * 12.7 + cos(v) * 9.1 + 1.0;
    }
    let y = u32(clamp(abs(v), 0.0, 255.0));
    outp[i] = y | (y << 8u) | (y << 16u) | (0xffu << 24u);
}
"#;

// ---- CPU 参考实现：与 WGSL 逐字对应，用来给 GPU 结果“对拍” ----
#[inline]
fn cpu_gray(px: u32) -> u32 {
    let r = (px & 0xff) as f32;
    let g = ((px >> 8) & 0xff) as f32;
    let b = ((px >> 16) & 0xff) as f32;
    let y = (0.299 * r + 0.587 * g + 0.114 * b) as u32;
    y | (y << 8) | (y << 16) | (0xff << 24)
}
#[inline]
fn cpu_heavy(px: u32) -> u32 {
    let mut v = (px & 0xff) as f32;
    for _ in 0..300 {
        v = v.sin() * 12.7 + v.cos() * 9.1 + 1.0;
    }
    let y = v.abs().clamp(0.0, 255.0) as u32;
    y | (y << 8) | (y << 16) | (0xff << 24)
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // 合成一张 1920x1080 的 RGBA 图（渐变），自包含、可复现。
    let n: usize = 1920 * 1080;
    let mut input = vec![0u32; n];
    for (i, p) in input.iter_mut().enumerate() {
        let r = (i % 256) as u32;
        let g = ((i / 256) % 256) as u32;
        let b = ((i / 512) % 256) as u32;
        *p = r | (g << 8) | (b << 16) | (0xff << 24);
    }

    // 1) 拿适配器（显卡句柄）。headless/无卡环境这里会 Err —— 优雅退出，不 panic。
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("拿不到 GPU 适配器：{e:?}");
            eprintln!("（headless / 无显卡 / CI 环境属正常，本示例跳过）");
            return;
        }
    };
    let info = adapter.get_info();
    println!("适配器：{} （后端 {:?}，类型 {:?}）", info.name, info.backend, info.device_type);

    // 2) 由适配器要一个逻辑 device 和一个命令队列 queue。
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("request_device 失败");

    // 3) 三个 GPU buffer：输入（只读）、输出（可写、可拷贝）、读回（可 map 到 CPU）。
    let bytes = (n * 4) as u64;
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 4) 把 WGSL 编译成 compute pipeline（layout=None 让 wgpu 自动推断绑定布局）。
    let make_pipe = |src: &str| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipe_light = make_pipe(WGSL_LIGHT);
    let pipe_heavy = make_pipe(WGSL_HEAVY);

    // 5) bind group：把 in_buf/out_buf 绑到 shader 的 @binding(0)/@binding(1)。
    let bind = |pipe: &wgpu::ComputePipeline| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        })
    };
    let bind_light = bind(&pipe_light);
    let bind_heavy = bind(&pipe_heavy);

    // 一次完整往返：encode -> submit -> map -> 读回。返回 (端到端毫秒, 可选读回数据)。
    let run_gpu = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, want: bool| {
        let t = Instant::now();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            // 每 64 个像素一个 workgroup，向上取整覆盖尾部。
            pass.dispatch_workgroups(((n + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes); // D2H 拷贝
        queue.submit(Some(enc.finish()));
        // map_async + poll(Wait) 阻塞直到 GPU 完成并可读。
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let data = slice.get_mapped_range().unwrap();
        let out = if want { bytemuck::cast_slice::<u8, u32>(&data).to_vec() } else { Vec::new() };
        drop(data);
        read_buf.unmap();
        (t.elapsed().as_secs_f64() * 1000.0, out)
    };

    // 返回 (冷启动首次毫秒, 预热后 best-of-7 毫秒, 一份读回数据用于对拍)。
    let bench_gpu = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
        let (cold, sample) = run_gpu(pipe, bg, true); // 第一次：含 shader 编译/驱动预热
        let mut warm = f64::MAX;
        for _ in 0..7 {
            warm = warm.min(run_gpu(pipe, bg, false).0);
        }
        (cold, warm, sample)
    };
    // CPU 参考：预分配输出、直接内联，让编译器能向量化（公平基线）。best-of-N 取最快。
    let bench_cpu = |f: fn(u32) -> u32, reps: usize| {
        let mut out = vec![0u32; n];
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            for (o, &px) in out.iter_mut().zip(input.iter()) { *o = f(px); }
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        (best, out)
    };
    let maxdiff = |a: &[u32], b: &[u32]| {
        let mut m = 0i32;
        for i in 0..n { m = m.max(((a[i] & 0xff) as i32 - (b[i] & 0xff) as i32).abs()); }
        m
    };

    println!("图像：1920x1080 = {n} 像素\n");

    // ---- 轻算子 ----
    let (cold, warm, g) = bench_gpu(&pipe_light, &bind_light);
    let (cpu, c) = bench_cpu(cpu_gray, 5);
    println!("[轻·灰度]");
    println!("  GPU 冷启动首次 = {cold:.2}ms （含 shader 编译/驱动预热）");
    println!("  GPU 预热后     = {warm:.2}ms   CPU（向量化）= {cpu:.2}ms   加速 = {:.2}x", cpu / warm);
    println!("  GPU↔CPU 最大差 = {}（f32 舍入，正常）\n", maxdiff(&g, &c));

    // ---- 重算子 ----
    let (cold, warm, g) = bench_gpu(&pipe_heavy, &bind_heavy);
    let (cpu, c) = bench_cpu(cpu_heavy, 3);
    println!("[重·每像素 300 次迭代]");
    println!("  GPU 冷启动首次 = {cold:.2}ms");
    println!("  GPU 预热后     = {warm:.2}ms   CPU = {cpu:.2}ms   加速 = {:.2}x", cpu / warm);
    println!("  GPU↔CPU 最大差 = {}（300 次混沌迭代放大了两端数学库的微小差异）", maxdiff(&g, &c));
}
