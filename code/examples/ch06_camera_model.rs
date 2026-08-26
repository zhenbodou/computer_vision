//! 第 6 章 相机模型：用 nalgebra 定义内参矩阵 K，把 3D 点投影成像素坐标 (u,v)。
//! 验证：cargo check --example ch06_camera_model / cargo run --example ch06_camera_model
#![allow(unused)]

use nalgebra::{Matrix3, Vector3};

fn main() {
    // 内参：焦距（像素单位）与光心（主点）
    let fx = 800.0f32;
    let fy = 800.0f32;
    let cx = 320.0f32; // 图像宽 640 的一半
    let cy = 240.0f32; // 图像高 480 的一半

    // 内参矩阵 K（3×3），Matrix3::new 按「行优先」填入 9 个数
    let k = Matrix3::new(
        fx, 0.0, cx, //
        0.0, fy, cy, //
        0.0, 0.0, 1.0,
    );

    // 相机坐标系下的一个 3D 点 (X, Y, Z)，单位米。Z 是到相机的距离（深度）
    let point_cam = Vector3::new(0.1f32, 0.05, 2.0);

    // 投影：先做 K * P 得到齐次坐标，再除以第三分量做「透视除法」
    let homogeneous = k * point_cam;
    let u = homogeneous.x / homogeneous.z;
    let v = homogeneous.y / homogeneous.z;

    println!(
        "3D 点 (X,Y,Z) = ({}, {}, {})",
        point_cam.x, point_cam.y, point_cam.z
    );
    println!("投影到像素 (u, v) = ({:.1}, {:.1})", u, v);

    // 手算核对：u = fx*X/Z + cx = 800*0.1/2 + 320 = 360
    //          v = fy*Y/Z + cy = 800*0.05/2 + 240 = 260
    assert!((u - 360.0).abs() < 1e-3);
    assert!((v - 260.0).abs() < 1e-3);
    println!("与手算一致 ✓");
}
