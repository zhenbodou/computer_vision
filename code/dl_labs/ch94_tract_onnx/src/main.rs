// 第 94 章配套：用 tract（纯 Rust）加载 ONNX 并前向推理
// 运行：cd code/dl_labs/ch94_tract_onnx && cargo run --release
//
// 本例分两步，全程在纯 Rust 内完成、结果可复现：
//   Part A：用 Rust 亲手造一个极小的 .onnx（一个线性层 y = x·W + b），
//           目的是让本章"免下载、即刻可跑"——真实项目里 .onnx 由
//           PyTorch 的 torch.onnx.export 导出，不需要在 Rust 里手搓。
//   Part B：像加载任何真实模型一样，用 tract 读这个 .onnx，设定输入 fact，
//           优化成可执行模型，喂入张量，取回输出。
//
// Part B 的 API 流程（onnx() -> model_for_path -> with_input_fact ->
// into_optimized -> into_runnable -> run -> to_plain_array_view）
// 对任何 .onnx 都一模一样，换成真实 ResNet/YOLO 只是形状与预处理不同。

use prost::Message; // 把 protobuf 结构体编码成字节
use tract_onnx::pb; // tract 内置的 ONNX protobuf 类型（ModelProto 等）
use tract_onnx::prelude::*;

// ---------- Part A：用 Rust 构造一个极小 ONNX ----------

/// 造一个固定形状的 f32 输入/输出描述（ValueInfoProto）
fn f32_io(name: &str, shape: &[i64]) -> pb::ValueInfoProto {
    let dim = shape
        .iter()
        .map(|&d| pb::tensor_shape_proto::Dimension {
            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: pb::tensor_proto::DataType::Float as i32,
                shape: Some(pb::TensorShapeProto { dim }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// 造一个 f32 初始化张量（权重/偏置就存在这里，相当于训练学到的参数）
fn f32_init(name: &str, shape: &[i64], data: Vec<f32>) -> pb::TensorProto {
    pb::TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: pb::tensor_proto::DataType::Float as i32,
        float_data: data,
        ..Default::default()
    }
}

/// 造一个算子节点（NodeProto）：op 类型 + 输入名 + 输出名
fn node(op: &str, inputs: &[&str], output: &str) -> pb::NodeProto {
    pb::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.to_string()],
        op_type: op.to_string(),
        ..Default::default()
    }
}

/// 造一个 y = input·W + b 的最小模型，序列化成 .onnx 文件
fn build_tiny_onnx(path: &str) -> TractResult<()> {
    // 权重 W [4,3]（行优先）：让输出便于手算验证
    let w = vec![
        1.0, 0.0, 0.0, // 行 0
        0.0, 1.0, 0.0, // 行 1
        0.0, 0.0, 1.0, // 行 2
        1.0, 1.0, 1.0, // 行 3
    ];
    let b = vec![0.1, 0.2, 0.3]; // 偏置 [3]

    // 计算图：两个节点，MatMul 后接 Add（这就是一个全连接/分类头）
    let graph = pb::GraphProto {
        name: "tiny_linear".to_string(),
        node: vec![
            node("MatMul", &["input", "W"], "mm"), // mm = input·W
            node("Add", &["mm", "B"], "output"),   // output = mm + B（广播）
        ],
        initializer: vec![f32_init("W", &[4, 3], w), f32_init("B", &[3], b)],
        input: vec![f32_io("input", &[1, 4])],
        output: vec![f32_io("output", &[1, 3])],
        ..Default::default()
    };
    // 模型头：IR 版本、算子集版本（opset）、生产者名——真实导出工具也会填这些
    let model_proto = pb::ModelProto {
        ir_version: 7,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "rust_cv_book".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    let mut buf = Vec::new();
    model_proto.encode(&mut buf).unwrap(); // protobuf 序列化
    std::fs::write(path, &buf)?;
    println!("[A] 已写出 {} （{} 字节的 protobuf）", path, buf.len());
    Ok(())
}

// ---------- Part B：用 tract 加载并推理 ----------

fn main() -> TractResult<()> {
    let path = "tiny_linear.onnx";
    build_tiny_onnx(path)?;

    // 标准 tract 推理流程：加载 -> 设输入 fact -> 优化 -> runnable
    let model = tract_onnx::onnx()
        .model_for_path(path)? // 读 .onnx（纯 Rust 解析 protobuf）
        .with_input_fact(0, f32::fact([1, 4]).into())? // 指定第 0 个输入的 dtype+形状
        .into_optimized()? // 图优化（常量折叠、算子融合等）
        .into_runnable()?; // 得到可反复调用的可执行模型

    // 构造输入张量 [1,4]，喂入推理
    let input = Tensor::from_shape(&[1, 4], &[1.0f32, 2.0, 3.0, 4.0])?;
    let result = model.run(tvec!(input.into()))?; // 输入用 tvec!(...) 包起来

    // 取第 0 个输出，转成 ndarray view 读数
    // ⚠️ tract 0.23 用 to_plain_array_view（旧版的 to_array_view 已移除）
    let view = result[0].to_plain_array_view::<f32>()?;
    println!("[B] 输出形状 = {:?}", view.shape());
    println!("[B] 输出值   = {:?}", view.as_slice().unwrap());
    println!("    手算校验：x=[1,2,3,4]·W=[5,6,7]，加 b=[0.1,0.2,0.3] => [5.1, 6.2, 7.3]");
    Ok(())
}
