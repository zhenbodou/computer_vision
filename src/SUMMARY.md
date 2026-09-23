# 目录

[前言：写给零基础的你](preface.md)

# 第一部分 · 打地基：图像与 Rust 工具

- [第 1 章 · 机器视觉到底是什么](ch01-overview.md)
- [第 2 章 · 图像的本质：像素、色彩与数据结构](ch02-image-basics.md)
- [第 3 章 · Rust 图像处理上手：image 与 ndarray](ch03-rust-image.md)
- [第 4 章 · 色彩空间与内存布局：RGB / BGR / YUV / NV12](ch04-colorspace.md)

# 第二部分 · 数据的进出：编解码

- [第 5 章 · 图像编解码：JPEG/PNG 是如何压缩的](ch05-image-codec.md)
- [第 6 章 · 视频编解码与流：帧、码流与 RTSP](ch06-video-codec.md)

# 第三部分 · 传统图像处理：图像级过滤器

- [第 7 章 · 空间滤波：降噪、模糊与锐化](ch07-spatial-filters.md)
- [第 8 章 · 边缘、形态学与轮廓](ch08-edge-morphology.md)
- [第 9 章 · 几何变换：缩放、旋转、仿射与透视](ch09-geometry.md)

# 第四部分 · 喂给模型：预处理

- [第 10 章 · 模型预处理全解：letterbox、归一化与张量布局](ch10-preprocessing.md)

# 第五部分 · 深度学习与模型工程

- [第 11 章 · 深度学习速成：神经网络与模型文件](ch11-dl-basics.md)
- [第 12 章 · 读懂模型：输入输出、shape 与元数据](ch12-model-io.md)
- [第 13 章 · 模型转换：格式、Batch 与跨平台适配](ch13-model-conversion.md)
- [第 14 章 · 模型推理：用 Rust 把模型跑起来](ch14-inference.md)

# 第六部分 · 模型家族：各类模型详解

- [第 15 章 · 分类模型：Softmax、Top-K 与多标签](ch15-classification.md)
- [第 16 章 · 检测模型：YOLO 家族与 NMS 后处理](ch16-detection.md)
- [第 17 章 · 关键点与姿态模型：骨架与人脸关键点](ch17-keypoint.md)
- [第 18 章 · 分割模型：语义与实例分割](ch18-segmentation.md)
- [第 19 章 · 差分模型：帧差、背景建模与变化检测](ch19-diff-models.md)
- [第 20 章 · 人脸识别：检测、对齐、特征与比对](ch20-face.md)

# 第七部分 · 视频分析：跟踪与业务过滤器

- [第 21 章 · 多目标跟踪：SORT、卡尔曼与 ByteTrack](ch21-tracking.md)
- [第 22 章 · 空间过滤器：ROI、Mask 与静态过滤](ch22-roi-mask.md)
- [第 23 章 · 拌线检测：单拌线、双拌线与方向判断](ch23-tripwire.md)
- [第 24 章 · 时间与去重过滤器：Cooldown、Window 与去重](ch24-cooldown-window.md)
- [第 25 章 · 脸人绑定与跨模型关联](ch25-face-person-binding.md)

# 第八部分 · 综合实战：多模型结合

- [第 26 章 · 实战一：端到端目标检测程序](ch26-project-detection.md)
- [第 27 章 · 实战二：周界入侵报警系统](ch27-project-perimeter.md)
- [第 28 章 · 实战三：人流统计与人脸抓拍](ch28-project-people.md)
- [第 29 章 · 工程化：并发、性能与部署](ch29-engineering.md)

# 附录

- [附录 A · 数学补漏：矩阵、均值方差与 Softmax](appendix-a-math.md)
- [附录 B · 常用 crate 速查表](appendix-b-crates.md)
- [附录 C · 术语表](appendix-c-glossary.md)
- [附录 D · 踩坑与常见问题 FAQ](appendix-d-faq.md)
