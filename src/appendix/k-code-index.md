# 附录 K：逐章代码入口与运行边界

> 此表由 `python3 scripts/check_book.py --write-index` 从仓库生成。存在源码不表示本次已运行或完成真实数据验证。

## 运行前先选对目录

主工程示例：在仓库根目录执行 `cd code`，再运行表中的 example。独立工程：从仓库根目录进入表中的目录，再运行 `cargo run --locked --release`。每次更换工程前先回到仓库根目录，不要连续照抄 `cd`。

`--manifest-path` 只选择清单，不会把程序的工作目录切到工程目录；相对输入输出路径仍相对于启动位置。见 [Cargo 官方运行说明](https://doc.rust-lang.org/cargo/commands/cargo-run.html)。第一次运行需要下载依赖；`--locked` 固定解析结果，并不使缺失的依赖自动离线可用。

表中“主工程 example”执行 `cargo run --locked --example 名称`。GPU/WASM 示例使用专门后端或目标，按照相应章节运行，不套用普通 CPU 命令。`standalone/` 是片段保留区，不等同于独立 Cargo 工程。

## 逐章索引

| 章节 | 已有入口（仓库相对路径） | 运行类型 |
|---|---|---|
| [第 1 章 什么是机器视觉（CV 全景任务地图）](../p01-start/ch01-what-is-cv.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 2 章 为什么用 Rust 做机器视觉](../p01-start/ch02-why-rust.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 3 章 搭建开发环境，跑通第一个程序](../p01-start/ch03-setup.md) | `ch03_first_image` | 主工程 example |
| [第 4 章 Rust 速成营（够用就行）](../p01-start/ch04-rust-crash-course.md) | `ch04_rust_crash` | 主工程 example |
| [第 5 章 图像到底是什么](../p01-start/ch05-what-is-an-image.md) | `ch05_image_basics` | 主工程 example |
| [第 6 章 相机成像原理与相机模型](../p01-start/ch06-camera-model.md) | `ch06_camera_model` | 主工程 example |
| [第 7 章 读取、显示与保存图像](../p02-image-basics/ch07-load-save.md) | `ch07_load_save` | 主工程 example |
| [第 8 章 访问和修改像素](../p02-image-basics/ch08-pixels.md) | `ch08_pixels` | 主工程 example |
| [第 9 章 颜色空间与灰度化（RGB/HSV/YUV/Lab）](../p02-image-basics/ch09-color-spaces.md) | `ch09_color_spaces` | 主工程 example |
| [第 10 章 几何变换：缩放、裁剪、旋转、仿射、透视](../p02-image-basics/ch10-geometry.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 11 章 点运算：亮度、对比度、Gamma 与查找表](../p02-image-basics/ch11-point-ops.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 12 章 直方图与直方图均衡化](../p02-image-basics/ch12-histogram.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 13 章 卷积原理与线性滤波（模糊）](../p03-classic/ch13-convolution.md) | `ch13_convolution` | 主工程 example |
| [第 14 章 非线性滤波：中值与双边滤波（去噪）](../p03-classic/ch14-nonlinear-filter.md) | `ch14_nonlinear_filter` | 主工程 example |
| [第 15 章 梯度与边缘检测（Sobel、Canny）](../p03-classic/ch15-edges.md) | `ch15_edges` | 主工程 example |
| [第 16 章 阈值分割与二值化（Otsu、自适应）](../p03-classic/ch16-threshold.md) | `ch16_threshold` | 主工程 example |
| [第 17 章 形态学操作（腐蚀、膨胀、开闭运算）](../p03-classic/ch17-morphology.md) | `ch17_morphology` | 主工程 example |
| [第 18 章 连通域与轮廓分析](../p03-classic/ch18-contours.md) | `ch18_contours` | 主工程 example |
| [第 19 章 霍夫变换：检测直线与圆](../p03-classic/ch19-hough.md) | `ch19_hough` | 主工程 example |
| [第 20 章 模板匹配：在大图里找小图](../p03-classic/ch20-template-matching.md) | `ch20_template_matching` | 主工程 example |
| [第 21 章 图像金字塔与多尺度处理](../p03-classic/ch21-pyramids.md) | `ch21_pyramids` | 主工程 example |
| [第 22 章 频域处理：傅里叶变换与频域滤波](../p03-classic/ch22-frequency.md) | `ch22_frequency` | 主工程 example |
| [第 23 章 角点检测（Harris、FAST、Shi-Tomasi）](../p04-features/ch23-corners.md) | `ch23_corners` | 主工程 example |
| [第 24 章 局部特征描述子（SIFT、ORB、BRIEF）](../p04-features/ch24-descriptors.md) | `ch24_descriptors` | 主工程 example |
| [第 25 章 特征匹配与几何验证（RANSAC）](../p04-features/ch25-matching-ransac.md) | `ch25_matching_ransac` | 主工程 example |
| [第 26 章 单应矩阵、图像对齐与全景拼接](../p04-features/ch26-homography-stitching.md) | `ch26_homography_stitching` | 主工程 example |
| [第 27 章 经典手工特征：HOG、LBP、Haar](../p04-features/ch27-handcrafted-features.md) | `ch27_handcrafted_features` | 主工程 example |
| [第 28 章 词袋模型与图像检索基础（BoVW）](../p04-features/ch28-bovw-retrieval.md) | `ch28_bovw_retrieval` | 主工程 example |
| [第 29 章 ndarray：多维数组与张量运算](../p05-numeric/ch29-ndarray.md) | `ch29_ndarray` | 主工程 example |
| [第 30 章 nalgebra：线性代数与几何变换](../p05-numeric/ch30-nalgebra.md) | `ch30_nalgebra` | 主工程 example |
| [第 31 章 把图像当数据：图像与张量互转、批处理](../p05-numeric/ch31-image-as-tensor.md) | `ch31_image_as_tensor` | 主工程 example |
| [第 32 章 机器学习是怎么"学"的](../p06-ml/ch32-ml-intro.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 33 章 用 linfa 做分类、回归与聚类](../p06-ml/ch33-linfa.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 34 章 传统目标检测：滑动窗口 + HOG + SVM](../p06-ml/ch34-sliding-window.md) | `ch34_sliding_window` | 主工程 example |
| [第 35 章 实战：手写数字识别](../p06-ml/ch35-digit-recognition.md) | `ch35_digit_recognition` | 主工程 example |
| [第 36 章 神经网络与反向传播（图解版）](../p07-dl-basics/ch36-neural-nets.md) | `ch36_mlp` | 主工程 example |
| [第 37 章 卷积神经网络 CNN 原理](../p07-dl-basics/ch37-cnn.md) | `ch37_cnn` | 主工程 example |
| [第 38 章 经典网络结构演进（LeNet→ResNet→Transformer）](../p07-dl-basics/ch38-architectures.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 39 章 Rust 深度学习生态全景（candle/burn/tch/tract/ort）](../p07-dl-basics/ch39-dl-ecosystem.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 40 章 用 burn 从零训练一个 CNN：合成形状入门与 MNIST 迁移](../p07-dl-basics/ch40-burn-train.md) | `code/dl_labs/ch40_burn_train` | 独立 CPU 工程 |
| [第 41 章 训练技巧：数据增强、优化器、迁移学习](../p07-dl-basics/ch41-training-tricks.md) | `ch41_augment` | 主工程 example |
| [第 42 章 用 Candle 加载预训练模型做图像分类](../p08-classification/ch42-candle-classify.md) | `code/dl_labs/ch42_candle_classify` | 独立 CPU 工程 |
| [第 43 章 微调自己的分类器：数据、策略与消融](../p08-classification/ch43-finetune.md) | `code/dl_labs/ch43_finetune` | 独立 CPU 工程 |
| [第 44 章 细粒度、多标签与层级分类](../p08-classification/ch44-finegrained.md) | `code/dl_labs/ch44_multilabel` | 独立 CPU 工程 |
| [第 45 章 目标检测原理：匹配、损失、解码与 NMS](../p09-detection/ch45-detection-principles.md) | `ch45_detection_principles` | 主工程 example |
| [第 46 章 用 YOLO 做实时目标检测：可靠的 Rust 推理管线](../p09-detection/ch46-yolo-inference.md) | `code/dl_labs/ch46_yolo_infer` | 独立 CPU 工程 |
| [第 47 章 训练与微调检测器：数据闭环与受控实验](../p09-detection/ch47-train-detector.md) | `code/dl_labs/ch47_train_detector` | 独立 CPU 工程 |
| [第 48 章 检测评估、标注质量与统计不确定性](../p09-detection/ch48-metrics-annotation.md) | `ch48_metrics_annotation` | 主工程 example |
| [第 49 章 语义分割：从像素损失到边界质量](../p10-segmentation/ch49-semantic-seg.md) | `code/dl_labs/ch49_unet_seg` | 独立 CPU 工程 |
| [第 50 章 实例分割与全景分割](../p10-segmentation/ch50-instance-seg.md) | `code/dl_labs/ch50_maskrcnn` | 独立 CPU 工程 |
| [第 51 章 交互式分割与 SAM（Segment Anything）](../p10-segmentation/ch51-sam.md) | `code/dl_labs/ch51_interactive_seg` | 独立 CPU 工程 |
| [第 52 章 关键点检测原理（热图回归、PAF）](../p11-keypoints/ch52-keypoint-basics.md) | `code/dl_labs/ch52_heatmap` | 独立 CPU 工程 |
| [第 53 章 人体姿态估计（OpenPose/HRNet/MoveNet）](../p11-keypoints/ch53-human-pose.md) | `code/dl_labs/ch53_pose` | 独立 CPU 工程 |
| [第 54 章 人脸关键点与人脸对齐](../p11-keypoints/ch54-face-landmarks.md) | `code/dl_labs/ch54_face_align` | 独立 CPU 工程 |
| [第 55 章 手部关键点与手势识别](../p11-keypoints/ch55-hand-gesture.md) | `code/dl_labs/ch55_hand` | 独立 CPU 工程 |
| [第 56 章 人脸检测（MTCNN/RetinaFace）](../p12-face/ch56-face-detection.md) | `code/dl_labs/ch56_face_detect` | 独立 CPU 工程 |
| [第 57 章 人脸识别与比对（embedding、ArcFace、度量学习）](../p12-face/ch57-face-recognition.md) | `code/dl_labs/ch57_face_recog` | 独立 CPU 工程 |
| [第 58 章 活体检测与人脸属性分析](../p12-face/ch58-liveness-attributes.md) | `code/dl_labs/ch58_liveness_attr` | 独立 CPU 工程 |
| [第 59 章 光流：假设、数值求解与失效诊断](../p13-tracking/ch59-optical-flow.md) | `ch59_optical_flow` | 主工程 example |
| [第 60 章 卡尔曼滤波：状态、噪声与一致性](../p13-tracking/ch60-kalman.md) | `ch60_kalman` | 主工程 example |
| [第 61 章 粒子滤波：非线性、多峰与退化](../p13-tracking/ch61-particle-filter.md) | `ch61_particle_filter` | 主工程 example |
| [第 62 章 多目标跟踪（SORT / DeepSORT）](../p13-tracking/ch62-mot.md) | `ch62_mot` | 主工程 example |
| [第 63 章 背景建模与运动检测](../p13-tracking/ch63-background-motion.md) | `ch63_background_motion` | 主工程 example |
| [第 64 章 相机标定（棋盘格、张正友标定法）](../p14-3d/ch64-calibration.md) | `ch64_calibration` | 主工程 example |
| [第 65 章 立体视觉与深度：几何、匹配和置信度](../p14-3d/ch65-stereo-depth.md) | `ch65_stereo_depth` | 主工程 example |
| [第 66 章 对极几何、PnP 与鲁棒位姿估计](../p14-3d/ch66-epipolar-pnp.md) | `ch66_epipolar_pnp` | 主工程 example |
| [第 67 章 点云处理入门](../p14-3d/ch67-pointcloud.md) | `ch67_pointcloud` | 主工程 example |
| [第 68 章 运动恢复结构 SfM 与视觉 SLAM 概览](../p14-3d/ch68-sfm-slam.md) | `ch68_sfm_slam` | 主工程 example |
| [第 69 章 自编码器与表示学习](../p15-generative/ch69-autoencoder.md) | `code/dl_labs/ch69_autoencoder` | 独立 CPU 工程 |
| [第 70 章 生成对抗网络 GAN](../p15-generative/ch70-gan.md) | `code/dl_labs/ch70_gan` | 独立 CPU 工程 |
| [第 71 章 扩散模型：手写 DDPM 与理解 Stable Diffusion](../p15-generative/ch71-diffusion.md) | `code/dl_labs/ch71_diffusion` | 独立 CPU 工程 |
| [第 72 章 超分辨率、去噪与风格迁移](../p15-generative/ch72-superres-style.md) | `code/dl_labs/ch72_restoration` | 独立 CPU 工程 |
| [第 73 章 视觉-语言多模态（CLIP、图文检索、图像描述）](../p15-generative/ch73-clip-vlm.md) | `code/dl_labs/ch73_clip` | 独立 CPU 工程 |
| [第 74 章 视频读写与帧处理](../p16-video/ch74-video-io.md) | `ch74_video_io` | 主工程 example |
| [第 75 章 RTSP / 流媒体接入与实时处理管线](../p16-video/ch75-rtsp-streaming.md) | `ch75_rtsp_pipeline` | 主工程 example |
| [第 76 章 动作识别与时序模型](../p16-video/ch76-action-recognition.md) | `ch76_action_recognition` | 主工程 example |
| [第 77 章 视频目标检测与跟踪落地](../p16-video/ch77-video-detection.md) | `ch77_video_detection` | 主工程 example |
| [第 78 章 亚像素边缘、精密测量与不确定度](../p17-industrial/ch78-metrology.md) | `ch78_metrology` | 主工程 example |
| [第 79 章 视觉定位、机器人引导与手眼标定](../p17-industrial/ch79-alignment-guidance.md) | `ch79_alignment` | 主工程 example |
| [第 80 章 表面缺陷检测：从成像到量产验收](../p17-industrial/ch80-defect.md) | `ch80_defect` | 主工程 example |
| [第 81 章 工业读码：一维码 / 二维码 / DataMatrix 与字符 OCR](../p17-industrial/ch81-code-reading.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 82 章 光源、镜头与成像方案（硬件与打光常识）](../p17-industrial/ch82-imaging-hardware.md) | `ch82_optics` | 主工程 example |
| [第 83 章 检测结果清洗：置信度/类别/尺寸/长宽比过滤](../p18-events/ch83-filtering-basics.md) | `ch83_filtering_basics` | 主工程 example |
| [第 84 章 ROI 区域过滤与多边形内外判断](../p18-events/ch84-roi-filter.md) | `ch84_roi_filter` | 主工程 example |
| [第 85 章 时间窗口过滤与状态机（N/M 投票、去抖）](../p18-events/ch85-temporal-window.md) | `ch85_temporal_window` | 主工程 example |
| [第 86 章 冷却（cooldown）、跨帧去重、告警聚合与节流](../p18-events/ch86-cooldown-dedup.md) | `ch86_cooldown_dedup` | 主工程 example |
| [第 87 章 绊线检测（line-crossing）与方向计数](../p18-events/ch87-tripwire.md) | `ch87_tripwire` | 主工程 example |
| [第 88 章 区域入侵、停留（dwell）与目标计数](../p18-events/ch88-intrusion-dwell.md) | `ch88_intrusion_dwell` | 主工程 example |
| [第 89 章 轨迹与行为分析（速度、方向、逆行、越界、热力图）](../p18-events/ch89-trajectory.md) | `ch89_trajectory` | 主工程 example |
| [第 90 章 布防排班、相机遮挡（tamper）与场景鲁棒性](../p18-events/ch90-arming-tamper.md) | `ch90_arming_tamper` | 主工程 example |
| [第 91 章 事件联动与推送：抓拍、录像、Webhook/MQTT/Kafka](../p18-events/ch91-actions-push.md) | `ch91_event_sink` | 主工程 example |
| [第 92 章 多路流并发调度与工程架构：线程、汇聚、背压与抽帧](../p18-events/ch92-multistream.md) | `ch92_concurrency` | 主工程 example |
| [第 93 章 设计一个可配置的规则引擎（过滤器管线）](../p18-events/ch93-rule-engine.md) | `ch93_rule_engine` | 主工程 example |
| [第 94 章 ONNX 跨框架推理：契约、验证与兼容性](../p19-deploy/ch94-onnx.md) | `code/dl_labs/ch94_tract_onnx` | 独立 CPU 工程 |
| [第 95 章 模型量化、剪枝与压缩：精度—性能共同验证](../p19-deploy/ch95-quantization.md) | `ch95_quantization` | 主工程 example |
| [第 96 章 Axum 视觉服务：并发、背压与 SLO](../p19-deploy/ch96-web-service.md) | `code/service_demo` | 独立服务工程 |
| [第 97 章 项目：实时摄像头处理管线（采集→检测→跟踪）](../p20-projects/ch97-realtime-pipeline.md) | `code/projects/ch97_pipeline` | 独立 CPU 工程 |
| [第 98 章 项目：二维码与条形码识别](../p20-projects/ch98-qrcode.md) | `code/projects/ch98_qrcode` | 独立 CPU 工程 |
| [第 99 章 项目：OCR 文字检测与识别](../p20-projects/ch99-ocr.md) | `code/projects/ch99_ocr` | 独立 CPU 工程 |
| [第 100 章 项目：以图搜图引擎（特征 + 向量检索）](../p20-projects/ch100-image-search.md) | `code/projects/ch100_image_search` | 独立 CPU 工程 |
| [第 101 章 项目：人脸考勤 / 门禁 Demo](../p20-projects/ch101-face-attendance.md) | `code/projects/ch101_face_attendance` | 独立 CPU 工程 |
| [第 102 章 项目：工业缺陷检测系统](../p20-projects/ch102-defect-system.md) | `code/projects/ch102_defect` | 独立 CPU 工程 |
| [第 103 章 项目：周界入侵报警（ROI + 绊线 + cooldown）](../p20-projects/ch103-perimeter-intrusion.md) | `code/projects/ch103_perimeter` | 独立 CPU 工程 |
| [第 104 章 项目：客流 / 车流计数（绊线 + 跟踪 + 窗口）](../p20-projects/ch104-people-counting.md) | `code/projects/ch104_counting` | 独立 CPU 工程 |
| [第 105 章 项目：健身姿态计数器（关键点应用）](../p20-projects/ch105-pose-counter.md) | `code/projects/ch105_pose_counter` | 独立 CPU 工程 |
| [第 106 章 项目：全景图拼接工具](../p20-projects/ch106-panorama.md) | `code/projects/ch106_panorama` | 独立 CPU 工程 |
| [第 107 章 项目：端到端把模型部署成 Web 服务](../p20-projects/ch107-deploy-service.md) | `code/projects/ch107_deploy` | 独立 CPU 工程 |
| [第 108 章 性能工程：测量、内存、并行与 SIMD](../p21-mastery/ch108-performance.md) | `ch108_parallel` | 主工程 example |
| [第 109 章 GPU 加速：Metal、CUDA 与 wgpu 的性能模型](../p21-mastery/ch109-gpu.md) | `code/gpu_demo` | GPU：按正文运行 |
| [第 110 章 WebAssembly 视觉：浏览器边界与可测性能](../p21-mastery/ch110-wasm.md) | `code/wasm_demo` | WASM：按正文构建 |
| [第 111 章 边缘与嵌入式部署：资源、热设计与故障恢复](../p21-mastery/ch111-embedded.md) | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |
| [第 112 章 生产工程：架构、测试、观测与发布](../p21-mastery/ch112-engineering.md) | `ch112_engineering` | 主工程 example |
| [第 113 章 数据与 MLOps：从血缘到线上质量闭环](../p21-mastery/ch113-datasets-mlops.md) | `ch113_datasets_mlops` | 主工程 example |

## 验证范围

主工程的 `cargo check --examples` 不覆盖独立工程。按各工程自己的清单和锁文件验证；成功编译也不能代替执行、数值对齐或真实数据实验。完整学习目标见 [附录 J](j-capstone.md)。
