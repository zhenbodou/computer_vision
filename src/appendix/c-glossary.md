# 附录 C：中英术语对照表

> **怎么用这份对照表**
> - 按主题分组，每条给出**中文 · English · 一句话说明**，方便你查词、对齐中英文献、看懂论文里的术语。
> - 只做对照与一句话释义，**不展开**——想深入某个概念，去正文对应章节；数学名词的直觉见 [附录 B](b-math.md)。
> - 术语中英译法以本书正文为准（对应写作规范 STYLE §6），全书统一。

## 图像基础（Image Basics）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 图像 | image | 由像素排成的二维数字阵列 |
| 像素 | pixel | 图像的最小单元，一个带颜色/亮度的小方格 |
| 分辨率 | resolution | 图像的宽 × 高像素数 |
| 通道 | channel | 颜色分量层，如 RGB 三通道 |
| 灰度图 | grayscale image | 单通道、只有明暗的图 |
| 位深 | bit depth | 每像素用多少位表示，如 8-bit |
| 色彩空间 | color space | 表示颜色的坐标系，如 RGB/HSV/Lab |
| RGB | RGB | 红绿蓝三原色的加色模型 |
| HSV | HSV | 色相/饱和度/明度模型，更贴近人对颜色的描述 |
| YUV | YUV | 亮度 + 色度模型，视频编码常用 |
| 直方图 | histogram | 像素值的分布统计 |
| 直方图均衡化 | histogram equalization | 拉开对比度的经典手法 |
| 亮度 | brightness | 图像整体的明暗 |
| 对比度 | contrast | 明暗差异的程度 |
| Gamma 校正 | gamma correction | 非线性地调整亮度 |
| 查找表 | LUT (lookup table) | 预存的像素映射表，加速点运算 |
| 插值 | interpolation | 缩放/旋转时估算新像素值（最近邻/双线性/双三次） |
| 感兴趣区域 | ROI (region of interest) | 只处理图像中的某块区域 |
| 掩膜 | mask | 标记「哪些像素参与运算」的二值图 |
| 二值图 | binary image | 只有 0/255 两种取值的图 |

## 经典图像处理（Classic Image Processing）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 卷积 | convolution | 用一个小核在图上滑动、加权求和 |
| 卷积核 | kernel / filter | 卷积用的小权重方阵 |
| 滤波 | filtering | 用核处理图像（模糊、锐化等） |
| 高斯模糊 | Gaussian blur | 用高斯核平滑去噪 |
| 中值滤波 | median filter | 取邻域中位数，专治椒盐噪声 |
| 双边滤波 | bilateral filter | 保住边缘的去噪 |
| 梯度 | gradient | 像素变化的快慢与方向 |
| 边缘检测 | edge detection | 找亮度突变的地方 |
| Sobel 算子 | Sobel operator | 常用的梯度/边缘算子 |
| Canny 边缘 | Canny edge detector | 经典的多步边缘检测 |
| 阈值分割 | thresholding | 按灰度值把图分成前景/背景 |
| 大津法 | Otsu's method | 自动选最佳全局阈值 |
| 自适应阈值 | adaptive threshold | 逐区域各定各的阈值 |
| 形态学 | morphology | 基于形状的运算（腐蚀/膨胀等） |
| 腐蚀 | erosion | 缩小白色区域 |
| 膨胀 | dilation | 扩大白色区域 |
| 开运算 | opening | 先腐蚀后膨胀，去小噪点 |
| 闭运算 | closing | 先膨胀后腐蚀，填小孔洞 |
| 连通域 | connected component | 相连的一片同类像素 |
| 轮廓 | contour | 物体的边界线 |
| 霍夫变换 | Hough transform | 投票法检测直线/圆 |
| 模板匹配 | template matching | 在大图里滑动找小图 |
| 图像金字塔 | image pyramid | 同一图像的多尺度栈 |
| 傅里叶变换 | Fourier transform | 把图像转到频域 |
| 频域 | frequency domain | 用频率成分描述图像 |

## 特征工程（Features）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 特征 | feature | 能描述图像内容的信息 |
| 关键点 | keypoint / interest point | 图中显著、可重复检测的点 |
| 角点 | corner | 两个方向都变化剧烈的点 |
| Harris 角点 | Harris corner | 经典的角点检测法 |
| FAST | FAST | 快速角点检测 |
| 描述子 | descriptor | 刻画关键点邻域的特征向量 |
| SIFT | SIFT | 尺度不变特征变换，经典描述子 |
| ORB | ORB | 快速二进制描述子（FAST + BRIEF） |
| BRIEF | BRIEF | 二进制描述子 |
| 特征匹配 | feature matching | 找两图中相互对应的特征点 |
| 汉明距离 | Hamming distance | 两个二进制串不同的位数 |
| 随机采样一致性 | RANSAC | 从含外点的数据里稳健估计模型 |
| 内点/外点 | inlier / outlier | 符合/不符合模型的点 |
| 单应矩阵 | homography | 平面到平面的 3×3 投影变换 |
| 仿射变换 | affine transform | 保持平行的线性变换 + 平移 |
| 图像配准 | image registration | 把两张图对齐到同一坐标系 |
| 全景拼接 | image stitching / panorama | 多张图拼成一张宽幅 |
| 方向梯度直方图 | HOG | 用梯度方向直方图描述形状 |
| 局部二值模式 | LBP | 编码局部明暗关系的纹理特征 |
| Haar 特征 | Haar-like feature | 类矩形明暗差特征 |
| 词袋模型 | BoVW (bag of visual words) | 把图像表示成「视觉单词」的频率 |
| 尺度不变 | scale-invariant | 物体放大缩小后仍能识别 |

## 机器学习（Machine Learning）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 机器学习 | machine learning | 从数据里自动学出规律 |
| 监督学习 | supervised learning | 用带标签的数据训练 |
| 无监督学习 | unsupervised learning | 无标签，自己找数据结构 |
| 分类 | classification | 预测离散类别 |
| 回归 | regression | 预测连续数值 |
| 聚类 | clustering | 把相似样本分成组 |
| 特征向量 | feature vector | 描述一个样本的一串数 |
| 训练集/测试集 | training / test set | 用于学习/用于评估的数据 |
| 过拟合 | overfitting | 记死了训练集，泛化差 |
| 欠拟合 | underfitting | 没学够，训练集都做不好 |
| 泛化 | generalization | 在没见过的数据上的表现 |
| K 均值 | K-means | 经典聚类算法 |
| 决策树 | decision tree | 按规则层层分叉的树模型 |
| 支持向量机 | SVM (support vector machine) | 找最大间隔的分界面 |
| 朴素贝叶斯 | naive Bayes | 基于贝叶斯的简单分类器 |
| 逻辑回归 | logistic regression | 输出概率的线性分类器 |
| 降维 | dimensionality reduction | 减少特征的维度 |
| 主成分分析 | PCA | 沿方差最大的方向降维 |
| 滑动窗口 | sliding window | 在图上逐块扫描来做检测 |

## 深度学习（Deep Learning）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 神经网络 | neural network | 多层「神经元」堆成的模型 |
| 神经元 | neuron | 加权求和 + 激活的基本单元 |
| 权重/偏置 | weight / bias | 网络要学的参数 |
| 激活函数 | activation function | 给网络引入非线性 |
| ReLU | ReLU | 最常用的激活，负数归零 |
| 反向传播 | backpropagation | 从损失反推每个参数的梯度 |
| 梯度下降 | gradient descent | 顺梯度反方向更新参数 |
| 学习率 | learning rate | 每步更新的步长 |
| 优化器 | optimizer | 更新参数的策略（SGD/Adam） |
| 损失函数 | loss function | 衡量预测错了多少 |
| 交叉熵 | cross-entropy | 分类任务常用的损失 |
| 卷积神经网络 | CNN | 用卷积提特征的网络 |
| 池化 | pooling | 下采样，缩小特征图 |
| 感受野 | receptive field | 一个输出像素「看到」的输入范围 |
| 特征图 | feature map | 卷积层输出的张量 |
| 批 | batch | 一次喂进网络的一组样本 |
| 批归一化 | batch normalization | 稳定训练的归一化层 |
| 数据增强 | data augmentation | 变换图像来扩充训练数据 |
| 迁移学习 | transfer learning | 拿预训练模型改造到新任务 |
| 微调 | fine-tuning | 在预训练权重上继续训练 |
| 预训练模型 | pretrained model | 别人已训好的现成权重 |
| 张量 | tensor | 多维数组，深度学习的基本数据 |
| 注意力机制 | attention | 让模型聚焦到重要部分 |
| Transformer | Transformer | 基于注意力的网络架构 |
| 残差连接 | residual / skip connection | 跨层直连，缓解深网退化 |
| 轮次 | epoch | 把训练集完整过一遍 |
| 推理 | inference | 用训好的模型做预测 |

## 检测·分割·关键点（Detection / Segmentation / Keypoints）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 目标检测 | object detection | 找出图中物体的类别与位置框 |
| 边界框 | bounding box | 框住物体的矩形 |
| 锚框 | anchor / anchor box | 预设的候选框模板 |
| 单阶段/两阶段 | one-stage / two-stage | 检测器的两大流派 |
| YOLO | YOLO | 经典的单阶段实时检测器 |
| 非极大值抑制 | NMS (non-maximum suppression) | 去掉相互重叠的多余框 |
| 交并比 | IoU (intersection over union) | 两个框重叠的程度 |
| 平均精度均值 | mAP (mean average precision) | 检测任务的主评测指标 |
| 置信度 | confidence score | 模型对预测结果的把握 |
| 语义分割 | semantic segmentation | 逐像素分类，不区分个体 |
| 实例分割 | instance segmentation | 区分同类的不同个体 |
| 全景分割 | panoptic segmentation | 语义 + 实例合而为一 |
| U-Net | U-Net | 经典的分割网络 |
| SAM | Segment Anything Model | 可交互的通用分割大模型 |
| 关键点检测 | keypoint detection | 定位物体上的特征点 |
| 姿态估计 | pose estimation | 估计人体/物体的关节位置 |
| 热图回归 | heatmap regression | 用概率热图来定位关键点 |
| 人脸检测 | face detection | 找出画面里人脸的位置 |
| 人脸识别 | face recognition | 判断「这张脸是谁」 |
| 人脸嵌入 | face embedding | 人脸的特征向量 |
| 度量学习 | metric learning | 学习「相似度」的表示 |

## 跟踪·三维视觉（Tracking / 3D Vision）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 目标跟踪 | object tracking | 在视频里持续锁定同一目标 |
| 多目标跟踪 | MOT (multi-object tracking) | 同时跟踪多个目标 |
| 光流 | optical flow | 像素在相邻帧间的运动 |
| Lucas-Kanade | Lucas-Kanade | 经典的稀疏光流法 |
| 卡尔曼滤波 | Kalman filter | 预测 + 观测融合的最优估计 |
| 粒子滤波 | particle filter | 用一群样本近似分布的滤波 |
| SORT / DeepSORT | SORT / DeepSORT | 检测 + 卡尔曼的多目标跟踪 |
| 背景建模 | background modeling | 学出背景以检出运动前景 |
| 相机标定 | camera calibration | 求相机的内参与外参 |
| 内参/外参 | intrinsic / extrinsic | 相机自身/相对世界的参数 |
| 畸变 | distortion | 镜头造成的图像变形 |
| 对极几何 | epipolar geometry | 双视图之间的几何约束 |
| 本质/基础矩阵 | essential / fundamental matrix | 编码双目相对位姿 |
| 透视 n 点 | PnP (perspective-n-point) | 由 3D-2D 点对求相机位姿 |
| 立体视觉 | stereo vision | 用双目求深度 |
| 视差 | disparity | 左右图对应点的水平差，反推深度 |
| 深度图 | depth map | 每像素到相机的距离 |
| 点云 | point cloud | 一堆三维点的集合 |
| 同时定位与建图 | SLAM | 边定位边建地图 |
| 运动恢复结构 | SfM (structure from motion) | 从多张图恢复三维结构 |

## 事件·规则·工程（Events / Rules / Engineering）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 后处理 | post-processing | 把模型原始输出加工成业务结果 |
| ROI 过滤 | ROI filtering | 只在指定区域内报事件 |
| 点在多边形内 | point-in-polygon | 判断一个点是否落在区域内 |
| 绊线/越线 | tripwire / line-crossing | 目标穿越设定线时触发事件 |
| 方向计数 | directional counting | 按穿越方向分别计数 |
| 时间窗口 | temporal window | 在一段时间内统计或投票 |
| N/M 投票 | N-of-M voting | M 帧里有 N 帧命中才算数，去抖 |
| 状态机 | state machine | 用状态转移管理事件逻辑 |
| 冷却 | cooldown | 触发后一段时间内不再重复报警 |
| 去重 | deduplication | 合并同一事件的重复上报 |
| 节流 | throttling | 限制事件上报的频率 |
| 区域入侵 | intrusion | 目标进入禁区 |
| 停留 | dwell | 目标在区域内滞留超时 |
| 轨迹 | trajectory / track | 目标随时间的位置序列 |
| 逆行 | wrong-way | 逆着规定方向运动 |
| 热力图 | heatmap | 累积活动密度的可视化 |
| 布防/撤防 | arming / disarming | 开启/关闭事件检测 |
| 排班 | scheduling | 按时间表布防 |
| 遮挡篡改 | tamper | 相机被遮挡、移动或失焦 |
| 事件联动 | action / linkage | 事件触发抓拍、录像、推送 |
| 抽帧 | frame skipping | 隔帧处理以节省算力 |
| 隐私打码 | privacy masking | 给敏感区域打马赛克 |

## 模型部署（Deployment）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 开放神经网络交换 | ONNX | 跨框架的开放模型交换格式 |
| 量化 | quantization | 用低精度（如 int8）压缩并加速模型 |
| 剪枝 | pruning | 去掉冗余权重来压缩模型 |
| 知识蒸馏 | knowledge distillation | 用大模型「教」小模型 |
| 推理引擎 | inference engine | 跑模型的运行时，如 tract/ort |
| 执行提供器 | execution provider | ONNX Runtime 的后端（CPU/GPU 等） |
| 边缘部署 | edge deployment | 部署到嵌入式/终端设备 |
| 服务化 | serving | 把模型包成 Web API |
| 批处理 | batching | 多个请求合并推理以提吞吐 |
| 延迟/吞吐 | latency / throughput | 单次耗时/单位时间处理量 |

## Rust 与生态（Rust / Ecosystem）

| 中文 | English | 一句话说明 |
|------|---------|-----------|
| 包 | crate | Rust 的库/包单元 |
| 构建工具 | Cargo | Rust 的构建与包管理工具 |
| 特质 | trait | 定义共享行为的接口 |
| 所有权 | ownership | Rust 内存管理的核心机制 |
| 借用 | borrowing | 临时使用而不夺走所有权 |
| 生命周期 | lifetime | 引用保持有效的时间范围 |
| 泛型 | generics | 对多种类型通用的代码 |
| 结果/可选值 | `Result` / `Option` | 表达「可能失败/可能为空」的类型 |
| image | `image` | 图像读写与像素操作 crate |
| imageproc | `imageproc` | 经典图像处理算法 crate |
| ndarray | `ndarray` | 多维数组/张量 crate |
| nalgebra | `nalgebra` | 线性代数与几何 crate |
| candle | `candle` | Hugging Face 的纯 Rust 深度学习框架 |
| burn | `burn` | 多后端、可训练的纯 Rust DL 框架 |
| tract | `tract` | 纯 Rust 的 ONNX 推理引擎 |
| ort | `ort` | ONNX Runtime 的 Rust 绑定（底层 C++） |
| linfa | `linfa` | Rust 的经典机器学习工具箱 |
| rayon | `rayon` | 数据并行库 |
| WebAssembly | WASM | 可在浏览器里运行的编译目标 |
| 单指令多数据 | SIMD | CPU 的一种并行加速方式 |
