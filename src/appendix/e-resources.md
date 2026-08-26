# 附录 E：进阶学习资源与论文清单

> **本附录导读**
> - 给你一张**靠谱的资源地图**：继续深入机器视觉，工具、论文、书籍、数据集分别去哪儿找。
> - Rust 视觉/ML 生态怎么找官方文档、怎么核对版本，一次说清。
> - 一份**按主题整理的经典论文清单**（论文名 + 作者/年份 + 一句话贡献），你可以照着自行检索精读。
> - 一个重要观念：这些论文和课程**大多是 Python/C++ 语境的，但原理是通用的**——你完全可以用本书的纯 Rust 思路把它们复现出来。

学到这里，你已经不再是那个"不知道从哪儿写第一行代码"的新手了。接下来的路，很大一部分要靠**自己读、自己查、自己复现**。这一节我不打算再讲新知识，而是把师兄这些年攒下的"藏宝图"交给你——哪些工具值得投入，哪些论文是绕不开的地基，哪些书和数据集能让你少走弯路。

> 💡 **提示**：网络链接会失效，但**crate 名、论文标题、作者、年份**不会。所以下面我尽量给你"**可检索的关键词**"而不是一堆随时会 404 的死链。工具去 [crates.io](https://crates.io) / [docs.rs](https://docs.rs) 搜 crate 名，论文去 Google Scholar 或 arXiv 搜标题，准没错。

## 一、Rust 视觉 / ML 生态：你的工具箱

本书用到的每一个 crate，都有活跃维护的官方仓库和文档。**统一的获取方式**：在 `docs.rs/<crate 名>` 看 API 文档，在 `crates.io` 看版本与下载量，在对应 GitHub 组织看源码与 issue。下面按"定位一句话 + 去哪找"整理。

**图像与经典处理**

- **`image`** —— 纯 Rust 的图像读写、像素访问、几何与颜色变换基础库，本书第二部分的主力。GitHub 组织 `image-rs`。（用法见 [第 7 章](../p02-image-basics/ch07-load-save.md)）
- **`imageproc`** —— 建立在 `image` 之上的经典算法库：滤波、边缘、形态学、轮廓、霍夫、模板匹配、角点、HOG、绘制。GitHub `image-rs/imageproc`。本书第三、四部分几乎章章都用它。

**数值与线代**

- **`ndarray`** —— 多维数组/张量运算（类比 Python 的 NumPy）。GitHub 组织 `rust-ndarray`。见 [第 29 章](../p05-numeric/ch29-ndarray.md)。
- **`nalgebra`** —— 线性代数与几何变换（矩阵、SVD、四元数、齐次坐标），三维视觉与标定的基石。GitHub 组织 `dimforge`。见 [第 30 章](../p05-numeric/ch30-nalgebra.md)。

**传统机器学习**

- **`linfa`** —— Rust 的"scikit-learn"：分类、回归、聚类、SVM、决策树一站式。GitHub `rust-ml/linfa`。见 [第 33 章](../p06-ml/ch33-linfa.md)。

**深度学习与推理**（本书第七部分及以后，选型详见 [第 39 章](../p07-dl-basics/ch39-dl-ecosystem.md)）

- **`candle`** —— Hugging Face 出品的极简 ML 框架，safetensors 原生、CPU/CUDA/Metal，偏 LLM/Transformer 与手搓&微调模型。纯 Rust。GitHub `huggingface/candle`。
- **`burn`** —— 带自动微分、多后端可切换的**训练型**框架（PyTorch 式体验，一套代码切 CPU/WGPU/CUDA）。纯 Rust。GitHub `tracel-ai/burn`。
- **`tract`** —— Sonos 出品的纯 Rust ONNX/NNEF 推理引擎，**零 C/C++ 依赖**、交叉编译友好，"要纯 Rust 跑现成 ONNX"的首选。GitHub 搜 `tract`。
- **`ort`** —— ONNX Runtime 的 Rust 绑定，**底层链接 C++ onnxruntime（非纯 Rust）**，换来最强性能与最全算子。GitHub `pykeio/ort`，文档 `ort.pyke.io`。跨框架部署见 [第 94 章](../p19-deploy/ch94-onnx.md)。

**rust-cv 组织与更多视觉专用 crate**

- **`rust-cv`** —— 一个专注计算机视觉的 GitHub 组织，聚集了多个几何视觉与特征相关 crate。其中：
  - **`akaze`** —— AKAZE 特征提取，纯 Rust 的旋转/尺度不变特征（`rust-cv/akaze`），可补 `imageproc` 里 BRIEF 不抗旋转的短板。
  - **`arrsac` / `sample-consensus`** —— RANSAC 家族的采样一致性估计，做特征匹配几何验证时很有用（见 [第 25 章](../p04-features/ch25-matching-ransac.md)）。用前请在 crates.io 核对版本与维护状态。
- **`kornia-rs`** —— Rust 的低层三维计算机视觉库（对标 Python 的 Kornia），GitHub `kornia/kornia-rs`，主页 `kornia.org`。生态较新、发展中，适合关注三维方向的读者跟进。

> ⚠️ **注意**：Rust 视觉生态**发展很快**，版本迭代频繁。本书成书时的实测版本是 `image 0.25`、`imageproc 0.27`、`ndarray 0.17`、`nalgebra 0.35`、`candle 0.11`、`burn 0.21`、`tract 0.23.5`、`ort 2.0-rc`、`linfa 0.8`。你上手时**务必以 crates.io 上的最新稳定版为准**，API 若有出入，一切以 `docs.rs` 对应版本页面为权威。crate 速查见 [附录 A](a-crates.md)。

## 二、经典论文清单（按主题）

下面这份清单不求全，只求"**绕不开**"——每一篇都是它那个方向的地标。格式为：**论文名**（作者，年份）— 一句话贡献。读法建议：先看本书对应章节建立直觉，再回来精读原文，最后试着用纯 Rust 复现核心步骤。

> 📖 **术语**：论文里常见的 **et al.** 是拉丁语"及其他人"的缩写，下面用"等"表示。**arXiv** 是免费的论文预印本网站，绝大多数深度学习论文都能在上面免费下载。

### 边缘检测

- **A Computational Approach to Edge Detection**（Canny，1986）— 提出边缘检测的三条最优准则（检测率、定位精度、单一响应），配合非极大值抑制与双阈值，即经典 Canny 算子。对应 [第 15 章](../p03-classic/ch15-edges.md)。

### 局部特征与匹配

- **Distinctive Image Features from Scale-Invariant Keypoints**（Lowe，2004；会议版 1999）— SIFT，尺度与旋转不变的局部特征，特征工程时代的巅峰之作。
- **SURF: Speeded Up Robust Features**（Bay 等，2006）— 用积分图加速的近似 SIFT，更快。
- **ORB: An Efficient Alternative to SIFT or SURF**（Rublee 等，2011）— FAST 角点 + 带方向的 BRIEF 描述子，快且无专利，工程界宠儿。
- **KAZE / Fast Explicit Diffusion...（AKAZE）**（Alcantarilla 等，2012/2013）— 在非线性尺度空间上提特征，边缘保持更好。对应 [第 24 章](../p04-features/ch24-descriptors.md)、[第 25 章](../p04-features/ch25-matching-ransac.md)。

### 目标检测

- **Rapid Object Detection using a Boosted Cascade of Simple Features**（Viola & Jones，2001）— Haar 特征 + 积分图 + AdaBoost 级联，第一个能实时跑的人脸检测器。
- **Histograms of Oriented Gradients for Human Detection**（Dalal & Triggs，2005）— HOG 特征 + 线性 SVM，行人检测的经典组合（本书 [第 27 章](../p04-features/ch27-handcrafted-features.md)、[第 34 章](../p06-ml/ch34-sliding-window.md) 会实现）。
- **R-CNN / Fast R-CNN / Faster R-CNN**（Girshick 等，2014–2015）— 两阶段检测的奠基三部曲，Faster R-CNN 引入 RPN 实现端到端。
- **You Only Look Once (YOLO)**（Redmon 等，2016）— 把检测当成单次回归，开创单阶段实时检测；后续 YOLO 系列长盛不衰。
- **SSD: Single Shot MultiBox Detector**（Liu 等，2016）— 多尺度特征图上的单阶段检测。
- **End-to-End Object Detection with Transformers (DETR)**（Carion 等，2020）— 用 Transformer + 集合预测，去掉了 anchor 与 NMS。以上对应 [第九部分](../p09-detection/ch45-detection-principles.md)。

### 图像分割

- **Fully Convolutional Networks (FCN)**（Long 等，2015）— 把分类网改成全卷积，实现像素级语义分割。
- **U-Net**（Ronneberger 等，2015）— 编码器-解码器 + 跳跃连接，医学影像分割的常青树。
- **Mask R-CNN**（He 等，2017）— 在 Faster R-CNN 上加一条掩码分支，实例分割标杆。
- **Segment Anything (SAM)**（Kirillov 等，Meta，2023）— 可提示的通用分割基础模型，"分割一切"。对应 [第十部分](../p10-segmentation/ch49-semantic-seg.md) 与 [第 51 章](../p10-segmentation/ch51-sam.md)。

### 关键点与姿态估计

- **Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (OpenPose)**（Cao 等，2017）— 用部件亲和场自底向上估计多人姿态。
- **Deep High-Resolution Representation Learning (HRNet)**（Sun 等，2019）— 全程维持高分辨率表示，关键点定位更准。对应 [第 53 章](../p11-keypoints/ch53-human-pose.md)。

### 目标跟踪

- **High-Speed Tracking with Kernelized Correlation Filters (KCF)**（Henriques 等，2015）— 利用循环矩阵 + FFT，把相关滤波跟踪做到极快。
- **Simple Online and Realtime Tracking (SORT)**（Bewley 等，2016）— 卡尔曼滤波 + 匈牙利算法，极简高效的多目标跟踪。
- **SORT with a Deep Association Metric (DeepSORT)**（Wojke 等，2017）— 给 SORT 加上外观特征，缓解 ID 频繁跳变。对应 [第 60 章](../p13-tracking/ch60-kalman.md)、[第 62 章](../p13-tracking/ch62-mot.md)。

### 三维视觉与 SLAM

- **A Flexible New Technique for Camera Calibration**（Zhang，2000）— 张正友标定法，用棋盘格标定相机内参，工业界事实标准。对应 [第 64 章](../p14-3d/ch64-calibration.md)。
- **Structure-from-Motion Revisited (COLMAP)**（Schönberger & Frahm，2016）— 现代增量式 SfM 的代表实现，从多张照片重建三维结构。
- **ORB-SLAM / ORB-SLAM2**（Mur-Artal 等，2015/2017）— 基于 ORB 特征的通用、精确的实时视觉 SLAM 系统。对应 [第 68 章](../p14-3d/ch68-sfm-slam.md)。

### 分类骨干网络（Backbone 演进）

- **LeNet-5 / Gradient-Based Learning...**（LeCun 等，1998）— 最早的卷积网络，手写数字识别。
- **AlexNet / ImageNet Classification with Deep CNNs**（Krizhevsky 等，2012）— 引爆深度学习的一战，ImageNet 上碾压传统方法。
- **VGG / Very Deep Convolutional Networks**（Simonyan & Zisserman，2014）— 用堆叠 3×3 卷积把网络做深。
- **ResNet / Deep Residual Learning**（He 等，2016）— 残差连接解决深层网络退化，几乎所有现代骨干的基础。
- **MobileNets**（Howard 等，2017）— 深度可分离卷积，为移动端/边缘设备而生（呼应本书 [第 111 章](../p21-mastery/ch111-embedded.md)）。
- **Vision Transformer (ViT) / An Image is Worth 16x16 Words**（Dosovitskiy 等，2021）— 把 Transformer 搬到图像块上，开启视觉 Transformer 时代。对应 [第 38 章](../p07-dl-basics/ch38-architectures.md)。

### 生成式与多模态

- **Generative Adversarial Nets (GAN)**（Goodfellow 等，2014）— 生成器与判别器对抗博弈，图像生成的里程碑。对应 [第 70 章](../p15-generative/ch70-gan.md)。
- **Denoising Diffusion Probabilistic Models (DDPM)**（Ho 等，2020）— 扩散模型的奠基工作，用"加噪-去噪"学习生成。
- **High-Resolution Image Synthesis with Latent Diffusion Models**（Rombach 等，2022）— 潜空间扩散，即 Stable Diffusion 的基础。对应 [第 71 章](../p15-generative/ch71-diffusion.md)。
- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**（Radford 等，OpenAI，2021）— 用图文对比学习对齐视觉与语言，撑起以文搜图与零样本分类。对应 [第 73 章](../p15-generative/ch73-clip-vlm.md)。

> 💡 **提示**：别被清单吓到。**不必一次读完，更不必读懂每个公式**。挑一个你最感兴趣的方向（比如检测），把那一组 4–5 篇按时间顺序读下来，你就能清晰看到一个领域是怎么一步步演进的——这种"看着一个方向长大"的体验，比零散读一百篇都值。

## 三、书籍与课程

**经典教材**

- **《Computer Vision: Algorithms and Applications》**（Richard Szeliski）— 机器视觉的"大部头"综述，覆盖面极广。作者在个人主页长期提供**免费电子版**（搜书名 + Szeliski 即可）。
- **《Multiple View Geometry in Computer Vision》**（Hartley & Zisserman）— 多视图几何的"圣经"，三维视觉方向必备，对应本书第十四部分。
- **《Digital Image Processing》**（Gonzalez & Woods）— 图像处理经典教材，第二、三部分的理论后盾。
- **《Computer Vision: A Modern Approach》**（Forsyth & Ponce）— 另一本广受欢迎的综合教材。

**深度学习资源**

- **《Deep Learning》**（Goodfellow、Bengio、Courville）— 深度学习理论奠基读物，官方提供免费在线版（搜 `deeplearningbook.org`）。
- **《动手学深度学习》(Dive into Deep Learning)**（李沐 等）— **中英文皆有、边讲边写代码**的免费在线书（搜 `d2l.ai`），对中文读者尤其友好。
- **Stanford CS231n**（视觉方向深度学习公开课）— 讲义与作业质量极高，搜 `CS231n` 可得课程主页与笔记。

> ⚠️ **注意**：上面这些资源**基本都用 Python + PyTorch/TensorFlow**。这没关系——**它们教的是原理，不是语言**。反向传播、卷积、注意力机制的数学是通用的；你在本书里已经用 `burn`/`candle` 把这些原理落到了纯 Rust 上。把它们当"理论教材"读，用本书当"Rust 实践手册"练，两者互补。

## 四、常用数据集

做视觉离不开数据。下面几个是各任务的"标准考场"，你在论文里会反复看到它们：

- **MNIST** — 7 万张 28×28 手写数字灰度图，10 类。深度学习的"Hello World"（本书 [第 35 章](../p06-ml/ch35-digit-recognition.md)、[第 40 章](../p07-dl-basics/ch40-burn-train.md) 用它）。
- **CIFAR-10 / CIFAR-100** — 32×32 彩色小图，10 类 / 100 类，小模型分类实验的常用基准。
- **ImageNet (ILSVRC)** — 上千类、百万级图像，分类骨干网络的"试金石"，AlexNet/ResNet 都在它上面证明自己。
- **PASCAL VOC** — 20 类，早期目标检测与分割的经典基准。
- **MS COCO** — 80 类，含检测框、实例掩码与人体关键点标注，是当今检测/分割领域的**主流基准**。
- **Cityscapes** — 城市街景语义分割，自动驾驶方向常用。
- **WIDER FACE / LFW** — 分别是人脸检测与人脸验证的标准基准（对应本书第十二部分）。
- **KITTI** — 自动驾驶场景的综合数据集（检测、立体、光流、里程计），三维视觉方向常用。

> 💡 **提示**：这些数据集都有官方主页提供下载与评测协议（搜数据集名即可）。用它们时**务必留意授权协议**（学术免费不等于可商用）。数据、标注与 MLOps 的工程实践，见本书 [第 113 章](../p21-mastery/ch113-datasets-mlops.md)。

## 结语：清单是地图，路要自己走

这份清单会跟着你走很久。刚开始你可能只认得其中几个名字，但随着你一个方向一个方向地深入，你会发现自己能在这张地图上标注出越来越多"我读过、我复现过"的点。

再强调一次那个贯穿本书的信念：**这些论文和课程大多诞生在 Python/C++ 的世界，但算法原理是语言无关的**。当别人还在"调包"时，你已经能用纯 Rust 把一篇论文的核心算法从零搭出来——这种"知其所以然"的能力，正是从"会用"迈向"精通"的分水岭。

带着这张地图，去 [附录 F](f-roadmap.md) 看看接下来这条成长路线该怎么走吧。🚀
