# 附录 B：数学基础速查（线代、概率、优化、几何）

> **怎么用这份速查**
> - 这不是数学教材，而是一张「够用就好」的**速查卡**：重直觉、重「在机器视觉里怎么用」，不追求严格证明。
> - 正文遇到看不懂的公式时，翻回这里对照，几分钟补上直觉再回去读。
> - 分四块——线性代数、概率统计、优化、几何。每条都标注了**本书哪里用到**，以及**Rust 里用哪个 crate 算**。

动手算数学，底层工具就两件：[第 29 章 ndarray](../p05-numeric/ch29-ndarray.md)（多维数组/张量）管「大块数据批量运算」，[第 30 章 nalgebra](../p05-numeric/ch30-nalgebra.md)（线性代数）管「小矩阵精密几何」。本书实测版本 `ndarray 0.17`、`nalgebra 0.35`。

## 一、线性代数：几乎所有几何问题的骨架

一张灰度图就是一个数字矩阵，一个特征就是一个向量。相机投影、单应、位姿、标定，剥开外壳全是**矩阵乘法、求逆、解方程**。

**向量与矩阵（vector / matrix）**：向量是一列数（一个点、一个特征），记 \\( \mathbf{v} \\)；矩阵是一个数表（一张图、一个变换），记 \\( A \\)。本书里向量默认是**列向量**。

**点积与范数（dot product / norm）**：两个向量的点积 \\( \mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i \\) 衡量它们「有多同向」；向量的范数（长度）是 \\( \lVert \mathbf{v} \rVert = \sqrt{\sum_i v_i^{2}} \\)。把点积除以两者的范数做归一化，就得到**余弦相似度**——以图搜图（[第 100 章](../p20-projects/ch100-image-search.md)）、人脸比对（[第 57 章](../p12-face/ch57-face-recognition.md)）判断两个特征向量像不像，靠的正是它。

**矩阵乘法（matrix multiplication）**：\\( C = AB \\)，其中 \\( C_{ij} = \sum_k A_{ik} B_{kj} \\)（C 的第 i 行 j 列等于 A 的第 i 行与 B 的第 j 列做点积）。一句直觉：**连续做两个变换，等于把两个矩阵乘起来**。注意它**不满足交换律**（\\( AB \ne BA \\)），所以「先旋转后平移」和「先平移后旋转」结果不同。

**转置（transpose）**：\\( (A^{T})_{ij} = A_{ji} \\)，把行和列互换。

**逆（inverse）**：满足 \\( A A^{-1} = I \\) 的矩阵，是解方程的钥匙——\\( A\mathbf{x} = \mathbf{b} \\) 的解就是 \\( \mathbf{x} = A^{-1}\mathbf{b} \\)。但**不是所有矩阵都可逆**。

**行列式（determinant）**：一个标量，衡量矩阵把空间「放大」了多少倍。2×2 的算法：

\\[ \det \begin{bmatrix} a & b \\\\ c & d \end{bmatrix} = ad - bc \\]

记住两点直觉就够用：**行列式为 0 就等价于矩阵不可逆**（它把空间压扁到了低维）；行列式的绝对值等于面积/体积的缩放倍数。

**特征值与特征向量（eigenvalue / eigenvector）**：满足 \\( A\mathbf{v} = \lambda \mathbf{v} \\) 的特殊方向 \\( \mathbf{v} \\) 与倍数 \\( \lambda \\)。一句直觉：**被矩阵作用后方向不变、只被拉伸 λ 倍的那些方向**。用途：Harris 角点用局部梯度矩阵的两个特征值判断「角/边/平面」（[第 23 章](../p04-features/ch23-corners.md)）；PCA 降维取协方差矩阵最大的几个特征向量。

**SVD（奇异值分解，singular value decomposition）**：任何矩阵都能拆成 \\( A = U \Sigma V^{T} \\)（\\( \Sigma \\) 对角线上是从大到小排列的「奇异值」）。它是线代里最万能的工具，三大用途在本书都会遇到：

- **最小二乘**：方程比未知数多（超定）、没有精确解时，SVD 给出「整体误差最小」的解——直线拟合、[第 30 章](../p05-numeric/ch30-nalgebra.md) 都靠它；
- **解单应 / 基础矩阵**：DLT 求单应矩阵 H 的线性方程组用 SVD 解最稳（[第 26 章](../p04-features/ch26-homography-stitching.md)、对极几何 [第 66 章](../p14-3d/ch66-epipolar-pnp.md)）；
- **降维**：保留大奇异值、丢掉小的，就是压缩与去噪，也是 PCA 的另一副面孔。

**齐次坐标（homogeneous coordinates）**：给 2D 点 \\( (x, y) \\) 补一个 1，写成 \\( (x, y, 1) \\)。好处是**平移、旋转、缩放、透视统统能塞进一个矩阵乘法**（否则平移只能「加」、没法「乘」）。用完把前两维**除以第三维**还原真实坐标，即「透视除法」 \\( (x, y, w) \to (x/w, y/w) \\)。这是相机模型（[第 6 章](../p01-start/ch06-camera-model.md)）、几何变换（[第 10 章](../p02-image-basics/ch10-geometry.md)）、单应矩阵（[第 26 章](../p04-features/ch26-homography-stitching.md)）的共同地基。

> 💡 **Rust 里用哪个**：小矩阵（3×3 相机矩阵、4×4 位姿）用 `nalgebra`——`Matrix3`/`Vector3` 创建、`*` 相乘、`.transpose()`、`.determinant()`、`.try_inverse()`（返回 `Option`）、`a.svd(true, true).solve(&b, eps)` 做最小二乘。大块数据/批量张量用 `ndarray` 的 `Array2` 与 `.dot()`。用法详见 [第 30 章](../p05-numeric/ch30-nalgebra.md)。

## 二、概率与统计：和噪声、不确定性打交道

现实里的测量都带噪声。概率就是量化「有多不确定」、并在新证据到来时更新判断的语言——跟踪、滤波、朴素贝叶斯全靠它。

**均值与方差（mean / variance）**：均值是「中心在哪」，方差是「散得多开」。

\\[ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i, \qquad \sigma^{2} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^{2} \\]

标准差 \\( \sigma \\) 是方差开根号，量纲和数据一致。算一块区域的亮度均值/方差，是缺陷检测、自适应阈值里最常用的统计量。

**协方差（covariance）**：衡量两个变量「一起变」的倾向（同涨同跌为正，此消彼长为负）；把多维变量两两之间的协方差排成矩阵，就是**协方差矩阵**。卡尔曼滤波用它表示状态估计的不确定性（[第 60 章](../p13-tracking/ch60-kalman.md)），PCA 则对它做特征分解来降维——这也是特征值/特征向量在统计里的登场方式。

**高斯分布（正态分布，Gaussian / normal distribution）**：那条钟形曲线，自然界最常见的分布。

\\[ p(x) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left( -\frac{(x-\mu)^{2}}{2\sigma^{2}} \right) \\]

记住「68–95–99.7」法则：数据落在 \\( \mu \pm \sigma \\)、\\( \mu \pm 2\sigma \\)、\\( \mu \pm 3\sigma \\) 内的比例约为 68%、95%、99.7%。用途：高斯模糊核的权重（[第 13 章](../p03-classic/ch13-convolution.md)）、卡尔曼滤波假设噪声服从高斯（[第 60 章](../p13-tracking/ch60-kalman.md)）。

**贝叶斯公式（Bayes）**：一句话——**用新证据更新旧信念**。

\\[ P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)} \\]

\\( P(A) \\) 是先验（看到证据前的信念），\\( P(A \mid B) \\) 是后验（看到证据 B 后更新的信念）。卡尔曼滤波、粒子滤波（[第 60](../p13-tracking/ch60-kalman.md)、[61 章](../p13-tracking/ch61-particle-filter.md)）本质都是「预测 → 用观测做贝叶斯更新」的循环；朴素贝叶斯分类器（[第 33 章](../p06-ml/ch33-linfa.md)）直接拿它算类别概率。

**似然（likelihood）**：给定一组参数时，观测到当前这批数据的「可能性」。直觉：**哪套参数最可能生成我看到的数据，就选哪套**（最大似然估计 MLE）——这是很多模型「学参数」的思想源头。

> 💡 **Rust 里用哪个**：均值/方差直接用 `ndarray` 的 `.mean()`、`.std()` 或手算；要从各种分布采样用 `rand` + `rand_distr`（如 `Normal`）；朴素贝叶斯用 `linfa-bayes`（注意 linfa 锁定 `ndarray 0.16`，版本坑见 [第 33 章](../p06-ml/ch33-linfa.md)）。

## 三、优化：模型是怎么「学」出来的

训练一个模型，就是不断调参数、让「损失」变小的过程。怎么调？顺着让损失下降最快的方向挪一小步——这就是优化。

**损失函数（loss function）**：一个衡量「模型现在错得多离谱」的数，越小越好。两个最常见的：

\\[ L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^{2}, \qquad L_{\text{CE}} = -\sum_{i} y_i \log \hat{y}_i \\]

均方误差 MSE 用于回归（预测连续值），交叉熵 CE 用于分类（\\( \hat{y}_i \\) 是预测的概率）。

**梯度（gradient）**：损失对每个参数求偏导、再拼成的向量 \\( \nabla L \\)，指向**损失上升最快**的方向。所以想让损失下降，就往它的**反方向**走一步。

**梯度下降（gradient descent）**：训练神经网络的引擎，核心就一条更新式：

\\[ \theta \leftarrow \theta - \eta\, \nabla L(\theta) \\]

\\( \theta \\) 是参数，\\( \eta \\) 是**学习率（learning rate）**，控制每步迈多大。太大会越过最低点甚至发散，太小则训练得慢。实战常用带动量的变体（SGD+Momentum、Adam）。

**凸与非凸（convex / non-convex）**：凸函数像一只碗，只有一个最低点，梯度下降一定能到底；非凸像连绵起伏的山谷，有很多「局部最低点」，可能卡在半山腰。深度网络的损失几乎都是**非凸**的——好在实践中找到的局部最优通常「够好用」。

这些概念集中用在深度学习：神经网络与反向传播（[第 36 章](../p07-dl-basics/ch36-neural-nets.md)）、从零训练 CNN（[第 40 章](../p07-dl-basics/ch40-burn-train.md)）、训练技巧与优化器（[第 41 章](../p07-dl-basics/ch41-training-tricks.md)）。

> 💡 **Rust 里用哪个**：手推梯度容易错，深度学习框架用**自动微分**替你算——`candle` 和 `burn` 都内置反向传播与 SGD/Adam 等优化器（`burn` 的 Autodiff 后端、`candle` 的 `backward()`）。你只管写前向和损失，梯度它包了。

## 四、几何：点、线、变换与「框」

机器视觉一半是「看懂内容」，另一半是「算清位置」。这里是位置计算的常用工具箱。

**2D/3D 变换（transformation）**：由弱到强分三档，能力一档比一档强。

**刚体 / 相似变换**：平移 + 旋转 +（等比）缩放，保持物体形状不变形。2D 旋转矩阵是：

\\[ R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \\]

**仿射变换（affine）**：在上面基础上再允许错切与非等比缩放，特点是**平行线变换后仍平行**。用齐次坐标写成一个 3×3 矩阵，最后一行固定为 `0 0 1`：

\\[ \begin{bmatrix} u \\\\ v \\\\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\\\ c & d & t_y \\\\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix} \\]

**透视变换 / 单应（perspective / homography）**：最强的一档，能把矩形拍成梯形（模拟近大远小）。它的齐次矩阵最后一行不再是 `0 0 1`，算完必须做一次**透视除法**才是真实像素坐标。三档详见 [第 10 章 几何变换](../p02-image-basics/ch10-geometry.md) 与 [第 26 章 单应](../p04-features/ch26-homography-stitching.md)。

**点到直线距离（point-line distance）**：直线写成 \\( ax + by + c = 0 \\)，点 \\( (x_0, y_0) \\) 到它的距离是：

\\[ d = \frac{|a x_0 + b y_0 + c|}{\sqrt{a^{2} + b^{2}}} \\]

霍夫直线（[第 19 章](../p03-classic/ch19-hough.md)）、判断目标离绊线还有多远，都要用它。

**叉积（cross product）**：在 2D 里它退化成一个标量 \\( \mathbf{a} \times \mathbf{b} = a_x b_y - a_y b_x \\)。它的**符号定方向**：正说明 b 在 a 的逆时针一侧，负则顺时针，零则共线。这正是**绊线检测判断「从哪个方向穿过」**（[第 87 章](../p18-events/ch87-tripwire.md)）、以及判断点在多边形 ROI 内外（[第 84 章](../p18-events/ch84-roi-filter.md)）的核心技巧。

**IoU（交并比，intersection over union）**：两个框重叠程度的度量，取值从 0（毫不相干）到 1（完全重合）：

\\[ \text{IoU} = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} \\]

它是目标检测的命根子：NMS 去重叠框、mAP 评测、判断预测框是否命中真值，全靠它（[第 45 章](../p09-detection/ch45-detection-principles.md)）。

> 💡 **Rust 里用哪个**：几何变换矩阵用 `nalgebra` 拼装，图像 warp 用 `imageproc` 的 `Projection` + `warp`/`warp_into`（[第 10](../p02-image-basics/ch10-geometry.md)、[26 章](../p04-features/ch26-homography-stitching.md)）；点线距离、叉积、IoU 都是几行算术，直接手写即可，无需额外 crate。

遇到具体公式卡壳时，先回这里补一补直觉，再回正文——**数学在这里是工具，会用比会证更重要**。
