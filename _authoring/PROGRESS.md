# 写作进度追踪（PROGRESS.md）

> 不属于书稿。记录 113 章 + 6 附录 + 前言/后记的撰写状态，便于分批生成与断点续写。
> 状态：⬜ 未写 · 🟡 生成中 · ✅ 已完成 · 🔍 待复核

## 前置件
- ✅ book.toml / SUMMARY.md / theme / 目录骨架 / 占位文件 / `mdbook build` 通过
- ✅ 配套代码工程 code/（image0.25.10 imageproc0.27.0 ndarray0.17.2 nalgebra0.35.0 rustfft6.4.1 rayon1.12 thiserror2 anyhow1 serde1(derive) serde_json1 + dev:criterion0.8；已 lib+bin；bench ch108_bench.rs 在位，`cargo metadata` 通过）
- ✅ _authoring/STYLE.md · _authoring/API_NOTES.md（candle0.11/tract0.23.5纯R/ort2rc❌C++/burn0.21 选型表 + 第10节 candle 训练推理实测签名+safetensors存取 + 第11节视频crate边界 + 第12节 axum/tokio 服务栈）
- ✅ 黄金范例：preface.md · p01-start/ch01 · DL 范例 ch40(burn)/ch42-43(candle 推理闭环+冻结微调)

## 关键约定（复用，勿再踩）
- 🧭 重依赖代码（candle/burn/tract/axum+tokio/wasm）→ **独立工程**（code/dl_labs/chNN_*/、code/wasm_demo/、code/service_demo/，各自带 Cargo.toml **含空 `[workspace]` 隔离**）；**不改主 code/Cargo.toml、不进 code/examples/**。纯依赖示例才放 code/examples/。**绝不**在任何 Cargo.toml 加 `[[bench]]`/`[[example]]` 却不建源文件（会破坏所有 cargo 命令）。
- 📌 **收尾待办（全书写完一次性做）**：SUMMARY.md 标题同步 —— 多个 stub 首行 H1 已更新为专家版标题、章内容据此保留，但 src/SUMMARY.md 仍是原始标题 → 侧边栏≠页面 H1（非构建错误，链接按路径正常）。**最终以各章文件首行 H1 为准统一回写 SUMMARY.md**。
- 🌙 2026-08-27 曾因机器休眠一次性中断 11 个在跑 agent（非内容问题）；已从各自 transcript 原地续跑，无内容丢失。当前保活：caffeinate **PID 8496（6h，08-27 起，覆盖 P20 阶段）**；另有 80466(~3.2h)/43925(~2h) 叠加。⚠️ 起 caffeinate 请用 run_in_background 直接跑 `caffeinate -dimsu -t N`，别加尾随 &（会孤儿化，虽仍存活但语义乱）。

## 章节状态（P1–P21 + 附录）
- P1 起步 ch01-06 ✅
- P2 图像基础 ch07-12 ✅
- P3 经典算法 ch13-22 ✅
- P4 特征 ch23-28 ✅（ch26/28 已确认完成）
- P5 数值 ch29-31 ✅
- P6 传统ML ch32-35 ✅（linfa 独立工程验证，主 code/ 未引入）
- P7 DL基础 ch36-41 ✅（ch40 burn 独立工程 loss 1.04→0.0001）
- P8 分类 ch42-44 ✅（candle 推理闭环端到端验证：safetensors 存取 + 冻结微调 25%→100% + 多标签 macroF1 0.955；code/dl_labs/ch4{2,3,4}_*/）
- P9 检测 ch45-48 ✅（**第九部分完成**；ch45原理+ch48评估[纯算法 examples/：mAP@.5=0.6379、NMS 12→3框]、ch46 YOLO推理[loss2.25→0.24、10/10检出 IoU0.775、类别感知NMS]、ch47训练[划分泄漏 mAP0.945→0.705差24pt、λ_box消融]；ch46/47→candle dl_labs/）
- P10 分割 ch49-51 ✅（**第十部分完成**；ch49 U-Net语义分割[loss1.38→0.024·像素acc99.35%·mIoU0.970·跳连消融 0.970 vs 0.414·DeepLab空洞卷积RF演示]+ch50 实例分割Mask R-CNN[two-stage 复用ch46检测+ch49掩码头·检出99.3%·掩码头RoI帧IoU0.954 vs 整图0.811卡在框精度·非填满形状让掩码IoU有意义]+ch51 交互式/SAM[种子生长点提示IoU1.000·迷你SAM前景IoU0.951·3点→3掩码]；均 code/dl_labs/。发现 **upsample 反向仅支持整数倍、bilinear 无反向**→RoIAlign训练不可行→RoI resize放Rust图像侧，已记 API_NOTES §17）
- P11 关键点 ch52-55 ✅（**第十一部分完成**；ch52 热图回归原理[loc 0.488→0.370px]+ch53 人体姿态[PCK@0.2 84.2%·划分泄漏→95.8%]+ch54 人脸关键点/对齐[**FC头**直接回归 MSE 0.283831→0.000074·定位1.113px·nalgebra Umeyama 摆正 残差0.887px **含镜像校正**]+ch55 手部关键点/手势[热图 定位1.263px·PCK@0.2 95.9%·几何手势 GT 100%/预测 89% 瓶颈在感知]；均 candle dl_labs/。**确认 FC头能回归绝对坐标而 1×1 卷积头不能**、关键点须空间分离否则热图重叠[ch55 2.29px/47%→1.26px/89%]→API_NOTES §16）
- P12 人脸 ch56-58 ✅（**第十二部分完成**；ch56 检测[100%检出·IoU0.870·五点1.26px]+ch57 ArcFace[消融 margin0→0.5 开集TAR@FAR1% 49.1%→69.3%]+ch58 活体[手工阈值69.0% vs 小CNN 99.0%·翻拍最难84/86]+属性多任务[眼镜/微笑/帽子各100%]；均 code/dl_labs/ch5{6,7,8}_*/ 实测。**发现 LCG 低位退化坑→API_NOTES §18 全书防**）
- P13 跟踪 ch59-63 ✅（迭代LK 0.007px、卡尔曼RMSE-55.7%、粒子60→6px）
- P14 三维 ch64-68 ✅（标定RMS~1e-13、立体99.89%、PnP 0.000°）
- P15 生成多模态 ch69-73 ✅（**第十五部分完成**；ch69 自编码器✅[16x16→瓶颈8维 重建PSNR23.21dB·冻结编码器线性探针 latent8维85.0%>raw256像素75.8%·DAE 15.70→22.55dB]+ch70 GAN✅[6离散模式小GAN,数值稳定BCE,3对照:朴素不稳(D_loss跌0.069)/稳定技巧精度0.551≈真0.326但仍崩1/6(召回7.59)/条件GAN 6/6覆盖(召回→0.67);均衡D_loss≈2ln2]+ch71 扩散✅[极小DDPM 9高斯团3x3网格,预测ε MSE0.99→0.28,采样9/9均衡覆盖·均值对齐·88%落流形,轨迹图PNG;√ᾱ_T=0.075;谱偏置坑(环形只覆盖一半→网格全覆盖);诚实SD边界(潜空间+CLIP+U-Net,candle官方example需下权重)]+ch72 超分/去噪/风格✅[SR 双线性15.23→SRCNN残差29.93dB·去噪 DnCNN32.32 vs 中值19.59/高斯16.72(<含噪16.95,糊边)·风格 纯R颜色统计迁移,真NST需下载VGG]+ch73 CLIP多模态✅[零样本180/180=100%·双向检索recall@1 9/9·组合泛化 颜色100%/形状55%·index_select可微查表/对称InfoNCE→API_NOTES§19,code/dl_labs/ch73_clip]；均实测可复现,code/dl_labs/ch69_autoencoder+ch70_gan+ch71_diffusion+ch72_restoration+ch73_clip）
- P16 视频 ch74-77 ✅（帧差运动峰值448精确、实时管线丢帧不变式、动作识别100%、视频检测跟踪ID切换0；均 code/examples/）
- P17 工业专题 ch78-82 ✅（亚像素0.025px、手眼RMS0.025mm、缺陷4/4、读码闭环、光学FOV）
- P18 事件规则 ch83-90✅ · ch91-93✅(a3a46ce0 = 事件联动Sink[fan-out 3路/重试2次:8事件无重试丢3、加重试后7达1入死信]+多路并发mpsc[4路×100=400帧/16事件、背压cap16零丢、丢最新vs丢最旧各丢42留8]+规则引擎JSON[52框，改阈值0.5→0.8: 3条→2条、换ROI右→左: 3条同数不同源]，code/examples/ std+serde_json，均 cargo run 实测)（共享 EVENTS_MODEL.md；ch87绊线去抖后 in2/out1、ch88停留32.5s≥30s触发）
- P19 部署 ch94-96 ✅（**第十九部分完成**；ch94 纯R造218字节.onnx+tract加载真跑[`tract_onnx::pb`+prost0.14]、ch95 int8量化 per-channel改善8911×/对称省3/4体积、ch96 axum service 冒烟5/5；ch94→dl_labs、ch95→examples、ch96→service_demo）
- P20 项目 ch97-107 ✅（**第二十部分完成·完成11**：ch100 以图搜图✅[学习嵌入32维 precision@5 98.6% vs 原始像素1728维45.3%·揭示原始像素按位置乱配·报precision@K非top-1]+ch105 健身姿态计数✅[姿态0.475px·滞回状态机 σ=1px 单阈值0%→滞回97.5%·关节角近伸直病态→时间平滑救回·感知vs逻辑瓶颈]+ch102 工业缺陷✅[两路共享后处理(分割掩码→阈值→8邻域BFS连通域→缺陷框+严重度)：A监督(复用ch49 U-Net+加权CE w=4) loss0.70→0.18·缺陷像素IoU0.512·缺陷级召回99.1%(113/114)·**0误报**(不加权仅89.5%)；B异常检测(复用ch69自编码器·仅正常样本训·重建误差>μ+4σ) recon0.0075→0.0011·IoU0.495·召回100%(114/114)·36误报·工业取舍:缺陷罕见/形态未知→选异常检测·**candle cross_entropy 无类别权重→自写加权逐像素CE→§17**]+ch101 人脸考勤门禁✅[级联 ch56检测→ch54五点对齐→ch58活体→ch57 ArcFace·开集12人未见·员工放行83.3%(25/30)·陌生人/假体拒止100%·**0安全事故**·**级联域偏移**(下游须在上游真实输出上训:0%→40.7%)+多点对齐鲁棒(40.7%→83.3%)→§20·APCER/BPCER阈值偏安全侧]+ch97/103/104 事件集成三章✅[ch97实时管线 45帧/120框/3轨/0 ID切换/3事件·丢帧不变式35+10=45·**丢帧平移事件时间戳920→880ms→计时/测速类须用采集真实时间戳**·Stage升级吃TrackedFrame;ch103周界入侵 ROI+绊线方向+cooldown 收敛245→218→2·**脚点压线浮点退化 d1·d2<0 端点得0漏判越线→避开边界/放宽等号/迟滞**;ch104客流计数 朴素穿线10→(track_id+1500ms窗口)去重6·**绊线是线段非无限直线,须"两组叉积严格异号"判据**·去重窗只防抖动不防id断裂重计]（均 std-only zero-dep,code/projects/）+ch107 部署Web服务✅[axum0.8.9+tract0.23.5+tokio1.53 冒烟7/7:健康检查/红绿蓝分类conf0.97-0.99/非法base64→400/空body→400/背压Semaphore→503·模型`Arc<TypedRunnableModel>`只load一次入AppState共享·`model_for_read`从内存字节加载省落盘·`spawn_blocking`+双`??`·`Router`即Service→`oneshot`进程内测试冒烟=CI复用·**tract纯Rust星号:tract-linalg构建期cc仅汇编自带SIMD内核.S(仓库0个cpp)、无外部C++运行库、自包含单二进制≠ort链onnxruntime→§14/附录A据此措辞**,code/projects/ch107_deploy]+ch98 二维码/条码✅[qrcode0.14.1生成→rqrr0.10.1读回闭环:369×369px/33×33模块/M级逐字一致·旋转5-45°全成·缩到≈90px(每模块2.7px)才崩·椒盐噪声M+3×3中值全档100%,H更密更脆1%噪声反26.7%→对症预处理>堆纠错等级·**rqrr ecc_level是格式信息原始2bit(M=00/L=01/H=10/Q=11)非序号→§7**·诚实边界rqrr仅QR/一维码指rxing,code/projects/ch98_qrcode]+ch106 全景拼接✅[2×600×420(重叠240px)→FAST各500角点→BRIEF 140匹配→RANSAC3000迭代/3px内点111/140=79.3%重投影0.471px·估计vs真值单应四角1.965px·全景970×420无缝·**瘦SVD解8×9单应Ah=0翻车(V^T仅8行取错向量→四角513px)→改AᵀA(9×9)对称特征分解取最小特征向量→1.965px→§6:终检spot-check三维DLT**·**BRIEF自相似纹理→RANSAC自洽错模型(内点0.4px却差真值237px):内点漂亮≠模型对,须独立真值复核+中等尺度纹理甜点→§4**·warp_into内部求逆须传源→目标H·两图共用确定性TestPairs,code/projects/ch106_panorama]+ch99 文字检测识别(OCR)✅[受限13类"0123456789OIS":连通域切分+candle小CNN分类·分割成功率95.5%(191/200)·**§20级联域偏移OCR变体:居中保长宽比字形训练→接真实分割器(紧贴框+拉伸失真)整串85%→20.5%且不报错→改用分割真实产出训练→逐字69.6%→93.6%/整串20.5%→85.0%**·**反直觉混淆全压0→O(21次):0/O只差内部一暗点轻模糊即抹平,1/I·5/S判别特征跨多像素糊不掉→受限字符集最该防"唯一区别落在成像退化会吃掉之处"的类**·**粘连击穿连通域:字距为负99%串并块整串仅1%(1/100)→纯连通域方案天花板**·模糊+阈值两难(定1×模糊+THR0.35+最小面积15,过高一度崩到6.5%)·诚实边界:纯R无成熟通用OCR引擎(Tesseract=C++/PaddleOCR=Py+C++),务实=CRNN导ONNX+tract(代码纯R但依赖外部模型非自包含)→§7/§20,code/projects/ch99_ocr]）
- P21 精通 ch108-113 ✅（**第二十一部分完成**；rayon 7.90×/101×、criterion、cargo test 6/6；ch109 GPU M1 Metal 重算子≈884×[wgpu30，code/gpu_demo/ 独立]、ch113 数据划分按 image_id 0泄漏 vs 随机 5泄漏[code/examples/ch113]）
- 附录 A✅ B✅ C✅ D✅ E✅ F✅ G✅ · 后记✅（2026-08-27 收尾写完：A crate速查表[九节·纯度四级·诚实边界总账·含tract纯R星号·三大版本冲突] / D 编译错误排查[九节·一二三四五六七=编译报错·第八节=能编译却跑错的静默陷阱表·汇 §6-§21] / 后记[bookend 前言三承诺·诚实重构"大佬"·怀疑精神·纯R边界=待填的坑]）

## 收尾/终检 阶段（正文 113 章全 ✅ · 附录 A-G 全 ✅ · 后记 ✅ · 终检 全 ✅）
- **已完成收尾件**：附录 A✅ / 附录 D✅ / 后记✅（2026-08-27 写）；SUMMARY.md 首行 H1 回写✅（上一段已同步 28 处专家版标题，路径未动，0 build 风险）。
- **终检 checklist（全部完成 2026-08-27）**：
  - ① `mdbook build` 全书 ✅（HTML 写出，0 报错/警告）。
  - ② opencv/`opencv` leak grep ✅（code/ 下 **0** crate 依赖、**0** 使用；src/ 仅诚实边界处的散文提名，符合硬约束）。
  - ③ 各独立工程结构 + 抽样编译 ✅（37 个 Cargo.toml；projects/ 11 个全含空 `[workspace]` + src/main.rs；抽样 `cargo check` ch103_perimeter[std-only zero-dep] 0.06s 通过。重依赖 candle/burn/tract 各章授权时已 `cargo run` 实测，此处只做结构性 + 抽样复核）。
  - ④ **LCG §18 低位退化 spot-check ✅ 已发现「教学 vs 代码」不一致并解决**：ch58 原文把规则写成绝对（"离散随机选择**绝不能**用低位 / LCG 低位是**垃圾**"），但 14 个 dl_labs（ch42/43/46/47/49/50/51/56/69/70/71/72/73 等）的 `range()`/`rand_i32()` 用 `self.next() % n`（64-bit LCG）。**技术真相更精确**：真正致命的是 `& 1`（周期 2）和**对 2 的幂取模**（ch58 踩的 `% 4` 就是只取最烂低 2 位）——这两种 **14 个 lab 都没用**（布尔全走 `unit()<0.5`）；对**非 2 的幂**的一般 `n` 做 `% n` 只是质量平平、不退化，故那些 lab 结果稳定可复现。**决策**：不动 RNG 代码（改序列会打乱全书已实测引用的数字，且重依赖 lab 无法在会话内廉价重跑复核 → 反而违背"数字都是真实 cargo 输出"原则）；改为**把教学改精确**——refine ch58 坑段 + 小结①、appendix D §8 LCG 行，锁定"`& 1`/2 的幂取模=真地雷、高位=永远安全的默认"，一般 `% n`=质量平平但不退化。既消除自相矛盾，又零改数字。
  - ⑤ §6 三维 ch64-68 DLT「过定才能用 `v_t` 末行解 Ax=0」spot-check ✅（上一段：标定 2N≫9 行、F 8+点+RANSAC、三角化 m=n=4、点云法向 N×3，残差 1e-13/1e-15/0.000° 全过定/恰定；最小情形陷阱在 ch106 已用 AᵀA 对称特征分解正确处理）。
  - ⑥（附带）ch94「零 C/C++ 依赖」措辞 ✅：原文绝对，已加 tract 纯 Rust 星号（构建期 `cc` 仅汇编器编自带 SIMD .S、0 cpp、无外部 C++ 运行库、自包含，≠ort），与 [附录 A](../src/appendix/a-crates.md)/ch107/§14 措辞一致，交叉链接已验证解析。
- **书稿状态：全书完成并终检通过。** 113 章正文 + 附录 A-G + 前言/后记；1274 条内部链接 0 死链；纯 Rust / 无 OpenCV 硬约束满足。

## 下一波（待放行，分批勿一次全发以免机器过载）
- ✅ ac4926f3（检测 candle ch46/47）已落地，检测经验已固化进 A 波 prompt（1×1 卷积头/narrow-cat 多通道解码/数值稳定 BCE/lr 衰减/**平移等变致划分泄漏不显形**的合成数据注意）
- **A 波（已发，5 agent 在跑）**：ch49,51 分割 · ch52,53 关键点 · ch56,57 人脸 · ch69,72 生成 · ch87,88 事件
- **B 波（全部在飞，无待发）**：ch70 GAN+ch71 扩散(ac1d4035)、ch54+ch55 关键点(a4fd2e7a 续)、ch58 活体/属性(ac2dd313 续)、ch50 实例分割 Mask R-CNN(abfdfd90 续，检测✅+分割✅经验齐备)、ch73 CLIP 多模态(af7fd05e 新)。**至此 DL 全波(A+B) 章节全部完成或在飞**；余下＝P20 集成 ch97-107 + 附录 A/D + 后记 + 终检
- ✅ ch87-88 已回收 → P18 收尾 ch91-93 已发(a3a46ce0)
- **P20 项目 ch97-107 计划（待 DL 波腾槽后分批发）**：每个项目做成**独立工程 `code/projects/chNN_*/` 含空 `[workspace]`** → 并行安全（勿挤 code/examples/ 免 target 锁串行）。
  - **依赖已就绪可即发(9)**：ch97 实时管线(视频✅+检测✅+跟踪✅)、ch98 二维码/条码、ch99 OCR(诚实边界:纯R无成熟OCR,走tract/小模型)、ch102 工业缺陷(P17✅)、ch103 周界入侵(ROI+绊线+cooldown,P18✅)、ch104 客流/车流计数(绊线+跟踪+窗口)、ch105 健身姿态计数(ch53人体姿态✅)、ch106 全景拼接(特征+单应✅)、ch107 部署Web服务(扩 service_demo/axum✅)
  - **依赖已全部就绪**：ch73 CLIP 已完成 → ch100（软依赖 ch73/ch57/ch69 嵌入）已解锁并已发；ch99 OCR 亦解锁。P20 余下 ch97-107 依赖均满足，无「等在飞」项。
  - **P20 完成(11) ✅**：ch100(af7fd05e)、ch105(a4fd2e7a)、ch102(abfdfd90)、ch101(ac2dd313)、ch97/103/104(a7d17594)、ch107(a6c296ad 部署Web服务)、ch98+106(af37a1d3 QR读回闭环+全景四角1.965px)、ch99(a3241431 受限OCR连通域+candle)。**在飞(0)·待发(0)——全书 113 章正文全部收官。** 转入收尾：附录A/D + 后记 + SUMMARY回写 + 终检。
- 全书稳定后 → 附录 A/D + 后记 + SUMMARY 标题回写 + 全书 mdbook build / opencv grep / cargo build 终检 + **抽查各章自造 Lcg 离散采样是否踩 §18 低位退化（已完成章数字均合理，风险低，终检时 spot-check）**
