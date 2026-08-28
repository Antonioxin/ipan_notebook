# Sharp Monocular View Synthesis in Less Than a Second

## 1. Introduction

### 1.1 领域背景：从新视图合成到单图新视图合成

**新视图合成（Novel View Synthesis, NVS）**的目标是：给定场景的一些观测图像，生成从未拍摄相机位置看到的图像。传统方法通常从多视图数据中恢复相机与几何，再通过 NeRF 或 3D Gaussian Splatting（3DGS）优化每一个场景。它们可达到很高的保真度，但需要多张图像、已知或估计的相机位姿，且逐场景优化耗时较长。

近年来的前馈式方法尝试把流程改写为一次网络前向：输入图像，直接输出可渲染的三维表示。其中，3DGS 用大量带位置、尺度、旋转、颜色和不透明度的 Gaussian 表示场景，能以光栅化方式实时渲染，因此适合低延迟 NVS。

SHARP 进一步收紧了输入条件：**只输入一张 RGB 图像**。这使它适用于将已有照片快速转换为可轻微浏览的 3D 内容，例如 AR/VR 中的自然头部移动、照片库中的交互式查看。

### 1.2 单图任务为何困难

单张图像没有多视图视差，三维结构并不唯一：小物体靠近相机与大物体远离相机，可以产生近似相同的二维投影；背面与遮挡区域更没有直接观测。因此，从单图恢复的不是严格可验证的完整几何，而是由视觉证据和训练先验共同决定的合理 3D 假设。

对普通单目深度估计而言，局部深度误差有时仍可接受；对 NVS 而言，深度会决定相机移动后的视差，深度错误会立刻变成边界撕裂、重影、透明孔洞或漂浮 Gaussian（floaters）。

### 1.3 SHARP 的目标与边界

SHARP 的目标是从一张图像在少于一秒内生成具度量尺度的 3D Gaussian 表示，并以高分辨率实时渲染 **nearby views / headbox**。它并不承诺：

- 大幅度相机移动后的真实感；
- 背面与完全遮挡内容的真实恢复；
- 对强 view-dependent 或 volumetric effects 的系统建模；
- 360°、可自由漫游的完整场景重建。

这种任务边界是理解论文实验与优势的前提：它追求的是**单图、近邻视角、清晰且低延迟**，而不是与扩散模型争夺任意远视角的生成能力。

### 1.4 论文的主要贡献

1. 提出端到端的单图到高分辨率 3D Gaussian 回归框架；输入一张图，生成约 120 万个 Gaussian。
2. 用两层深度、Gaussian refinement 和一组渲染/正则损失，优先优化附近新视图的视觉保真度。
3. 引入仅在训练期使用的 learned depth adjustment，在单目深度不确定或深度标签与视图合成目标冲突时，为监督提供受限的调整空间。
4. 在多个未参与训练的数据集上报告较强的 DISTS、LPIPS 与推理效率结果。

## 2. 问题的形式化定义

### 2.1 输入、输出与渲染

输入为单张 RGB 图像：

$$
I\in\mathbb R^{3\times H\times W}.
$$

网络输出一组 3D Gaussian：

$$
G\in\mathbb R^{K\times N},
\qquad K=14,
$$

其中单个 Gaussian 的属性包括：三维位置（3）、尺度（3）、旋转四元数（4）、颜色（3）与 opacity（1）。实现中输出两层、每层 $768\times768$ 个 Gaussian：

$$
N=2\times768\times768\approx1.2\text{M}.
$$

给定目标相机投影参数 $P$，可微 Gaussian renderer 生成图像：

$$
\hat I=R(G,P).
$$

训练时，每个样本包含 input view 与 novel view。模型从 input view 预测 $G$，再渲染回两种视角并施加图像与几何监督。

### 2.2 深度与视差的关系

相机发生横向移动 $t$ 时，物点的近似像素位移满足：

$$
\Delta u\propto\frac{t}{z},
$$

其中 $z$ 为物点深度。于是，将物体预测得偏近或偏远并非只是 depth metric 的误差：它会直接改变前景与背景在新视图中的相对移动，破坏遮挡关系和渲染结果。

## 3. 关键概念的定义

### 3.1 单目深度歧义与“折中解”

给定单张图像 $I$，深度分布 $p(D\mid I)$ 可能有多个合理解。确定性回归网络只能输出一张 $\hat D$；若训练中多种尺度/几何都与相似图像对应，平方误差下的单一最优预测会趋向条件均值（L1 下则更接近中位数）。这种折中解可能不对应任一真实三维场景。

论文以 Figure 5 作不确定性诊断：分别对原图和其水平翻转图做深度预测，再将后者翻回，计算两张预测的相对绝对误差。物体边界、枝叶等复杂区域的不一致更明显。

![Ambiguity in depth estimation](/assets/Ambiguity_in_depth_estimation.png)

**证据边界**：翻转预测不一致支持“模型在这些区域不稳定”，但并不能严格证明真实分布必然多峰，也不能直接证明模型确实输出了所有可能尺度的平均值；不一致还可能受到网络非完全 flip-equivariance 或数据偏置影响。

### 3.2 具度量尺度（metric）

SHARP 试图输出可与真实相机平移相结合的尺度化表示，而不是仅保留相对深度排序。这里的度量能力来自训练先验和监督，不是单图从几何上天然可唯一恢复的物理尺度；遇到罕见尺度、特殊镜头或域外场景时仍可能失准。

### 3.3 两层深度（layered depth）

Depth decoder 输出：

$$
\hat D\in\mathbb R^{2\times H\times W}.
$$

- 第一层主要服务于输入图像可见的主表面，并接受直接逆深度监督。
- 第二层为局部遮挡后的内容、补全与部分视角相关外观预留容量；它受平滑正则约束，不能被理解为已经恢复了完整不可见世界。

### 3.4 3D Gaussian Splatting

3DGS 用一组可微渲染的椭球 Gaussian 表示场景。它比隐式场更便于实时光栅化；但若位置、尺度或 opacity 不合理，仍会出现大块模糊、噪点、漂浮物和边界错误。因此 SHARP 不只预测 Gaussian，还对其偏移与投影尺寸施加约束。

### 3.5 DPT-style dense decoder

DPT（Dense Prediction Transformer）原始架构用 ViT encoder 处理 token，再将不同层特征 reassemble 成二维 feature map，并逐级 fusion + upsample，得到高分辨率稠密输出。

SHARP **不使用原始 DPT 的 ViT backbone**；它的 visual backbone 是 Depth Pro。SHARP 借用的是 DPT 的 dense decoder 思想：将四组多尺度特征融合成空间对齐的 depth map 或 Gaussian 属性图，兼顾全局布局与局部边缘。

### 3.6 Learned depth adjustment

训练期的小型 U-Net 接收预测逆深度与真值逆深度，产生一个逐像素 scale map：

$$
(\hat D^{-1},D^{-1})\xrightarrow{\text{U-Net}}S\in\mathbb R^{H\times W}.
$$

调整后深度为：

$$
\bar D=S(\hat D,D)\odot\hat D.
$$

它的功能是让训练期深度监督更服务于新视图保真，而不是让网络机械拟合可能不利于渲染的深度标签。推理期直接令 $S=1$，移除该模块，不需要真值深度。

## 4. 模型架构的细节

![SHARP Pipeline](/assets/SHARP-pipeline.png)

### 4.1 总体数据流

```text
单张 RGB
  │
  ▼
Depth Pro encoder（预训练，输出四组多尺度特征）
  ├──────────► DPT-style depth decoder ─► 两层预测深度 D-hat
  │                                                │
  │                  训练期：depth adjustment U-Net + 真值深度
  │                                                ▼
  │                                          调整深度 D-bar
  │                                                │
  └──────────► DPT-style Gaussian decoder ─► Gaussian 属性增量 Delta-G
                                                   │
                         base Gaussian ◄──────────┘
                                │
                                ▼
                   differentiable Gaussian renderer → 两种视图的监督
```

### 4.2 Feature encoder 与 depth decoder

Depth Pro encoder 从输入图提取四组不同分辨率的特征。SHARP 使用 DPT-style depth decoder 融合这些特征；其最终卷积层被复制，从而输出两通道深度而非单通道深度。训练时，低分辨率 image encoder 与 depth decoder 被微调；patch encoder 和 normalization layer 冻结。

### 4.3 Depth adjustment 的信息瓶颈

小型 U-Net 约有 2M 参数。它仅在训练时使用 $hat D^{-1}$ 与 $D^{-1}$ 计算 $S$，并由以下约束限制：

$$
\mathcal L_{scale}=\mathbb E_{p\sim\Omega}|S(p)-1|,
$$

$$
\mathcal L_{\nabla scale}=\sum_{k=1}^{6}\mathbb E_{p\sim\Omega_{\downarrow k}}|\nabla S_{\downarrow k}(p)|.
$$

第一项要求调整尽量接近恒等映射，第二项要求调整在多尺度上平滑。二者构成信息瓶颈，避免 U-Net 以任意的高频逐像素修改替主网络“作弊”。

### 4.4 Gaussian 初始化与细化

调整后的两层深度与输入图像先形成 base Gaussian。Gaussian decoder 采用与 depth decoder 相似的 DPT-style 多尺度融合框架，但最后的 prediction head 输出所有 Gaussian 属性的增量：

$$
\Delta G=\{\Delta G_{pos},\Delta G_{scale},\Delta G_{rot},\Delta G_{color},\Delta G_{alpha}\}.
$$

Gaussian composer 将 base Gaussian 与增量按属性特定的激活函数组合。位置增量在 NDC 风格的 $(x/z,y/z,1/z)$ 空间中处理，以提高训练稳定性；尺度、颜色与 opacity 等属性采用相应的正值/范围约束。

### 4.5 训练策略

1. **Stage 1：synthetic training**。在具有输入/新视图 RGB 和深度真值的合成数据上训练，学习几何和渲染关系。
2. **Stage 2：self-supervised fine-tuning（SSFT）**。面对无 NVS 真值的现实单图，先由已训练模型生成伪新视图，再将伪新视图作为 input、真实原图作为 novel view 训练。该反向配对使模型适应真实图像，而不要求真实立体对。

### 4.6 损失函数

总损失为：

$$
\mathcal L=\sum_{d\in\mathcal D}\lambda_d\mathcal L_d+
\sum_{r\in\mathcal R}\lambda_r\mathcal L_r+
\sum_{s\in\mathcal S}\lambda_s\mathcal L_s.
$$

| 类别 | 损失 | 约束与作用 |
|---|---|---|
| 视图保真 | $\mathcal L_{color}$ | input 与 novel view 的 RGB L1，保证基础颜色和内容一致。 |
| 视图保真 | $\mathcal L_{percep}$ | novel view 的 feature 与 Gram matrix 匹配，增强纹理统计、锐利度与合理补全。 |
| 视图保真 | $\mathcal L_{alpha}$ | 渲染 alpha 对 1 的 BCE，抑制透明孔洞。 |
| 几何 | $\mathcal L_{depth}$ | 第一层 $\bar D_{(1)}^{-1}$ 与真值逆深度的 L1，约束可见主表面。 |
| 深度正则 | $\mathcal L_{tv}$ | 第二层逆深度的 TV，抑制补全层高频噪声。 |
| Gaussian 正则 | $\mathcal L_{grad}$ | 惩罚大逆深度梯度附近、高 opacity 的 Gaussian，抑制 floaters。 |
| Gaussian 正则 | $\mathcal L_{delta}$ | 限制屏幕平面增量 $\Delta G_x,\Delta G_y$，避免大幅横向漂移。 |
| Gaussian 正则 | $\mathcal L_{splat}$ | 将投影 Gaussian 方差限制在合理范围，防止过模糊或碎点化。 |
| Adjustment 正则 | $\mathcal L_{scale}$ | 要求 $S$ 接近 1。 |
| Adjustment 正则 | $\mathcal L_{\nabla scale}$ | 要求 $S$ 在多尺度上平滑。 |

论文给出的总权重为：

$$
(\lambda_{color},\lambda_{alpha},\lambda_{percep},\lambda_{depth},\lambda_{tv},\lambda_{grad},\lambda_{delta},\lambda_{splat},\lambda_{scale},\lambda_{\nabla scale})
=(1,1,3,0.2,1,0.5,1,1,0.1,5).
$$

## 5. 实验结果

### 5.1 评测目标与协议

论文在 Middlebury、Booster、ScanNet++、WildRGBD、Tanks and Temples、ETH3D 上做 zero-shot 评测。它主要采用 LPIPS 与 DISTS，因为 PSNR/SSIM 对很小的像素平移异常敏感，而 NVS 中的小几何误差常表现为轻微错位，未必对应人眼感知上的显著退化。

论文报告：相对最佳先前模型，LPIPS 下降约 25--34%，DISTS 下降约 21--43%；在 A100 上表示生成少于一秒，附近视图渲染超过 100 FPS。应将这些结果解释为对“单图、近邻视角、低延迟、高分辨率”的支持，不应外推到任意相机位移或完整场景重建。

### 5.2 Loss 消融

补充材料 Table 8 逐步加入深度、感知和正则项：

- 加入 $\mathcal L_{depth}$ 显著减少几何失真。
- 加入 $\mathcal L_{percep}$ 显著改善补全与清晰度，并改善感知指标。
- 正则项在所用指标上影响不总是很大，但定性上改善复杂几何与远处背景，同时减少退化/过大的 Gaussian，论文报告其也改善渲染速度。

### 5.3 Depth adjustment 消融

Table 11 对 learned depth adjustment 的独立贡献进行测试：

| 设置 | ScanNet++：DISTS / LPIPS | Tanks and Temples：DISTS / LPIPS |
|---|---:|---:|
| 不使用 learned adjustment | 0.077 / 0.154 | 0.148 / 0.444 |
| 使用 learned adjustment | **0.064 / 0.147** | **0.126 / 0.419** |

Figure 10 显示使用 adjustment 训练后的结果更锐利。该实验证明 adjustment 有助于感知保真；但它不能单独证明改善完全来自“消除了平均尺度”，也可能同时缓解标签噪声、反射/透明材质，以及深度监督与渲染目标的冲突。

一个值得讨论的现象是：在 ScanNet++ 上，启用 adjustment 后 PSNR/SSIM 从 22.89/0.838 变为 22.61/0.829，略有下降；DISTS/LPIPS 却改善。这体现了论文的评测立场：对 nearby-view synthesis，感知保真比严格逐像素对齐更重要。

### 5.4 其他消融结论

- 加入 perceptual loss 中的 Gram-matrix 项可进一步改善指标。
- SSFT 的指标收益并不在所有数据集上一致，但定性结果更清晰；作者将其归因于合成数据中复杂 view-dependent effects 不足。
- 解冻 monodepth backbone 改善反射、边界与复杂几何区域。
- 提高 Gaussian 数量能提升质量；这也意味着内存、存储与部署成本值得进一步量化。

## 6. 附录信息总结

### 6.1 补充材料中最值得看的图表

- **Figure 4 / Table 2**：展示 PSNR/SSIM 对仅 1% 图像平移仍非常敏感；这解释了为何论文以 DISTS/LPIPS 为主指标。
- **Figure 5**：用原图/翻转图的深度预测不一致性可视化单目深度的不确定区域。
- **Figure 9 / Table 8**：不同损失组合的质量与速度影响。
- **Figure 10 / Table 11**：depth adjustment 对图像锐度和感知指标的贡献。
- **Table 12**：SSFT 的指标与定性结果；效果不在所有数据集上完全一致。
- **Table 13**：解冻 monodepth backbone 的收益。
- **Table 14**：输出 Gaussian 数量与质量的关系。

### 6.2 实现与训练信息

- 输入分辨率为 $1536\times1536$；输出为两层 $768\times768$ Gaussian 网格。
- Stage 1 使用 128 张 A100 训练 100K steps；Stage 2 使用 32 张 A100 训练 60K steps。
- 模型总参数约 702M，其中可训练参数约 340M；Gaussian decoder 约 7.8M，depth-adjustment U-Net 约 2M。
- 不使用 spherical harmonics，避免颜色系数随阶数平方增长导致输出量明显扩大。

### 6.3 局限与开放问题

- 远视角、大位移、背面与重遮挡内容仍是根本未解问题。
- view-dependent / volumetric effects 尚缺乏系统表示。
- “metric scale”依赖学习到的先验，单图不具备从几何上唯一恢复绝对尺度的充分信息。
- 两层深度与约 120 万 Gaussian 的实际分工、质量提升来源及部署代价，仍值得更细致的定量分析。
- 一个可延伸的问题是：能否以不确定性预测、多假设深度或有限 diffusion prior 取代训练期真值深度 adjustment，并兼顾 nearby-view 清晰度与远视角合理性？
