# RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

论文：Guosheng Lin、Anton Milan、Chunhua Shen、Ian Reid，CVPR 2017  
论文原文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)；[arXiv](https://arxiv.org/abs/1611.06612)

## 一句话定位

RefineNet 让粗分辨率但语义强的深层特征，沿着残差式多路径逐步吸收高分辨率细节，形成高分辨率的语义分割表示。

它是 DPT、进而 DA3 Fusion 模块最直接的**命名技术祖先**。

## 它要解决什么问题

CNN 的深层特征通常有两种相互矛盾的性质：

- 深层、低分辨率特征：知道物体类别与场景上下文，但边界粗；
- 浅层、高分辨率特征：保留边缘和准确位置，但语义较弱。

RefineNet 的目标是让这两类信息不只相遇一次，而是从粗到细反复进行 refinement（细化）。

## 总体架构

ResNet backbone 输出不同 stage 的特征。RefineNet 从最深、最粗的 stage 开始，逐级向更高分辨率传播：

$$
F_\ell
=
\operatorname{Refine}_\ell
\left(
E_\ell,\,
\operatorname{Up}(F_{\ell+1})
\right).
$$

其中：

- $E_\ell$：当前尺度的 ResNet 特征；
- $F_{\ell+1}$：更粗一级、已包含更强全局语义的 RefineNet 输出；
- $\operatorname{Up}$：把粗特征上采样到当前尺度；
- $F_\ell$：既有当前局部细节、又吸收全局语义的细化结果。

## 一个 RefineNet block 的四个组成部分

![RefineNet_pipeline](/assets/RefineNet_f3.png)

### 1. Residual Convolutional Unit（RCU）

![[RefineNet_RCU.png|383]]

每条输入路径先经过两次 RCU。RCU 用卷积和残差连接适配特征，其核心形式可概念化为：

$$
y=x+g(x).
$$

它让模块学习“应在原有特征上补充什么”，而不是从零重写特征；这也给梯度提供了更直接的传播通路。

### 2. Multi-Resolution Fusion（MRF）

![[RefineNet_MRF.png]]

各输入路径先被卷积投影到共同通道数；较小的特征图再上采样到最大空间尺度，最后逐元素相加：

$$
m
=
\sum_i
\operatorname{Resize}_{i\to r}\!
\left(\phi_i(\widetilde x_i)\right).
$$

这里的 $\phi_i$ 负责通道适配，$\operatorname{Resize}$ 负责空间尺寸对齐，$\widetilde x_i$ 是经过 RCU 的输入。相加本身不增加通道数，因此计算简洁，但要求对齐步骤正确。

### 3. Chained Residual Pooling（CRP）

![[RefineNet_CRP.png]]

CRP 串联多个池化—卷积路径，以较低成本扩大感受野并汇聚多尺度上下文。概念上可写作：

$$
z=x+\sum_{j=1}^{J}\psi_j\!\left(\operatorname{Pool}^{(j)}(x)\right).
$$

它的重点不是恢复空间尺寸，而是让高分辨率特征也能利用更大范围的背景信息。

### 4. Output Convolution

最后的卷积将细化结果整理为可传给下一尺度或分类 head 的特征。整体顺序可概括为：

$$
\text{RCU}
\rightarrow
\text{多分辨率融合}
\rightarrow
\text{CRP}
\rightarrow
\text{输出卷积}.
$$

## 它为何与 DA3 最相关

[[basic paper/DPT|DPT]] 明确将自己的 decoder 称为 **RefineNet-based feature fusion block**：它同样从较粗的多尺度表示开始，使用残差卷积单元、横向细特征融合和逐级上采样，生成适合稠密预测的高分辨率特征。

[[key baseline/Depth Anything|Depth Anything 3（DA3）]] 的 Dual-DPT 又继承了 DPT 的这条路线：depth 与 ray 共用 Reassemble，却各自拥有 Fusion 链和输出 head。

需要避免一个误解：DA3 / DPT 是 **RefineNet-based**，不是将原始 RefineNet 一字不改地搬来。例如，原始 RefineNet 中的 CRP 并不是 DPT / DA3 Fusion block 必须完整保留的部件。

## 与 FPN 的关系

RefineNet 与 [[basic paper/FPN|FPN]] 都将“粗而语义强”的特征传向“细而定位准”的尺度，但它们是 2016—2017 年间的同期不同路线：

- RefineNet 面向高分辨率稠密分割，强调残差式多路径**细化**；
- FPN 面向检测，强调建立一组每层都有强语义的**输出金字塔**。

DA3 的具体 Fusion block 更接近 RefineNet；其粗到细、横向相加的数据流则具有 FPN 风格。

## 局限

多条高分辨率路径会增加显存和计算量；逐元素相加也要求各路径预先做精确的尺度与通道对齐。原始设计依赖 CNN 自然产生的多尺度 stage，Transformer 需要像 DPT 那样先做 Reassemble。
