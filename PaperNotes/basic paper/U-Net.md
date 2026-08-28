# U-Net: Convolutional Networks for Biomedical Image Segmentation

论文：Olaf Ronneberger、Philipp Fischer、Thomas Brox，MICCAI 2015  
论文原文：[arXiv](https://arxiv.org/abs/1505.04597)；[作者项目页](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## 一句话定位

U-Net 用一个对称的“编码器—解码器”结构，将下采样获得的全局上下文和同尺度跳连提供的精确位置结合起来，是稠密分割网络最经典的架构原型之一。

## 它要解决什么问题

医学图像分割不仅要判断“这是什么”，还要准确描出细胞或器官的边界。连续下采样可扩大感受野（一个特征位置可见的原图范围），帮助模型理解上下文，却会使像素位置变得粗糙；若仅将粗特征上采样，细小边界难以恢复。

## 核心思想：U 形的编码与解码

U-Net 的左半边是 contracting path（编码器）：

- 重复卷积以抽取特征；
- 通过 max pooling 下采样，使空间尺寸变小、感受野变大；
- 通道数逐层增加，令每个较粗的位置可以携带更丰富的语义。

右半边是 expanding path（解码器）：

- 逐级上采样，使特征图回到更细的空间网格；
- 将编码器中**同一空间尺度**的特征直接传给解码器；
- 让后续卷积把“全局判断”与“边界 / 位置细节”重新组合。

这条从左到右再向上的路径看起来像字母 U，因此得名 U-Net。

## 架构

![U-Net_architecture](/assets/U-Net_architecture.png)

原始论文中，每个编码器 stage 包含两次 $3\times3$ 卷积和 ReLU，随后是 $2\times2$ max pooling；每下采样一次，通道数翻倍。解码器先用 $2\times2$ up-convolution（转置卷积）上采样、减少通道，再与对应编码器特征融合，最后经过两次 $3\times3$ 卷积。

可以把第 $\ell$ 层的解码过程抽象为：

$$
D_\ell
=
\psi_\ell\!\left(
\left[
\operatorname{Up}(D_{\ell+1})
\;\Vert\;
E_\ell
\right]
\right).
$$

其中：

- $E_\ell$：编码器在第 $\ell$ 个尺度保留下来的高分辨率特征；
- $D_{\ell+1}$：更粗尺度的解码器特征；
- $\operatorname{Up}$：上采样；
- $\Vert$：沿**通道维**拼接（concatenation），不是相加；
- $\psi_\ell$：后续卷积模块，学习如何使用两路信息。

原始 U-Net 使用 valid convolution，因此卷积会缩小特征图；为能精确拼接，论文会裁剪 encoder 特征。现代复现常使用 padding 保持尺寸，这属于实现选择，而不是 U-Net 思想的本体。

## 为什么拼接而不是相加

相加会立即将两个通道向量压成一个混合结果；拼接则把两路信息并排保留：

$$
[a\Vert b]\in\mathbb R^{C_a+C_b}.
$$

随后卷积自己学习哪些通道应被采用。代价是通道数与显存开销更大。

这与 FCN 的 score map 相加不同，也与 DPT / DA3 的残差式融合不同；它们解决的是同一个“语义与定位如何兼得”的问题，但信息合并的算子不同。

## 训练配套

原论文面向小样本医学数据，使用较强的数据增强，尤其是弹性形变；同时让边界附近像素在损失中拥有更高权重，以鼓励模型分开相邻细胞。

## 与 Depth Anything 3 的双链关系

U-Net 提供了理解 DA3 Fusion 的一个好心智模型：

> 先在粗尺度获得全局理解，再回到细尺度，并重新引入高分辨率定位线索。

但 DA3 并不是 U-Net：它的主干是 Transformer，四组特征先经 Reassemble 由 token 变为二维特征图；其 Fusion 使用更接近 RefineNet 的残差加法与逐级上采样，而非原始 U-Net 的通道拼接。

- 后续阅读：[[key baseline/Depth Anything|Depth Anything 3（DA3）]]
- 早期跳连范例：[[basic paper/FCN|FCN]]
- 更直接的 Fusion 谱系：[[basic paper/RefineNet|RefineNet]] $\rightarrow$ [[basic paper/DPT|DPT]]

## 局限

高分辨率 encoder feature 必须被保存并拼接，显存代价较高；卷积的长程关系建模也有限。它原本针对医学分割设计，不能直接等同于多视图几何模型。
