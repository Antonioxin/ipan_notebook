# Fully Convolutional Networks for Semantic Segmentation（FCN）

论文：Jonathan Long、Evan Shelhamer、Trevor Darrell，CVPR 2015  
论文原文：[arXiv](https://arxiv.org/abs/1411.4038)；[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)

## 一句话定位

FCN 将用于“整张图分类”的 CNN 改造成可以为**每一个像素**输出类别的网络，并以 skip architecture 将“深层但粗糙的语义”与“浅层但精细的定位”结合起来。

它是现代稠密预测网络中 skip fusion 的重要早期范例。

## 它要解决什么问题

图像分类的目标是：

$$
I\longmapsto \text{一个类别}.
$$

而语义分割的目标是：

$$
I\longmapsto Y,\qquad
Y(u,v)=\text{像素 }(u,v)\text{ 的类别}.
$$

分类 CNN 经过多次下采样后，最深层特征具有很强的“这是什么物体”能力，却只保留很粗的空间网格。例如，步幅（stride，特征图相邻位置在原图中相隔的像素数）为 $32$ 时，$32\times32$ 的原图区域可能只对应一个预测位置。直接放大这种粗网格，物体边界往往会模糊。

## 核心思想

### 1. 全连接层改为卷积层

![FCN_f2](/assets/FCN_f2.png)

全连接层要求固定长度的输入；卷积层则在每个空间位置复用同一组局部计算。FCN 将分类网络末端的全连接层改写为卷积层，于是网络可接受任意尺寸图像，并输出仍带空间布局的类别得分图（score map）。

score map 的每个位置存放一个 $K$ 维向量，$K$ 是类别数；它经过 softmax 后才成为各类别的概率。

### 2. 学习式上采样

FCN 使用转置卷积（论文中称为 deconvolution）将低分辨率得分图放大到像素级大小。它并不是严格数学意义上的“卷积逆运算”，而是一种参数可学习的上采样层。

### 3. 用跳连补回定位细节

![FCN_f3](/assets/FCN_f3.png)

最深层特征的语义强，却过于粗糙；较浅层特征保留边缘和位置，却不那么“懂物体”。FCN 将二者融合：

$$
S_{16}=\operatorname{Up}(S_{32})+\phi(\mathrm{pool4}),
\qquad
S_{8}=\operatorname{Up}(S_{16})+\phi(\mathrm{pool3}).
$$

其中：

- $S_{32}$：最深层、步幅为 $32$ 的类别得分图；
- $\operatorname{Up}$：上采样；
- $\mathrm{pool4}$、$\mathrm{pool3}$：更浅、更高分辨率的特征；
- $\phi$：$1\times1$ 卷积，将不同通道数投影成可相加的类别得分；
- 加号表示逐元素相加，因此两路特征的空间尺寸和通道数必须先对齐。

FCN-32s 只上采样最深层结果；FCN-16s 再融合 pool4；FCN-8s 进一步融合 pool3。随着跳连加入，预测边界更精细。

## 架构

$$
\text{图像}
\rightarrow
\text{分类 CNN 骨干}
\rightarrow
\text{粗粒度类别得分图}
\xrightarrow[\text{浅层定位细节}]{\text{上采样 + skip fusion}}
\text{像素级分类图}.
$$

它的关键不只是“放大”，而是让浅层特征为深层结果补回在下采样中丢失的空间位置线索。

## 与 U-Net、DPT / DA3 的区别

- FCN 的融合通常发生在低维的**类别得分图**上；
- U-Net 会直接拼接高分辨率的隐藏特征，因此给后续卷积留下更大的信息容量；
- DPT / DA3 则在隐藏的多尺度特征图上进行逐级融合，最终才由 task head 输出 depth 或 ray。

所以 FCN 提供的是一条基础直觉，而不是 DA3 Fusion block 的直接实现祖先。

## 局限与后续发展

从后续架构的发展看，FCN 只使用少数几条跳连，且融合空间是类别得分而非丰富的隐藏特征；经过上采样后，细边界仍可能不足。这推动了更完整的编码器—解码器设计（[[basic paper/U-Net|U-Net]]）、残差式高分辨率细化（[[basic paper/RefineNet|RefineNet]]）和多尺度金字塔（[[basic paper/FPN|FPN]]）。

## 与 Depth Anything 3 的双链关系

FCN 奠定了“深层全局语义必须与浅层空间细节结合，才能做好稠密预测”的基本直觉。DA3 将这一思想带入 Transformer decoder：先把 token 重组为多尺度特征，再为 depth 与 ray 分别进行融合。

- 后续阅读：[[key baseline/Depth Anything|Depth Anything 3（DA3）]]
- 更直接的架构前身：[[basic paper/DPT|Vision Transformers for Dense Prediction（DPT）]]
