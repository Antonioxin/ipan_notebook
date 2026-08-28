# Vision Transformers for Dense Prediction（DPT）

论文：René Ranftl、Alexey Bochkovskiy、Vladlen Koltun，ICCV 2021  
论文原文：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf)；[arXiv](https://arxiv.org/abs/2103.13413)；[官方代码](https://github.com/isl-org/DPT)

![DPT_pipeline](/assets/DPT_f1.png)
## 一句话定位

DPT 将 Vision Transformer（ViT）产生的 token 序列重新组织为多尺度二维特征图，再使用 RefineNet 式 decoder 从粗到细融合、上采样，得到适合深度估计或语义分割的高分辨率稠密预测。

它是 [[key baseline/Depth Anything|Depth Anything 3（DA3）]] 中 Dual-DPT decoder 的直接架构前身。

## 它要解决什么问题

ViT 很擅长让远距离 patch 通过 self-attention 交换信息，因而具备强全局上下文；但它的输出首先是 token 序列，而不是 CNN 那样天然存在的多层二维特征金字塔。

稠密预测要求为每个像素给出结果。只把最终 token 直接上采样，难以兼顾全局结构与清晰边界。因此 DPT 需要回答：

> 如何把 Transformer 的中间 token 重新变成多尺度的“图像式特征图”，并用 decoder 恢复像素级细节？

## 总体架构

![DPT_pipeline](/assets/DPT_f1.png)
$$
\text{图像}
\rightarrow
\text{Patch / Hybrid Embedding}
\rightarrow
\text{ViT encoder}
\rightarrow
\text{多个 Transformer 层的 token}
\rightarrow
\text{Reassemble}
\rightarrow
\text{多尺度特征}
\rightarrow
\text{RefineNet-based Fusion}
\rightarrow
\text{任务 head}.
$$

论文将这条思路用于单目相对深度与语义分割。

## Reassemble：token 如何重新成为二维特征图

假设第 $\ell$ 个读出层得到 patch token：

$$
t_\ell\in\mathbb R^{N\times C}.
$$

其中：

- $N$ 是 patch 数量，例如 $N=\frac{H}{P}\frac{W}{P}$；
- $P$ 是 patch 的边长；
- $C$ 是每个 token 的通道维度。

Reassemble 可概括为：

$$
\operatorname{Reassemble}_{s}^{\widehat D}(t)
=
\left(
\operatorname{Resample}_{s}
\circ
\operatorname{Concatenate}
\circ
\operatorname{Read}
\right)(t).
$$

它包含三步：

1. **Read**：处理 ViT 的 readout / class token，使其中的全局信息可以影响 patch token；
2. **Concatenate / projection**：将 token 变换到 decoder 所需通道数，并按 patch 原本的位置排回二维网格；
3. **Resample**：用卷积或转置卷积调整空间尺寸，构造不同分辨率的特征图。

关键点是：ViT 的不同层并不像 CNN 一样天然具有不同空间分辨率。DPT 的多尺度特征是通过 Reassemble 中的 Resample 对不同层 token **显式构造**出来的。

例如，论文从多个 Transformer 中间层读取 token；ViT-Base 读取 $\{3,6,9,12\}$，ViT-Large 读取 $\{5,12,18,24\}$。默认 decoder 通道宽度为 $\widehat D=256$。

## Fusion：如何从粗到细恢复稠密表示

DPT 明确称其为 **RefineNet-based feature fusion block**。它从较粗尺度开始，递归地写成：

$$
F_\ell
=
\operatorname{Fuse}_\ell
\left(
A_\ell,\,
\operatorname{Up}_2(F_{\ell+1})
\right).
$$

每一级大致包含：

1. 对当前尺度横向特征做残差卷积适配；
2. 融合来自更粗尺度、但更具全局语义的上采样特征；
3. 用残差卷积进一步细化；
4. 再上采样并投影，交给更高分辨率的下一层。

因此，粗尺度负责“场景整体几何 / 语义是否合理”，细尺度负责“边界和像素位置是否准确”。最后的 task head 才把高分辨率隐藏特征变成深度或类别等输出。

## 与 RefineNet、FPN 的关系

- 直接来源：[[basic paper/RefineNet|RefineNet]]。DPT 论文明确使用其 RefineNet-based fusion block 思路；
- 结构近亲：[[basic paper/FPN|FPN]]。二者都有“粗尺度信息上采样 + 同尺度横向特征融合”的自顶向下数据流；
- 早期心智模型：[[basic paper/FCN|FCN]] 与 [[basic paper/U-Net|U-Net]]，它们说明了为何稠密预测不能只依赖最深层的粗特征。

需要注意：DPT 不是完整复刻 RefineNet。它保留残差细化、多尺度融合、逐级上采样等核心精神，但其 decoder 为 Transformer 稠密预测进行了适配和简化。

## 与 Depth Anything 3 的双链关系

DA3 将 DPT 扩展为 **Dual-DPT**：

$$
\text{共享 Reassemble}
\longrightarrow
\begin{cases}
\text{depth Fusion + head},\\
\text{ray Fusion + head}.
\end{cases}
$$

这意味着 depth 与 ray 可以共享前段视觉 / 几何表征，却分别学习最适合各自输出形式的融合和读出方式。DA3 的新意不在于首次发明 Fusion，而在于将成熟的 DPT decoder 改造成适合联合几何预测的双分支设计。

- 后续发展：[[key baseline/Depth Anything|Depth Anything 3（DA3）]]

## 局限与阅读边界

DPT 的全局 attention 和高分辨率 dense decoder 都需要可观的计算；其表现也依赖预训练与输入分辨率。深度任务中，论文主要讨论相对深度：预测值的整体尺度并不自动等于真实世界米制尺度，这与 DA3 同时建模 ray、相机和跨视图几何的目标不同。
