# Feature Pyramid Networks for Object Detection（FPN）

论文：Tsung-Yi Lin 等，CVPR 2017  
论文原文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)；[arXiv](https://arxiv.org/abs/1612.03144)

## 一句话定位

FPN 用一次 backbone 前向传播构造一组“分辨率不同、但都具有较强语义”的特征金字塔，使检测器能更好地处理大小不同的目标。

它将 **top-down pathway + lateral connections（自顶向下路径 + 横向连接）** 带成了极具影响力的通用多尺度融合范式。

## 它要解决什么问题

同一张图中，小目标与大目标需要不同分辨率的特征：

- 高分辨率浅层特征适合看小目标，却语义不足；
- 低分辨率深层特征语义强，却难以精确定位小目标。

传统图像金字塔需要把同一张图缩放成多个版本并分别跑 CNN，代价很高。FPN 希望只运行一次 backbone，就得到多尺度、强语义的特征。

## 架构：bottom-up、top-down 与 lateral connections

设 CNN 自底向上产生：

$$
C_2,C_3,C_4,C_5,
$$

其中编号越大，特征图越小、语义通常越强。

FPN 再自顶向下构建中间金字塔 $M_\ell$：

$$
M_5=\phi_5(C_5),
$$

$$
M_\ell
=
\phi_\ell(C_\ell)
+
\operatorname{Up}_2(M_{\ell+1}),
\qquad
\ell=4,3,2,
$$

$$
P_\ell=\psi_{3\times3}(M_\ell).
$$

其中：

- **bottom-up pathway**：普通 CNN 从高分辨率输入逐步下采样，得到 $C_2$ 到 $C_5$；
- **top-down pathway**：把高层语义强的特征逐级上采样；
- **lateral connection**：当前尺度的 $\phi_\ell(C_\ell)$ 与上采样的高层特征相加；
- $\phi_\ell$：横向 $1\times1$ 卷积，主要用于通道对齐；
- $\psi_{3\times3}$：每次相加后的 $3\times3$ 卷积，用于整理特征并减轻上采样后可能出现的混叠（aliasing）。

因此，$P_2,P_3,P_4,P_5$ 的空间分辨率不同，却都获得了来自更深层的语义信息。

## Fusion 的直观意义

在第 $\ell$ 层，FPN 做的是：

$$
\text{当前尺度的局部细节}
+
\text{来自更粗尺度的全局语义}.
$$

这里的逐元素相加要求两者已被投影到相同通道数、上采样到相同高度和宽度。这也是我们在 DA3 的 Reassemble / Fusion 中反复遇到的“先对齐、再融合”的原因。

## FPN 与 DA3 / DPT 的区别

[[basic paper/DPT|DPT]] 和 [[key baseline/Depth Anything|Depth Anything 3（DA3）]] 的 Fusion 同样会把粗尺度特征上采样，并与当前尺度特征结合。因此其数据流是 **FPN-like（具有 FPN 风格）** 的。

但目标不同：

| FPN                         | DPT / DA3                                       |
| --------------------------- | ----------------------------------------------- |
| 保留 $P_2,\ldots,P_5$ 一组输出金字塔 | 逐级融合，主要形成一个高分辨率稠密预测特征                           |
| 服务于不同大小目标的检测 head           | 服务于每个像素的 depth、ray 等预测 head                     |
| 输入通常是 CNN 的天然多尺度 stage      | 输入可以是 Transformer 中间 token 经 Reassemble 后的多尺度特征 |

所以 FPN 是 DA3 的重要结构近亲和通用范式成熟者，但不是其 Fusion block 的最直接命名来源；后者是 [[basic paper/RefineNet|RefineNet]] 经由 DPT 的谱系。

## 局限

FPN 的跨尺度交互主要是固定的相邻层相加，较浅层本身仍可能语义不足；保留多个输出层也会给后续 head 带来额外计算。它主要为目标检测设计，不能直接等同于高分辨率语义分割或几何解码器。
