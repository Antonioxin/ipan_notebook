# Surflo: Consistent 3D Surface Flow Model with Global State

论文：[Surflo.pdf](/Users/pro/Desktop/🤖.ai/暑期论文阅读/key_baseline/Surflo.pdf)

## 开始前复习一下知识点

### 1. Perceiver-style Cross-Attention

输入图像数量为 $N$，每张图又有大量 patch token。若令全部 token 两两 self-attention，代价会随 token 数平方增长。Perceiver 的做法是准备一个固定数量的、可学习的 latent query：

$$
z_p^{(0)}\in\mathbb R^{K\times D}.
$$

令图像 token 的集合为 $X\in\mathbb R^{M\times D}$，其中 $M$ 随视图数增长。Cross-attention 中：

$$
Q=z_pW_Q,\qquad K_X=XW_K,\qquad V_X=XW_V,
$$

$$
\operatorname{CrossAttn}(z_p,X)
=
\operatorname{softmax}
\left(\frac{QK_X^\top}{\sqrt d}\right)V_X.
$$

也就是说，$K$ 个 latent token 充当固定数量的“信息槽位”：它们作为 Query，从所有图像 patch 的 Key/Value 中有选择地读取证据。随后 latent token 之间进行 self-attention，从而交换跨区域、跨视图的信息。

Surflo 中：

- $K=128$：128 个几何 latent slot；
- $D=512$：每个 slot 的特征维度；
- 进行 $L_e=4$ 轮“cross-attention + $L_s=4$ 个 self-attention block”；
- $K$ 不等于输入视图数、空间网格数或输出点数；$D$ 也不是三维坐标，而是抽象特征通道。

初始的 $z_p^{(0)}$ 是对所有场景共用的可学习 query 参数，并不由当前图像直接生成；经过 cross-attention 更新后的 $z_p$ 才是当前场景特有的内容。它们应被理解为场景的压缩记忆，而不是具有固定语义或固定空间位置的 128 个“体素”。

### 2. Fourier Features（Fourier 位置编码）

直接把三维坐标 $p=(x,y,z)$ 交给网络时，网络往往不易表达高频的空间变化。Fourier features 用多组不同频率的正弦、余弦将坐标编码为更丰富的向量：

$$
\gamma(x)=
[\sin(2\pi f_1x),\cos(2\pi f_1x),\ldots,
\sin(2\pi f_Lx),\cos(2\pi f_Lx)].
$$

对三维坐标同样编码后，可得到 $\gamma(p)$。它不是对图像做傅里叶变换，而是为一个位置贴上多尺度的“坐标条形码”。

Surflo 将每个 VGGT patch 中心的三维位置 $p_n$ 编码后加到 patch token：

$$
T_n+\gamma(p_n).
$$

decoder 对查询点 $x_t$ 也使用同一套 $\gamma$。因此，encoder 所读到的局部证据与 decoder 所提出的空间查询使用共同的空间语言：查询点附近的位置更容易通过 attention 找到相关的编码信息。论文的消融表明，去掉该 3D positional encoding 会降低 DL3DV 与 Tanks & Temples 上的 CD 和 F1。

### 3. Flow Matching 与 ODE 采样

Flow matching 学习的不是一个固定点坐标，而是一个随时间变化的速度场，将简单的源分布 $p_0$ 连续运输为目标分布 $p_1$。

训练时，先采样：

$$
x_0\sim p_0,\qquad x_1\sim p_1,\qquad
t\sim\operatorname{LogitNormal}(1,1.6).
$$

然后构造从 $x_0$ 到 $x_1$ 的线性插值路径：

$$
x_t=(1-t)x_0+t x_1=x_0+t(x_1-x_0).
$$

在这条直线路径上，速度恒为：

$$
u(x_t,t)=x_1-x_0.
$$

因此模型只需学习：

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E\left[
\left\|v_\theta(x_t,t\mid z)-(x_1-x_0)\right\|_2^2
\right].
$$

直观上，$t=0$ 时点仍在噪声源；$t=1$ 时到达真实表面。推理时不再给模型真实的 $x_1$，而是从 $p_0$ 采样许多 $x_0$，通过 ODE

$$
\frac{dx_t}{dt}=v_\theta(x_t,t\mid z)
$$

逐步积分到 $t=1$。Surflo 使用 Euler solver。大量 query 可并行处理，因此同一 $z$ 可输出从少量预览点到百万级密集点的表面采样。

注意，训练中 $x_0$ 与 $x_1$ 的配对是构造监督路径的方法，并不代表某个初始噪声点在现实中天然对应某个特定表面点。

### 4. Ada-LN（Adaptive Layer Normalization）

普通 LayerNorm 只依据当前 token $x$ 做归一化；Ada-LN 还根据条件 $c$ 动态生成缩放与平移：

$$
\operatorname{AdaLN}(x,c)
=
(1+\operatorname{scale}(c))\odot\operatorname{LN}(x)
+\operatorname{shift}(c).
$$

Surflo 的条件为：

$$
c=\tau(t)+\operatorname{MLP}_{\mathrm{cam}}(z_c),
$$

其中 $\tau(t)$ 是 flow 时间编码，$z_c$ 是由 camera token 汇总而来的相机/世界坐标条件。实际 block 还会预测 gate，典型残差形式为：

$$
y=x+\operatorname{gate}(c)\odot
F\bigl(\operatorname{AdaLN}(x,c)\bigr).
$$

它让 decoder 在不同积分阶段采取不同策略：早期大范围地从噪声靠近几何，后期细调表面；同时根据当前 VGGT 坐标系解释点的位置。论文将调制参数的输出投影零初始化，使训练初始时 block 近似恒等映射，从而提高稳定性。

## Introduction

### 1. 研究背景：多视图的冗余与全局几何

从 Klein 的 Erlangen program 出发，论文将三维几何视为在视角变换下保持不变的内容。多张场景图像只是对同一个三维状态的不同投影：视图变多时，像素信息线性增多，但几何本体不应随之膨胀。

这对于 real2sim、机器人和导航很重要，因为它们最终需要的是可操作的统一几何，而非若干彼此重叠的视图相关点云。

### 2. 现有前馈重建的两类矛盾

现有方法大致落在两端：

1. **逐视图方法**，如 VGGT。每张图输出自己的 pointmap，局部精度高，但输出 token 数随视图数线性增加；不同 pointmap 会重叠、错位，难以融合成干净网格。
2. **全局 latent 方法**，如 NOVA3R。它们有统一场景码，却常被固定输出规模限制，例如固定 10K 点，无法按需要增减细节。

D4RT 也能从场景表示查询独立点，但它面向密集视频和像素锚定的时空查询，并非由稀疏图片生成自由的带法线表面点。

### 3. Surflo 的核心主张

Surflo 要同时满足三件事：

1. 将任意数量的无位姿输入图像压缩为固定大小的 scene-specific global state；
2. 从该状态生成任意数量的带法线表面点；
3. 让独立生成的点仍形成连续、一致的表面。

其对应设计为：

> 多视图图像  
> $\downarrow$  
> 冻结的 VGGT：产生几何感知 patch token 与 camera token  
> $\downarrow$  
> Perceiver：压缩成全局 latent $z$  
> $\downarrow$  
> 以 $z$ 为条件的独立 flow-matching decoder：运输任意数量噪声点  
> $\downarrow$  
> 推理后期的 rendering guidance：让点通过共同损失通信  
> $\downarrow$  
> 带法线的表面点 $\rightarrow$ Delaunay 三角化 $\rightarrow$ mesh

### 4. 论文的主要贡献

1. 用冻结 VGGT 加 Perceiver，将可变数量的无位姿 RGB 图像压缩为固定大小的全局场景状态；
2. 用在 $\mathbb R^3\times\mathbb S^2$ 上的 flow matching，从同一 latent 解码任意数量的带法线表面点；
3. 在 ODE 积分中加入基于渲染损失的 guidance，使原本独立的点通过共享的全局梯度协调；
4. 构建带水密网格与定向点监督的 meshed DL3DV：约 $10.5$K 个真实场景，每个场景约 $10^7$ 个带法线点。

## Method

### 1. 问题设定

输入是未给定相机位姿的多视图 RGB 图像：

$$
\{I_n\}_{n=1}^{N}.
$$

输出为场景表面 $\mathcal S\subset\mathbb R^3$ 上的带法线点。系统先计算固定大小的 $z$，再对每个查询 $x_t\in\mathbb R^3\times\mathbb S^2$ 和时间 $t\in[0,1]$ 预测速度。最后由 guided ODE solver 积分得到一致表面。

![Surflo_pipeline](/assets/Surflo_f2.png)

### 2. Encoder：从图像到固定大小的全局状态

#### 2.1 冻结的 VGGT 与几何感知 token

VGGT-1B 是冻结 backbone。每张图产生多个 patch token 与 camera token；Surflo 取第 $4,11,17,23$ 层的 patch token 并拼接、线性投影到工作维度 $D=512$。

VGGT patch token 不是三维坐标本身，而是局部外观、跨视图对应与几何线索的高维描述。Surflo 从相应 VGGT pointmap 中读取 patch 中心的三维点 $p_n$，并加入 $3$D Fourier position encoding。于是每个编码 token 同时携带：

- “该局部区域看起来是什么、与其他视图可能如何对应”的特征；
- “它在当前 VGGT 世界坐标中位于哪里”的显式空间线索。

这对后续 decoder 的空间查询尤其关键：它无需完全从数据中隐式学习“原始坐标”和 VGGT 坐标框架如何关联。

#### 2.2 Perceiver 压缩

将所有视图、所有选取层的 position-encoded patch token 作为 Key/Value，固定的 learned latent tokens 作为 Query：

$$
z_p^{(l+1)}
=
\operatorname{SelfAttn}^{L_s}
\Bigl(
\operatorname{CrossAttn}
\bigl(z_p^{(l)},\{T_n+\gamma(p_n)\}_n\bigr)
\Bigr).
$$

重复 $L_e$ 次后得到 $z_p$。它表示场景的主要几何记忆。camera token 则经一个更轻的 Perceiver 汇总为单个 $z_c\in\mathbb R^{1\times D}$，表达当前世界空间/相机条件。

概念上论文写作：

$$
z=[z_p,z_c].
$$

但实现中两条路径保持分离：$z_p$ 作为 decoder cross-attention 的 Key/Value，$z_c$ 与时间编码一起通过 Ada-LN 调制 decoder。

### 3. Flow-Matching Decoder：独立运输查询点

#### 3.1 目标分布与 source distribution

目标分布 $p_1$ 是训练网格表面上采样得到的带法线点。源分布 $p_0$ 的坐标部分不是简单的全局高斯 $\mathcal N(0,I_3)$，而是以当前场景 VGGT pointmap 为中心的高斯混合。

![Surflo_f3](/assets/Surflo_f3.png)

设 $\{q_i\}$ 为由 VGGT 预测深度、结合估计相机参数反投影得到的世界点。每个初始位置可写为：

$$
m=q_i+\epsilon,\qquad
\epsilon\sim\mathcal N(0,\sigma_s^2I_3).
$$

等价地：

$$
p_0(m)=\sum_iw_i\,\mathcal N(m;q_i,\sigma_s^2I_3).
$$

附录的实现使用场景归一化坐标下 $\sigma_s=0.1$。法线独立地从单位球面均匀采样：

$$
n\sim\operatorname{Uniform}(\mathbb S^2).
$$

**这种设计的原因是**：

1. 初始点大多落在已有粗几何附近，不将大量计算浪费在远离场景的空白区域；
2. 高斯噪声仍会把一部分点扩散到可见表面之外，为补全遮挡区保留空间；
3. 若 $\sigma_s$ 过小，模型难以生成遮挡面；若过大，又退化为不利用粗几何先验的全局随机采样。

实践中，为了在欧氏空间中稳定地表示法线，论文将概念上的 $(m,n)$ 写为：

$$
x_0=(m,m+\varepsilon n)\in\mathbb R^6,
$$

其中 $\varepsilon=10^{-3}$ 乘以场景尺度。推理后以两半之差并归一化恢复法线。

#### 3.2 条件速度场与训练目标

decoder 输入 Fourier-encoded query point $\gamma(x_t)$，通过 cross-attention 查询 $z_p$，并由 $[\tau(t),z_c]$ 的 Ada-LN 条件控制。它是一个 12-layer Transformer：前 6 层交替执行对 latent 的 cross-attention 与 MLP，后 6 层只使用 Ada-LN + MLP；不存在 query point 之间的 self-attention。

不使用 query self-attention 使各点可独立、批量生成，因而支持极大的输出点数；代价是模型训练目标只约束单点边缘分布，未直接约束点集的联合一致性。这一问题由下一节 guidance 处理。

训练时随机采样时间：

$$
t=\operatorname{sigmoid}(y),\qquad
y\sim\mathcal N(1.0,1.6^2).
$$

因此 $t$ 严格位于 $(0,1)$，但推理 ODE 从 $0$ 积分到 $1$。该时间分布覆盖整个路径，同时使监督略偏向靠近目标表面的后期阶段。

#### 3.3 推理与任意分辨率输出

给定场景 latent $z$，从 $p_0$ 采样 $P$ 个 $x_0$，并用 Euler 积分：

$$
x_{t+dt}=x_t+v_\theta(x_t,t\mid z)\,dt.
$$

论文默认以 150 个 ODE steps 解码 $100$K 个点，但 $P$ 是可自由调整的。图 5 展示了同一个 latent 分别解码为 8K、32K、128K 个点；latent 不变，增大 $P$ 只是更密地采样同一场景表面。最终按 Gaussian Wrapping 的方案以 Delaunay triangulation 将带法线点转为三角网格。

### 4. Communication via Guidance：让独立点形成共同表面

#### 4.1 为什么独立解码不够

每个 query point 单独经过 decoder，因此高效且可扩展；但它们不直接看见彼此。尤其在遮挡区或不确定区域，多个点可能落到稍有分歧的位置，表现为漂浮点、局部噪声或表面断裂。

这不是 flow matching 无法生成单点，而是单点损失没有显式建模点集的联合分布。

#### 4.2 基于渲染的耦合项

在 ODE 后期（$t\ge 0.95$），先用普通 decoder 速度估计最终位置：

$$
\hat x_1=x_t+(1-t)v_\theta(x_t,t\mid z).
$$

将整批 $\{\hat x_1\}$ 视为一组小的定向 Gaussian，用 VGGT 恢复的相机渲染出输入视图对应的图像 $\hat I_n(t)$。主要的渲染损失为：

$$
\mathcal L_{\mathrm{render}}
=
\frac1N\sum_{n=1}^{N}
\left[
\lambda\|\hat I_n(t)-I_n\|_1
+(1-\lambda)\operatorname{DSSIM}(\hat I_n(t),I_n)
\right],
$$

并加入渲染深度图上的正则项；可选地，使用单目深度提供深度顺序约束。

对整批候选点做 $M$ 次梯度更新，得到 $\hat x_1^g$，再将修正结果转回实际使用的速度：

$$
v_g=\frac{\hat x_1^g-x_t}{1-t}.
$$

最后进行 Euler 更新：

$$
x_{t+dt}=x_t+dt\,v_g.
$$

关键不在于将每个点分别拉近某个像素，而在于 $\mathcal L_{\mathrm{render}}$ 同时依赖整批点的共同渲染结果。因此梯度让点通过一个共享的全局信号“通信”：相邻却彼此矛盾的点会共同被推向更一致、也更符合输入图像的表面。

### 5. 训练数据：Meshed DL3DV

真实场景的、规模足够大的水密表面数据较少。DL3DV 有多视图图像与 COLMAP 位姿，却没有表面真值。论文对约 $10.5$K 个 DL3DV 场景运行 Gaussian Wrapping，得到：

- watertight mesh；
- 每个场景约 $10^7$ 个从网格表面均匀采样的带法线点；
- 经输入图可见性检查后去除离群点。

训练时，每一步采样 12 个场景，每场景取 $N\in[2,16]$ 个输入视图与约 8K 个表面点。Ground-truth 点原本在 COLMAP 坐标系，而模型点遵循 VGGT 坐标约定；论文用基于渲染深度点的加权仿射拟合，将前者对齐到后者后再计算 flow-matching 损失。

## Experiments

### 1. 实验设置

模型在 meshed DL3DV 上训练，在 ML-Hypersim、BlendedMVS、DTU、SCRREAM 等带原生表面真值的数据集上测试；对 DL3DV test、Tanks & Temples、Mip-NeRF 360、DeepBlending，则用密集视图运行 Gaussian Wrapping 产生 pseudo-GT，以评价稀疏视图前馈方法与密集采集优化表面之间的差距。

默认评价从 16 张无位姿图像重建。指标为：

- Chamfer Distance（CD，越低越好），以场景空间尺度归一化；
- F1-score（越高越好），阈值为场景对角线的 $1\%$。

对比对象包括：

1. 逐视图前馈：VGGT、Depth Anything 3 的 raw pointmap，以及 TSDF 融合版本；
2. 全局 latent 前馈：NOVA3R；
3. 稀疏视图逐场景优化：2DGS、RaDe-GS、Gaussian Wrapping。

raw pointmap 被单独标灰并不参与排名：它们可有不错的局部点距离，却不是一个单一、物理一致的表面。

### 2. 主结果：统一表面而非逐视图拼接

定性结果中，VGGT pointmap 仍存在按视图重复、错位的层；从稀疏输入启动 Gaussian Wrapping 也难以稳定得到完整干净网格。Surflo 从共享 latent 输出的点分布更均匀，网格更完整。

在 16-view 的 DL3DV / Tanks & Temples / Mip-NeRF 360 / DeepBlending 评估中，无 guidance 的 Surflo CD 分别为：

$$
0.0072,\quad 0.0053,\quad 0.0068,\quad 0.0116,
$$

显著优于同表的稀疏视图优化基线。原生表面真值的 ML-Hypersim、BlendedMVS、DTU、SCRREAM 也显示 Surflo 能保持有竞争力的跨数据集泛化。

应注意 guidance 的作用不应只用单一 CD/F1 排序来判断：它主要抑制可见离群点、补回细节、提升感知的表面连续性。表中有些数据集无 guidance 的 CD/F1 更优，有些数据集 guidance 更优；论文对它的主要论断是提升视觉质量与图像一致性，而非所有指标、所有数据集都严格单调改善。

### 3. 输入视图数与输出分辨率的可扩展性

在 Tanks & Temples 与 Mip-NeRF 360 上，作者将输入视图数从 2 增至 32。Surflo 在各视图数下表现最佳或最具竞争力，且 $K=128$ 的 latent 大小始终不变。更多视图会逐步补全遮挡部分、锐化细节，但不会要求重新训练或改变架构。

输出端也独立可调：图 5 以 8K、32K、128K 点展示同一 latent 的多分辨率解码。低点数适合快速预览或粗碰撞查询；密集点集提供更好的细节和背景几何，以便网格化。

### 4. 消融实验与效率

Table 5 的关键结论：

| 设计 | DL3DV | Tanks & Temples | 含义 |
|---|---:|---:|---|
| $K=32$ | CD 0.0089 / F1 72.76 | CD 0.0068 / F1 79.51 | 压缩槽位过少会损失场景细节 |
| $K=128$ | CD 0.0073 / F1 81.21 | CD 0.0055 / F1 88.02 | 更大的 latent 容量更有效 |
| 无 3D PE | CD 0.0083 / F1 76.05 | CD 0.0068 / F1 80.78 | 坐标关系不应完全靠网络隐式学习 |
| Gaussian Fourier 3D PE | CD 0.0073 / F1 81.21 | CD 0.0055 / F1 88.02 | 共享位置编码改善空间查询 |
| 纯全局 Gaussian 源 | CD 0.0088 / F1 73.57 | CD 0.0074 / F1 77.09 | 在空域开始运输的效率较低 |
| VGGT 点图 Gaussian mixture | CD 0.0073 / F1 81.21 | CD 0.0055 / F1 88.02 | 粗几何先验显著有益 |

训练使用 4 张 H100、400K iterations、每步 12 个场景及每场景 8K query。推理时，16-view encoding 只需一次 VGGT + Perceiver 前向；缓存 latent 后，解码 $10^5$ 点需数秒，论文称相较 2DGS 或 Gaussian Wrapping 等逐场景优化快约两个数量级。

guidance 会增加额外时间：论文报告约 30 秒到 3 分钟，取决于 guidance 更新次数；默认从 $t=0.95$ 开始，每个 ODE step 做 $M=32$ 次损失更新。因此“带 guidance”不再是纯粹的一次前馈解码。

### 5. 局限性与阅读判断

1. **依赖 VGGT 的粗几何。** 在极少视图或超大基线下，VGGT pointmap 若错误，source distribution 与 tokenization 都会受影响，decoder 很难完全恢复。
2. **独立生成与全局一致性的矛盾。** 独立 point query 带来灵活性，却必须靠推理时 rendering guidance 补足联合约束；这增加了时间成本。
3. **监督表面并非绝对真值。** meshed DL3DV 的网格由 Gaussian Wrapping 得到，对透明、无纹理等结构本身可能不可靠，模型会继承这些数据偏差。
4. **几何而非外观。** 当前 decoder 不预测 view-dependent radiance；输出可形成网格，但不是完整的可新视角渲染外观表示。

## Conclusion

Surflo 的关键不是单独使用 VGGT、Perceiver 或 flow matching，而是将三者组织成一条统一链路：

> 可变数量的图像证据  
> $\rightarrow$ 固定大小、全局一致的场景 latent  
> $\rightarrow$ 可变数量的独立表面查询  
> $\rightarrow$ 通过渲染梯度获得点集一致性

它回答了一个很实际的问题：能否从稀疏、无位姿的多视图输入中得到既全局统一、又可按需提高分辨率的几何表示？论文给出的答案是以“全局 latent + per-point flow”为核心，并用 inference-time guidance 解决独立采样的表面不一致问题。

对后续工作而言，最值得复用的思路是：将**输入观测数、全局表示容量和输出采样密度**分别控制，而不要把每张图、每个像素与最终几何元素一一绑定。
