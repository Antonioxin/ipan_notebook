# C3G: Learning Compact 3D Representations with 2K Gaussians

## 开始前复习一下知识点

### 1. 前馈式三维重建（Feed-forward 3D Reconstruction）

前馈式三维重建指的是：模型在大量场景上完成预训练后，面对一个新的测试场景，只需把该场景的单张或多张图像输入网络，经过一次或有限次数的前向计算，就能直接预测这个场景的三维表示。

输出的三维表示可以是：

- depth map 或 point map；
- point cloud；
- voxel；
- mesh；
- NeRF；
- 3D Gaussians。

它的典型流程是：

```text
大量训练场景
    ↓
学习跨场景共享的几何与外观先验
    ↓
训练完成的通用重建网络

新的测试场景图像
    ↓
一次网络前向推理
    ↓
直接得到该场景的三维表示
```


---

### 2. 测试时优化（Test-Time Optimization, TTO）

TTO 属于一种测试时扩展（Test-Time Scaling）方法，TTS 主要有两种优化方向：
1. 优化输入数量，增加信息输入，比如增加输入视图
2. 增加测试时的计算，比如更多的优化步数，也是本文（C3G）采用的方法

TTO 指模型在测试阶段面对一个具体场景时，利用该场景现有的输入观测，对前馈预测结果再进行一小段梯度优化。


在 C3G 中，TTO 不是从随机 Gaussians 开始重建，而是把前馈生成的约 2K 个 Gaussians 当作高质量初始化：

```text
测试场景的多视图图像
    ↓
C3G 前馈预测
    ↓
约 2K 个初始 Gaussians
    ↓
利用测试场景的输入视图优化 1,000 steps
+ Gaussian densification
    ↓
约 26K–28K 个优化后 Gaussians
    ↓
更高质量的新视角渲染
```

因此，TTO 的作用可以理解为：**用通用网络迅速给出一个不错的场景初值，再允许模型针对当前场景修正几何、外观和局部细节。**


## Introduction

### 1. 研究背景：从多视图图像快速获得三维场景表示

从若干张多视图图像中恢复三维场景，是计算机视觉与计算机图形学中的基础问题。一个理想的系统不仅要能够完成 novel view synthesis（新视角合成），还应该生成可被机器人、场景理解模型和视觉语言模型继续使用的三维表示，例如在三维空间中完成语义分割、跨视图匹配和开放词汇理解。

传统三维重建策略通常采用 **逐场景优化（per-scene optimization）**：面对一个新场景，需要利用该场景的多张带位姿图像，对辐射场或 Gaussian 参数进行数千乃至数万步优化。这类方法可以获得较好的重建质量，但每个新场景都要重新求解，计算成本较高，也不适合需要快速感知的应用。

近年来出现的 **前馈式三维重建（feed-forward 3D reconstruction）** 试图改变这一流程。模型预先在大量场景上训练，测试时输入一个从未见过的新场景，只经过一次网络前向计算，就直接预测该场景的点云、NeRF 或 3D Gaussians。进一步地，NoPoSplat、AnySplat 等工作开始研究 **无相机位姿的前馈式重建（pose-free feed-forward reconstruction）**：模型只接收多张 RGB 图像，不要求用户显式提供输入相机位姿，而是依靠预训练视觉模型中蕴含的几何先验推断场景结构。

3DGS 很适合这种前馈式重建。一方面，Gaussian 是显式的三维 primitive，每个 Gaussian 都有中心、协方差、opacity 和颜色属性；另一方面，它可以通过高效的 differentiable rasterization 完成实时或近实时渲染。因此，许多前馈方法选择直接让网络从图像特征预测一组 3D Gaussians。

然而，C3G 认为，现有前馈 3DGS 方法虽然省去了逐场景优化，却继承了一个并不理想的表示假设：**把 Gaussian 与输入图像中的像素绑定起来。** 当 Gaussian 默认与场景中的像素进行绑定时，场景中就会不可避免地出现冗余高斯体，而这些高斯体又往往会进一步影响重建质量（因为同一场景的不同视角输入图可能存在重叠的部分）。

---

### 2. 现有方法的核心问题：per-pixel Gaussian prediction 产生了大量冗余

主流前馈式 3DGS 方法通常采用 pixel-aligned prediction。网络首先为输入图像中的每个像素或 patch 估计深度与三维位置，然后为它解码一个或多个 Gaussians：


这种设计实现简单，而且保留了二维像素与三维 primitive 之间的直接对应，但它会带来三个相互关联的问题。

#### 2.1 Gaussian 数量随视图数和分辨率快速增长

如果每个输入像素都生成一个 Gaussian，那么 Gaussian 数量大致与

```text
输入视图数量 × 每张图像的像素或 patch 数量
```

成正比。随着输入图像数量增加，模型可能生成数十万甚至数百万个 Gaussians。问题在于，这些 Gaussians 并不都对应独立的三维信息：不同视图经常重复观察同一个墙面、桌子或物体表面，但 pixel-aligned decoder 会分别为这些重复观测生成新的 Gaussians。

因此，增加视图虽然带来了更多观测，也带来了大量重复 primitive。表示规模越来越大，却不一定获得相应比例的有效三维信息。

#### 2.2 跨视图误差会被固化为空间错位

不同视图对同一表面的深度估计和几何预测不可能完全一致。若每个视图独立或近似独立地产生 Gaussians，那么同一个真实表面可能被重建成多层相互错开的 Gaussian：

- 同一位置被多个视图重复表示；
- 不同视图的深度误差造成 Gaussian misalignment；
- 错误的深度预测会直接把 Gaussian 放到不正确的三维位置；
- 输入视图越多，误差和冗余可能不断累积。

换句话说，pixel-aligned prediction 把二维采样网格强行带入了三维表示。它优先保证“每个像素都有一个 Gaussian”，却不能精确回答“这个场景真正需要在哪些三维位置放置 Gaussian”。

#### 2.3 密集 Gaussians 使高维语义特征 lifting 变得昂贵

三维重建不仅用于渲染 RGB，还可以把 LSeg、MaskCLIP、DINO、VGGT 等视觉模型的二维特征提升到三维空间，使三维场景能够支持：

- open-vocabulary segmentation（开放词汇场景分割）；
- 3D scene understanding；
- multiview correspondence（多视角一致性）；
- feature upsampling；
- 其他需要三维一致特征的下游任务。

问题在于，语义特征往往具有数百甚至上千维。如果场景中存在几十万或数百万个 Gaussians，并且每个 Gaussian 都携带一个高维特征向量，显存与计算开销会非常大。因此，已有方法通常需要先用 autoencoder 把语义特征压缩到较低维度，再附着到 Gaussians 上，渲染后再通过 decoder 恢复。

这种做法虽然降低了内存，却会造成两类损失：

1. **特征压缩损失**：高维 foundation-model features 被压缩后，细粒度语义与对应关系可能丢失；
2. **跨视图不一致**：同一三维位置在不同视图中的二维特征并不完全相同，方法仍需额外设计 correspondence、backward mapping 或多视图聚合策略。

因此，dense per-pixel Gaussians 不仅是重建表示的冗余问题，也成为二维 foundation features 进入三维空间时的主要瓶颈。

---

### 3. 论文提出的核心问题

基于上述观察，C3G 提出了一个很直接但重要的问题：

> 为了重建和理解一个三维场景，我们真的需要与输入像素一一对齐的大量 Gaussians 吗？

作者认为答案是否定的。人类理解场景时，并不会在头脑中保存每个视角下每个像素的独立副本，而是形成更紧凑的空间抽象：识别重要物体、表面区域及其空间关系。受到这一思路启发，C3G 不再从输入图像的像素网格出发，而是从一个固定大小的三维表示预算出发：

> 预先给模型一组数量有限的 learnable queries，让这些 queries 自己决定应该从哪些视图、哪些图像区域收集信息，以及应该把 Gaussians 放在场景中的哪些位置。

这样一来，模型优化的目标不再是“为每个像素生成一个 Gaussian”，而是：

> 在只有大约 2K 个 Gaussians 的严格容量限制下，寻找最有利于重建整个场景的三维 primitive。

这也是 C3G 名称中 **Compact 3D Gaussians** 的核心含义。

---

### 4. 方法一：C3G-G——使用 learnable queries 直接生成紧凑 Gaussians

C3G 的第一个核心模块是 **C3G-G（Compact 3D Gaussian Decoder）**。

![C3G-G_pipeline](/assets/C3G-G.png)

#### 4.1 从多视图图像提取具有几何先验的特征

给定同一场景的多张无位姿图像，模型首先使用预训练视觉编码器提取多视图特征。论文默认采用 VGGT，因为 VGGT 在大规模几何任务上训练过，其 feature tokens 包含较强的三维结构和跨视图关系先验。

这一阶段可以理解为：

```text
多张无位姿 RGB 图像
        ↓
VGGT 等视觉编码器
        ↓
多视图 image feature tokens
```

C3G 不要求外部输入相机 pose，但并不意味着模型完全凭空恢复几何。它很大程度上利用了 VGGT 在预训练阶段学到的几何知识。

#### 4.2 用固定数量的 Gaussian queries 代替像素网格

模型引入固定数量的 learnable Gaussian queries，主模型设置为：

```text
N = 2048
```

这些 queries 不与某个固定像素绑定。每个 query 都是一个可学习的场景表示 slot，可以自由地从任意输入视图、任意图像区域中收集信息。

模型将 Gaussian queries 与所有多视图 image tokens 拼接，然后送入 Transformer self-attention blocks：

```text
[Gaussian queries（Q） ; multi-view image tokens（K）]
                    ↓
          Transformer self-attention
                    ↓
          refined Gaussian queries
```

通过 full self-attention：

- query 可以关注不同视图中的相关图像区域；
- query 可以聚合同一三维区域在多个视角下的信息；
- query 之间可以交换信息，减少多个 query 重复表示同一区域；
- 不同 queries 可以逐渐形成对场景不同空间区域的分工。

最后，每个 refined query 经过一个轻量 Gaussian head，直接解码成一个 3D Gaussian，包括中心、协方差、opacity 和颜色等属性。由于每个 query 对应一个 Gaussian，主模型最终始终输出大约 2K 个 Gaussians，而不会随输入分辨率或输入视图数量线性增加。

#### 4.3 只用 novel-view photometric loss 学习空间分配 “简单而有效的策略“

作者没有为 query 应该覆盖哪个物体、表面或空间区域提供显式监督，也没有使用 ground-truth scene decomposition。训练时，模型将预测出的 Gaussians 渲染到目标视角，并使用 RGB 重建误差和 LPIPS perceptual loss 监督：

```text
预测 Gaussians
      ↓
在训练目标相机位姿下渲染
      ↓
与真实目标图像计算 MSE + LPIPS
```

这意味着 query 的空间分工是由 novel view synthesis 目标自行形成的。由于模型只有 2K 个 Gaussians，如果多个 queries 重复覆盖同一区域，或者把 Gaussians 放在无效位置，剩余区域便无法被正确重建。固定容量因此产生了一种隐式优化压力，迫使 queries：

- 减少冗余；
- 覆盖重要场景区域；
- 聚合跨视图一致信息；
- 把有限表示能力分配到最有价值的位置。

#### 4.4 emergent attention：有限表示预算带来的跨视图对应

论文观察到一个重要的 emergent property：即使没有显式 correspondence supervision，每个 Gaussian query 的 attention map 也会在不同输入视图中聚焦到空间上相互对应的区域。

例如，一个最终位于桌子附近的 Gaussian，其 query 可能同时关注多张输入图像中的桌子区域；位于墙角的 Gaussian query，则会在不同视图中关注对应的墙角位置。

作者据此认为，C3G-G 学到的不只是“如何输出 Gaussian 参数”，还隐式学到了：

```text
某个三维 Gaussian
        ↕
它在不同输入视图中对应哪些二维区域
```

这种 Gaussian-to-Image attention 是后续 C3G-F 能够进行高效特征 lifting 的关键。

需要注意的是，这里的“空间上有意义”主要由 attention 可视化和下游 correspondence 实验支持。论文没有证明每个 query 都对应稳定的语义物体，也没有证明同一 query index 在不同场景中具有固定语义。因此，更准确的理解是：queries 学到了对重建有用的跨视图空间分工，而不一定学到了显式、可解释的 object parts。

---

### 5. 方法二：C3G-F——复用几何 attention 完成任意二维特征的三维提升

C3G 的第二个核心模块是 **C3G-F（Compact 3D Feature Decoder）**。它要解决的问题是：如何把任意视觉编码器产生的二维特征，高效地聚合并附着到 C3G-G 生成的紧凑 Gaussians 上。

![C3G-F_training_schema](/assets/C3G-F_training_schema.png)

#### 5.1 传统 feature lifting 的两个难点

要把二维特征提升到三维，首先需要知道：

1. 图像中的某个 patch 对应哪些 3D Gaussians；
2. 同一个三维区域在多个视图中出现时，应该怎样聚合这些并不完全一致的二维特征。

已有方法通常依赖显式投影、backward mapping、可见性计算或额外的多视图聚合模块。这些操作本身较昂贵，而且面对不同 foundation models 时，往往需要重新设计压缩和聚合方式。

#### 5.2 将 C3G-G 的 attention 当作 correspondence

C3G-F 的关键想法是：既然 C3G-G 的 Gaussian queries 已经学会在不同视图中关注空间对应区域，就可以直接把这组 attention 当作 feature aggregation weights。

假设希望提升另一个视觉编码器 \(E'\) 的特征，例如：

- LSeg；
- MaskCLIP；
- DINOv2；
- DINOv3；
- VGGT tracking features。

C3G-F 保留 C3G-G 提供的 query/key attention 关系，不让 feature loss 反向修改这套几何 correspondence；同时，用目标视觉编码器的特征构造新的 value features，并训练对应的 value projection 与 feature head。

可以把它理解成：

```text
C3G-G 决定“去哪里取特征”
目标视觉编码器 E' 决定“取到的内容是什么”
```

在 attention 公式中：

- \(Q\) 和 \(K\) 来自几何分支 C3G-G，负责建立 Gaussian 与多视图图像区域之间的对应；
- \(V'\) 来自目标视觉特征，负责把真正需要保存的语义或几何信息传给 Gaussian；
- attention 聚合后，每个 Gaussian 获得一个跨视图融合的高维 feature attribute。

最终，C3G-F 输出与 2K Gaussians 一一对应的多视图聚合特征。这些 feature attributes 可以像颜色一样通过 Gaussian rasterization 渲染到任意视角，从而生成 novel-view feature maps。

#### 5.3 紧凑表示为什么有利于 feature lifting

C3G 的优势不仅是把 RGB Gaussians 从十几万、几十万压缩到 2K，更重要的是：只有 2K 个 Gaussians 时，可以直接为每个 Gaussian 保存较高维的原始视觉特征，而不必强制使用 autoencoder 压缩。

因此，C3G-F 试图同时解决：

- dense Gaussians 带来的高内存；
- autoencoder 带来的特征损失；
- 多视图 feature inconsistency；
- 显式 backward mapping 的额外计算。

这使同一组紧凑 Gaussians 不只是 RGB 场景表示，也可以成为统一的 3D feature field，服务于开放词汇分割、跨视图 correspondence 和 feature upsampling 等任务。

---

### 6. 论文的主要贡献

综合来看，C3G 的贡献可以概括为以下四点。

#### 贡献一：把前馈式 3DGS 从 pixel prediction 改写为 set prediction

论文不再从输入像素出发生成 Gaussians，而是使用固定数量的 learnable queries 直接预测一个场景级 Gaussian set。这个改变使模型可以主动选择最值得表示的空间位置，从根源上避免 primitive 数量随输入像素和视图数增长。

最重要的思想不是“2K”这个具体数字，而是 **representation budget first**：先限定场景可以使用的表示预算，再让网络学习如何分配这份预算。

#### 贡献二：只依靠重建目标学出紧凑、跨视图一致的空间分工

C3G-G 不依赖 ground-truth depth、三维分割或 scene decomposition 标签，而是在 novel-view photometric supervision 下，让 queries 自发学习覆盖场景中的重要区域。

固定 query 数量既是一种压缩约束，也是一种归纳偏置：它迫使模型减少重复、利用跨视图信息，并形成对重建有用的空间分工。

#### 贡献三：利用 emergent attention 实现任意二维特征的三维聚合

C3G-F 复用 C3G-G 学到的 Gaussian-to-Image attention，把几何 correspondence 与目标特征内容解耦：

- 几何分支提供 Q/K；
- 任意视觉编码器提供新的 V；
- 每个 Gaussian 得到多视图一致的聚合特征。

这使模型无需为每种视觉特征重新设计复杂的 correspondence 或压缩模块，也减少了 autoencoder 的信息损失。

#### 贡献四：验证紧凑表示可以同时支持重建与理解

论文在以下任务中验证了 C3G：

- pose-free novel view synthesis；
- open-vocabulary 3D scene segmentation；
- multiview correspondence；
- feature upsampling；
- 任意视觉基础模型特征的 2D-to-3D lifting。

主模型只使用约 2K 个 Gaussians，相比部分 per-pixel 方法减少约 65 倍，同时显著降低 feature-field 内存。在纯前馈 NVS 中，它以一定高频细节和 SSIM 为代价获得接近部分密集基线的 PSNR；经过可选 TTO 后，重建质量可以进一步提升。对于三维理解和跨视图特征任务，紧凑 Gaussians 的优势更加明显，因为模型能够避免在大量冗余 primitives 上存储压缩后的特征。

---

### 7. 快速回忆：C3G 到底做了什么？

可以用下面这条主线快速回忆全文：

```text
现有前馈 3DGS：
每个像素预测 Gaussian
→ 数量巨大
→ 跨视图重复和错位
→ 高维语义特征存不起，只能压缩

C3G-G：
用 2048 个全局 learnable queries 读取多视图特征
→ 每个 query 直接解码一个 Gaussian
→ 在固定预算下学习覆盖关键空间区域
→ query attention 自发形成跨视图对应

C3G-F：
冻结几何分支学到的 Q/K correspondence
→ 把任意视觉编码器特征作为新的 V
→ 聚合成每个 Gaussian 的多视图一致特征
→ 支持新视角 feature rendering 与三维理解
```

一句话概括：

> C3G 的核心不是事后把大量 Gaussians 压缩到 2K，而是从一开始就只给模型 2K 个全局表示 slots，让它直接学习“场景中哪些位置值得被表示”；随后再把这些 slots 学到的跨视图 attention 复用于高维特征聚合。

---

### 8. 精读时需要记住的概念边界

1. **Pose-free 不等于训练和评估完全不使用 pose。**  
   C3G 的推理输入不要求显式相机位姿，但训练 novel-view loss 时使用已知 target pose；评估 pose-free 方法时还会优化 target camera pose 来处理尺度与坐标对齐问题。

2. **2K 是纯前馈主模型的表示规模，不是所有高质量结果的最终 Gaussian 数。**  
   `C3G w/ TTO` 会以 2K Gaussians 为初始化，对当前测试场景进行约 1,000 步优化和 densification，最终通常增长到约 26K–28K。它依然远小于数百万 Gaussian 的基线，但不能再解释为“最终仅使用 2K”。

3. **C3G 更准确的定位是 compactness-first。**  
   在两视图高保真 NVS 中，纯前馈 C3G 的 PSNR、SSIM 和 LPIPS 并不全面领先强基线；它最突出的优势是表示规模、渲染效率，以及高维 feature lifting 的成本与质量。

4. **learnable queries 不应直接理解为语义物体。**  
   论文证明的是 queries 学到了对重建有用、在多视图中具有空间一致性的 attention。它们是否对应稳定的物体部件、是否跨场景保持相同语义，仍没有被证明。

5. **论文真正可迁移的贡献是“query slots 与 rendering primitives 解耦”。**  
   附录表明，可以保持 2,048 个 queries 不变，让每个 query 解码多个局部 Gaussians，从而在不破坏跨视图 slot 分工的情况下提高细节容量。因此，2K queries 更像紧凑的全局 correspondence slots，而最终用于渲染的 primitive 数可以按任务需要扩展。
