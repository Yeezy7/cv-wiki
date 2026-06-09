---
title: 图像分割综述
description: 语义分割、实例分割、全景分割的区别、代表方法与评估指标一文讲清
category: cv
tags: [segmentation, overview]
status: stable
order: 1
---

# 图像分割综述

## 一句话解释

图像分割（Image Segmentation）是将图像中的每个像素分配到一个类别标签的任务，根据粒度不同分为语义分割（Semantic Segmentation）、实例分割（Instance Segmentation）和全景分割（Panoptic Segmentation）三种。

## 它解决什么问题

目标检测只给出目标的矩形框，无法精确描述目标的形状和边界。在自动驾驶、医学影像、遥感分析等场景中，我们需要知道"哪些像素属于道路"、"哪些像素属于肿瘤"——这就是图像分割要解决的问题。

## 核心思想

### 三种分割任务的区别

用一张包含 3 辆车和 2 个行人的街景图来说明：

| 任务 | 输出 | 特点 |
|------|------|------|
| **语义分割** | 每个像素一个类别标签 | 知道"这是车"，但无法区分"哪辆车" |
| **实例分割** | 每个目标实例一个掩码 | 知道"这是第 1 辆车、第 2 辆车"，但不处理背景/stuff |
| **全景分割** | 每个像素一个 (类别, 实例 ID) | 既区分实例，又覆盖所有像素（包括背景） |

关键区别：
- **语义分割**：分类所有像素，同类目标不区分实例
- **实例分割**：只分割"可数目标"（Things，如车、人），不管"不可数区域"（Stuff，如天空、道路）
- **全景分割**：统一处理 Things 和 Stuff，每个像素都要有标签

### 为什么全景分割最难

它同时要求：
1. 对 Things 类别做实例级区分（像实例分割）
2. 对 Stuff 类别做语义级标注（像语义分割）
3. 两类输出要无缝拼接，不能有像素遗漏

## 算法流程

### 语义分割代表方法

#### FCN（Fully Convolutional Network，2015）

开创性工作，将分类网络改为全卷积网络：
1. 用 VGG/ResNet 等骨干网络提取特征
2. 用 $1 \times 1$ 卷积替代全连接层，输出逐像素分类
3. 通过反卷积（Transposed Convolution）上采样到原图尺寸
4. 跳跃连接（Skip Connection）融合浅层和深层特征，恢复细节

**问题**：上采样结果粗糙，边界模糊。

#### U-Net（2015）

为医学影像设计的编码器-解码器结构：
1. **编码器（下采样）**：逐步缩小特征图，提取高层语义
2. **解码器（上采样）**：逐步恢复分辨率
3. **跳跃连接**：编码器的特征直接拼接到解码器对应层，保留空间细节
4. 输出与输入同尺寸的分割掩码

U-Net 的对称结构非常优雅，在医学影像中效果极好，因为医学图像通常目标边界清晰、数据量小，跳跃连接能有效保留细节。

#### DeepLab 系列（2015-2018）

针对语义分割的三大挑战提出系统性解决方案：

**DeepLabv1/v2：空洞卷积（Atrous/Dilated Convolution）**
- 问题：连续池化和下采样导致分辨率下降
- 方案：用空洞卷积替代部分池化，在不增加参数的情况下扩大感受野
- 空洞卷积公式：在标准卷积核的元素间插入"空洞"，膨胀率 $r$ 控制间隔

**DeepLabv2：空洞空间金字塔池化（ASPP）**
- 并行使用多个不同膨胀率的空洞卷积
- 捕获多尺度信息：小膨胀率关注局部细节，大膨胀率关注全局上下文

**DeepLabv3+：编码器-解码器**
- 在 DeepLabv3 基础上加解码器
- 编码器用 ASPP 提取多尺度特征
- 解码器逐步上采样并融合低层特征

### 实例分割代表方法

#### Mask R-CNN（2017）

在 Faster R-CNN 基础上增加一个掩码预测分支：
1. **骨干网络 + FPN**：提取多尺度特征
2. **RPN（Region Proposal Network）**：生成候选区域
3. **RoI Align**：精确的区域特征提取（替代 RoI Pooling，避免量化误差）
4. **三个并行头**：
   - 分类头：预测类别
   - 回归头：精修边界框
   - 掩码头：对每个 RoI 预测一个 $28 \times 28$ 的二值掩码

**RoI Align 的关键改进**：RoI Pooling 在量化时会引入像素级对齐误差，RoI Align 用双线性插值（Bilinear Interpolation）精确计算每个位置的特征值，对分割任务至关重要。

### 全景分割代表方法

#### Panoptic FPN（2019）

将 FPN 扩展到全景分割：
1. **Stuff 分支**：在 FPN 的各层特征上用语义分割头处理背景区域
2. **Things 分支**：用 Mask R-CNN 的方式处理可数目标
3. **后处理融合**：将两个分支的输出合并，解决 Things 和 Stuff 的重叠冲突

#### Mask2Former（2021）

基于 Transformer 的统一架构，一个模型同时处理三种分割任务：
1. **像素解码器（Pixel Decoder）**：提取多尺度像素特征
2. **Transformer 解码器**：用可学习的查询（Queries）通过交叉注意力关注不同区域
3. **掩码分类（Mask Classification）**：每个查询预测一个掩码和对应的类别，不逐像素分类

## 数学定义

### 交并比（Intersection over Union, IoU）

$$ 
IoU = \frac{|M \cap G|}{|M \cup G|}
$$

- $M$：预测掩码（Predicted Mask）
- $G$：真实掩码（Ground Truth Mask）
- 对于二值掩码，交集和并集就是像素级别的与和或

### 平均交并比（mean Intersection over Union, mIoU）

$$ 
mIoU = \frac{1}{k+1} \sum_{i=0}^{k} \frac{TP_i}{TP_i + FP_i + FN_i}
$$

- $k$：类别总数
- $TP_i$：类别 $i$ 的真正例（正确预测为该类的像素数）
- $FP_i$：类别 $i$ 的假正例（错误预测为该类的像素数）
- $FN_i$：类别 $i$ 的假负例（属于该类但未被预测到的像素数）
- $k+1$：包含背景类别

mIoU 是语义分割最核心的指标，取所有类别 IoU 的平均值。

### 平均精度（mean Average Precision, mAP）

用于实例分割和全景分割，计算方式与目标检测类似，但用掩码 IoU 替代框 IoU：

$$ 
AP = \int_0^1 p(r) \, dr
$$

- $p$：精确率（Precision）
- $r$：召回率（Recall）
- 实际计算时，在不同的 IoU 阈值（0.5, 0.55, ..., 0.95）下计算 AP，再取平均

### 全景质量（Panoptic Quality, PQ）

$$ 
PQ = \underbrace{\frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}}_{\text{SQ (Segmentation Quality)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{RQ (Recognition Quality)}}
$$

- $TP$：匹配的预测-真实对（IoU > 0.5）
- $FP$：未匹配的预测
- $FN$：未匹配的真实标注
- SQ 衡量分割质量，RQ 衡量识别质量

## 代码示例

### 语义分割输出格式

```python
import torch
import torch.nn as nn

# 语义分割模型输出
# 输入: [B, 3, H, W] 的 RGB 图像
# 输出: [B, num_classes, H, W] 的逐像素分类 logits

class SimpleSemanticSeg(nn.Module):
    def __init__(self, num_classes=21):  # 21 = 20 类 + 背景
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(128, num_classes, 1)  # 1x1 卷积做逐像素分类

    def forward(self, x):
        features = self.backbone(x)            # [B, 128, H, W]
        logits = self.head(features)           # [B, num_classes, H, W]
        return logits

# 模拟推理
model = SimpleSemanticSeg(num_classes=21)
image = torch.randn(1, 3, 224, 224)
logits = model(image)                        # [1, 21, 224, 224]

# 获取每个像素的类别
pred_mask = torch.argmax(logits, dim=1)      # [1, 224, 224]，值为 0~20

# 输出格式：每个像素一个整数类别标签
print(f"预测掩码形状: {pred_mask.shape}")
print(f"出现的类别: {torch.unique(pred_mask).tolist()}")
```

### 实例分割输出格式

```python
import torch

# 实例分割模型输出（如 Mask R-CNN）
# 每个检测到的目标包含：边界框、类别、置信度、实例掩码

# 模拟 Mask R-CNN 输出（3 个检测结果）
num_instances = 3
image_h, image_w = 224, 224

# 边界框: [N, 4]，格式为 [x1, y1, x2, y2]
boxes = torch.tensor([
    [10, 20, 100, 150],
    [50, 60, 200, 180],
    [120, 30, 210, 200],
], dtype=torch.float32)

# 类别标签: [N]
labels = torch.tensor([1, 1, 3])  # 1=人, 3=车

# 置信度: [N]
scores = torch.tensor([0.95, 0.88, 0.76])

# 实例掩码: [N, 1, H, W]，每个实例一个二值掩码
masks = torch.zeros(num_instances, 1, image_h, image_w)
for i in range(num_instances):
    x1, y1, x2, y2 = boxes[i].int()
    masks[i, 0, y1:y2, x1:x2] = 1.0  # 在框内区域设为 1

# 输出格式：每个实例独立的掩码
print(f"检测到 {num_instances} 个实例")
for i in range(num_instances):
    print(f"  实例 {i}: 类别={labels[i].item()}, "
          f"置信度={scores[i]:.2f}, "
          f"掩码非零像素数={masks[i].sum().int().item()}")
```

### 语义分割 vs 实例分割的关键区别

```python
import torch

h, w = 128, 128

# --- 语义分割输出 ---
# 只有类别，不区分实例
# 图中有 3 辆车，语义分割只标注"这是车"
semantic_mask = torch.zeros(h, w, dtype=torch.long)
semantic_mask[10:40, 10:40] = 3   # 车（所有车都是同一个标签 3）
semantic_mask[50:80, 50:80] = 3   # 另一辆车，标签也是 3
semantic_mask[90:120, 20:50] = 1  # 人

print("语义分割输出形状:", semantic_mask.shape)  # [H, W]
print("语义分割类别:", torch.unique(semantic_mask).tolist())  # [0, 1, 3]

# --- 实例分割输出 ---
# 每个实例有独立掩码，能区分不同的车
instance_masks = [
    {"label": 3, "mask": torch.zeros(h, w)},  # 车 1
    {"label": 3, "mask": torch.zeros(h, w)},  # 车 2
    {"label": 1, "mask": torch.zeros(h, w)},  # 人 1
]
instance_masks[0]["mask"][10:40, 10:40] = 1
instance_masks[1]["mask"][50:80, 50:80] = 1
instance_masks[2]["mask"][90:120, 20:50] = 1

print(f"实例分割输出: {len(instance_masks)} 个实例，每个有独立掩码")

# --- 全景分割输出 ---
# 每个像素一个 (类别, 实例 ID)，统一处理 stuff 和 things
panoptic_mask = torch.zeros(h, w, dtype=torch.long)
# 编码方式：pixel_label = class_id * 1000 + instance_id
panoptic_mask[10:40, 10:40] = 3 * 1000 + 1   # 车, 实例 1
panoptic_mask[50:80, 50:80] = 3 * 1000 + 2   # 车, 实例 2
panoptic_mask[90:120, 20:50] = 1 * 1000 + 1  # 人, 实例 1
panoptic_mask[0:128, 100:128] = 5 * 1000 + 0  # 天空(stuff), 实例 0
# stuff 类别的实例 ID 通常为 0

print("全景分割输出形状:", panoptic_mask.shape)  # [H, W]
print("能同时区分实例和覆盖所有像素（包括背景）")
```

## 面试标准回答

**第一段（概念区分）**：图像分割根据任务粒度分为三种。语义分割给每个像素分配类别标签，但不区分同类目标的不同实例。实例分割在语义分割基础上进一步区分每个目标实例，但只处理可数目标（Things），不管背景区域。全景分割是最完整的方案，统一处理 Things 和 Stuff，每个像素都要有 (类别, 实例) 标签。

**第二段（代表方法）**：语义分割的经典方法是 FCN（全卷积网络，开创性地用卷积替代全连接层）、U-Net（编码器-解码器加跳跃连接，医学影像首选）和 DeepLab 系列（空洞卷积加 ASPP 多尺度特征）。实例分割的代表是 Mask R-CNN，在 Faster R-CNN 基础上加掩码分支，关键改进是 RoI Align 用双线性插值避免量化误差。全景分割早期用 Panoptic FPN 分别处理 Things 和 Stuff，后来 Mask2Former 用 Transformer 统一了三种分割任务。

**第三段（评估指标）**：语义分割用 mIoU（各类别 IoU 的平均值），实例分割用 mAP（与目标检测类似但用掩码 IoU），全景分割用 PQ（Panoptic Quality，等于分割质量 SQ 乘以识别质量 RQ）。mIoU 是最核心的指标，计算时要注意包含背景类别。

## 高频追问

**Q1：语义分割和实例分割的本质区别是什么？**
A：本质区别在于是否区分同类目标的不同实例。语义分割只关心"这个像素属于什么类别"，所有同类目标共享一个标签；实例分割进一步区分"这个像素属于哪个具体的目标个体"。例如一张图中有 3 辆车，语义分割输出中它们都是"车"，实例分割能区分"车1、车2、车3"。

**Q2：为什么 U-Net 在医学影像中效果特别好？**
A：三个原因：一是医学图像目标边界通常比较清晰，跳跃连接能有效融合浅层细节和高层语义；二是医学数据集通常很小，U-Net 的编码器-解码器结构参数量适中，不容易过拟合；三是医学分割对边界精度要求极高，U-Net 的对称结构在恢复空间细节方面表现优异。

**Q3：DeepLab 的空洞卷积解决了什么问题？**
A：标准卷积网络通过池化和步长卷积来扩大感受野，但会降低特征图分辨率，导致分割结果粗糙。空洞卷积通过在卷积核元素间插入间隔（膨胀率 $r$），在不增加参数和计算量的情况下扩大感受野。例如 $3 \times 3$ 卷积核在 $r=2$ 时感受野变为 $5 \times 5$，但仍然只有 9 个参数。

**Q4：RoI Align 和 RoI Pooling 有什么区别？**
A：RoI Pooling 将候选区域划分为固定网格，取每个网格的最大值，但划分时有量化取整操作，会导致像素级对齐误差（最多 1 像素偏差）。对分类任务影响不大，但对分割任务影响显著。RoI Align 去掉了量化操作，用双线性插值精确计算每个采样点的特征值，位置对齐更准确。

**Q5：全景质量 PQ 的设计思路是什么？**
A：PQ = SQ × RQ，把"分割准不准"和"识别对不对"解耦。SQ 是匹配对（IoU > 0.5 的预测-真实对）的平均 IoU，衡量分割边界的精度。RQ 类似 F1 Score，衡量目标是否被正确检测到。这种设计避免了单纯用 mIoU 时，大面积 Stuff 类别主导指标的问题。

## 工程实践

### 模型选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 医学影像（器官/病灶分割） | U-Net / nnU-Net | 小数据集效果好，边界精度高 |
| 自动驾驶（街景理解） | DeepLabv3+ / Mask2Former | 多尺度目标，需要全景理解 |
| 遥感图像分析 | DeepLabv3+ / U-Net 变体 | 大尺寸图像，需要高效推理 |
| 通用实例分割 | Mask R-CNN / Mask2Former | 成熟稳定，生态丰富 |
| 全景分割 | Mask2Former | 统一架构，三种任务通用 |

### 常用数据集

| 数据集 | 任务类型 | 类别数 | 特点 |
|--------|----------|--------|------|
| PASCAL VOC 2012 | 语义/实例分割 | 20 | 入门级，类别少 |
| COCO | 实例/全景分割 | 80 | 最主流的检测分割数据集 |
| Cityscapes | 语义/全景分割 | 19 | 自动驾驶场景，精细标注 |
| ADE20K | 语义分割 | 150 | 类别多，场景复杂 |
| ISBI (医学) | 语义分割 | 2 | 医学影像经典数据集 |

### 评估指标计算

```python
import numpy as np

def compute_miou(pred_mask, gt_mask, num_classes):
    """计算 mIoU（平均交并比）"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        intersection = (pred_cls & gt_cls).sum()
        union = (pred_cls | gt_cls).sum()
        if union == 0:
            continue  # 跳过该图中不存在的类别
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# 示例
pred = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]])
gt = np.array([[0, 0, 1], [0, 0, 1], [2, 2, 0]])
print(f"mIoU: {compute_miou(pred, gt, num_classes=3):.4f}")
```

### 常用优化手段

1. **数据增强**：随机裁剪、翻转、颜色抖动、尺度变换。对分割任务，裁剪时必须同步变换图像和标注
2. **类别平衡**：使用加权交叉熵损失或 Dice Loss，解决背景/前景像素比例失衡问题
3. **多尺度推理**：测试时对多个尺度的预测取平均（Multi-Scale Test），提升精度
4. **预训练骨干**：使用在 ImageNet 上预训练的骨干网络，不要从零训练
5. **混合精度训练**：使用 FP16 加速训练，节省显存

## 常见误区

**误区 1："语义分割和实例分割只是粒度不同"**
不只是粒度问题，两者的建模方式完全不同。语义分割是逐像素分类问题，输出一个 $H \times W$ 的标签图；实例分割是检测+分割问题，输出一组 (掩码, 类别, 置信度) 三元组。两者的模型架构和损失函数都不同。

**误区 2："mIoU 高就代表模型好"**
mIoU 对大类别（如道路、天空）和小类别（如交通标志）一视同仁。如果数据集中类别严重不均衡，mIoU 可能掩盖小类别上的糟糕表现。实际中还需要关注每个类别的 IoU 分布。

**误区 3："上采样用反卷积效果最好"**
反卷积（Transposed Convolution）容易产生棋盘格伪影（Checkerboard Artifacts）。实践中双线性插值 + 卷积的效果通常更平滑、更稳定。

**误区 4："实例分割一定要用 Mask R-CNN"**
Mask R-CNN 是经典方案，但 Mask2Former 用统一架构在三种分割任务上都达到了 SOTA，且不需要针对任务设计不同头部。对于新项目，建议优先考虑 Mask2Former 或类似的 Transformer 架构。

**误区 5："全景分割的 PQ 可以直接比较不同数据集"**
PQ 的数值受类别数量和 Things/Stuff 比例影响。COCO 和 Cityscapes 的 PQ 不能直接比较。同一数据集上的 PQ 比较才有意义。

## 参考资料

- Long et al., "Fully Convolutional Networks for Semantic Segmentation" (FCN, 2015)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Chen et al., "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs" (DeepLabv1, 2015)
- Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (DeepLabv3, 2017)
- Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (DeepLabv3+, 2018)
- He et al., "Mask R-CNN" (2017)
- Kirillov et al., "Panoptic Feature Pyramid Networks" (2019)
- Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation" (Mask2Former, 2021)
- Kirillov et al., "Panoptic Segmentation" (PQ 指标定义, 2019)
