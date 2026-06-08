---
title: 目标检测综述
description: 从 R-CNN 到 DETR，全面理解目标检测的发展脉络与核心范式
category: cv
tags: [object-detection, overview]
status: stable
order: 1
---

# 目标检测综述

## 一句话解释

目标检测（Object Detection）是在图像中同时定位目标的位置（用边界框表示）并识别其类别的任务。

## 它解决什么问题

图像分类只告诉你"图里有猫"，目标检测则告诉你"图里有一只猫，在左上角，坐标是 (50, 30) 到 (200, 180)"。它是自动驾驶、安防监控、工业质检等场景的核心能力。

## 核心思想

目标检测 = 定位 + 分类。模型需要输出两样东西：

1. **边界框（Bounding Box）**：目标在图像中的位置，通常用 (x, y, w, h) 或 (x1, y1, x2, y2) 表示
2. **类别标签（Class Label）**：目标属于哪个类别
3. **置信度（Confidence Score）**：模型对这个检测结果有多大把握

## 发展脉络

### 两阶段检测器（Two-Stage Detectors）

两阶段方法先生成候选区域，再对候选区域分类和回归。

**R-CNN（2014）**

- 用 Selective Search 生成约 2000 个候选区域
- 每个候选区域分别过 CNN 提取特征
- 用 SVM 分类 + 线性回归修正框
- 缺点：极慢，一张图要几十秒

**Fast R-CNN（2015）**

- 整张图过一次 CNN 得到特征图
- 候选区域在特征图上用 RoI Pooling 提取固定大小特征
- 分类和回归用多任务损失同时训练
- 改进：快了很多，但候选区域生成还是瓶颈

**Faster R-CNN（2015）**

- 提出区域提议网络（Region Proposal Network, RPN）
- RPN 直接在特征图上生成候选区域，替代 Selective Search
- 整个流程端到端训练
- 标志着两阶段检测器成熟

### 一阶段检测器（One-Stage Detectors）

一阶段方法直接在特征图上预测框和类别，不需要候选区域生成步骤。

**YOLO（You Only Look Once, 2016）**

- 将图像划分为 S x S 网格，每个网格直接预测框和类别
- 速度极快，实现实时检测
- 从 YOLOv1 到 YOLOv8 不断迭代优化

**SSD（Single Shot MultiBox Detector, 2016）**

- 在多个不同尺度的特征图上检测
- 多尺度检测提升了小目标的检测能力
- 兼顾速度和精度

**RetinaNet（2017）**

- 提出焦点损失（Focal Loss），解决正负样本不平衡问题
- 一阶段检测器的精度首次超越两阶段
- 证明了一阶段方法的潜力

### Anchor-Free 方法

Anchor-Free 方法摒弃了预定义锚框（Anchor Box）的设计。

**FCOS（Fully Convolutional One-Stage, 2019）**

- 每个特征图位置直接预测到边界框四条边的距离
- 无需锚框，设计更简洁
- 无需 NMS 后处理的变体也被探索

**CenterNet（2019）**

- 将目标检测转化为关键点检测
- 用目标中心点表示目标，预测中心点偏移和尺寸

### 基于 Transformer 的方法

**DETR（Detection Transformer, 2020）**

- 用 Transformer 的注意力机制替代手工设计的组件
- 将检测建模为集合预测问题
- 无需锚框、NMS 等后处理
- 端到端训练，设计极简

## Anchor-Based vs Anchor-Free

| 特性 | Anchor-Based | Anchor-Free |
|------|-------------|-------------|
| 预定义框 | 需要预设不同尺度和比例的锚框 | 不需要 |
| 超参数 | 锚框的数量、大小、比例需要调参 | 更少的超参数 |
| 代表方法 | Faster R-CNN, SSD, RetinaNet | FCOS, CenterNet |
| 正负样本分配 | 基于 IoU 匹配锚框和真值框 | 基于点或中心区域 |
| 灵活性 | 锚框设计对数据敏感 | 更通用 |

## 一阶段 vs 两阶段

| 特性 | 一阶段 | 两阶段 |
|------|--------|--------|
| 流程 | 直接预测 | 先生成候选区域，再分类回归 |
| 速度 | 快 | 较慢 |
| 精度 | 早期较低，RetinaNet 后追平 | 通常更高 |
| 代表方法 | YOLO, SSD, RetinaNet | R-CNN 系列 |
| 适用场景 | 实时检测 | 精度要求高 |

## 评估指标概述

- **IoU（Intersection over Union）**：预测框与真值框的重叠程度
- **Precision（精确率）**：预测为正的样本中有多少是真正例
- **Recall（召回率）**：所有正样本中有多少被正确预测
- **AP（Average Precision）**：单个类别的 PR 曲线下面积
- **mAP（mean Average Precision）**：所有类别 AP 的平均值

详见 [mAP 详解](../map/)。

## 面试标准回答

目标检测是计算机视觉中的核心任务，需要同时完成定位和分类。从发展脉络来看，最早是 R-CNN 系列的两阶段方法，先用 Selective Search 或 RPN 生成候选区域，再对候选区域分类和回归。Faster R-CNN 通过引入 RPN 实现了端到端训练，是两阶段方法的里程碑。

后来 YOLO 和 SSD 提出了一阶段方法，直接在特征图上预测框和类别，速度大幅提升。RetinaNet 通过 Focal Loss 解决了正负样本不平衡问题，使一阶段精度追平两阶段。再后来 FCOS 等 Anchor-Free 方法摒弃了锚框设计，简化了模型。DETR 则用 Transformer 彻底替代了手工组件，实现了真正的端到端检测。

核心范式可以总结为：Anchor-Based 两阶段 -> Anchor-Based 一阶段 -> Anchor-Free -> 基于 Transformer。每一步都在追求更简洁的设计和更好的性能。

## 高频追问

**Q: 为什么一阶段方法早期精度不如两阶段？**

A: 主要原因是正负样本极度不平衡。一阶段方法在每个位置都预测，导致负样本远多于正样本。RetinaNet 的 Focal Loss 通过降低易分样本的权重解决了这个问题。

**Q: Anchor-Free 方法为什么能替代 Anchor-Based？**

A: Anchor-Based 需要精心设计锚框的尺度和比例，且正负样本分配依赖 IoU 阈值。Anchor-Free 用点或中心区域分配样本，设计更简洁，泛化性更好。

**Q: DETR 的优势和劣势是什么？**

A: 优势是端到端、无需 NMS、设计简洁。劣势是训练收敛慢（需要几百个 epoch）、小目标检测效果相对较差。后续的 Deformable DETR 改善了这些问题。

## 工程实践

1. 选择方法时，实时场景优先考虑 YOLO 系列，精度优先考虑两阶段或 DETR 变体
2. 数据集较小或类别单一时，Fine-tune 预训练模型比从头训练效果好得多
3. 多尺度检测对小目标很重要，FPN（Feature Pyramid Network）是标配组件
4. 数据增强（如 Mosaic、MixUp）对检测任务效果显著

## 常见误区

1. **"一阶段一定比两阶段快"** — 不一定，YOLO 的大模型可能比轻量化的 Faster R-CNN 还慢
2. **"mAP 高就是好模型"** — 还要看推理速度、模型大小、部署难度
3. **"锚框越多检测越好"** — 过多锚框会增加计算量和正负样本不平衡问题
4. **"DETR 完全替代了传统方法"** — DETR 训练成本高，在小数据集上不一定有优势

## 参考资料

- Girshick et al., "Rich feature hierarchies for accurate object detection and semantic segmentation", CVPR 2014
- Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019
- Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
