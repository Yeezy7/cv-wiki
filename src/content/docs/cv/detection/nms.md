---
title: 非极大值抑制（NMS）
description: 目标检测后处理的核心算法，去除冗余检测框
category: cv
tags: [nms, object-detection, post-processing]
status: stable
order: 3
---

# 非极大值抑制（NMS）

## 一句话解释

非极大值抑制（Non-Maximum Suppression, NMS）是目标检测的后处理步骤，用于去除对同一目标的多个重复检测框，只保留置信度最高的那个。

## 它解决什么问题

目标检测模型通常会对同一个物体生成多个重叠的检测框。比如检测一只猫，模型可能输出 5 个框，位置略有不同，置信度也不同。NMS 的作用就是把这 5 个框合并成 1 个最准确的框。

## 核心思想

贪心策略：每次选置信度最高的框，然后删掉所有和它重叠度（IoU）超过阈值的框。重复这个过程，直到没有框可以处理。

## 算法流程

1. 将所有检测框按置信度从高到低排序
2. 选置信度最高的框加入结果集
3. 计算该框与剩余所有框的 IoU
4. 删除 IoU 超过阈值（如 0.5）的框
5. 对剩余框重复步骤 2-4，直到没有框剩余

## 数学定义

### IoU（Intersection over Union，交并比）

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{交集面积}}{\text{并集面积}}$$

其中：
- $A$、$B$ 分别表示两个边界框
- $|A \cap B|$ 是两个框的交集面积
- $|A \cup B|$ 是两个框的并集面积
- IoU 的取值范围是 $[0, 1]$，1 表示完全重叠，0 表示完全不重叠

### 交集面积计算

给定两个框 $B_1 = (x_1, y_1, x_2, y_2)$ 和 $B_2 = (x_1', y_1', x_2', y_2')$：

$$x_{inter} = \max(0, \min(x_2, x_2') - \max(x_1, x_1'))$$

$$y_{inter} = \max(0, \min(y_2, y_2') - \max(y_1, y_1'))$$

$$\text{交集面积} = x_{inter} \times y_{inter}$$

取 $\max(0, \cdot)$ 是因为当两个框不重叠时，交集面积为 0。

## 代码示例

### 手写 NMS 实现

```python
import torch

def compute_iou(box, boxes):
    """
    计算一个框与一组框的 IoU
    Args:
        box: shape (4,)，格式 [x1, y1, x2, y2]
        boxes: shape (N, 4)，格式 [x1, y1, x2, y2]
    Returns:
        iou: shape (N,)
    """
    # 交集的左上角和右下角
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    # 交集面积
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 各自的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 并集面积 = 两个面积之和 - 交集面积
    union_area = box_area + boxes_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)  # 加 1e-6 防止除零
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制
    Args:
        boxes: shape (N, 4)，格式 [x1, y1, x2, y2]
        scores: shape (N,)，置信度分数
        iou_threshold: IoU 阈值，超过此值的框被抑制
    Returns:
        keep: 保留的框的索引
    """
    # 按置信度降序排序
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        # 选置信度最高的框
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算该框与剩余框的 IoU
        remaining = order[1:]
        iou = compute_iou(boxes[i], boxes[remaining])

        # 保留 IoU 小于阈值的框
        mask = iou < iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)
```

### 使用示例

```python
# 模拟检测结果
boxes = torch.tensor([
    [100, 100, 210, 210],
    [105, 108, 215, 218],
    [102, 105, 212, 215],
    [300, 300, 400, 400],
    [305, 302, 405, 402],
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.85, 0.8, 0.95, 0.88])

keep_indices = nms(boxes, scores, iou_threshold=0.5)
print(f"保留的框索引: {keep_indices}")
print(f"保留的框: {boxes[keep_indices]}")
```

### Soft-NMS

标准 NMS 直接删除重叠框，Soft-NMS 则降低重叠框的置信度，避免误删相邻目标。

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS：降低重叠框的置信度而非直接删除
    Args:
        boxes: shape (N, 4)
        scores: shape (N,)
        iou_threshold: 线性衰减的 IoU 阈值
        sigma: 高斯衰减的参数
        score_threshold: 分数低于此值的框被移除
    Returns:
        keep: 保留的框的索引
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        remaining = order[1:]
        iou = compute_iou(boxes[i], boxes[remaining])

        # 高斯衰减：exp(-iou^2 / sigma)
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[remaining] *= decay

        # 移除分数过低的框
        mask = scores[remaining] > score_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)
```

## 为什么 DETR 不需要 NMS

传统检测器在密集采样点上预测框，导致同一个目标产生大量重复检测，所以需要 NMS 去重。

DETR（Detection Transformer）将检测建模为集合预测（Set Prediction）问题：

1. DETR 使用一组可学习的物体查询（Object Query），每个查询负责预测一个目标
2. 通过二分图匹配（Hungarian Matching），每个查询与最多一个真值框匹配
3. 训练时用匈牙利算法找到最优的一对一匹配，天然避免重复检测

这就是 DETR 被称为"端到端"检测器的核心原因之一 —— 不需要任何手工后处理。

## 面试标准回答

NMS 是目标检测中最常用的后处理算法。核心思想很简单：按置信度排序，每次取最高的，然后删掉和它重叠度超过阈值的框。重叠度用 IoU 衡量，就是交集面积除以并集面积。

NMS 的主要问题是它是一个硬阈值操作，当两个目标距离很近时，可能会误删其中一个。Soft-NMS 通过降低重叠框的置信度来缓解这个问题。另外，NMS 的阈值选择也很关键，太大会保留过多重复框，太小会漏检。

DETR 通过集合预测的方式彻底避免了 NMS，它用匈牙利算法做一对一匹配，天然不会产生重复检测。这是端到端检测的一个重要优势。

## 高频追问

**Q: NMS 的时间复杂度是多少？**

A: 标准 NMS 是 O(N^2)，其中 N 是检测框数量。对于实时检测场景，可以用矩阵运算优化，或者用 Batched NMS 处理多类别情况。

**Q: 多类别检测时 NMS 怎么处理？**

A: 两种策略：一是按类别分别做 NMS（Class-wise NMS），不同类别的框互不影响；二是所有类别一起做 NMS（Cross-class NMS），适用于类别间有互斥关系的场景。

**Q: NMS 阈值一般怎么选？**

A: COCO 数据集常用 0.5 或 0.6。不同任务需要调整，密集场景用较大阈值避免误删，稀疏场景可以用较小阈值。

**Q: Soft-NMS 和标准 NMS 的区别？**

A: 标准 NMS 直接删除 IoU 超过阈值的框，Soft-NMS 则根据 IoU 大小降低置信度。Soft-NMS 在密集目标场景下表现更好，但计算量略大。

## 工程实践

1. PyTorch 提供了 `torchvision.ops.nms`，生产环境直接用这个，比手写快得多
2. 批量推理时用 `batched_nms` 处理多类别，避免不同类别的框互相干扰
3. 对于实时检测场景，可以在 NMS 之前先用分数阈值过滤掉低置信度框，减少 NMS 的计算量
4. 密集场景（如行人检测）可以考虑用 Soft-NMS 或 DIoU-NMS

```python
# PyTorch 内置 NMS
from torchvision.ops import nms, batched_nms

# 单类别
keep = nms(boxes, scores, iou_threshold=0.5)

# 多类别，class_ids 表示每个框的类别
keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
```

## 常见误区

1. **"NMS 阈值越大越好"** — 不是，太大会保留大量重复框
2. **"NMS 是检测必须的"** — DETR 等端到端方法不需要 NMS
3. **"Soft-NMS 一定比标准 NMS 好"** — 在稀疏场景下差异不大，密集场景才有明显优势
4. **"IoU 只用于 NMS"** — IoU 也是评估指标、损失函数（IoU Loss、GIoU Loss）的基础

## 参考资料

- Neubeck & Van Gool, "Efficient Non-Maximum Suppression", ICPR 2006
- Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of Code", ICCV 2017
- Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
