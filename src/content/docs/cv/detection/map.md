---
title: 平均精度均值（mAP）
description: 目标检测最核心的评估指标，理解 AP 和 mAP 的计算方式
category: cv
tags: [map, object-detection, metrics]
status: stable
order: 4
---

# 平均精度均值（mAP）

## 一句话解释

平均精度均值（mean Average Precision, mAP）是目标检测中最常用的评估指标，它是所有类别平均精度（Average Precision, AP）的平均值，综合衡量模型的定位和分类能力。

## 它解决什么问题

单独看 Precision（精确率）或 Recall（召回率）都不全面。Precision 高可能是因为只报了最有把握的框，漏检了很多；Recall 高可能是框报了太多，误检也多。AP 通过 PR 曲线下面积综合考虑两者，mAP 再对所有类别取平均，给出一个整体评估。

## 核心思想

1. 对每个类别，画出 Precision-Recall 曲线
2. 计算曲线下面积，得到该类别的 AP
3. 对所有类别的 AP 取平均，得到 mAP

## 数学定义

### 基础概念

先理解四个基础量：

- **真正例（True Positive, TP）**：预测正确，检测框与真值框的 IoU >= 阈值
- **假正例（False Positive, FP）**：预测错误，检测框没有匹配到真值框，或 IoU < 阈值
- **真反例（True Negative, TN）**：正确地没有检测（目标检测中一般不直接使用）
- **假反例（False Negative, FN）**：漏检，真值框没有被任何检测框匹配

### Precision 和 Recall

$$ 
Precision = \frac{TP}{TP + FP} = \frac{\text{正确检测数}}{\text{所有检测数}}
$$

$$ 
Recall = \frac{TP}{TP + FN} = \frac{正确检测数}{\text{所有真值框数}}
$$

其中：
- $TP$ 是 IoU 达到阈值的检测框数量
- $FP$ 是未匹配到真值框的检测框数量
- $FN$ 是未被检测到的真值框数量

### PR 曲线

将检测结果按置信度从高到低排序，依次将每个检测结果加入，计算当前的 Precision 和 Recall，就得到 PR 曲线上的一个点。

### AP 计算

**11 点插值法（PASCAL VOC 2007）**

在 Recall = 0, 0.1, 0.2, ..., 1.0 这 11 个点上取 Precision 的最大值，然后求平均：

$$ 
AP = \frac{1}{11} \sum_{r \in \{0, 0.1, ..., 1.0\}} \max_{\tilde{r} \geq r} p(\tilde{r})
$$

**所有点插值法（PASCAL VOC 2010+ / COCO）**

在每个 Recall 变化点上取 Precision 的最大值，然后计算面积：

$$ 
AP = \sum_{n} (r_{n+1} - r_n) \cdot p_{\text{interp}}(r_{n+1})
$$

其中 $p_{\text{interp}}(r) = \max_{\tilde{r} \geq r} p(\tilde{r})$，即在 Recall 大于等于 $r$ 的所有点中取最大的 Precision。

### mAP

$$ 
mAP = \frac{1}{C} \sum_{c=1}^{C} AP_c
$$

其中：
- $C$ 是类别总数
- $AP_c$ 是第 $c$ 个类别的 AP

### COCO 评估指标

COCO 数据集使用多种 IoU 阈值来评估：

| 指标 | 含义 |
|------|------|
| AP@0.5 | IoU 阈值为 0.5 时的 AP（即 PASCAL VOC 标准） |
| AP@0.75 | IoU 阈值为 0.75 时的 AP（更严格的定位要求） |
| AP@[0.5:0.95] | IoU 阈值从 0.5 到 0.95，步长 0.05，取 10 个阈值的平均 |
| AP_small | 面积 < 32x32 的小目标的 AP |
| AP_medium | 面积在 32x32 到 96x96 之间的中等目标的 AP |
| AP_large | 面积 > 96x96 的大目标的 AP |

其中 AP@[0.5:0.95] 是 COCO 的主要排名指标，比单一 IoU 阈值更全面。

## 代码示例

### 计算 AP 的 Python 实现

```python
import numpy as np

def compute_ap(recalls, precisions, method='voc2010'):
    """
    计算 Average Precision
    Args:
        recalls: Recall 值数组，已按置信度排序
        precisions: Precision 值数组
        method: 'voc2007' 使用 11 点插值，'voc2010' 使用所有点插值
    Returns:
        ap: Average Precision
    """
    if method == 'voc2007':
        # 11 点插值法
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            precisions_at_recall = precisions[recalls >= t]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall)
        ap /= 11.0
    else:
        # 所有点插值法（VOC 2010+ / COCO）
        # 在每个 recall 变化点取 precision 的最大值
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # 从右往左取最大值（确保 precision 单调递减）
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # 找到 recall 变化的点
        recall_changes = np.where(recalls[1:] != recalls[:-1])[0]

        # 计算面积
        ap = np.sum(
            (recalls[recall_changes + 1] - recalls[recall_changes]) *
            precisions[recall_changes + 1]
        )

    return ap


def compute_detection_ap(detections, ground_truths, iou_threshold=0.5):
    """
    计算单个类别的 AP
    Args:
        detections: list of dict，每个 dict 包含 'bbox', 'score', 'image_id'
        ground_truths: list of dict，每个 dict 包含 'bbox', 'image_id'
        iou_threshold: IoU 阈值
    Returns:
        ap: Average Precision
        precisions: Precision 数组
        recalls: Recall 数组
    """
    # 按置信度降序排序
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    # 统计每张图片的真值框数量
    num_gt_per_image = {}
    for gt in ground_truths:
        img_id = gt['image_id']
        num_gt_per_image[img_id] = num_gt_per_image.get(img_id, 0) + 1
    total_gt = len(ground_truths)

    # 标记已匹配的真值框
    matched = set()

    tp_list = []
    fp_list = []

    for det in detections:
        img_id = det['image_id']
        best_iou = 0.0
        best_gt_idx = -1

        # 找到与该检测框 IoU 最大的真值框
        for idx, gt in enumerate(ground_truths):
            if gt['image_id'] != img_id:
                continue
            if idx in matched:
                continue
            iou = compute_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # 判断 TP 或 FP
        if best_iou >= iou_threshold and best_gt_idx not in matched:
            tp_list.append(1)
            fp_list.append(0)
            matched.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # 累积 TP 和 FP
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    # 计算 Precision 和 Recall
    recalls = tp_cumsum / (total_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 计算 AP
    ap = compute_ap(recalls, precisions, method='voc2010')

    return ap, precisions, recalls


def compute_iou(box1, box2):
    """
    计算两个框的 IoU
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)
```

### 使用示例

```python
# 模拟检测结果
detections = [
    {'bbox': [100, 100, 200, 200], 'score': 0.9, 'image_id': 1},
    {'bbox': [105, 108, 205, 208], 'score': 0.8, 'image_id': 1},
    {'bbox': [300, 300, 400, 400], 'score': 0.7, 'image_id': 1},
    {'bbox': [50, 50, 150, 150], 'score': 0.6, 'image_id': 2},
]

ground_truths = [
    {'bbox': [100, 100, 200, 200], 'image_id': 1},
    {'bbox': [300, 300, 400, 400], 'image_id': 1},
    {'bbox': [50, 50, 150, 150], 'image_id': 2},
]

ap, precisions, recalls = compute_detection_ap(
    detections, ground_truths, iou_threshold=0.5
)
print(f"AP@0.5 = {ap:.4f}")
print(f"Precision: {precisions}")
print(f"Recall: {recalls}")
```

### 计算 mAP

```python
def compute_map(all_detections, all_ground_truths, categories, iou_threshold=0.5):
    """
    计算所有类别的 mAP
    Args:
        all_detections: dict，key 为类别，value 为检测结果列表
        all_ground_truths: dict，key 为类别，value 为真值框列表
        categories: 类别列表
        iou_threshold: IoU 阈值
    Returns:
        mAP: mean Average Precision
        ap_per_class: 每个类别的 AP
    """
    ap_per_class = {}
    for cat in categories:
        dets = all_detections.get(cat, [])
        gts = all_ground_truths.get(cat, [])
        if len(gts) == 0:
            continue
        ap, _, _ = compute_detection_ap(dets, gts, iou_threshold)
        ap_per_class[cat] = ap

    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class
```

## 面试标准回答

mAP 是目标检测最核心的评估指标。它的计算分三步：首先，对每个类别，把检测结果按置信度排序，逐个计算 Precision 和 Recall，画出 PR 曲线；然后，计算 PR 曲线下的面积作为该类别的 AP；最后，对所有类别的 AP 取平均，得到 mAP。

AP 的计算有两种主流方式：PASCAL VOC 的 11 点插值法在 11 个固定的 Recall 点上取最大 Precision 求平均；COCO 的方式是在每个 Recall 变化点上取最大 Precision，然后计算曲线下面积。

COCO 数据集还会在多个 IoU 阈值下计算 AP，主要排名指标是 AP@[0.5:0.95]，也就是 IoU 从 0.5 到 0.95 步长 0.05 的 10 个阈值的平均。IoU 阈值越高，对定位精度的要求越严格。

## 高频追问

**Q: 为什么 mAP 比单纯的 Precision 或 Recall 更好？**

A: Precision 只看预测准不准，可能漏检很多；Recall 只看检出率，可能误检很多。mAP 通过 PR 曲线综合考虑两者，而且是在不同置信度阈值下的整体表现，更全面。

**Q: IoU 阈值对 mAP 有什么影响？**

A: IoU 阈值越高，要求预测框和真值框重叠度越大，mAP 通常会下降。AP@0.5 到 AP@0.75 可能会下降很多，说明很多检测框的定位精度不够高。

**Q: COCO 的 AP@[0.5:0.95] 为什么比 AP@0.5 更公平？**

A: AP@0.5 对定位精度要求较粗，框稍微偏一点也能算对。AP@[0.5:0.95] 综合了 10 个不同的 IoU 阈值，对定位精度的评估更全面、更严格。

**Q: 如何提升 mAP？**

A: 几个方向：提升数据质量和数量、改进网络结构（如 FPN 多尺度检测）、使用更好的损失函数（如 CIoU Loss）、优化后处理（如 Soft-NMS）、使用更强的数据增强。

## 工程实践

1. 评估时用 COCO 官方的 `pycocotools`，不要自己写，避免计算差异
2. 训练时可以关注 mAP 的变化趋势来判断模型是否收敛
3. 如果 AP@0.5 高但 AP@0.75 低，说明定位精度不够，可以优化回归损失
4. 如果某个类别的 AP 特别低，检查该类别的数据量和标注质量

```python
# 使用 pycocotools 评估
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt = COCO('annotations.json')
coco_dt = coco_gt.loadRes('detections.json')

coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
```

## 常见误区

1. **"mAP 高就是好模型"** — 还要看推理速度、模型大小、部署难度
2. **"AP 和 mAP 是一回事"** — AP 是单个类别的，mAP 是所有类别的平均
3. **"不同论文的 mAP 可以直接比较"** — 不同数据集、不同 IoU 阈值、不同 AP 计算方式都会影响结果
4. **"IoU 阈值 0.5 是通用标准"** — 这只是 PASCAL VOC 的标准，COCO 用的是 0.5:0.95 的平均

## 参考资料

- Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge", IJCV 2010
- Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
- COCO Evaluation: https://cocodataset.org/#detection-eval
