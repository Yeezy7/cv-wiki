---
title: 损失函数
description: 深度学习中常用损失函数的原理、分类与工程实践
category: cv
tags: [loss, optimization, basics]
status: stable
order: 4
---

# 损失函数

## 一句话解释

损失函数（Loss Function）是衡量模型预测值与真实值之间差距的函数，它为模型优化提供了明确的方向——模型通过最小化损失函数来学习。

## 它解决什么问题

训练神经网络的核心问题是：如何告诉模型它的预测是好是坏？损失函数就是这个"裁判"。

- **量化误差**：将"预测准不准"这个模糊概念变成一个可计算的数值
- **提供梯度**：反向传播需要损失函数对参数的梯度，损失函数的设计直接决定了梯度的质量
- **引导优化**：不同的损失函数会让模型关注不同类型的错误，从而学到不同的特征

没有损失函数，模型就没有学习的信号，就像考试没有评分标准一样。

## 核心思想

损失函数的核心思想可以归纳为：**定义"好"与"坏"的标准，然后让模型朝着"好"的方向优化**。

设计损失函数时需要考虑几个关键原则：

1. **可微性**：损失函数需要对模型参数可微，这样才能用梯度下降（Gradient Descent）优化
2. **非负性**：通常损失函数的值大于等于零，完美预测时为零
3. **任务匹配**：回归任务和分类任务需要不同的损失函数
4. **鲁棒性**：对异常值（Outlier）不过度敏感
5. **梯度质量**：梯度既不能太大（导致训练不稳定）也不能太小（导致学习停滞）

## 算法流程

损失函数在训练中的工作流程：

```
前向传播
  ↓
模型输出预测值 ŷ
  ↓
计算损失 L = Loss(ŷ, y)    ← 损失函数在这里发挥作用
  ↓
反向传播：计算 ∂L/∂W        ← 梯度来自于损失函数
  ↓
更新参数：W = W - lr × ∂L/∂W
  ↓
重复以上步骤直到收敛
```

## 数学定义

### 回归损失

#### MSE（均方误差，Mean Squared Error）

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **变量说明**：$n$ 为样本数量，$y_i$ 为第 $i$ 个样本的真实值，$\hat{y}_i$ 为预测值
- **特点**：对大误差惩罚重（平方项），对异常值敏感

#### MAE（平均绝对误差，Mean Absolute Error）

$$L_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **特点**：对异常值更鲁棒，但在零点处不可微

#### Smooth L1（Huber Loss）

$$L_{\text{SmoothL1}} = \begin{cases} 0.5(y_i - \hat{y}_i)^2 & |y_i - \hat{y}_i| < 1 \\ |y_i - \hat{y}_i| - 0.5 & |y_i - \hat{y}_i| \geq 1 \end{cases}$$

- **特点**：结合了 MSE 和 MAE 的优点，误差小时像 MSE，误差大时像 MAE

### 分类损失

#### Cross Entropy（交叉熵损失，Cross Entropy Loss）

**二分类**：

$$L_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

- **变量说明**：$y_i \in \{0, 1\}$ 为真实标签，$\hat{y}_i \in (0, 1)$ 为预测概率

**多分类**：

$$L_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

- **变量说明**：$C$ 为类别数，$y_{i,c}$ 为 one-hot 编码的真实标签，$\hat{y}_{i,c}$ 为 softmax 后的预测概率

#### Focal Loss

$$L_{\text{Focal}} = -\frac{1}{n} \sum_{i=1}^{n} \alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **变量说明**：
  - $p_t = \begin{cases} p & \text{if } y = 1 \\ 1-p & \text{if } y = 0 \end{cases}$，是模型对正确类别的预测概率
  - $\alpha_t$ 为类别权重，用于平衡正负样本
  - $\gamma$ 为聚焦参数（通常取 2），控制对易分类样本的降权程度

### 目标检测损失

#### IoU Loss（交并比损失）

$$L_{\text{IoU}} = 1 - \text{IoU}(B, B^{gt})$$

其中 IoU 的计算：

$$\text{IoU} = \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|} = \frac{\text{交集面积}}{\text{并集面积}}$$

- **变量说明**：$B$ 为预测框，$B^{gt}$ 为真实框，$|B \cap B^{gt}|$ 为交集面积，$|B \cup B^{gt}|$ 为并集面积

#### DIoU Loss（距离交并比损失）

$$L_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2}$$

- **变量说明**：
  - $b$ 和 $b^{gt}$ 分别为预测框和真实框的中心点
  - $\rho(b, b^{gt})$ 为两个中心点之间的欧氏距离
  - $c$ 为同时包含两个框的最小闭包区域的对角线长度

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 回归损失函数
# ============================================================

# 1. MSE Loss — 均方误差，回归任务默认选择
mse_loss = nn.MSELoss()

# 2. MAE / L1 Loss — 平均绝对误差，对异常值鲁棒
mae_loss = nn.L1Loss()

# 3. Smooth L1 Loss — 结合 MSE 和 MAE 的优点
# 目标检测中回归 bounding box 常用
smooth_l1_loss = nn.SmoothL1Loss()

# 测试回归损失
pred = torch.tensor([2.5, 0.0, 2.0])
target = torch.tensor([3.0, -0.5, 2.0])

print(f"MSE Loss: {mse_loss(pred, target):.4f}")       # 0.1667
print(f"MAE Loss: {mae_loss(pred, target):.4f}")       # 0.3333
print(f"Smooth L1 Loss: {smooth_l1_loss(pred, target):.4f}")  # 0.1583


# ============================================================
# 分类损失函数
# ============================================================

# 4. Cross Entropy Loss — 多分类任务的标准损失函数
# 注意：nn.CrossEntropyLoss 内部已包含 Softmax，输入应该是 logits
ce_loss = nn.CrossEntropyLoss()

logits = torch.tensor([[2.0, 1.0, 0.1],   # 样本 1 的 logits
                        [0.5, 2.5, 0.3]])  # 样本 2 的 logits
labels = torch.tensor([0, 1])               # 真实类别

print(f"Cross Entropy Loss: {ce_loss(logits, labels):.4f}")

# 5. Binary Cross Entropy — 二分类任务
# BCEWithLogitsLoss 内部包含 Sigmoid，比手动加 Sigmoid 更数值稳定
bce_loss = nn.BCEWithLogitsLoss()

logits_binary = torch.tensor([1.5, -0.5, 0.3])
labels_binary = torch.tensor([1.0, 0.0, 1.0])

print(f"BCE Loss: {bce_loss(logits_binary, labels_binary):.4f}")


# ============================================================
# Focal Loss — 解决类别不平衡问题
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss 实现，用于处理类别不平衡问题。

    在目标检测中，正负样本比例可能达到 1:1000，
    标准交叉熵会被大量简单负样本主导，Focal Loss 通过
    降低易分类样本的权重来聚焦于难分类样本。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算标准交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算 p_t
        pt = torch.exp(-ce_loss)
        # 计算 focal 权重
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        # 加权
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 测试 Focal Loss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
print(f"Focal Loss: {focal_loss(logits, labels):.4f}")


# ============================================================
# IoU Loss — 目标检测中的回归损失
# ============================================================

def iou_loss(pred_boxes, target_boxes):
    """计算 IoU Loss。

    Args:
        pred_boxes: 预测框 [x1, y1, x2, y2]，shape (N, 4)
        target_boxes: 真实框 [x1, y1, x2, y2]，shape (N, 4)

    Returns:
        IoU Loss，值越小表示框越接近
    """
    # 计算交集
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算各自面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])

    # 计算并集
    union_area = pred_area + target_area - inter_area

    # 计算 IoU
    iou = inter_area / (union_area + 1e-6)

    return 1 - iou


def diou_loss(pred_boxes, target_boxes):
    """计算 DIoU Loss。

    相比 IoU Loss，DIoU Loss 额外考虑了中心点距离，
    即使两个框没有重叠，也能提供梯度进行优化。

    Args:
        pred_boxes: 预测框 [x1, y1, x2, y2]，shape (N, 4)
        target_boxes: 真实框 [x1, y1, x2, y2]，shape (N, 4)

    Returns:
        DIoU Loss
    """
    # 计算 IoU
    iou = 1 - iou_loss(pred_boxes, target_boxes)  # 复用上面的函数

    # 计算中心点
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    # 中心点距离的平方
    center_distance = (pred_center_x - target_center_x) ** 2 + \
                      (pred_center_y - target_center_y) ** 2

    # 最小闭包区域的对角线长度平方
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    diagonal_distance = (enclose_x2 - enclose_x1) ** 2 + \
                        (enclose_y2 - enclose_y1) ** 2

    # DIoU Loss
    diou = iou - center_distance / (diagonal_distance + 1e-6)

    return 1 - diou


# 测试 IoU 和 DIoU Loss
pred_boxes = torch.tensor([[50, 50, 150, 150],
                            [30, 30, 120, 120]]).float()
target_boxes = torch.tensor([[60, 60, 160, 160],
                              [50, 50, 140, 140]]).float()

print(f"IoU Loss: {iou_loss(pred_boxes, target_boxes).mean():.4f}")
print(f"DIoU Loss: {diou_loss(pred_boxes, target_boxes).mean():.4f}")


# ============================================================
# 在训练中使用损失函数
# ============================================================

# 分类任务示例
model = nn.Linear(784, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模拟训练步骤
x = torch.randn(32, 784)    # batch_size=32
y = torch.randint(0, 10, (32,))  # 10 个类别

pred = model(x)              # 前向传播
loss = criterion(pred, y)    # 计算损失
loss.backward()              # 反向传播
optimizer.step()             # 更新参数

print(f"训练损失: {loss.item():.4f}")
```

## 损失函数分类总览

| 类型 | 损失函数 | 典型任务 | PyTorch API |
|------|---------|---------|-------------|
| 回归 | MSE | 预测连续值 | `nn.MSELoss()` |
| 回归 | MAE / L1 | 鲁棒回归 | `nn.L1Loss()` |
| 回归 | Smooth L1 | 目标检测 bbox 回归 | `nn.SmoothL1Loss()` |
| 分类 | Cross Entropy | 多分类 | `nn.CrossEntropyLoss()` |
| 分类 | BCE | 二分类 | `nn.BCEWithLogitsLoss()` |
| 分类 | Focal Loss | 类别不平衡分类 | 需自定义 |
| 检测 | IoU Loss | bbox 回归 | 需自定义 |
| 检测 | DIoU Loss | bbox 回归 | 需自定义 |
| 检测 | CIoU Loss | bbox 回归 | 需自定义 |

## 面试标准回答

**"请介绍一下常用的损失函数及其适用场景。"**

损失函数是模型优化的目标函数，根据任务类型可以分为回归损失和分类损失两大类。

回归任务中最基本的是 MSE（均方误差），它对误差取平方，对大误差惩罚重，但对异常值敏感。MAE（平均绝对误差）对异常值更鲁棒，但在零点处不可微。Smooth L1 结合了两者的优点，误差小时用平方项（梯度平滑），误差大时用线性项（对异常值鲁棒），这是目标检测中 bounding box 回归的标准损失函数。

分类任务的标准损失函数是交叉熵（Cross Entropy），它的本质是衡量两个概率分布之间的差异。二分类用 BCE（Binary Cross Entropy），多分类用 Categorical Cross Entropy。PyTorch 的 `nn.CrossEntropyLoss` 内部已经包含了 Softmax，输入应该是 logits 而不是概率。

在类别极度不平衡的场景（如目标检测中正负样本比例 1:1000），标准交叉熵会被大量简单负样本主导。Focal Loss 通过 $(1-p_t)^\gamma$ 这个调制因子，降低易分类样本的权重，让模型聚焦于难分类样本。这是 RetinaNet 论文中提出的核心贡献。

在目标检测中，bounding box 的回归损失也在不断演进。从 L1/L2 到 IoU Loss，再到 DIoU Loss 和 CIoU Loss，核心思想是让损失函数更好地反映框之间的几何关系。IoU Loss 直接优化交并比，DIoU 额外考虑了中心点距离，CIoU 进一步考虑了宽高比。

## 高频追问

**Q1：为什么分类任务用交叉熵而不用 MSE？**

从梯度的角度看，MSE 在 Sigmoid 输出接近 0 或 1 时梯度很小（因为 Sigmoid 的导数本身就小），会导致学习缓慢。交叉熵在同样的情况下梯度更大，训练更高效。从概率角度看，交叉熵对应的是最大似然估计（MLE），在分类任务中有更好的概率解释。

**Q2：Focal Loss 的 gamma 参数有什么作用？**

gamma 控制对易分类样本的降权程度。gamma=0 时退化为标准交叉熵；gamma 越大，对易分类样本的惩罚越小。论文中 gamma=2 效果最好。直觉上理解：当模型对某个样本预测很准（p_t 接近 1），$(1-p_t)^\gamma$ 接近零，这个样本对损失的贡献几乎被消除。

**Q3：IoU Loss 相比 L1/L2 Loss 有什么优势？**

L1/L2 Loss 将框的四个坐标独立看待，但实际中我们关心的是框的整体重叠程度。两个框可能 L1 距离相同，但 IoU 差异很大（比如一个完全不重叠，一个部分重叠）。IoU Loss 直接优化我们真正关心的指标，而且具有尺度不变性（scale invariant）。

**Q4：DIoU 比 IoU 好在哪里？**

IoU Loss 有两个问题：一是当两个框不重叠时 IoU 为零，梯度也为零，无法优化；二是 IoU 只考虑重叠面积，不考虑框的相对位置。DIoU 通过加入中心点距离项，即使框不重叠也能提供梯度，而且能更快地将预测框拉向目标框。

**Q5：`nn.CrossEntropyLoss` 和 `nn.BCEWithLogitsLoss` 为什么要带 Logits 后缀？**

带 Logits 后缀的版本接受未经激活的原始输出（logits），内部会自动处理 Softmax/Sigmoid。这比手动先加激活再算损失更数值稳定，因为合并后可以使用 log-sum-exp 技巧避免数值溢出。

## 工程实践

### 1. 损失函数与输出层的正确搭配

```python
# 错误：重复激活
model = nn.Sequential(
    nn.Linear(256, 10),
    nn.Softmax(dim=1),       # 手动加 Softmax
)
criterion = nn.CrossEntropyLoss()  # 内部又算了一次 Softmax

# 正确：直接用 logits
model = nn.Sequential(
    nn.Linear(256, 10),      # 直接输出 logits
)
criterion = nn.CrossEntropyLoss()
```

### 2. 类别不平衡的处理策略

```python
# 方法 1：类别权重
# 假设类别 0:1:2 的样本比例为 100:10:1
class_weights = torch.tensor([1.0, 10.0, 100.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 方法 2：Focal Loss（更推荐）
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)

# 方法 3：采样策略（配合损失函数使用）
from torch.utils.data import WeightedRandomSampler
weights = [1.0, 10.0, 100.0]  # 每个类别的采样权重
sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
```

### 3. 多任务损失的权重平衡

```python
# 目标检测中同时有分类损失和回归损失
cls_loss = F.cross_entropy(cls_pred, cls_target)
reg_loss = F.smooth_l1_loss(reg_pred, reg_target)

# 简单加权
total_loss = cls_loss + 10.0 * reg_loss

# 更高级的方法：自适应权重（Uncertainty Weighting）
# 可学习的权重参数
log_var_cls = torch.nn.Parameter(torch.zeros(1))
log_var_reg = torch.nn.Parameter(torch.zeros(1))

total_loss = (1 / (2 * torch.exp(log_var_cls))) * cls_loss + \
             (1 / (2 * torch.exp(log_var_reg))) * reg_loss + \
             0.5 * (log_var_cls + log_var_reg)
```

### 4. 损失函数的数值稳定性

```python
# 不要这样做：先 Sigmoid 再 BCE
pred_prob = torch.sigmoid(logits)
loss = F.binary_cross_entropy(pred_prob, target)  # 可能数值溢出

# 应该这样做：直接用 logits
loss = F.binary_cross_entropy_with_logits(logits, target)  # 数值稳定

# 手动实现交叉熵时要注意 log(0) 的问题
def stable_cross_entropy(logits, targets):
    # log_softmax 内部使用了 log-sum-exp 技巧，数值稳定
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_probs, targets)
    return loss
```

### 5. 标签平滑（Label Smoothing）

```python
# 标签平滑可以防止模型过度自信
# 将 one-hot 标签从 [0, 0, 1, 0] 变为 [0.025, 0.025, 0.925, 0.025]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 常见误区

### 误区 1：CrossEntropyLoss 输入前加 Softmax

**错误**：`pred = softmax(logits); loss = CrossEntropyLoss(pred, label)`

**正确**：`loss = CrossEntropyLoss(logits, label)`。`nn.CrossEntropyLoss` 内部已经包含了 Softmax，重复计算会导致梯度异常。

### 误区 2：MSE 万能论

**真相**：MSE 在分类任务中效果很差，因为 MSE 假设误差服从高斯分布，而分类任务的标签是离散的。此外，MSE 在分类任务中的梯度特性不如交叉熵。

### 误区 3：损失越小模型越好

**真相**：损失值本身没有绝对意义，不同损失函数的量级不同。更重要的是：训练损失和验证损失的差距（过拟合程度）、在实际评估指标（如 mAP、准确率）上的表现。

### 误区 4：Focal Loss 只用于目标检测

**真相**：Focal Loss 最初在 RetinaNet 中提出用于目标检测，但其思想适用于任何类别不平衡的场景，如医学图像分割、异常检测等。

### 误区 5：IoU Loss 可以直接替代 L1 Loss

**真相**：IoU Loss 在框完全不重叠时梯度为零，无法优化。在实际使用中，通常需要配合其他损失（如 L1 Loss）或使用改进版本（DIoU、CIoU、SIoU）。

## 参考资料

- Lin, T. Y., et al. (2017). Focal loss for dense object detection. ICCV.
- Rezatofighi, H., et al. (2019). Generalized intersection over union: A metric and a loss for bounding box regression. CVPR.
- Zheng, Z., et al. (2020). Distance-IoU loss: Faster and better learning for bounding box regression. AAAI.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
