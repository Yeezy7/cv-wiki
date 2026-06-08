---
title: YOLO 系列
description: 从 YOLOv1 到 YOLOv11，一文搞懂 YOLO 家族的核心思想与演进脉络
category: cv
tags: [yolo, object-detection]
status: stable
order: 2
---

# YOLO 系列

## 一句话解释

YOLO（You Only Look Once）是一系列将目标检测（Object Detection）转化为单次回归问题的算法，通过一个端到端的神经网络同时预测所有目标的位置和类别，速度极快。

## 它解决什么问题

在 YOLO 出现之前，主流目标检测方法如 R-CNN 系列采用"两阶段"策略：先生成候选区域（Region Proposals），再对每个区域分类和回归。这类方法精度不错，但速度很慢（每张图需要数秒），无法满足实时检测的需求。

YOLO 的核心贡献是：**把检测问题变成一个回归问题**，只需一次前向传播就能得到所有目标的边界框（Bounding Box）和类别概率，从根本上解决了速度瓶颈。

## 核心思想

### 网格划分（Grid Cell）

YOLOv1 将输入图像划分为 $S \times S$ 的网格。如果某个目标的中心落入某个网格单元（Grid Cell），该单元就负责预测这个目标。

每个网格单元预测：
- **B 个边界框**：每个框包含 5 个值 $(x, y, w, h, confidence)$
  - $(x, y)$：框中心相对于该网格单元的偏移
  - $(w, h)$：框的宽高（相对于整张图归一化）
  - $confidence$：置信度 = $P(Object) \times IoU$，表示"这里有没有目标"以及"框得准不准"
- **C 个类别概率**：该网格单元属于各类别的概率 $P(Class_i | Object)$

最终输出张量形状为 $S \times S \times (B \times 5 + C)$。

### 为什么 YOLO 快

1. **单次检测**：不需要生成候选区域，一次前向传播完成所有预测
2. **全卷积结构**：整个网络可以高效并行计算
3. **看到全局信息**：每个网格单元都能看到整张图，背景误检（Background False Positives）比两阶段方法少一半

## 算法流程

### YOLOv1（2016）

1. 将图像缩放到 $448 \times 448$
2. 通过 24 层卷积 + 2 层全连接提取特征
3. 输出 $7 \times 7 \times 30$ 的张量（$S=7, B=2, C=20$，针对 VOC 数据集）
4. 通过非极大值抑制（Non-Maximum Suppression, NMS）过滤冗余框

**局限**：每个网格只预测 2 个框，对密集小目标和长宽比异常的目标检测效果差。

### YOLOv2 / YOLO9000（2017）

关键改进：
- **Batch Normalization**：每个卷积层后加 BN，收敛更快，mAP 提升 2%
- **高分辨率分类器**：先在 $448 \times 448$ 上微调分类网络，再做检测
- **引入锚框（Anchor Boxes）**：借鉴 Faster R-CNN 的 Anchor 机制，每个网格预设 5 个锚框，召回率大幅提升
- **维度聚类（Dimension Clusters）**：用 K-Means 对训练集的框尺寸聚类，得到更好的锚框先验
- **多尺度训练（Multi-Scale Training）**：训练时随机改变输入尺寸（320~608），增强鲁棒性
- **Darknet-19 骨干网络**：19 层卷积 + 5 层最大池化，比 VGG 更快

### YOLOv3（2018）

关键改进：
- **多尺度预测（Multi-Scale Prediction）**：在 3 个不同尺度上做检测（类似特征金字塔网络 FPN），分别检测大、中、小目标
- **Darknet-53 骨干网络**：引入残差连接（Residual Connection），53 层卷积，特征提取能力更强
- **逻辑回归替代 Softmax**：每个类别独立做二分类，支持多标签（Multi-Label）分类

### YOLOv4（2020）

系统性地整合了大量训练技巧：
- **骨干网络 CSPDarknet53**：跨阶段局部网络（Cross Stage Partial Network），减少计算量同时保持精度
- **SPP（Spatial Pyramid Pooling）**：多尺度池化，增大感受野
- **PANet（Path Aggregation Network）**：双向特征融合，比 FPN 多一条自底向上的路径
- **训练技巧**：Mosaic 数据增强、CutMix、DropBlock 正则化、CIoU Loss

### YOLOv5（2020，Ultralytics）

并非学术论文，而是 Ultralytics 公司的工程化产品：
- **PyTorch 原生实现**，部署极其方便
- **自适应锚框计算**：根据数据集自动聚类锚框尺寸
- **Focus 结构**：对输入做切片操作，减少计算量
- **工程化优势**：自带数据增强、超参搜索、模型导出（ONNX/TensorRT/CoreML）

### YOLOv8（2023，Ultralytics）

标志性变化：**Anchor-Free**（无锚框）。
- **解耦头（Decoupled Head）**：分类和回归分支完全分离，各自独立优化
- **无锚框设计**：直接预测目标中心点偏移和宽高，不再依赖预设锚框
- **C2f 模块**：改进的 CSP 结构，梯度流动更顺畅
- **Task-Aligned Assigner**：动态标签分配策略，同时考虑分类得分和 IoU 质量
- **DFL Loss**：分布焦点损失（Distribution Focal Loss），对框的回归更精确

### YOLO11（2024，Ultralytics）

最新一代，重点优化效率和精度的平衡：
- **C3k2 模块**：更轻量的跨阶段局部模块，用两个小卷积核替代大卷积核
- **C2PSA（Cross Stage Partial with Spatial Attention）**：引入空间注意力，聚焦重要特征
- **大核深度可分离卷积**：在颈部网络中使用大核可分离卷积，扩大感受野同时控制计算量
- **SPPF 优化**：更高效的空间金字塔池化
- **精度进一步提升**：在 COCO 数据集上达到新的 SOTA（State of the Art）

## 数学定义

### 交并比（Intersection over Union, IoU）

$$IoU = \frac{|A \cap B|}{|A \cup B|} = \frac{交集面积}{并集面积}$$

- $A$：预测框
- $B$：真实框（Ground Truth）
- IoU 越接近 1 表示框越准

### 置信度（Confidence）

$$Confidence = P(Object) \times IoU_{pred}^{truth}$$

- $P(Object)$：该网格/锚框中是否包含目标
- $IoU_{pred}^{truth}$：预测框与真实框的 IoU

### YOLOv1 损失函数

$$\mathcal{L} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$
$$+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

变量说明：
- $\mathbb{1}_{ij}^{obj}$：第 $i$ 个网格的第 $j$ 个锚框是否负责预测目标（0 或 1）
- $(x, y)$：框中心坐标
- $(w, h)$：框的宽高，取平方根是为了小框的误差更敏感
- $C$：置信度
- $p(c)$：类别概率
- $\lambda_{coord} = 5$：坐标损失的权重，强调定位精度
- $\lambda_{noobj} = 0.5$：无目标置信度损失的权重，降低负样本影响

### CIoU Loss（YOLOv4+ 使用）

$$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

- $\rho(b, b^{gt})$：预测框中心与真实框中心的欧氏距离
- $c$：最小外接框的对角线长度
- $v = \frac{4}{\pi^2} (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h})^2$：衡量长宽比的一致性
- $\alpha = \frac{v}{(1 - IoU) + v}$：平衡因子

## 代码示例

### 使用 Ultralytics 进行推理

```python
# pip install ultralytics
from ultralytics import YOLO

# 加载预训练模型（自动下载权重）
model = YOLO("yolo11n.pt")  # YOLO11 Nano 版本，速度最快

# 推理
results = model("street.jpg")

# 解析结果
for result in results:
    boxes = result.boxes          # 边界框
    for box in boxes:
        xyxy = box.xyxy[0]        # [x1, y1, x2, y2] 格式
        conf = box.conf[0]        # 置信度
        cls = box.cls[0]          # 类别索引
        name = model.names[int(cls)]  # 类别名称
        print(f"{name}: {conf:.2f} at {xyxy.tolist()}")

    # 可视化保存
    result.save(filename="result.jpg")
```

### 使用 Ultralytics 进行训练

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 在自定义数据集上微调
# 数据集需要按 YOLO 格式组织（images/ + labels/，txt 标注文件）
results = model.train(
    data="coco128.yaml",   # 数据集配置文件
    epochs=100,            # 训练轮数
    imgsz=640,             # 输入图像尺寸
    batch=16,              # 批大小
    device="mps",          # Mac 上用 MPS，CUDA 设备用 "0" 或 "0,1"
)
```

### YOLO 输出格式说明

```python
import torch

# YOLO 模型输出的原始张量形状示例（以 YOLOv8 为例）
# 对于 640x640 输入，80 个类别：
# output shape: [batch, 84, 8400]
# 84 = 4 (xywh) + 80 (class scores)
# 8400 = 80x80 + 40x40 + 20x20 三个尺度的预测点总数

output = torch.randn(1, 84, 8400)  # 模拟输出
boxes = output[:, :4, :]     # 前 4 维是边界框
scores = output[:, 4:, :]    # 后 80 维是类别得分

# ultralytics 内部会自动做 NMS 和后处理，用户直接用 results 对象即可
```

## 面试标准回答

**第一段（概述）**：YOLO 系列是单阶段目标检测的经典代表，核心思想是将目标检测转化为回归问题，通过一次前向传播同时预测所有目标的位置和类别。相比两阶段方法如 Faster R-CNN，YOLO 的速度优势非常明显，可以做到实时检测。

**第二段（演进）**：YOLO 从 v1 到 v11 经历了几个重要阶段。早期版本（v1-v3）主要在骨干网络和多尺度检测上做改进，引入了锚框机制和特征金字塔。中期（v4-v5）侧重于训练技巧和工程化，Mosaic 增强、CSP 结构等大幅提升了精度。近期（v8、v11）则回归简洁，采用 Anchor-Free 设计和解耦头，去掉锚框后模型更简洁、泛化性更好。

**第三段（关键优势）**：YOLO 的速度快主要因为两点：一是单阶段检测，不需要生成候选区域；二是全卷积结构可以高效并行。另外 YOLO 看到的是整张图的全局信息，相比滑动窗口或区域提议方法，背景误检更少。在工程部署上，YOLO 生态非常成熟，Ultralytics 提供了从训练到导出的一站式工具链。

## 高频追问

**Q1：YOLO 为什么比 Faster R-CNN 快？**
A：Faster R-CNN 是两阶段：先用 RPN 生成约 300 个候选区域，再逐个分类和回归。YOLO 直接在特征图上做一次回归，省去了候选区域生成和逐区域处理的开销。

**Q2：Anchor-Free 相比 Anchor-Based 有什么优势？**
A：Anchor-Free 去掉了预设锚框，好处有三：一是减少了超参数（锚框数量、尺寸、比例），模型更简洁；二是避免了正负样本分配不均衡的问题；三是对新数据集不需要重新聚类锚框，泛化性更好。

**Q3：YOLO 的 NMS 是做什么的？**
A：非极大值抑制用于过滤冗余检测框。同一目标可能被多个网格或锚框检测到，NMS 会保留置信度最高的框，然后抑制（删除）与它 IoU 超过阈值的其他框。YOLOv8 后也出现了无 NMS 的版本（如 RT-DETR），通过其他机制避免冗余。

**Q4：YOLOv3 的多尺度检测是怎么做的？**
A：YOLOv3 在三个不同尺度的特征图上做检测：$13 \times 13$（检测大目标）、$26 \times 26$（检测中目标）、$52 \times 52$（检测小目标）。小尺度特征图分辨率低但感受野大，适合检测大目标；大尺度特征图分辨率高但感受野小，适合检测小目标。

**Q5：如何选择 YOLO 的不同版本（n/s/m/l/x）？**
A：n（Nano）适合边缘设备和实时场景，速度最快但精度最低；s（Small）是速度和精度的平衡点；m（Medium）适合大多数场景；l（Large）和 x（Extra Large）精度最高，适合对精度要求高的场景，但需要更强的算力。

## 工程实践

### 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 边缘设备（Jetson、树莓派） | YOLOv8n / YOLO11n | 体积小，推理快 |
| 实时视频流 | YOLOv8s / YOLO11s | 速度与精度平衡 |
| 通用检测任务 | YOLOv8m / YOLO11m | 综合表现最优 |
| 精度优先 | YOLOv8x / YOLO11x | 牺牲速度换精度 |

### 数据集准备

YOLO 标注格式（每个图像对应一个 txt 文件）：

```
# class_id  x_center  y_center  width  height（全部归一化到 0~1）
0 0.512 0.483 0.215 0.392
1 0.731 0.295 0.108 0.156
```

### 常用优化手段

1. **数据增强**：Mosaic、MixUp、Copy-Paste，对小目标和遮挡目标效果显著
2. **预训练权重**：始终从官方预训练权重微调，不要从零训练
3. **输入尺寸**：小目标多时用更大的输入尺寸（1280），但计算量会增大
4. **TensorRT 加速**：部署时用 TensorRT（NVIDIA）或 CoreML（Apple）加速，速度可提升 2-5 倍

### 推理速度参考（RTX 3080, 640x640）

| 模型 | 参数量 | FPS | mAP (COCO) |
|------|--------|-----|------------|
| YOLO11n | 2.6M | ~500 | 39.5 |
| YOLO11s | 9.4M | ~350 | 47.0 |
| YOLO11m | 20.1M | ~200 | 51.5 |
| YOLO11l | 25.3M | ~150 | 53.4 |
| YOLO11x | 56.9M | ~80 | 54.7 |

*数据为近似值，实际速度因硬件和环境而异。*

## 常见误区

**误区 1："YOLO 精度不如两阶段方法"**
早期版本确实如此，但 YOLOv8/v11 在 COCO 上的精度已经接近甚至超过一些两阶段方法。精度差距已经不是选择 YOLO 的障碍。

**误区 2："YOLO 只能做目标检测"**
YOLO 的架构可以扩展到多种任务：实例分割（YOLO-Seg）、姿态估计（YOLO-Pose）、分类（YOLO-cls）、旋转目标检测（OBB）等。Ultralytics 的 YOLOv8/v11 已经内置了这些任务。

**误区 3："Anchor-Free 就是不用任何先验"**
YOLOv8 虽然是 Anchor-Free，但它仍然在特征图的每个位置预测一个"默认点"，然后通过偏移和宽高来确定框。这本质上是一种隐式的先验，只是不再需要手动设定锚框尺寸。

**误区 4："输入越大效果越好"**
增大输入尺寸确实能提升小目标检测效果，但会显著增加计算量和显存占用。需要根据实际场景权衡。通常 $640 \times 640$ 是默认选择，小目标密集场景用 $1280 \times 1280$。

**误区 5："NMS 阈值越低越好"**
NMS 阈值过低会误删正确检测（漏检），过高会保留冗余框（误检）。一般 IoU 阈值设在 0.5~0.7 之间，需要根据具体场景调优。

## 参考资料

- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLOv1, 2016)
- Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger" (YOLOv2, 2017)
- Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018)
- Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection" (2020)
- Jocher et al., "ultralytics/yolov5" (GitHub, 2020)
- Jocher et al., "ultralytics/ultralytics" (GitHub, 2023)
- Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024)
- Ultralytics 官方文档：https://docs.ultralytics.com
