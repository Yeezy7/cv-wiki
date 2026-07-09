---
title: ResNet
description: 残差网络入门
category: cv
tags:
    - resnet
status: review
order: 2
---

## 简介

ResNet，全称 **Residual Network，残差网络**，是 CNN 中非常重要的经典结构。

![ResNet 残差学习](/images/multimodal/resnet/resnet_architecture.png)

*ResNet 核心思想：残差学习框架，让网络学习残差映射而非完整映射（来源：ResNet 论文 Figure 1）*

它的核心贡献是提出了 **残差连接**，解决了深层神经网络难以训练的问题。ResNet 之后，网络可以稳定地堆到几十层甚至上百层，因此被大量用于图像分类、目标检测、图像分割和多模态视觉编码器中。

---

## 为什么需要 ResNet

在 ResNet 之前，直接加深 CNN 层数并不一定带来更好效果。

深层网络常见问题：

1. 梯度传播困难；
    
2. 训练误差反而升高；
    
3. 网络难以学习恒等映射；
    
4. 参数更多但优化更困难。
    

注意这里主要不是普通的过拟合问题，而是 **退化问题**：

> 网络变深后，理论表达能力更强，但实际训练误差反而更高。

ResNet 的目标就是让深层网络更容易优化。

---

## 核心思想：残差连接

![ResNet Shortcut Connections](/images/multimodal/resnet/resnet_shortcut.png)

*ResNet Shortcut Connections：残差连接的两种形式（来源：ResNet 论文 Figure 2）*

普通网络学习的是：

```text
H(x)
```

ResNet 不直接学习完整映射，而是学习残差：

```text
F(x) = H(x) - x
```

因此输出变成：

```text
H(x) = F(x) + x
```

也就是：

```text
y = F(x) + x
```

其中：

- `x` 是输入；
    
- `F(x)` 是卷积层学习到的残差；
    
- `+ x` 是跳跃连接，也叫 shortcut connection。
    

直观理解：

> ResNet 不要求网络直接学出目标结果，而是学习“在输入基础上需要修改什么”。

如果某些层暂时没学到有用信息，网络至少可以通过 shortcut 保留原始输入，使深层网络更容易训练。

---

## 残差块结构

ResNet 的基本单元是 **Residual Block**。

典型结构：

```text
输入 x
  │
  ├─────────────── shortcut ───────────────┐
  │                                        │
  ↓                                        │
Conv → BN → ReLU → Conv → BN               │
  │                                        │
  └──────────────── + x ←──────────────────┘
                  ↓
                 ReLU
```

可以简化理解为：

```text
输出 = 卷积分支输出 + 原始输入
```

---

## BasicBlock 和 Bottleneck

ResNet 常见两种残差块。

![ResNet Bottleneck Block](/images/multimodal/resnet/resnet_bottleneck.png)

*ResNet 残差块对比：BasicBlock（左）和 Bottleneck Block（右）（来源：ResNet 论文 Figure 5）*

### 1. BasicBlock

BasicBlock 通常用于较浅的 ResNet，例如 ResNet-18、ResNet-34。

结构：

```text
3×3 Conv
3×3 Conv
```

特点是结构简单，计算量适中。

---

### 2. Bottleneck

Bottleneck 通常用于更深的 ResNet，例如 ResNet-50、ResNet-101、ResNet-152。

结构：

```text
1×1 Conv  降维
3×3 Conv  提取特征
1×1 Conv  升维
```

其中：

|卷积|作用|
|---|---|
|`1×1` 降维|减少通道数，降低计算量|
|`3×3` 卷积|提取空间特征|
|`1×1` 升维|恢复通道数|

Bottleneck 的核心是：在加深网络的同时控制计算量。

---

## 维度匹配问题

残差连接要求两条分支的输出形状一致，才能相加。

也就是说：

```text
F(x) 和 x 的 shape 必须相同
```

如果通道数或特征图尺寸不同，就不能直接相加。

常见解决方法是使用 `1×1` 卷积调整 shortcut 分支：

```text
shortcut = 1×1 Conv(x)
```

这种操作通常出现在：

1. 通道数变化时；
    
2. stride 不为 1，下采样时；
    
3. stage 切换时。
    

---

## ResNet 网络结构

![ResNet 网络架构](/images/multimodal/resnet/resnet_architecture.png)

*ResNet 网络架构对比：VGG-19、Plain Network、Residual Network（来源：ResNet 论文 Figure 3）*

以 ResNet-50 为例，整体结构可以概括为：

```text
输入图像
→ Conv + BN + ReLU
→ MaxPool
→ Stage 1
→ Stage 2
→ Stage 3
→ Stage 4
→ Global Average Pooling
→ Fully Connected
→ 分类结果
```

其中每个 Stage 由多个残差块组成。

不同 ResNet 的主要区别在于残差块数量不同：

|模型|常用块|深度|
|---|---|---|
|ResNet-18|BasicBlock|18 层|
|ResNet-34|BasicBlock|34 层|
|ResNet-50|Bottleneck|50 层|
|ResNet-101|Bottleneck|101 层|
|ResNet-152|Bottleneck|152 层|

---

## PyTorch 示例：简化残差块

下面是一个简化版 BasicBlock。

```python
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


if __name__ == "__main__":
    block = BasicBlock(64, 128, stride=2)
    x = torch.randn(4, 64, 32, 32)
    y = block(x)

    print(y.shape)  # torch.Size([4, 128, 16, 16])
```

这里 `stride=2`，所以特征图尺寸从：

```text
32×32 → 16×16
```

同时通道数从：

```text
64 → 128
```

因此 shortcut 分支需要用 `1×1` 卷积进行维度匹配。

---

## 工程注意点

1. ResNet 常作为视觉 backbone，用于分类、检测、分割等任务。
    
2. 目标检测中常用 ResNet + FPN 提取多尺度特征。
    
3. 小 batch 训练时，BatchNorm 可能不稳定，可以考虑冻结 BN 或使用 GroupNorm。
    
4. 迁移学习时，常加载 ImageNet 预训练权重。
    
5. ResNet 越深不一定越适合部署，实际要考虑速度、显存和延迟。
    

---

## 常见误区

### 误区一：ResNet 只是把输入加到输出上

残差连接不只是简单相加，它改变了网络的优化方式，使深层网络更容易学习有效映射。

### 误区二：ResNet 解决的是过拟合

ResNet 主要解决的是深层网络的退化问题和优化困难，不是单纯的过拟合问题。

### 误区三：网络越深越好

更深的 ResNet 表达能力更强，但计算量更大，部署成本更高。在实际任务中，ResNet-50 经常比 ResNet-101 更实用。

---

## 面试问题

### Q1：ResNet 解决了什么问题？

ResNet 主要解决深层网络训练困难和退化问题。网络加深后，训练误差可能反而升高。ResNet 通过残差连接让网络学习残差映射，使深层网络更容易优化。

---

### Q2：什么是残差连接？

残差连接是指将输入 `x` 直接加到卷积分支输出 `F(x)` 上：

```text
y = F(x) + x
```

这样网络不需要直接学习完整映射，而是学习输入和目标之间的差异。

---

### Q3：为什么残差连接能缓解深层网络训练困难？

因为 shortcut 提供了一条更直接的信息和梯度传播路径。即使中间卷积层暂时学不到有效特征，输入信息也可以通过 shortcut 传递到后面，使网络更容易优化。

---

### Q4：BasicBlock 和 Bottleneck 有什么区别？

BasicBlock 通常由两个 `3×3` 卷积组成，用于 ResNet-18 和 ResNet-34。

Bottleneck 由 `1×1`、`3×3`、`1×1` 三个卷积组成，先降维、再提取特征、最后升维，常用于 ResNet-50 及更深网络。

---

### Q5：shortcut 分支什么时候需要 `1×1` 卷积？

当输入和输出 shape 不一致时，需要用 `1×1` 卷积调整 shortcut 分支。

常见情况包括：

1. 通道数变化；
    
2. 特征图尺寸变化；
    
3. stride 不为 1；
    
4. stage 切换。
    

---

### Q6：ResNet 为什么常用于检测和分割？

因为 ResNet 可以作为稳定的特征提取网络，提取不同层次的视觉特征。结合 FPN 后，可以进一步获得多尺度特征，适合目标检测和图像分割任务。

---

## 小结

ResNet 的重点不是网络有多深，而是 **残差连接**。

需要掌握：

```text
退化问题
残差映射
shortcut connection
y = F(x) + x
BasicBlock
Bottleneck
1×1 卷积维度匹配
ResNet 作为视觉 backbone
```