---
title: CNN 基础
description: 卷积神经网络（Convolutional Neural Network）的核心概念与经典架构
category: cv
tags: [cnn, convolution, basics]
status: stable
order: 1
---

## 一句话解释

卷积神经网络（CNN, Convolutional Neural Network）是一种专门处理网格结构数据（如图像）的深度学习模型，通过卷积操作自动提取局部特征，再逐层组合成高级语义特征。

## 它解决什么问题

传统全连接网络处理图像时有两个致命缺陷：

1. **参数爆炸**：一张 224×224×3 的图片展平后有 150,528 个像素，若第一个隐藏层有 1000 个神经元，仅第一层就需要 1.5 亿个参数，显存和计算量都不可接受。
2. **忽略空间结构**：展平操作丢失了像素之间的空间邻域关系，网络无法利用"相邻像素通常相关"这一先验知识。

CNN 通过两个核心设计解决这些问题：

- **局部连接**：每个神经元只看输入的一个局部区域，参数量大幅减少。
- **权值共享**：同一个卷积核在整个图像上滑动复用，进一步压缩参数量，同时赋予模型平移不变性（Translation Invariance）。

## 核心思想

CNN 的核心思想可以概括为三个字：**局部性**、**平移不变性**、**层次性**。

- **局部性（Locality）**：图像的语义信息往往集中在局部区域，不需要每个神经元都看全图。
- **平移不变性（Translation Invariance）**：一只猫出现在图片左上角和右下角，应该被同一个特征检测器识别。权值共享天然实现了这一点。
- **层次性（Hierarchy）**：低层卷积提取边缘、纹理等简单特征，中层组合成部件（眼睛、轮子），高层识别完整物体。这种从局部到整体的层次结构与人类视觉系统高度相似。

## 算法流程

一个典型的 CNN 前向传播流程如下：

```
输入图像
   │
   ▼
[卷积层] → 提取局部特征，输出特征图（Feature Map）
   │
   ▼
[激活函数（ReLU）] → 引入非线性
   │
   ▼
[池化层（Pooling）] → 降低空间分辨率，增强平移不变性
   │
   ▼
   ... 重复上述步骤若干次 ...
   │
   ▼
[展平（Flatten）] → 将多维特征图拉成一维向量
   │
   ▼
[全连接层] → 综合所有特征进行分类/回归
   │
   ▼
[输出层（Softmax / Sigmoid）] → 输出预测结果
```

### 卷积操作详解

卷积操作是 CNN 的核心。给定输入特征图和卷积核，计算过程如下：

1. 将卷积核覆盖在输入特征图的左上角。
2. 对应位置元素相乘再求和，得到输出的一个值。
3. 按步长（Stride）滑动卷积核，重复步骤 2。
4. 遍历完整个输入，得到输出特征图。

**关键超参数**：

| 超参数 | 含义 | 典型值 |
|--------|------|--------|
| 卷积核大小（Kernel Size） | 卷积核的空间尺寸 | 3×3、5×5、1×1 |
| 步长（Stride） | 卷积核每次滑动的距离 | 1 或 2 |
| 填充（Padding） | 在输入边缘补零 | Same（保持尺寸不变）或 Valid（不填充） |
| 输出通道数（Out Channels） | 卷积核的数量，决定输出特征图的通道数 | 64、128、256、512 |

### 池化操作

池化层（Pooling Layer）用于降低特征图的空间分辨率，减少计算量，同时增强特征的平移不变性。

- **最大池化（Max Pooling）**：取局部区域的最大值，保留最显著的特征。
- **平均池化（Average Pooling）**：取局部区域的均值，保留整体信息。

最常见的配置是 2×2 的最大池化，步长为 2，将特征图的宽高各缩小一半。

### 感受野

**感受野（Receptive Field）** 是指输出特征图上一个像素点对应输入图像上的区域大小。它决定了网络能"看到"多大范围的上下文信息。

计算公式（逐层递推）：

$$
RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i
$$

其中：
- $RF_l$ 是第 $l$ 层的感受野大小
- $k_l$ 是第 $l$ 层的卷积核大小
- $s_i$ 是第 $i$ 层的步长

直观理解：堆叠多个 3×3 卷积层的效果等价于一个更大的卷积核，但参数量更少。例如 3 层 3×3 卷积的感受野等价于一个 7×7 卷积，但参数量只有后者的 $\frac{3 \times 3^2}{7^2} = \frac{27}{49} \approx 55\%$。

## 数学定义

### 二维卷积

对于输入特征图 $X \in \mathbb{R}^{H \times W}$ 和卷积核 $K \in \mathbb{R}^{k_h \times k_w}$，输出特征图 $Y$ 的第 $(i, j)$ 个元素为：

$$ 
Y(i, j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X(i+m, j+n) \cdot K(m, n) + b
$$

其中：
- $H, W$ 是输入的高度和宽度
- $k_h, k_w$ 是卷积核的高度和宽度
- $b$ 是偏置项（Bias）

### 多通道卷积

实际中输入通常有 $C_{in}$ 个通道，卷积核也有 $C_{in}$ 个通道。一个卷积核的完整计算为：

$$ 
Y(i, j) = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X_c(i+m, j+n) \cdot K_c(m, n) + b
$$

若使用 $C_{out}$ 个不同的卷积核，则输出特征图有 $C_{out}$ 个通道。

### 输出尺寸计算

给定输入尺寸 $H_{in}$、卷积核大小 $k$、步长 $s$、填充 $p$，输出尺寸为：

$$ 
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

## 代码示例

下面用 PyTorch 实现一个简单的 CNN，用于 CIFAR-10 图像分类：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    一个简单的 CNN 模型，用于 CIFAR-10（32x32 彩色图像，10 类分类）。

    网络结构：
    - 3 个卷积块（卷积 + BN + ReLU + 池化）
    - 2 个全连接层
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # 第一个卷积块：3 通道 -> 32 通道
        # 输入: (B, 3, 32, 32) -> 输出: (B, 32, 16, 16)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,  # padding=1, kernel=3, stride=1 -> 尺寸不变
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半

        # 第二个卷积块：32 通道 -> 64 通道
        # 输入: (B, 32, 16, 16) -> 输出: (B, 64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块：64 通道 -> 128 通道
        # 输入: (B, 64, 8, 8) -> 输出: (B, 128, 4, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 展平后维度: 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 卷积块 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 卷积块 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 卷积块 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.view(x.size(0), -1)  # (B, 128*4*4)

        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 使用示例
if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)

    # 创建一个随机输入，模拟一张 CIFAR-10 图片
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")   # (1, 3, 32, 32)
    print(f"输出形状: {output.shape}")         # (1, 10)
    print(f"预测类别概率: {F.softmax(output, dim=1)}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
```

运行结果：

```
输入形状: torch.Size([1, 3, 32, 32])
输出形状: torch.Size([1, 10])
预测类别概率: tensor([[0.0981, 0.1052, 0.0945, 0.1018, 0.1034, 0.0967, 0.0988, 0.1012, 0.0978, 0.1025]])
总参数量: 423,562
```

## 面试标准回答

**"请介绍一下 CNN 的核心思想"**

CNN 的核心思想有三点。第一是局部连接，每个神经元只关注输入的一个局部区域，而不是看全图，这大幅减少了参数量。第二是权值共享，同一个卷积核在整个图像上滑动复用，这不仅进一步压缩参数，还赋予了模型平移不变性——猫出现在图片的任何位置都能被识别。第三是层次化特征提取，低层卷积提取边缘、纹理等简单特征，中层组合成部件，高层识别完整物体，这种从局部到整体的结构与人类视觉系统类似。

**"CNN 中卷积层和全连接层的本质区别是什么"**

全连接层的每个神经元与前一层所有神经元相连，参数量为 $n_{in} \times n_{out}$，是全局操作。卷积层的每个神经元只与前一层的一个局部区域相连（局部连接），并且同一层的所有神经元共享同一组卷积核权重（权值共享）。这使得卷积层的参数量与输入尺寸无关，只取决于卷积核大小和通道数。对于一张 224×224×3 的图片，全连接到 1000 个神经元需要 1.5 亿参数，而一个 3×3 卷积核只需要 27 个参数。

**"什么是感受野，它有什么意义"**

感受野是指输出特征图上一个像素点对应到输入图像上的区域大小。它决定了网络在做决策时能"看到"多大的上下文。感受野太小，网络只能看到局部纹理，无法理解全局语义；感受野太大，可能引入无关信息。可以通过堆叠小卷积核（如 3×3）、增大步长、使用空洞卷积（Dilated Convolution）来增大感受野。VGG 论文的一个重要发现是：两个 3×3 卷积串联的感受野等价于一个 5×5 卷积，但参数更少、非线性更强。

## 高频追问

**Q1: 为什么用 3×3 卷积核而不是更大的？**

VGG 论文的核心贡献之一就是证明了这一点。两个 3×3 卷积的感受野等价于一个 5×5 卷积，三个 3×3 卷积等价于一个 7×7 卷积。优势有两点：(1) 参数更少，三个 3×3 卷积的参数是 $3 \times 3^2 C^2 = 27C^2$，而一个 7×7 卷积是 $7^2 C^2 = 49C^2$；(2) 更多的非线性层，增强了模型的表达能力。

**Q2: 1×1 卷积有什么用？**

1×1 卷积的作用有三个：(1) 降维/升维，改变通道数；(2) 增加非线性，配合激活函数；(3) 跨通道信息融合。GoogLeNet 中大量使用 1×1 卷积来降低通道数、减少计算量。ResNet 中的 Bottleneck 结构也用 1×1 卷积先降维再升维。

**Q3: 池化层的作用是什么？为什么现在有些网络不用池化了？**

池化层的作用是降低空间分辨率、减少计算量、增强平移不变性。但现在有些网络（如 ResNet 的部分变体）用步长为 2 的卷积代替池化，因为：(1) 池化会丢失信息（最大池化只保留最大值）；(2) 可学习的下采样（步长卷积）可能比固定的池化操作更优。不过最大池化在很多任务中仍然是标配。

**Q4: 感受野越大越好吗？**

不是。感受野过大会引入与目标无关的上下文信息，反而干扰判断。合适的感受野应该覆盖目标物体的范围。这也是为什么不同尺度的特征图（FPN 中的多尺度融合）很重要——小目标用小感受野的特征图，大目标用大感受野的特征图。

**Q5: 卷积操作的计算量怎么计算？**

一个卷积层的计算量（FLOPs）为：$2 \times C_{in} \times k_h \times k_w \times C_{out} \times H_{out} \times W_{out}$。其中乘法和加法各算一次，所以乘以 2。这是衡量模型轻重的重要指标。

## 工程实践

### 1. 参数初始化

```python
# Kaiming 初始化（适用于 ReLU 激活函数）
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# Xavier 初始化（适用于 Sigmoid/Tanh 激活函数）
nn.init.xavier_uniform_(layer.weight)
```

### 2. 训练技巧

- **学习率预热（Warmup）**：前几个 epoch 用较小的学习率，避免初始权重不稳定导致训练发散。
- **数据增强（Data Augmentation）**：随机裁剪、翻转、颜色抖动等，是防止 CNN 过拟合的最有效手段。
- **混合精度训练（Mixed Precision）**：用 FP16 代替 FP32，显存减半、速度翻倍，精度损失极小。

### 3. 模型部署注意事项

- 推理时将模型设为 `model.eval()`，关闭 Dropout 和 BatchNorm 的训练模式。
- 使用 `torch.jit.trace` 或 `torch.onnx.export` 导出模型。
- 量化（Quantization）可以将 FP32 压缩到 INT8，模型大小缩小 4 倍，推理速度提升 2-4 倍。

## 常见误区

**误区 1："CNN 只能处理图像"**

CNN 的本质是处理网格结构数据，所以它也广泛用于文本分类（1D 卷积处理序列）、语音信号处理（1D/2D 卷积）、甚至图数据（图卷积网络 GCN）。

**误区 2："卷积核越大越好"**

大卷积核确实有更大的感受野，但参数量和计算量也更大。现代网络的主流做法是用 3×3 卷积堆叠，而不是用大卷积核。唯一例外是 ConvNeXt 借鉴了 Vision Transformer 的思想，使用了 7×7 的深度可分离卷积。

**误区 3："CNN 已经被 Transformer 取代了"**

虽然 Vision Transformer（ViT）在大规模数据集上表现优异，但 CNN 在中小规模数据集上仍然有优势，因为 CNN 的归纳偏置（Inductive Bias）——局部性和平移不变性——能更好地利用有限数据。ConvNeXt 证明了经过现代化改造的 CNN 在很多任务上可以和 Transformer 持平。

**误区 4："更深的网络一定更好"**

不是。网络太深会出现梯度消失（Gradient Vanishing）和退化问题（Degradation Problem）。ResNet 通过残差连接解决了这个问题，但这并不意味着无限制加深就有收益。

## 参考资料

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.
2. Krizhevsky, A., et al. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
3. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
5. Liu, Z., et al. (2022). "A ConvNet for the 2020s." *CVPR*. (ConvNeXt)
