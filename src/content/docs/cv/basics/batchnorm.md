---
title: BatchNorm
description: 批归一化（Batch Normalization）的原理、计算过程与工程实践
category: cv
tags: [batchnorm, normalization, basics]
status: stable
order: 2
---

# BatchNorm

## 一句话解释

批归一化（Batch Normalization, BatchNorm）是一种在训练过程中对每个 mini-batch 的特征进行归一化的技术，通过将中间层的输入分布稳定在均值为 0、方差为 1 的范围内，加速训练收敛并提升模型稳定性。

## 它解决什么问题

深度网络训练中有一个著名的难题：**内部协变量偏移（Internal Covariate Shift）**。

直观理解：网络第 $l$ 层的输入是由第 $l-1$ 层的输出决定的。当第 $l-1$ 层的权重更新后，第 $l$ 层看到的输入分布就变了。每一层都在不断适应变化的输入分布，就像在移动的地基上盖房子——训练效率极低。

具体表现：

1. **训练缓慢**：为了适应不断变化的输入分布，只能用很小的学习率，否则容易发散。
2. **梯度消失/爆炸**：深层网络中，信号经过多层变换后分布会严重偏移，导致梯度要么消失要么爆炸。
3. **初始化敏感**：不同的初始化方式可能导致完全不同的训练结果。

BatchNorm 的出现正是为了解决这个问题，让每一层都能接收到分布稳定的输入。

## 核心思想

BatchNorm 的核心思想可以用一个类比来理解：**想象一个班级参加考试，每次考试的难度不同，原始分数没有可比性。如果我们把每次考试的成绩标准化（减去平均分、除以标准差），就能公平地比较不同次考试的成绩。**

BatchNorm 对神经网络做的事情完全一样：

1. **归一化**：对每个 mini-batch 的特征减均值、除标准差，消除分布偏移。
2. **缩放和平移**：归一化后的特征分布被强制为标准正态分布，这可能损失网络的表达能力。因此引入两个可学习参数 $\gamma$（缩放）和 $\beta$（平移），让网络自己决定最优的分布。

这两个步骤保证了：每一层的输入分布是稳定的（有利于训练），同时又保留了网络的表达能力（不会因为归一化而降低模型容量）。

## 算法流程

### 训练阶段

给定一个 mini-batch $\mathcal{B} = \{x_1, x_2, ..., x_m\}$，BatchNorm 的计算步骤如下：

```
输入: mini-batch B = {x_1, x_2, ..., x_m}
输出: {y_1, y_2, ..., y_m}

Step 1: 计算 batch 均值
    μ_B = (1/m) * Σ x_i

Step 2: 计算 batch 方差
    σ²_B = (1/m) * Σ (x_i - μ_B)²

Step 3: 归一化
    x̂_i = (x_i - μ_B) / √(σ²_B + ε)
    其中 ε 是一个极小值（如 1e-5），防止除以零

Step 4: 缩放和平移（可学习参数）
    y_i = γ * x̂_i + β
    γ 和 β 是可学习参数，初始 γ=1, β=0
```

### 推理阶段

推理时没有 mini-batch，无法计算 batch 均值和方差。解决方案是使用训练过程中积累的**全局统计量**：

```
输入: 单个样本 x
输出: y

使用训练时通过指数移动平均（Exponential Moving Average）积累的:
    - 全局均值 μ_running
    - 全局方差 σ²_running

计算:
    x̂ = (x - μ_running) / √(σ²_running + ε)
    y = γ * x̂ + β
```

全局统计量的更新公式：

$$\mu_{running} = (1 - \alpha) \times \mu_{running} + \alpha \times \mu_B$$

$$\sigma^2_{running} = (1 - \alpha) \times \sigma^2_{running} + \alpha \times \sigma^2_B$$

其中 $\alpha$ 是动量（momentum），默认值通常为 0.1。

### 训练 vs 推理的区别总结

| 维度 | 训练阶段 | 推理阶段 |
|------|----------|----------|
| 均值/方差来源 | 当前 mini-batch 的统计量 | 训练积累的全局统计量 |
| 统计量是否更新 | 是，每个 batch 都更新 running mean/var | 否，使用固定的 running mean/var |
| Dropout | 生效 | 关闭 |
| 前向传播行为 | 同一输入不同 batch 结果不同 | 同一输入结果固定 |

**重要提醒**：推理前必须调用 `model.eval()`，否则 BatchNorm 会使用当前 batch 的统计量而非全局统计量，导致推理结果不稳定。

## 数学定义

### 前向传播

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

其中：
- $m$ 是 mini-batch 的大小
- $x_i$ 是第 $i$ 个样本的特征值
- $\mu_\mathcal{B}$ 是 batch 均值
- $\sigma^2_\mathcal{B}$ 是 batch 方差
- $\epsilon$ 是防止除零的小常数（如 $10^{-5}$）
- $\gamma$ 是可学习的缩放参数（Scale），初始值为 1
- $\beta$ 是可学习的偏移参数（Shift），初始值为 0

### 反向传播

BatchNorm 的反向传播需要计算 $\frac{\partial L}{\partial x_i}$、$\frac{\partial L}{\partial \gamma}$ 和 $\frac{\partial L}{\partial \beta}$：

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \left( m \frac{\partial L}{\partial y_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \cdot \hat{x}_j \right)$$

直观理解：反向传播时，梯度不仅与当前样本有关，还与整个 batch 的统计量有关，这意味着 BatchNorm 让 batch 内的样本产生了"信息交互"。

### BatchNorm 在 CNN 中的应用

对于卷积层的输出 $X \in \mathbb{R}^{B \times C \times H \times W}$，BatchNorm 是**按通道（per-channel）** 归一化的：

- 对每个通道 $c$，在 $(B, H, W)$ 维度上计算均值和方差
- 每个通道有独立的 $\gamma_c$ 和 $\beta_c$
- 总共有 $2C$ 个可学习参数（$C$ 个 $\gamma$ 和 $C$ 个 $\beta$）

## 代码示例

### 基础用法

```python
import torch
import torch.nn as nn


# ===== 1D BatchNorm（用于全连接层之后） =====
# 输入形状: (B, C)，如全连接层输出
batch_size, features = 32, 128
x = torch.randn(batch_size, features)

bn_1d = nn.BatchNorm1d(num_features=features)
y = bn_1d(x)

print(f"输入形状: {x.shape}")          # (32, 128)
print(f"输出形状: {y.shape}")          # (32, 128)
print(f"可学习参数: gamma={bn_1d.weight.shape}, beta={bn_1d.bias.shape}")  # (128,) (128,)


# ===== 2D BatchNorm（用于卷积层之后） =====
# 输入形状: (B, C, H, W)，如卷积层输出
batch_size, channels, height, width = 32, 64, 16, 16
x = torch.randn(batch_size, channels, height, width)

bn_2d = nn.BatchNorm2d(num_features=channels)
y = bn_2d(x)

print(f"\n输入形状: {x.shape}")         # (32, 64, 16, 16)
print(f"输出形状: {y.shape}")           # (32, 64, 16, 16)
print(f"可学习参数量: {sum(p.numel() for p in bn_2d.parameters())}")  # 128 (64个gamma + 64个beta)
```

### 训练和推理的区别

```python
import torch
import torch.nn as nn


model = nn.BatchNorm2d(64)

# ===== 训练模式 =====
model.train()
x = torch.randn(32, 64, 16, 16)
y_train = model(x)  # 使用当前 batch 的均值和方差

# 查看积累的全局统计量（初始值）
print("训练前 running_mean:", model.running_mean[:5])
print("训练前 running_var:", model.running_var[:5])

# 再跑几个 batch，running_mean 和 running_var 会逐步更新
for _ in range(100):
    x = torch.randn(32, 64, 16, 16)
    model(x)

print("\n训练后 running_mean:", model.running_mean[:5])
print("训练后 running_var:", model.running_var[:5])

# ===== 推理模式 =====
model.eval()
x_test = torch.randn(1, 64, 16, 16)
y_test1 = model(x_test)
y_test2 = model(x_test)

# 推理模式下，同一输入的输出完全一致
print(f"\n两次推理结果是否相同: {torch.allclose(y_test1, y_test2)}")  # True

# 训练模式下，同一输入的输出可能不同（因为 batch 统计量不同）
model.train()
y_train1 = model(x_test)
y_train2 = model(x_test)
print(f"两次训练前向结果是否相同: {torch.allclose(y_train1, y_train2)}")  # 通常 True（单样本 batch 方差为0）
```

### 在 CNN 中使用 BatchNorm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """
    卷积 + BatchNorm + ReLU 的标准组合模块。
    这是现代 CNN 中最常见的构建块。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # 注意：使用 BatchNorm 时，卷积层通常不需要 bias
        # 因为 BatchNorm 的 beta 已经起到了偏置的作用
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class CNNWithBN(nn.Module):
    """使用 BatchNorm 的 CNN 模型"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),        # (B, 3, 32, 32) -> (B, 32, 32, 32)
            nn.MaxPool2d(2),          # -> (B, 32, 16, 16)
            ConvBNReLU(32, 64),       # -> (B, 64, 16, 16)
            nn.MaxPool2d(2),          # -> (B, 64, 8, 8)
            ConvBNReLU(64, 128),      # -> (B, 128, 8, 8)
            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 使用示例
model = CNNWithBN(num_classes=10)
x = torch.randn(8, 3, 32, 32)
output = model(x)
print(f"输出形状: {output.shape}")  # (8, 10)

# 统计 BatchNorm 的参数量
bn_params = sum(p.numel() for name, p in model.named_parameters() if 'bn' in name)
total_params = sum(p.numel() for p in model.parameters())
print(f"BatchNorm 参数量: {bn_params}")
print(f"总参数量: {total_params}")
print(f"BatchNorm 占比: {bn_params / total_params * 100:.1f}%")
```

### 手动实现 BatchNorm

```python
import torch


class ManualBatchNorm2d:
    """
    手动实现 2D BatchNorm，帮助理解内部计算过程。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        # 全局统计量（推理时使用）
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):
        """
        x 形状: (B, C, H, W)
        """
        if training:
            # Step 1: 计算 batch 均值（沿 B, H, W 维度）
            mean = x.mean(dim=(0, 2, 3))  # (C,)

            # Step 2: 计算 batch 方差
            var = x.var(dim=(0, 2, 3), unbiased=False)  # (C,)

            # Step 3: 归一化
            x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)

            # Step 4: 缩放和平移
            out = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

            # 更新全局统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            return out
        else:
            # 推理时使用全局统计量
            x_norm = (x - self.running_mean[None, :, None, None]) / torch.sqrt(
                self.running_var[None, :, None, None] + self.eps
            )
            return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]


# 验证手动实现与 PyTorch 实现的一致性
torch.manual_seed(42)

x = torch.randn(16, 32, 8, 8)  # (B, C, H, W)

# PyTorch 官方实现
bn_pytorch = torch.nn.BatchNorm2d(32)
bn_pytorch.train()
y_pytorch = bn_pytorch(x)

# 手动实现
bn_manual = ManualBatchNorm2d(32)
bn_manual.gamma = bn_pytorch.weight.data.clone()
bn_manual.beta = bn_pytorch.bias.data.clone()
y_manual = bn_manual.forward(x, training=True)

print(f"手动实现与 PyTorch 结果是否一致: {torch.allclose(y_pytorch, y_manual, atol=1e-5)}")
```

## 面试标准回答

**"请解释 BatchNorm 的原理和作用"**

BatchNorm 的核心思想是对每一层的输入进行归一化，使其分布保持稳定。具体做法是：在训练时，对每个 mini-batch 计算均值和方差，将特征归一化到均值为 0、方差为 1 的分布，然后通过两个可学习参数 gamma 和 beta 进行缩放和平移，恢复网络的表达能力。推理时使用训练过程中积累的全局统计量。

BatchNorm 的作用有三个：(1) 加速训练收敛，因为每一层接收到的输入分布更稳定，可以用更大的学习率；(2) 一定程度上起到正则化作用，因为每个样本的归一化依赖于 batch 中其他样本，引入了噪声；(3) 缓解梯度消失问题，因为归一化后的特征分布更合理。

**"训练和推理时 BatchNorm 有什么区别"**

最关键的区别是统计量的来源不同。训练时，均值和方差来自当前 mini-batch，每个 batch 的统计量都会更新到全局的 running_mean 和 running_var 中（通过指数移动平均）。推理时，使用固定的全局统计量，不再更新。这就是为什么推理前必须调用 model.eval()，否则 BatchNorm 会用当前输入的统计量，导致结果不稳定。另一个区别是 Dropout：训练时生效，推理时关闭。

**"BatchNorm 为什么有效"**

这个问题学术界仍有争议，但主流解释有三个角度：(1) 缓解内部协变量偏移（Internal Covariate Shift），让每一层的输入分布更稳定；(2) 平滑损失曲面（Loss Landscape），使梯度更稳定、优化更容易，这是 2018 年一篇重要论文的结论；(3) 隐式正则化效果，因为 batch 统计量引入了噪声，类似 Dropout 的效果。实践中，BatchNorm 让我们能用 10 倍甚至更大的学习率训练，这本身就是巨大的收益。

## 高频追问

**Q1: BatchNorm 的 batch_size 很小（如 1 或 2）时怎么办？**

batch_size 太小时，batch 统计量噪声太大，BatchNorm 会失效甚至有害。解决方案有：
- **Group Normalization（GN）**：将通道分成若干组，在组内计算统计量，与 batch_size 无关。
- **Layer Normalization（LN）**：对单个样本的所有通道计算统计量，常用于 NLP 和 Transformer。
- **Instance Normalization（IN）**：对单个样本的每个通道单独计算统计量，常用于风格迁移。
- **Sync BatchNorm**：跨多个 GPU 同步计算 batch 统计量，等价于更大的 batch_size。

**Q2: 为什么 BatchNorm 的卷积层不需要 bias？**

因为 BatchNorm 的 $\beta$ 参数已经起到了偏置的作用。数学上，$y = \gamma \cdot \frac{Wx + b - \mu}{\sigma} + \beta$，其中 $b$ 和 $\beta$ 的作用是冗余的，去掉 $b$ 不影响表达能力，还能减少一个参数。

**Q3: BatchNorm 对学习率有什么影响？**

BatchNorm 使得损失曲面更平滑，这意味着可以用更大的学习率而不会发散。实验表明，使用 BatchNorm 后，可用学习率可以提升 10 倍甚至更多。但学习率太大仍然会出问题，建议配合学习率调度（如 Cosine Annealing）使用。

**Q4: BatchNorm 和 LayerNorm 的区别？**

两者的核心区别是归一化的维度不同：
- **BatchNorm**：沿 batch 维度归一化，即同一个特征在不同样本间归一化。适合 batch_size 较大的 CV 任务。
- **LayerNorm**：沿特征维度归一化，即同一个样本在不同特征间归一化。适合 batch_size 不固定的 NLP 任务。

在 Transformer 中，LayerNorm 是标配，因为序列长度不固定，且 batch_size 通常较小。

**Q5: BatchNorm 在推理时如果 batch_size=1 会出问题吗？**

不会。推理时使用的是训练积累的 running_mean 和 running_var，与当前输入的 batch_size 无关。只要正确调用了 model.eval()，batch_size=1 完全正常。

**Q6: BatchNorm 有正则化效果吗？**

有，但较弱。因为训练时每个样本的归一化依赖于 batch 中其他样本，这引入了随机噪声，起到了类似 Dropout 的正则化效果。但不要用 BatchNorm 替代 Dropout，两者的作用机制不同，通常可以同时使用。

## 工程实践

### 1. BatchNorm 的位置

常见的放置顺序有两种：

```python
# 方案 A：Conv -> BN -> ReLU（最常见，原始论文推荐）
x = F.relu(bn(conv(x)))

# 方案 B：Conv -> ReLU -> BN（某些场景效果更好）
x = bn(F.relu(conv(x)))
```

现代实践中方案 A 更主流。ResNet 的官方实现使用方案 A。

### 2. 冻结 BatchNorm

在微调（Fine-tuning）预训练模型时，有时需要冻结 BatchNorm 的统计量，防止小 batch_size 导致统计量不准：

```python
model.eval()  # 全局设为 eval 模式

# 只解冻特定层
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # 冻结 BN 层
        for param in module.parameters():
            param.requires_grad = False  # 冻结参数
```

### 3. 替代方案选择

| 场景 | 推荐方案 |
|------|----------|
| CV 任务，batch_size 较大 | BatchNorm |
| CV 任务，batch_size 很小 | Group Normalization |
| NLP / Transformer | Layer Normalization |
| 风格迁移 | Instance Normalization |
| 需要跨 GPU 同步 | Sync BatchNorm |

### 4. 常见 Bug

```python
# Bug：推理时忘记切换 eval 模型
model.train()  # 忘记切换
output = model(test_input)  # 结果不稳定！

# 修复：推理前必须调用
model.eval()
with torch.no_grad():
    output = model(test_input)

# Bug：加载 checkpoint 后忘记恢复训练模式
model.load_state_dict(checkpoint)
model.train()  # 记得切换回训练模式
```

## 常见误区

**误区 1："BatchNorm 能解决所有训练问题"**

BatchNorm 不是万能药。它对学习率、初始化仍然敏感，只是容忍度更高了。而且在某些场景（如强化学习、在线学习）中，batch_size 很小，BatchNorm 反而有害。

**误区 2："训练和推理时 BatchNorm 行为一样"**

这是最常见的错误。训练时用 batch 统计量，推理时用全局统计量。两者的行为完全不同，必须通过 model.train() 和 model.eval() 切换。

**误区 3："BatchNorm 的 epsilon 越小越好"**

epsilon 的作用是防止除以零，通常设为 1e-5。它对结果的影响很小，不需要特别调优。

**误区 4："BatchNorm 放在激活函数前面还是后面无所谓"**

有区别。原始论文推荐放在激活函数前面（Conv-BN-ReLU），但有些研究发现放在后面可能更好。实践中建议遵循主流做法，放在激活函数前面。

**误区 5："BatchNorm 的 running_mean 和 running_var 在训练时不会被使用"**

训练时它们会被更新（通过指数移动平均），但不会被用于前向传播。推理时才会被使用。如果在训练时想使用全局统计量（如某些特殊的微调策略），需要手动设置 `module.eval()`。

## 参考资料

1. Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML*.
2. Santurkar, S., et al. (2018). "How Does Batch Normalization Help Optimization?" *NeurIPS*. (证明 BatchNorm 平滑了损失曲面)
3. Wu, Y., & He, K. (2018). "Group Normalization." *ECCV*.
4. Ba, J. L., et al. (2016). "Layer Normalization." *arXiv*.
5. Ulyanov, D., et al. (2016). "Instance Normalization: The Missing Ingredient for Fast Stylization." *arXiv*.
