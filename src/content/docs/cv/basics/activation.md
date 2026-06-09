---
title: 激活函数
description: 深度学习中常用激活函数的原理、对比与工程实践
category: cv
tags: [activation, relu, basics]
status: stable
order: 3
---

# 激活函数

## 一句话解释

激活函数（Activation Function）是神经网络中对输入信号进行非线性变换的函数，它决定了神经元是否应该被"激活"以及输出多大的信号。

## 它解决什么问题

如果没有激活函数，无论神经网络有多少层，整个网络都只是输入的线性组合。数学上可以证明：多个线性变换的叠加仍然是线性变换。也就是说，没有激活函数的深层网络和一个单层线性模型等价，根本无法学习复杂的非线性模式。

激活函数的引入让神经网络具备了拟合任意复杂函数的能力（万能近似定理，Universal Approximation Theorem）。它解决了以下核心问题：

- **表达能力**：让网络能够学习非线性决策边界
- **梯度传播**：为反向传播（Backpropagation）提供梯度信号
- **特征映射**：将输入映射到特定范围，便于后续层处理

## 核心思想

激活函数的核心思想可以用一句话概括：**在保留信息的同时引入非线性**。

一个好的激活函数需要满足几个直觉上的要求：

1. **非线性**：这是最基本的要求，否则多层网络退化为单层
2. **可微性**：反向传播需要计算梯度，所以激活函数需要几乎处处可微
3. **单调性**（非必须）：单调的激活函数能让损失函数更容易优化
4. **输出范围有限**（非必须）：有限的输出范围能让梯度更稳定
5. **零中心**（非必须）：输出以 0 为中心时，梯度更新更高效

## 算法流程

激活函数在神经网络中的工作流程：

```
输入 x
  ↓
线性变换 z = Wx + b（权重 × 输入 + 偏置）
  ↓
非线性激活 a = f(z)
  ↓
输出 a 作为下一层的输入
```

在反向传播时，梯度链式法则需要激活函数的导数：

```
损失 L 对权重 W 的梯度 = ∂L/∂a × f'(z) × ∂z/∂W
                         ↑
                    激活函数的导数
```

## 数学定义

### Sigmoid

$$ 
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- **导数**：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **输出范围**：$(0, 1)$
- **变量说明**：$x$ 为输入值，$e$ 为自然对数底数

### Tanh

$$ 
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- **导数**：$\tanh'(x) = 1 - \tanh^2(x)$
- **输出范围**：$(-1, 1)$

### ReLU

$$ 
\text{ReLU}(x) = \max(0, x)
$$

- **导数**：$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$
- **输出范围**：$[0, +\infty)$

### LeakyReLU

$$ 
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

- **导数**：$\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$
- **变量说明**：$\alpha$ 通常取 0.01，是负半轴的斜率

### GELU

$$ 
\text{GELU}(x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）：

$$ 
\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

- **近似计算**：$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]$
- **输出范围**：$(-\infty, +\infty)$
- **变量说明**：$\text{erf}$ 为误差函数，$\pi$ 为圆周率

### Swish

$$ 
\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
$$

- **导数**：$\text{Swish}'(x) = \sigma(\beta x) + x \cdot \beta \sigma(\beta x)(1 - \sigma(\beta x))$
- **变量说明**：$\beta$ 为可学习参数或固定值 1（此时称为 SiLU）
- **输出范围**：$(-\infty, +\infty)$

## 代码示例

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 各激活函数的 PyTorch 实现
# ============================================================

# 1. Sigmoid — 将输出压缩到 (0, 1)
sigmoid = nn.Sigmoid()

# 2. Tanh — 将输出压缩到 (-1, 1)，零中心
tanh = nn.Tanh()

# 3. ReLU — 最常用的激活函数，计算简单
relu = nn.ReLU()

# 4. LeakyReLU — 解决 ReLU 的神经元死亡问题
leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# 5. GELU — Transformer 和 BERT 中的标准激活函数
gelu = nn.GELU()

# 6. Swish / SiLU — Google 提出的平滑激活函数
swish = nn.SiLU()  # SiLU 就是 beta=1 的 Swish

# ============================================================
# 生成测试数据并可视化
# ============================================================
x = torch.linspace(-5, 5, 500)

activations = {
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'LeakyReLU': leaky_relu(x),
    'GELU': gelu(x),
    'Swish (SiLU)': swish(x),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, y) in zip(axes.flat, activations.items()):
    ax.plot(x.numpy(), y.detach().numpy(), linewidth=2)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()


# ============================================================
# 在网络中使用激活函数
# ============================================================

# 方法 1：使用 nn.Sequential 搭建网络
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),           # 隐藏层用 ReLU
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Sigmoid(),        # 输出层根据任务选择
)

# 方法 2：在 forward 中手动调用
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 先线性变换，再激活
        x = self.fc2(x)
        return x


# ============================================================
# 手动实现激活函数（帮助理解原理）
# ============================================================

def sigmoid_manual(x):
    """手动实现 Sigmoid"""
    return 1.0 / (1.0 + torch.exp(-x))

def relu_manual(x):
    """手动实现 ReLU"""
    return torch.where(x > 0, x, torch.zeros_like(x))

def gelu_manual(x):
    """手动实现 GELU（近似版本）"""
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)
    ))

def swish_manual(x, beta=1.0):
    """手动实现 Swish"""
    return x * torch.sigmoid(beta * x)

# 验证手动实现与 PyTorch 内置实现的结果一致性
test_input = torch.randn(5)
print("Sigmoid 差异:", torch.abs(sigmoid_manual(test_input) - sigmoid(test_input)).max())
print("ReLU 差异:", torch.abs(relu_manual(test_input) - relu(test_input)).max())
print("GELU 差异:", torch.abs(gelu_manual(test_input) - gelu(test_input)).max())
```

## 激活函数对比

| 激活函数 | 输出范围 | 零中心 | 计算复杂度 | 梯度消失 | 典型场景 |
|---------|---------|--------|-----------|---------|---------|
| Sigmoid | (0, 1) | 否 | 中等 | 严重 | 二分类输出层 |
| Tanh | (-1, 1) | 是 | 中等 | 较严重 | RNN、早期 CNN |
| ReLU | [0, +∞) | 否 | 极低 | 负半轴死亡 | CNN 隐藏层（默认选择） |
| LeakyReLU | (-∞, +∞) | 否 | 极低 | 无 | 替代 ReLU 防止死亡 |
| GELU | (-∞, +∞) | 否 | 较高 | 无 | Transformer、BERT |
| Swish | (-∞, +∞) | 否 | 较高 | 无 | EfficientNet、大规模模型 |

## 面试标准回答

**"请介绍一下常用的激活函数及其优缺点。"**

激活函数是神经网络中引入非线性的关键组件。没有激活函数，多层网络等价于单层线性模型，无法学习复杂模式。

早期最常用的是 Sigmoid 和 Tanh。Sigmoid 将输出压缩到 0 到 1，适合做二分类的输出层，但它有两个严重问题：一是当输入值较大或较小时，梯度趋近于零，导致梯度消失（Vanishing Gradient）；二是输出不是零中心的，这会导致梯度更新效率低下。Tanh 解决了零中心的问题，但同样存在梯度消失。

ReLU 的出现是激活函数领域的重大突破，它在正半轴上梯度恒为 1，彻底解决了梯度消失问题，而且计算极其简单（只需一个比较操作）。但 ReLU 也有"神经元死亡"的问题：当某个神经元的输入始终为负时，它的梯度永远为零，这个神经元就再也无法被更新。LeakyReLU 通过给负半轴一个小的斜率（如 0.01）来解决这个问题。

近年来，随着 Transformer 架构的流行，GELU 和 Swish 成为了新的主流选择。GELU 在 BERT、GPT 等模型中被广泛使用，它的特点是平滑且在零点附近有随机正则化的效果。Swish 是 Google 通过搜索发现的激活函数，实验表明在深层网络中表现优于 ReLU。

**实际选择建议**：对于 CV 任务的 CNN，隐藏层默认用 ReLU，追求极致效果可以试 GELU 或 Swish；对于 NLP 的 Transformer，用 GELU；输出层根据任务选择，二分类用 Sigmoid，多分类用 Softmax。

## 高频追问

**Q1：为什么 Sigmoid 会导致梯度消失？**

Sigmoid 的导数最大值只有 0.25（在 x=0 处）。在反向传播中，梯度需要逐层连乘。假设网络有 n 层，每层的梯度最多是 0.25，那么传到第一层的梯度最多是 $0.25^n$，随着层数增加会指数级衰减，最终趋近于零，导致前面的层无法学习。

**Q2：ReLU 在 x<0 时梯度为零，这有什么问题？**

当某个神经元的输入始终落在负半轴时，它的梯度永远为零，权重永远不会更新，这个神经元就"死了"。在学习率设置过大的情况下，这个问题尤其严重，因为大梯度更新可能把权重推向一个让该神经元永远处于负输入区域的状态。

**Q3：GELU 为什么比 ReLU 效果好？**

GELU 的优势在于它是一个平滑函数，在零点附近不会像 ReLU 那样有突变。这种平滑性使得优化过程更稳定。此外，GELU 在零点附近有一个"软阈值"效应：输入接近零时，输出有一定的概率被丢弃，这相当于一种隐式的正则化，有助于防止过拟合。

**Q4：如何选择激活函数？**

经验法则：CNN 隐藏层用 ReLU（默认选择，简单高效）；如果遇到神经元死亡问题，换 LeakyReLU；Transformer 类模型用 GELU；输出层根据任务选择——二分类用 Sigmoid，多分类用 Softmax，回归任务可以用线性激活。

**Q5：Swish 和 SiLU 是什么关系？**

SiLU（Sigmoid Linear Unit）就是 $\beta = 1$ 时的 Swish 函数。PyTorch 中 `nn.SiLU()` 实现的就是这个版本。原始论文中 Swish 是带可学习参数 $\beta$ 的，但实践中固定 $\beta = 1$ 效果已经很好，所以 PyTorch 直接提供了 SiLU 实现。

## 工程实践

### 1. 初始化配合激活函数

激活函数的选择会影响权重初始化策略。使用 ReLU 时，推荐用 Kaiming 初始化（He 初始化），它考虑了 ReLU 会将一半输入置零的特性：

```python
# ReLU + Kaiming 初始化
layer = nn.Linear(256, 128)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Sigmoid/Tanh + Xavier 初始化
layer = nn.Linear(256, 128)
nn.init.xavier_normal_(layer.weight)
```

### 2. 输出层激活函数选择

```python
# 二分类：BCEWithLogitsLoss 内部已包含 Sigmoid，不需要手动加
model = nn.Sequential(
    nn.Linear(256, 1),
    # 不加 Sigmoid，直接用 BCEWithLogitsLoss
)
criterion = nn.BCEWithLogitsLoss()

# 多分类：CrossEntropyLoss 内部已包含 Softmax
model = nn.Sequential(
    nn.Linear(256, 10),
    # 不加 Softmax，直接用 CrossEntropyLoss
)
criterion = nn.CrossEntropyLoss()
```

### 3. 混合精度训练中的注意事项

GELU 和 Swish 在 FP16 混合精度训练中可能有数值稳定性问题，建议：

```python
# 使用自动混合精度时，让激活函数在 FP32 下计算
with torch.cuda.amp.autocast():
    x = layer(x)
    x = x.float()  # 转回 FP32
    x = gelu(x)
```

### 4. 推理时的优化

ReLU 在推理时可以被融合到前面的线性层或卷积层中，减少一次单独的激活操作：

```python
# 训练时
output = relu(conv(input))

# 导出 ONNX 时，ReLU 可以融合进 Conv，减少一层
# TensorRT 会自动做这个优化
```

## 常见误区

### 误区 1：输出层也要加激活函数

**错误**：多分类任务在输出层加 Softmax，然后用 `CrossEntropyLoss`。

**正确**：`nn.CrossEntropyLoss` 内部已经包含了 Softmax，不需要手动加。加了会导致 Softmax 被计算两次，训练不稳定。

### 误区 2：ReLU 永远是最好的选择

**真相**：ReLU 在大多数 CNN 任务中确实是默认选择，但在某些场景下（如 Transformer、轻量级网络），GELU 或 Swish 效果更好。不要盲目使用 ReLU。

### 误区 3：Sigmoid 只用于输出层

**真相**：Sigmoid 在隐藏层中确实不推荐使用（梯度消失问题），但在以下场景中仍然重要：
- 二分类任务的输出层
- 门控机制（如 LSTM 的遗忘门、输出门）
- 注意力机制中的权重归一化

### 误区 4：LeakyReLU 的 alpha 值很重要

**真相**：实验表明 LeakyReLU 对 alpha 值不敏感，0.01 和 0.1 的效果差异很小。不需要花太多时间调这个超参数。

### 误区 5：GELU 和 ReLU 可以随意替换

**真相**：虽然理论上可以替换，但 GELU 通常需要配合特定的初始化和学习率策略。在小数据集上，GELU 可能因为参数量不足而表现不如 ReLU；在大规模预训练模型中，GELU 的优势才能体现。

## 参考资料

- Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. ICML.
- Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. AISTATS.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
- Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. arXiv:1710.05941.
- He, K., et al. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. ICCV.
