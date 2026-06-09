---
title: Transformer
description: Transformer 架构的核心原理
category: llm
tags: [transformer, attention, architecture]
status: stable
order: 1
---

# Transformer

## 一句话解释

Transformer 是一种基于自注意力机制的神经网络架构，摒弃了传统的循环和卷积结构，完全依赖注意力机制来捕捉序列中的依赖关系。

## 它解决什么问题

传统的 RNN 和 LSTM 存在以下问题：
- **无法并行计算**：必须按顺序处理序列
- **长距离依赖困难**：梯度消失导致难以捕捉远距离关系
- **训练效率低**：序列越长，训练越慢

Transformer 通过自注意力机制解决了这些问题，实现了完全并行化计算。

## 核心思想

Transformer 的核心是**自注意力机制（Self-Attention）**，它允许序列中的每个位置直接关注其他所有位置，无需通过循环传递信息。

关键创新：
- **多头注意力**：从不同角度关注信息
- **位置编码**：注入序列位置信息
- **编码器-解码器结构**：分别处理输入和输出

## 算法流程

### 编码器（Encoder）

1. 输入嵌入 + 位置编码
2. 多头自注意力层
3. 前馈神经网络
4. 残差连接 + 层归一化
5. 重复 N 次

### 解码器（Decoder）

1. 输出嵌入 + 位置编码
2. 掩码多头自注意力（防止看到未来信息）
3. 编码器-解码器交叉注意力
4. 前馈神经网络
5. 残差连接 + 层归一化
6. 重复 N 次

## 数学定义

### 自注意力计算

$$ 
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$（Query）：查询矩阵，表示"我要找什么信息"
- $K$（Key）：键矩阵，表示"我有什么信息"
- $V$（Value）：值矩阵，表示"我的具体内容"
- $d_k$：Key 的维度，用于缩放防止梯度消失

### 多头注意力

$$ 
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

其中每个头：
$$ 
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 位置编码

$$ 
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$ 
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：
- $pos$：位置索引
- $i$：维度索引
- $d_{model}$：模型维度

## 代码示例

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

## 面试标准回答

**面试中可以这样回答：**

Transformer 是 2017 年由 Google 在论文 "Attention is All You Need" 中提出的神经网络架构。它的核心创新是完全基于自注意力机制，摒弃了传统的循环和卷积结构。

自注意力机制让序列中的每个位置都能直接关注其他所有位置，解决了 RNN 无法并行计算和长距离依赖困难的问题。通过多头注意力，模型可以从不同角度关注信息；通过位置编码，模型能够感知序列顺序。

Transformer 的编码器-解码器结构使其特别适合序列到序列的任务，如机器翻译。如今，Transformer 已成为 NLP 领域的基础架构，BERT、GPT、T5 等大模型都基于它构建。

## 高频追问

### 为什么用缩放点积注意力？

使用 $\sqrt{d_k}$ 缩放是为了防止 $d_k$ 较大时点积结果过大，导致 softmax 梯度消失。当 $d_k$ 较大时，点积的方差会增大，使得 softmax 的输出趋向于 one-hot 分布，梯度接近于 0。

### 为什么需要位置编码？

Transformer 没有循环或卷积结构，无法感知序列中元素的位置信息。位置编码通过将位置信息注入到输入嵌入中，让模型知道每个元素在序列中的位置。

### 为什么用多头注意力而不是单头？

多头注意力让模型能够从不同的表示子空间中学习信息。每个头可以关注不同类型的依赖关系（如语法关系、语义关系等），最后将它们组合起来得到更丰富的表示。

### Transformer 的复杂度是多少？

自注意力的时间复杂度是 $O(n^2 \cdot d)$，其中 $n$ 是序列长度，$d$ 是特征维度。这是因为每个位置都需要与所有其他位置计算注意力分数。这也是 Transformer 处理长序列的瓶颈。

## 工程实践

### 训练技巧

- **学习率预热**：先用较小的学习率预热，再逐渐增大
- **标签平滑**：防止模型过于自信
- **梯度裁剪**：防止梯度爆炸
- **混合精度训练**：使用 FP16 加速训练

### 推理优化

- **KV Cache**：缓存历史 Key-Value，避免重复计算
- **模型量化**：将 FP32 量化为 INT8，减少计算量
- **模型剪枝**：去除不重要的参数
- **知识蒸馏**：用大模型指导小模型训练

## 常见误区

- **误区**：Transformer 只能用于 NLP
  **事实**：Transformer 也可用于 CV（ViT）、音频（Wav2Vec）、多模态（CLIP）

- **误区**：注意力权重就是重要性
  **事实**：注意力权重只是模型的一种表示，不一定完全反映重要性

- **误区**：位置编码是唯一的位置信息来源
  **事实**：相对位置编码、旋转位置编码（RoPE）等也是常用方法

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [Harvard NLP: The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - 注释版实现
