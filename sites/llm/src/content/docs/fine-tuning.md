---
title: Fine-tuning 与 LoRA
description: 大语言模型微调技术：全参数微调、LoRA、QLoRA 的原理与实践
category: llm
tags: [llm, fine-tuning, lora, qlora, peft]
status: stable
order: 3
---

# Fine-tuning 与 LoRA

## 一句话解释

Fine-tuning（微调）是在预训练模型的基础上，用特定任务的数据继续训练以适配下游任务；LoRA 通过低秩分解大幅减少可训练参数，使微调大模型变得可行。

## 它解决什么问题

预训练大模型拥有通用知识，但直接用于特定任务（如医疗问答、代码生成）效果有限。全参数微调需要更新模型所有参数，对于 7B 参数的模型需要约 28GB 显存（FP32），普通 GPU 无法承受。

LoRA（Low-Rank Adaptation）的核心思想是：微调时权重变化量是低秩的，可以用两个小矩阵的乘积来近似。这将可训练参数从数十亿降低到数百万，显存需求降低 4-8 倍。

## 核心思想

### 全参数微调

最直接的微调方式，更新模型所有参数。优点是效果最好，缺点是：
- 显存需求大（需要存储模型参数、梯度、优化器状态）
- 容易过拟合（小数据集上灾难性遗忘）
- 每个任务需要一份完整模型副本

### LoRA

冻结预训练权重 $W_0$，只训练低秩增量 $\Delta W = BA$：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，秩 $r \ll d$。

可训练参数量从 $d \times d$ 降低到 $2 \times d \times r$。当 $d=4096$、$r=8$ 时，参数量减少 256 倍。

### QLoRA

在 LoRA 基础上引入量化：
- 将预训练权重量化为 4-bit（NF4 格式）
- LoRA 适配器保持 BF16/FP16
- 使用分页优化器处理显存峰值

这使得在单张 24GB GPU 上微调 65B 参数模型成为可能。

## 数学定义

### LoRA 前向传播

给定输入 $x$，LoRA 层的输出为：

$$
h = W_0 x + \frac{\alpha}{r} BAx
$$

其中 $\alpha$ 是缩放因子，$\frac{\alpha}{r}$ 控制低秩增量的权重。

### 初始化策略

- $A$ 使用高斯随机初始化
- $B$ 初始化为零矩阵
- 训练开始时 $\Delta W = BA = 0$，不改变预训练模型行为

### 参数效率对比

| 方法 | 可训练参数（7B 模型） | 显存需求 |
|------|----------------------|----------|
| 全参数微调 | 7B | ~28GB (FP32) |
| LoRA (r=8) | ~4M | ~8GB |
| QLoRA (r=8) | ~4M | ~6GB |

## 代码示例

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 加载基座模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 秩
    lora_alpha=32,          # 缩放因子
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 应用 LoRA 的层
)

# 包装模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# 训练配置
training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 保存 LoRA 适配器（只保存增量参数，通常只有几十 MB）
model.save_pretrained("./lora-adapter")
```

## 面试标准回答

**"LoRA 的核心思想是什么"**

LoRA 的核心假设是：微调时权重更新矩阵 $\Delta W$ 是低秩的。基于这个假设，LoRA 将 $\Delta W$ 分解为两个低秩矩阵 $B$ 和 $A$ 的乘积，冻结原始权重 $W_0$，只训练 $B$ 和 $A$。这样可训练参数量从 $d^2$ 降到 $2dr$，当 $r=8$、$d=4096$ 时参数量减少 256 倍。初始化时 $B=0$ 保证训练起点与预训练模型一致。推理时可以将 $BA$ 合并回 $W_0$，不增加推理延迟。

**"LoRA 的秩 r 怎么选"**

秩 $r$ 控制表达能力和参数效率的平衡。$r$ 越大，LoRA 能表达越复杂的权重变化，但参数量也越多。实践中 $r=8$ 或 $r=16$ 是常用起点。对于简单任务（如情感分类）$r=4$ 可能就够，复杂任务（如代码生成）可能需要 $r=64$。可以通过在验证集上对比不同 $r$ 值的效果来选择。

**"QLoRA 和 LoRA 有什么区别"**

QLoRA 在 LoRA 基础上做了三个改进：(1) 将预训练权重量化为 4-bit NF4 格式，大幅降低显存占用；(2) 双重量化，对量化常数再次量化；(3) 分页优化器，处理显存峰值。这使得在单张 24GB GPU 上微调 65B 模型成为可能，而 LoRA 在同样硬件上只能微调 7-13B 模型。

## 高频追问

**Q1: LoRA 应用在哪些层效果最好？**

研究表明，同时应用在 Query 和 Value 投影层（q_proj, v_proj）效果最好。有些实践也会加上 Key 和 Output 投影层。不建议只应用在 FFN 层，因为注意力层对任务适配更关键。

**Q2: 微调时学习率怎么设？**

LoRA 微调的学习率通常比全参数微调高一个数量级。全参数微调用 $1e-5$ 到 $5e-5$，LoRA 用 $1e-4$ 到 $3e-4$。因为 LoRA 只更新少量参数，需要更大的学习率才能有效学习。

**Q3: 如何避免微调时的灾难性遗忘？**

(1) 使用较小的学习率；(2) 混入通用数据（如 5-10% 的预训练数据）；(3) 使用 LoRA 等参数高效方法（天然保留预训练知识）；(4) 定期在通用基准上评估模型能力是否退化。

**Q4: LoRA 适配器可以合并到基座模型吗？**

可以。合并后推理时没有额外开销：$W = W_0 + BA$。合并后的模型与普通模型完全一样，可以用标准方式部署。这也是 LoRA 的一个优势——训练时高效，推理时无损。

## 工程实践

### 1. 多 LoRA 适配器管理

一个基座模型可以搭配多个 LoRA 适配器（不同任务），推理时动态切换：

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base-model")
# 加载不同任务的适配器
model_medical = PeftModel.from_pretrained(base_model, "lora-medical")
model_code = PeftModel.from_pretrained(base_model, "lora-code")
```

### 2. 数据质量比数据量重要

微调数据质量远比数量重要。1000 条高质量标注数据通常优于 10000 条噪声数据。建议：
- 数据去重
- 过滤低质量样本
- 保持格式一致性
- 验证标签正确性

### 3. 评估指标

微调后需要评估：
- 目标任务效果（accuracy/F1/BLEU 等）
- 通用能力保持（在基准测试上不退化）
- 推理延迟（合并适配器后应无变化）

## 常见误区

1. **"LoRA 效果一定比全参数微调差"** — 对于小数据集，LoRA 反而可能更好，因为它天然有正则化效果，不容易过拟合。

2. **"秩越大越好"** — 过大的秩会导致过拟合，且参数效率降低。需要在验证集上实验选择。

3. **"LoRA 只能用在 Transformer 上"** — LoRA 可以用在任何线性层，包括 CNN 的卷积层。只是在 Transformer 上效果最显著。

4. **"微调能弥补数据质量问题"** — 微调不能创造知识，只能激活和调整已有能力。垃圾数据进，垃圾结果出。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.15647)
