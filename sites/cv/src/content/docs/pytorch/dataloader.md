---
title: DataLoader
description: PyTorch 批量数据加载工具
category: cv
tags: [pytorch, dataloader, batch]
status: review
order: 2
---

## 简介

`DataLoader` 是 PyTorch 中用于批量读取数据的工具。

如果说 `Dataset` 负责“如何读取一个样本”，那么 `DataLoader` 负责：

> 如何把多个样本组成一个 batch，并送入训练循环。

典型用法：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
```

其中：

- `dataset` 提供单个样本；
    
- `DataLoader` 自动组成 batch；
    
- 训练循环每次拿到一批数据。
    

---

## DataLoader 解决什么问题

模型训练通常不是一个样本一个样本训练，而是一批一批训练。

例如图像分类中，`Dataset` 返回一个样本：

```python
image, label = dataset[0]
```

而 `DataLoader` 返回一个 batch：

```python
images, labels = next(iter(dataloader))
```

形状通常是：

```text
images: [B, C, H, W]
labels: [B]
```

例如：

```text
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

其中 `32` 就是 batch size。

---

## 最小示例

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4]
        self.labels = [0, 0, 1, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = MyDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for x, y in dataloader:
    print(x, y)
```

输出可能类似：

```text
tensor([3, 1]) tensor([1, 0])
tensor([2, 4]) tensor([0, 1])
```

由于设置了 `shuffle=True`，每次运行时样本顺序可能不同。

---

## 常用参数

### `batch_size`

`batch_size` 表示每个 batch 包含多少个样本。

```python
dataloader = DataLoader(dataset, batch_size=32)
```

如果数据集有 1000 个样本，`batch_size=32`，那么每个 epoch 大约有：

```text
1000 / 32 ≈ 32 个 batch
```

一般来说：

- batch 太小，训练不稳定；
    
- batch 太大，占用显存更多；
    
- 常见取值有 16、32、64、128。
    

---

### `shuffle`

`shuffle` 表示是否打乱样本顺序。

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

训练集通常设置：

```python
shuffle=True
```

验证集或测试集通常设置：

```python
shuffle=False
```

原因是训练时打乱顺序有助于减少模型对数据顺序的依赖，而验证和测试阶段只需要稳定评估结果。

---

### `num_workers`

`num_workers` 表示使用多少个子进程加载数据。

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

常见设置：

```text
num_workers=0：主进程加载数据，适合调试
num_workers>0：多进程加载数据，适合正式训练
```

如果数据读取或预处理比较慢，可以适当增大 `num_workers`。

但它不是越大越好。过大可能导致：

- CPU 占用过高；
    
- 内存占用增加；
    
- 进程启动开销变大；
    
- 在某些系统上报错更难排查。
    

初学阶段可以先用：

```python
num_workers=0
```

代码稳定后再尝试：

```python
num_workers=2
num_workers=4
```

---

### `drop_last`

`drop_last` 表示是否丢弃最后一个不完整的 batch。

例如有 100 个样本，`batch_size=32`：

```text
前 3 个 batch：32 + 32 + 32
最后 1 个 batch：4
```

默认情况下，最后这个 batch 会保留。

```python
DataLoader(dataset, batch_size=32, drop_last=False)
```

如果设置：

```python
DataLoader(dataset, batch_size=32, drop_last=True)
```

最后不足 32 个样本的 batch 会被丢弃。

一般任务中默认 `False` 即可。某些对 batch size 敏感的训练过程可以设置为 `True`。

---

### `pin_memory`

如果使用 GPU 训练，可以考虑开启：

```python
DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True
)
```

它可以让 CPU 到 GPU 的数据传输更高效。

典型写法：

```python
for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)
```

如果只在 CPU 上训练，不需要关心这个参数。

---

## 训练集和验证集写法

通常会分别创建训练集和验证集的 DataLoader。

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

区别：

|数据集|shuffle|
|---|---|
|训练集|`True`|
|验证集|`False`|
|测试集|`False`|

---

## 在训练循环中使用

```python
for epoch in range(num_epochs):
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

这里每次循环取出的 `images` 和 `labels` 都是一个 batch。

---

## 检查 DataLoader 是否正常

在正式训练前，建议先检查一个 batch。

```python
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
```

图像分类任务中，常见输出是：

```text
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

如果这里报错，问题通常来自：

- `Dataset.__getitem__` 返回格式不一致；
    
- 图像尺寸不一致；
    
- transform 没有把图片转成 Tensor；
    
- label 类型不正确；
    
- batch 中的数据无法自动拼接。
    

---

## `collate_fn`

默认情况下，`DataLoader` 会自动把多个样本拼成一个 batch。

例如多个样本：

```python
(image1, label1)
(image2, label2)
(image3, label3)
```

会被整理成：

```python
(images, labels)
```

但如果每个样本的结构不一样，默认拼接可能失败。

典型情况是目标检测：

```text
第一张图有 3 个框
第二张图有 8 个框
```

这时每张图的标注数量不同，不能直接堆叠成一个规则 Tensor。

可以自定义 `collate_fn`：

```python
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
```

使用：

```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)
```

初学阶段做分类任务时，一般不需要自定义 `collate_fn`。

---

## 常见注意点

### 1. 先用 `num_workers=0` 调试

如果 Dataset 代码有问题，`num_workers=0` 时错误信息通常更容易看懂。

```python
DataLoader(dataset, batch_size=32, num_workers=0)
```

调试通过后，再增加 `num_workers`。

---

### 2. 训练集通常要 shuffle

训练时建议：

```python
shuffle=True
```

否则模型可能受到数据原始顺序影响。

验证和测试时通常：

```python
shuffle=False
```

---

### 3. DataLoader 返回的是 batch，不是单个样本

`Dataset`：

```python
image, label = dataset[0]
```

`DataLoader`：

```python
images, labels = next(iter(dataloader))
```

区别在于：

```text
Dataset 返回单个样本；
DataLoader 返回一批样本。
```

---

### 4. batch size 过大可能显存不足

如果出现 CUDA out of memory，可以先减小 `batch_size`。

例如：

```python
batch_size=64
```

改为：

```python
batch_size=32
```

或者：

```python
batch_size=16
```

---

## 小结

`DataLoader` 是 PyTorch 训练流程中负责批量读取数据的工具。

重点记住：

```text
Dataset 负责单个样本；
DataLoader 负责组成 batch；
训练集通常 shuffle=True；
验证集通常 shuffle=False；
num_workers=0 适合调试；
batch_size 过大可能导致显存不足。
```

最常用模板：

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    pred = model(images)
    loss = loss_fn(pred, labels)
```

掌握 `DataLoader` 后，就可以把自定义 `Dataset` 接入标准训练流程。