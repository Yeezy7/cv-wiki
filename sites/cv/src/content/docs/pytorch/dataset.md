---
title: Dataset
description: PyTorch 数据集基础组件
category: cv
tags: [pytorch, dataset, data]
status: review
order: 3
---

## 简介

`Dataset` 是 PyTorch 中用于表示数据集的基础组件。

它的核心作用是：

> 定义如何根据索引 `idx` 取出一个样本。

在训练模型时，我们通常不会把数据读取逻辑直接写在训练循环里，而是把它封装到 `Dataset` 中。

典型流程如下：

```python
dataset = MyDataset(...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
```

其中：

* `Dataset` 负责读取单个样本；
* `DataLoader` 负责批量读取样本；
* 训练循环只负责模型训练。

---

## Dataset 的核心思想

`Dataset` 本质上回答两个问题：

```text
这个数据集有多少个样本？
给定一个索引 idx，如何返回第 idx 个样本？
```

所以自定义 Dataset 通常需要实现两个方法：

```python
__len__
__getitem__
```

---

## 最小示例

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4]
        self.labels = [0, 0, 1, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
```

使用方式：

```python
dataset = MyDataset()

print(len(dataset))  # 4
print(dataset[0])    # (1, 0)
```

这个例子已经包含了 Dataset 的基本结构。

---

## 三个核心部分

### `__init__`

`__init__` 用于初始化数据集。

常见操作包括：

* 保存数据路径；
* 读取标注文件；
* 构建样本列表；
* 保存 transform。

例如：

```python
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
```

一般不建议在 `__init__` 中读取全部图片，尤其是数据集很大时。更常见的做法是：

```text
__init__ 中保存路径；
__getitem__ 中按需读取图片。
```

---

### `__len__`

`__len__` 返回数据集大小。

```python
def __len__(self):
    return len(self.image_paths)
```

它会被 `len(dataset)` 调用，`DataLoader` 也会用它来计算 batch 数量。

---

### `__getitem__`

`__getitem__` 根据索引返回一个样本。

```python
def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    label = self.labels[idx]

    image = Image.open(image_path).convert("RGB")

    if self.transform is not None:
        image = self.transform(image)

    return image, label
```

对于图像分类任务，通常返回：

```python
return image, label
```

其中：

* `image` 是输入图像；
* `label` 是类别标签。

---

## 图像分类 Dataset 示例

假设数据路径和标签已经提前整理好：

```python
image_paths = [
    "data/cat/001.jpg",
    "data/dog/001.jpg",
]

labels = [0, 1]
```

可以这样写 Dataset：

```python
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
```

配合 `DataLoader` 使用：

```python
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageDataset(image_paths, labels, transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

训练时：

```python
for images, labels in dataloader:
    print(images.shape)
    print(labels.shape)
    break
```

输出通常类似：

```text
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

---

## Dataset 和 DataLoader 的区别

| 组件           | 作用         |
| ------------ | ---------- |
| `Dataset`    | 定义如何读取单个样本 |
| `DataLoader` | 定义如何批量读取样本 |

简单理解：

```text
Dataset 管一个样本怎么取；
DataLoader 管一批样本怎么取。
```

例如：

```python
dataset[0]
```

返回一个样本。

而：

```python
for images, labels in dataloader:
    ...
```

返回一个 batch。

---

## 常见注意点

### 1. 不要在 `__init__` 中读取所有大文件

不推荐：

```python
self.images = [Image.open(path) for path in image_paths]
```

推荐：

```python
self.image_paths = image_paths
```

然后在 `__getitem__` 中读取：

```python
image = Image.open(self.image_paths[idx])
```

这样可以减少内存占用。

---

### 2. 返回格式要稳定

不要有时返回两个值，有时返回一个值。

不推荐：

```python
if condition:
    return image, label
else:
    return image
```

推荐始终返回固定格式：

```python
return image, label
```

---

### 3. 图像任务中注意通道数

建议统一转成 RGB：

```python
image = Image.open(path).convert("RGB")
```

否则灰度图、RGBA 图可能导致 batch 拼接失败。

---

### 4. 分类标签通常是整数

对于分类任务，标签一般是类别索引：

```python
label = 0
label = 1
label = 2
```

如果使用 `CrossEntropyLoss`，通常不需要 one-hot 标签。

---

## 调试 Dataset

写完 Dataset 后，先不要急着训练模型。应该先检查数据是否正常。

```python
dataset = ImageDataset(image_paths, labels, transform=transform)

x, y = dataset[0]

print(type(x))
print(x.shape)
print(y)
```

再检查 batch：

```python
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

images, labels = next(iter(dataloader))

print(images.shape)
print(labels.shape)
```

如果这里报错，问题通常在：

* 图片路径；
* transform；
* 返回值格式；
* 图像 shape；
* label 类型。

---

## 小结

`Dataset` 是 PyTorch 数据读取流程中的基础抽象。

重点记住：

```text
Dataset 负责读取单个样本；
DataLoader 负责组成 batch；
自定义 Dataset 通常实现 __len__ 和 __getitem__。
```

最常用模板：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

掌握这个模板，就可以编写分类、检测、分割等任务的数据集。
