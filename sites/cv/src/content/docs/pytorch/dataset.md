---
title: Dataset
description: PyTorch 数据读取的核心抽象，定义如何读取单个样本
category: cv
tags: [pytorch, dataset, dataloader]
status: stable
order: 1
---

# Dataset

## 简介

在 PyTorch 中，`Dataset` 是数据读取流程的核心抽象之一。它负责描述“一个数据集里有什么数据，以及如何取出其中一个样本”。

通常来说，训练一个模型需要解决两个问题：

1. 数据存在哪里；
2. 每次训练时如何取出一个样本。

`Dataset` 主要解决第二个问题。它把数据读取逻辑从模型训练逻辑中分离出来，使代码结构更清晰。

一个典型的 PyTorch 训练流程如下：

```python
dataset = MyDataset(...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)
    ...
```

其中：

* `Dataset` 负责定义如何获取单个样本；
* `DataLoader` 负责批量读取、打乱顺序、多进程加载等；
* 训练循环只关心每次拿到的 `x` 和 `y`。

---

## Dataset 解决什么问题

假设我们要训练一个图像分类模型，原始数据可能是这样的：

```text
data/
  cat/
    001.jpg
    002.jpg
  dog/
    001.jpg
    002.jpg
```

模型训练时需要的数据形式通常是：

```python
image, label
```

也就是：

```text
一张图片 -> 一个类别标签
```

`Dataset` 的作用就是定义：

```text
给定一个索引 idx，返回第 idx 个样本。
```

例如：

```python
dataset[0]
```

可能返回：

```python
(image_tensor, 0)
```

其中：

* `image_tensor` 是图像张量；
* `0` 是类别标签，例如 cat。

---

## Dataset 和 DataLoader 的关系

`Dataset` 和 `DataLoader` 很容易混淆，但它们的职责不同。

| 组件                    | 作用                             |
| --------------------- | ------------------------------ |
| `Dataset`             | 定义如何读取单个样本                     |
| `DataLoader`          | 定义如何批量读取样本                     |
| `Dataset.__getitem__` | 返回一个样本                         |
| `Dataset.__len__`     | 返回数据集大小                        |
| `DataLoader`          | 负责 batch、shuffle、num_workers 等 |

简单理解：

```text
Dataset：告诉 PyTorch 数据怎么取。
DataLoader：告诉 PyTorch 数据怎么批量喂给模型。
```

如果没有 `DataLoader`，你也可以直接访问 `Dataset`：

```python
x, y = dataset[0]
```

但在训练模型时，通常不会一条一条手动取数据，而是使用 `DataLoader` 自动组成 batch：

```python
for x, y in dataloader:
    ...
```

---

## 最小 Dataset 示例

自定义 Dataset 通常需要继承 `torch.utils.data.Dataset`，并实现两个方法：

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

print(len(dataset))   # 4
print(dataset[0])     # (1, 0)
```

这个例子虽然简单，但已经包含了 Dataset 的核心结构。

---

## Dataset 的三个核心部分

一个标准的 Dataset 一般包含三个部分：

```python
class MyDataset(Dataset):
    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

### `__init__`

`__init__` 用来初始化数据集。

通常会在这里完成：

* 保存数据路径；
* 读取标注文件；
* 构建文件名列表；
* 构建标签列表；
* 保存 transform；
* 做一些不会频繁变化的预处理。

示例：

```python
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
```

注意：一般不建议在 `__init__` 中把所有大图像都读进内存，除非数据量很小。

更常见的做法是：

```text
__init__ 里保存路径；
__getitem__ 里按需读取图片。
```

这样可以避免一次性占用太多内存。

---

### `__len__`

`__len__` 返回数据集的样本数量。

```python
def __len__(self):
    return len(self.image_paths)
```

它会被 `len(dataset)` 调用。

例如：

```python
dataset = MyDataset(...)
print(len(dataset))
```

`DataLoader` 也会使用 `__len__` 来计算一个 epoch 中大约有多少 batch。

---

### `__getitem__`

`__getitem__` 定义如何根据索引读取一个样本。

```python
def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    label = self.labels[idx]

    image = Image.open(image_path).convert("RGB")

    if self.transform is not None:
        image = self.transform(image)

    return image, label
```

`__getitem__` 是 Dataset 最关键的方法。

它需要完成：

1. 根据 `idx` 找到样本；
2. 读取输入数据；
3. 读取或生成标签；
4. 应用 transform；
5. 返回模型训练需要的数据。

---

## 图像分类 Dataset 示例

下面是一个更接近真实项目的 Dataset 示例。

数据结构假设如下：

```text
data/
  cat/
    001.jpg
    002.jpg
  dog/
    001.jpg
    002.jpg
```

实现代码：

```python
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for image_path in cls_dir.glob("*"):
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    label = self.class_to_idx[cls_name]
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
```

使用方式：

```python
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageClassificationDataset(
    root_dir="data",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for images, labels in dataloader:
    print(images.shape)
    print(labels.shape)
    break
```

输出可能是：

```text
torch.Size([32, 3, 224, 224])
torch.Size([32])
```

其中：

* `32` 是 batch size；
* `3` 是 RGB 通道数；
* `224, 224` 是图像高宽；
* `labels` 是长度为 32 的类别标签。

---

## Dataset 返回值的常见形式

`__getitem__` 可以返回不同形式的数据，取决于任务类型。

### 图像分类

```python
return image, label
```

例如：

```python
image.shape = [3, 224, 224]
label = 0
```

### 目标检测

```python
return image, target
```

其中 `target` 可能是一个字典：

```python
target = {
    "boxes": boxes,
    "labels": labels
}
```

例如：

```python
{
    "boxes": tensor([[10, 20, 100, 200]]),
    "labels": tensor([1])
}
```

### 语义分割

```python
return image, mask
```

其中：

* `image` 是输入图像；
* `mask` 是每个像素的类别标签。

### 自监督学习

```python
return view1, view2
```

例如对同一张图片做两次不同的数据增强，得到两个视图。

### 多模态任务

```python
return image, text, label
```

例如图文匹配、视觉问答等任务。

---

## Transform 的作用

`transform` 通常用于对样本做预处理或数据增强。

常见操作包括：

* resize；
* crop；
* flip；
* normalization；
* 转 tensor；
* 颜色扰动；
* 随机旋转。

示例：

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

然后传入 Dataset：

```python
dataset = ImageClassificationDataset(
    root_dir="data/train",
    transform=transform
)
```

在 `__getitem__` 中使用：

```python
if self.transform is not None:
    image = self.transform(image)
```

注意：训练集和验证集通常使用不同的 transform。

训练集可以使用随机增强：

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
```

验证集一般不使用随机增强：

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

否则验证结果会不稳定。

---

## Dataset 中不要做什么

### 不要在 `__init__` 中读取全部大图像

不推荐：

```python
class BadDataset(Dataset):
    def __init__(self, image_paths):
        self.images = [Image.open(p) for p in image_paths]
```

如果图像数量很大，这会占用大量内存。

更推荐：

```python
class GoodDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return image
```

即：

```text
保存路径，而不是保存所有图像数据。
```

---

### 不要在 `__getitem__` 中做过重的计算

`__getitem__` 会在训练过程中被频繁调用。

如果在这里做非常耗时的操作，例如复杂图像处理、网络请求、重复解析大文件，会拖慢训练速度。

更合理的做法是：

* 能提前处理的就离线处理；
* 能缓存的就缓存；
* 能用 transform 表达的就放到 transform；
* 训练时只做必要的数据读取和轻量预处理。

---

### 不要返回不规则数据而不处理 collate

`DataLoader` 默认会把多个样本合成 batch。

如果每个样本 shape 不一致，默认拼接可能失败。

例如目标检测中，不同图片的目标框数量不同：

```python
image1_boxes.shape = [3, 4]
image2_boxes.shape = [8, 4]
```

这种情况下，默认 `collate_fn` 可能无法直接堆叠。

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

---

## 常见错误

### 错误一：忘记实现 `__len__`

错误示例：

```python
class MyDataset(Dataset):
    def __getitem__(self, idx):
        return self.data[idx]
```

问题：

```python
len(dataset)
```

无法正常工作，`DataLoader` 也可能受到影响。

---

### 错误二：`__getitem__` 返回类型不稳定

不推荐：

```python
def __getitem__(self, idx):
    if idx % 2 == 0:
        return image, label
    else:
        return image
```

同一个 Dataset 应该保持返回格式一致。

推荐始终返回：

```python
return image, label
```

或者：

```python
return {
    "image": image,
    "label": label
}
```

---

### 错误三：图像没有转成 RGB

有些图片可能是灰度图、RGBA 图或带透明通道的 PNG。

如果不统一通道数，后续组成 batch 时可能出错。

推荐：

```python
image = Image.open(image_path).convert("RGB")
```

这样可以保证输出是 3 通道图像。

---

### 错误四：标签类型不对

分类任务中，`CrossEntropyLoss` 通常需要类别索引作为标签，而不是 one-hot 向量。

推荐：

```python
label = 0
```

而不是：

```python
label = [1, 0, 0]
```

如果标签来自字符串，需要先映射成整数：

```python
class_to_idx = {
    "cat": 0,
    "dog": 1
}
```

---

### 错误五：训练集和验证集使用同一个随机增强

错误示例：

```python
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = MyDataset(..., transform=transform)
val_dataset = MyDataset(..., transform=transform)
```

验证集不应使用随机增强，否则每次验证的输入都可能不同。

推荐：

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

---

## Dataset 调试方法

写完 Dataset 后，不要急着训练模型。应该先单独检查 Dataset 是否正确。

### 检查长度

```python
print(len(dataset))
```

### 检查一个样本

```python
x, y = dataset[0]

print(type(x))
print(type(y))
print(y)
```

### 检查图像 shape

如果经过了 `ToTensor()`：

```python
print(x.shape)
```

图像分类中通常应为：

```text
[3, H, W]
```

### 检查 batch

```python
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, labels = next(iter(loader))

print(images.shape)
print(labels.shape)
```

图像分类中通常应为：

```text
images: [B, C, H, W]
labels: [B]
```

例如：

```text
torch.Size([4, 3, 224, 224])
torch.Size([4])
```

如果这里已经报错，说明问题通常在 Dataset、transform 或 collate_fn 中，而不是模型中。

---

## Map-style Dataset 和 Iterable-style Dataset

PyTorch 中常见 Dataset 可以分为两类：

```text
Map-style Dataset
Iterable-style Dataset
```

### Map-style Dataset

最常见的是 Map-style Dataset。

它支持通过索引访问样本：

```python
dataset[0]
dataset[1]
dataset[2]
```

通常需要实现：

```python
__len__
__getitem__
```

大多数图像分类、检测、分割任务都使用这种方式。

---

### Iterable-style Dataset

Iterable-style Dataset 适用于流式数据或无法随机索引的数据。

例如：

* 日志流；
* 网络数据流；
* 超大规模文件；
* 实时采集数据。

它通常继承 `IterableDataset`，并实现 `__iter__`：

```python
from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __iter__(self):
        for i in range(10):
            yield i
```

使用：

```python
dataset = MyIterableDataset()

for x in dataset:
    print(x)
```

初学阶段优先掌握 Map-style Dataset 即可。

---

## Dataset 的工程实践建议

### 1. 路径和标签提前整理好

推荐在 `__init__` 中把所有样本路径和标签整理成列表：

```python
self.samples = [
    ("data/cat/001.jpg", 0),
    ("data/dog/001.jpg", 1),
]
```

这样 `__getitem__` 会很清晰：

```python
image_path, label = self.samples[idx]
```

---

### 2. 保持 Dataset 简单

Dataset 不应该承担太多职责。

推荐职责：

```text
读取样本
读取标签
应用 transform
返回数据
```

不推荐职责：

```text
训练模型
计算 loss
更新参数
保存 checkpoint
复杂评估逻辑
```

这些应该放在训练脚本或评估脚本中。

---

### 3. 文件名和标签映射要固定

分类任务中，类别到索引的映射应该稳定。

推荐：

```python
self.classes = sorted(class_names)
self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
```

使用 `sorted` 可以避免不同系统中文件读取顺序不同导致标签映射变化。

---

### 4. 注意异常样本

真实数据集中经常存在：

* 损坏图片；
* 空文件；
* 标注缺失；
* 类别名错误；
* 文件路径不存在。

可以在构建样本列表时过滤明显非法的数据。

例如：

```python
valid_suffixes = [".jpg", ".jpeg", ".png", ".bmp"]

if image_path.suffix.lower() in valid_suffixes:
    self.samples.append((image_path, label))
```

更严格的项目中，可以提前写脚本检查数据集，而不是在训练时才暴露问题。

---

## 一个完整可运行示例

下面是一个完整的图像分类 Dataset 示例。

```python
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([
            p.name for p in self.root_dir.iterdir()
            if p.is_dir()
        ])

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []
        valid_suffixes = [".jpg", ".jpeg", ".png", ".bmp"]

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]

            for image_path in cls_dir.glob("*"):
                if image_path.suffix.lower() in valid_suffixes:
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageClassificationDataset(
        root_dir="data",
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    images, labels = next(iter(dataloader))

    print("classes:", dataset.classes)
    print("class_to_idx:", dataset.class_to_idx)
    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
```

如果数据结构正确，输出可能类似：

```text
classes: ['cat', 'dog']
class_to_idx: {'cat': 0, 'dog': 1}
images shape: torch.Size([8, 3, 224, 224])
labels shape: torch.Size([8])
```

---

## 高频面试题

### Q1：PyTorch 中 Dataset 和 DataLoader 的区别是什么？

`Dataset` 负责定义如何读取单个样本，通常实现 `__len__` 和 `__getitem__`。
`DataLoader` 负责把 Dataset 包装成可迭代对象，并提供 batch、shuffle、多进程加载等功能。

简单来说：

```text
Dataset 管样本怎么取；
DataLoader 管样本怎么批量取。
```

---

### Q2：自定义 Dataset 需要实现哪些方法？

对于最常见的 Map-style Dataset，一般需要实现：

```python
__len__
__getitem__
```

其中：

* `__len__` 返回数据集大小；
* `__getitem__` 根据索引返回一个样本。

---

### Q3：为什么不建议在 Dataset 的 `__init__` 中读取所有图片？

因为数据集可能很大，一次性读取所有图片会占用大量内存。

更常见的做法是：

```text
__init__ 中保存图片路径；
__getitem__ 中按需读取图片。
```

这样可以降低内存占用，也更适合大规模数据集。

---

### Q4：为什么训练集要 shuffle，而验证集通常不需要 shuffle？

训练集使用 shuffle 可以打乱样本顺序，减少模型对数据顺序的依赖。

验证集一般不需要 shuffle，因为验证阶段不更新模型参数，只评估模型表现。即使打乱顺序，整体指标通常也不变。

---

### Q5：Dataset 返回的 label 应该是什么类型？

分类任务中，标签通常是类别索引，例如：

```python
label = 0
label = 1
label = 2
```

如果使用 `CrossEntropyLoss`，通常不需要 one-hot 标签。

---

### Q6：什么时候需要自定义 `collate_fn`？

当不同样本无法被默认方式堆叠成 batch 时，需要自定义 `collate_fn`。

典型场景是目标检测：

```text
图片 A 有 3 个目标框；
图片 B 有 8 个目标框。
```

由于目标框数量不同，默认 batch 拼接可能失败。这时可以用自定义 `collate_fn` 返回 list。

---

## 小结

`Dataset` 是 PyTorch 数据读取流程的基础组件。

核心要点：

1. `Dataset` 定义如何读取单个样本；
2. 常见自定义 Dataset 需要实现 `__len__` 和 `__getitem__`；
3. `DataLoader` 基于 Dataset 生成 batch；
4. 图像任务中通常在 `__getitem__` 中读取图片并应用 transform；
5. Dataset 应该保持简单，只负责数据读取和必要预处理；
6. 训练前应先单独调试 Dataset 和 DataLoader。

最重要的代码模板是：

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

掌握这个模板，就可以开始为分类、检测、分割等任务编写自己的数据集。
