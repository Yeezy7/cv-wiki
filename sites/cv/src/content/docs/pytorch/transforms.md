---
title: Transforms
description: PyTorch 图像预处理与数据增强工具
category: cv
tags: [pytorch, transforms, augmentation]
status: review
order: 1
---

## 简介

`Transforms` 是 PyTorch 图像任务中常用的数据预处理和数据增强工具。

它通常用于：

```text
把原始数据转换成模型可以输入的 Tensor；
对训练数据做随机增强；
统一图像大小；
归一化图像数值。
```

在图像分类任务中，`Transforms` 常和 `Dataset` 一起使用：

```python
dataset = ImageDataset(
    image_paths=image_paths,
    labels=labels,
    transform=transform
)
```

然后在 `Dataset.__getitem__` 中调用：

```python
if self.transform is not None:
    image = self.transform(image)
```

---

## 为什么需要 Transforms

原始图片通常不能直接送入模型。

常见问题包括：

```text
图像尺寸不统一；
数据类型不是 Tensor；
像素值范围不合适；
训练数据太少，容易过拟合。
```

`Transforms` 可以解决这些问题。

例如，一张 PIL 图片经过 transform 后，可以变成形状为 `[C, H, W]` 的 Tensor：

```text
PIL Image -> Tensor
```

---

## 最常用写法

通常使用 `transforms.Compose` 把多个变换串起来。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

含义是：

```text
先把图片 resize 到 224 x 224；
再转换成 Tensor。
```

使用：

```python
image = transform(image)
```

---

## 常用 Transforms

### `Resize`

调整图像大小。

```python
transforms.Resize((224, 224))
```

常用于把不同尺寸的图片统一成模型输入尺寸。

---

### `ToTensor`

把 PIL Image 或 NumPy 数组转换成 Tensor。

```python
transforms.ToTensor()
```

转换后，图像形状通常从：

```text
[H, W, C]
```

变成：

```text
[C, H, W]
```

像素值也会从大致的 `0~255` 转换到 `0~1`。

---

### `Normalize`

对图像做归一化。 output[channel] = (input[channel] - mean[channel]) / std[channel]

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

它通常放在 `ToTensor()` 后面。

常见顺序：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

注意：`Normalize` 处理的是 Tensor，不是 PIL Image。

---

### `RandomHorizontalFlip`

随机水平翻转图像。

```python
transforms.RandomHorizontalFlip(p=0.5)
```

常用于训练集数据增强。

它的意思是：每张图片有 50% 的概率被水平翻转。

---

### `RandomCrop`

随机裁剪图像。

```python
transforms.RandomCrop((224, 224))
```

常用于训练集增强。

---

### `CenterCrop`

从图像中心裁剪。

```python
transforms.CenterCrop((224, 224))
```

常用于验证集或测试集。

---

## 训练集和验证集的区别

训练集通常使用随机增强：

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

验证集通常不使用随机增强：

```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

原因是：

```text
训练集需要增强数据多样性；
验证集需要稳定评估模型效果。
```

如果验证集也使用随机增强，每次验证输入可能不同，评估结果会不稳定。

---

## 在 Dataset 中使用

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

创建数据集：

```python
train_dataset = ImageDataset(
    image_paths=train_image_paths,
    labels=train_labels,
    transform=train_transform
)

val_dataset = ImageDataset(
    image_paths=val_image_paths,
    labels=val_labels,
    transform=val_transform
)
```

---

## 常见注意点

### 1. `Normalize` 要放在 `ToTensor` 后面

错误写法：

```python
transforms.Compose([
    transforms.Normalize(mean, std),
    transforms.ToTensor(),
])
```

推荐写法：

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

因为 `Normalize` 需要处理 Tensor。

---

### 2. 训练集和验证集不要共用随机增强

训练集可以使用：

```python
transforms.RandomHorizontalFlip()
transforms.RandomCrop()
```

验证集一般不要使用随机增强。

验证集更适合使用确定性变换：

```python
transforms.Resize()
transforms.CenterCrop()
transforms.ToTensor()
```

---

### 3. 图像分类中通常要统一尺寸

如果图像尺寸不同，`DataLoader` 默认无法把它们拼成一个 batch。

所以分类任务中通常需要：

```python
transforms.Resize((224, 224))
```

否则可能出现 batch 拼接错误。

---

### 4. 先检查 transform 输出

正式训练前，建议检查一个样本：

```python
x, y = train_dataset[0]

print(type(x))
print(x.shape)
print(y)
```

图像分类中，`x.shape` 通常应为：

```text
[3, 224, 224]
```

如果 shape 不对，问题通常在 transform 或图片通道数上。

---

## 最小模板

图像分类中最常用的 transform 模板：

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

如果使用 ImageNet 预训练模型，通常还会加上归一化：

```python
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

完整写法：

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
```

---

## 小结

`Transforms` 主要负责数据预处理和数据增强。

重点记住：

```text
Resize：统一图像尺寸；
ToTensor：把图片转成 Tensor；
Normalize：归一化；
RandomHorizontalFlip / RandomCrop：训练集数据增强；
训练集可以随机增强，验证集一般不要随机增强。
```

最常见使用方式：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

然后在 `Dataset.__getitem__` 中调用：

```python
image = self.transform(image)
```

掌握 `Transforms` 后，就能完成 PyTorch 图像任务中最基本的数据预处理流程。