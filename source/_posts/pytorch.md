---
title: PyTorch 面经
date: 2024-05-13 19:00:00
excerpt: PyTorch 面经整理
author: dydy
categories:
- 面试
- 机器学习
---

## PyTorch 概述

PyTorch 是一个开源的机器学习库，广泛应用于计算机视觉和自然语言处理等领域

PyTorch 的两个高级功能

- 强大的GPU加速的张量计算（Numpy）
- 包含自动求导系统的深度神经网络

PyTorch的核心特性

- 动态计算图：使用动态计算图（命令式编程模型），计算图在每次运行的时候都会重新构建。使得模型比较灵活
- 自动微分系统：PyTorch 的`torch.autograd` 提供了自动微分的功能，可以自动计算导数和梯度


## PyTorch 与深度学习

### Tensors 张量

在PyTorch中使用tensors来对模型的输入、输出和模型参数进行编码。
与Numpy的ndarrays相比，tensors可以运行在多GPU上以及其他的硬件上，拥有比Numpy更高的效率

#### Tensors 初始化

- 直接通过`data`

```python
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
```

- 通过Numpy 数组

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

- 从其他的tensor

```python
x_ones = torch.ones_like(x_data) # 维持原有的数据类型
>>> tensor([[1,1],
[1,1]])

x_rand = torch.rand_like(x_data, dtype=torch.float) # 重写了数据类型
>>> tensor([[0.8823, 0.9150],
        [0.3829, 0.9593]])
```

- 通过随机或常量值

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
>>> Random Tensor:
 tensor([[0.3904, 0.6009, 0.2566],
        [0.7936, 0.9408, 0.1332]])

>>> Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

>>> Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

#### Tensors 属性

Tensors的属性描述了他们的形状、数据类型以及他们存放的设备

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

>>> 
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

#### Tensor 操作

- 运行在GPU上

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
```

- 行列操作

```python
tensor = torch.ones(4,4)
tensor[:,1] = 0

>>> 
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

- 拼接操作

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

>>> 
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

- 置换操作

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)

>>>
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

### `torch.autograd`

#### 背景知识

神经网络（NN）是在某些输入数据上执行的嵌套函数的集合。这些函数（W）通过参数进行定义，在PyTorch中通过张量进行保存

训练一个神经网络通常有两个步骤

- **前向传播**：NN猜测输入的最可能输出，将输入传递到定义的几个函数
- **反向传播**：NN根据猜测值的误差调整自己的参数，通过梯度下降法优化参数

#### 在 PyTorch 的使用



```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weight=ResNet18_Weights.DEFAULT)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)
```

进行前向传播

```python
prediction = model(data)
```

获得与预测值的差值，然后惊醒方向传播

```python
loss = (prediction-labels).sum()
loss.backward()
```

然后加载一个优化器，通过学习率为0.01步长为0.9，注册在优化器中的所有参数

```python
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```
然后迭代参数下降法

```python
optim.step()
```


### 神经网络

一个神经网络定义的过程

- 定义神经网络的学习参数
- 迭代输入
- 在网络中传递
- 计算损失值
- 迭代反向梯度传播
- 更新权重