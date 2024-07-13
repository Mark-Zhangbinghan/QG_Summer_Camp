# pytorch

[TOC]

### 一、简介

Pytorch是torch的python版本，是由Facebook开源的神经网络框架，专门针对 GPU 加速的深度神经网络（DNN）编程。Torch 是一个经典的对多维矩阵数据进行操作的张量（tensor ）库，在机器学习和其他数学密集型应用有广泛应用。与Tensorflow的静态计算图不同，pytorch的计算图是动态的，可以根据计算需要实时改变计算图。但由于Torch语言采用 Lua，导致在国内一直很小众，并逐渐被支持 Python 的 Tensorflow 抢走用户。作为经典机器学习库 Torch 的端口，PyTorch 为 Python 语言使用者提供了舒适的写代码选择。



### 二、应用

<img src="C:\Users\24468\AppData\Roaming\Typora\typora-user-images\1720315604122.png" width="500" height="200" />

#### 1. 运用os库进行路径分析

```python
import os
path_list = os.listdir("路径") # 获取当前路径下的所有文件名
```

<img src="https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100909003.png" style="zoom:80%;"/>

<img src="https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100910767.png" style="zoom:80%;"/>

#### 2. 工具库的使用

| 名称        | 作用                                 |
| ----------- | ------------------------------------ |
| torch.utils | 数据载入器。具有训练器和其他便利功能 |

##### 1）Dataset的使用（储存数据）

```python
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class MyDataset(Dataset): # 定义类
    
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
```

##### 2）TensorBoard的使用（数据可视化）

```python
tensorboard --logdir "地址"
```

TensorBoard 是一组用于数据可视化的工具。它包含在流行的开源机器学习库 Tensorflow 中。TensorBoard 的主要功能包括：

1. 可视化模型的网络架构
2. 跟踪模型指标，如损失和准确性等
3. 检查机器学习工作流程中权重、偏差和其他组件的直方图
4. 显示非表格数据，包括图像、文本和音频
5. 将高维嵌入投影到低维空间

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('logs')
writer.add_image("test", img_array, 1,  dataformats='HWC') 
for i in range(100):
    writer.add_scalar("y=x", i, i)
writer.close()
```

##### 3）Transforms的使用（类型转换）

tranforms顾名思义是类型转换的意思

<img src="https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100910438.png" style="zoom:50%;"/>

```python
# 图片类型转换方法(PIL为python内置方法读取到的图片类型)
	#1. 变成array类型
    img_PIL = Image.open(img_path)
	img_array = np.array(img_PIL)
    #2. 变成tensor类型
    img_PIL = Image.open(img_path)
    tensor_trans = transforms.ToTensor()
    img_tensor = tensor_trans(img_PIL)
    #3. 变成PIL类型
    PIL_trans = transforms.ToPILImage()
    img_PIL = PIL_trans(img_tensor/img_array)
```

```python
#Normalize的张量数据归一化
	# input = (input - mean) / std
	# mean-通道的平均值	std-通道值的标准差 
	from torchvision import transforms
	trans_norm = transforms.Normalize(mean, std)
	image_norm = trans_norm(img_tensor)
```

```python
#Resize调整图像大小
	transforms.Resize((224, 224)),# 调整图像大小
    
#RandomCrop随机裁切
	transforms.RandomCrop((100, 100))
```

```python
#Compose进行图像预处理，包括调整尺寸、转换为张量并进行标准化，以准备模型输入
	data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),# 调整图像大小
        transforms.ToTensor(),#转化为张量
        # 归一化至 [0, 1] 范围内（假设图像为 RGB）
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

##### 4）Dataloader加载数据集

用来加载和存放数据集

```python
dataloader = DataLoader(dataset, batch_size=x， shuffle=False)
"""
dataset-数据
batch_size-每个批次的样本个数
shuffle-是否在每个周期开始时打乱数据
"""
```

#### 3.CNN卷积神经网络

> 可利用**Sequential**对框架进行整合
>
> ```python
> nn.Sequential(
> 	nn.Conv2d(3, 32, 5, padding=2),
>     nn.MaxPool2d(2),
>     nn.Conv2d(32, 32, 5, padding=2),
>     nn.MaxPool2d(2),
>     nn.Conv2d(32, 64, 5, padding=2),
>     nn.MaxPool2d(2),
>     nn.Flatten(),
>     nn.Linear(1024, 64),
>     nn.Linear(64, 10)
> )
> ```

##### 1）Container建立神经网络基本骨架（Module）

```python
import torch
from torch import nn

class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()

    def forward(self, input): # 运算步骤
        return input + 1

mark = Mark()
x = torch.tensor(1.0)
output = mark(x)
print(output)
```

##### 2）Conv2d实现卷积操作

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100911207.png)

```python
# 用reshape来改变类型 (样本数，通道数，行数，列数)
"""
stride-步长（卷积核移动速度）
padding-填充（补齐外环）
dilation-空洞（空洞卷积）
卷积后大小-size = (h - kernel_size + 2*padding)/stride + 1
"""

# torch.nn.function()
import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output_stride = F.conv2d(input, kernel, stride=1)
output_padding = F.conv2d(input, kernel, padding=1)
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100911935.png)

```python
# torch.nn()
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()
        self.conv1 = Conv2d(3, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


dataset = torchvision.datasets.CIFAR10("../nn_conv2d_data", train=False,transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

mark = Mark()
for data in dataloader: # 循环对每一个图像进行卷积操作
    imgs, targets = data
    output = mark(imgs)
    print(output)
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100913754.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100913327.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100913313.png)

##### 3）MaxPool2d实现最大池化盒

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100914539.png)

> 该图stride=3

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100914584.png)

```python
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))


class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()
        # 步长默认为池化层的大小
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

mark = Mark()
output = mark(input)
```

##### 4）ReLu实现非线性变化

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100915389.png)

```python
Mark = nn.ReLu()
input = torch.randn(2)
output = Mark(input)
```

##### 5）Linear实现BP神经网络

```python
# flatten函数摊平张量
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
torch.flatten(t)
# tensor([1, 2, 3, 4, 5, 6, 7, 8])
```

```python
# nn.Linear函数实现类似与BP神经网络 
# Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
nn.dropout(0.5)
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
# torch.Size([128, 30])
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100915081.png)

##### 6）损失函数和反向传播（loss）

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101001888.png)

> 调用的时候梯度会自动储存到.grad属性中
>
> 用 optimizer.step() 来对原参数进行修改

```python
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))
loss1 = nn.L1Loss()
loss2 = nn.MSELoss()
loss3 = nn.CrossEntropyLoss()

# nn.L1Loss求差和
	L = (yi-xi)
	result_L1 = loss1(reduction="方法") # sum-求和 mean-求平均
    
# nn.MSELoss求方差 (适合线性回归问题)
	L = (yi-xi)^2
    result_MSE = loss2(reduction="方法")
    
# nn.CrossEntropyLoss求交叉熵 (适合逻辑分类问题)
	x = torch.tensor([0.1, 0.2, 0.3])
	y = torch.tensor([1])
	x = torch.reshape(x, (1, 3))
    result_Cross = loss3(x, y)
    
# backward反向传播函数求偏导（梯度grad）
	result.backward()
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100915086.png)

##### 7）优化器的配置（optim）

```python
# torch.optim
"""
parameters-模型参数
lr-学习率
"""

loss = nn.CrossEntropyLoss()
mark = Mark()
optim = torch.optim.SGD(mark.parameters(), lr=0.01)	# 设置优化器
for data in dataloader:
    imgs, targets = data
    outputs = mark(imgs)
    result_loss = loss(outputs)		# 计算损失值
    optim.zero_grad() 				# 必不可少！！！！！重构梯度
    result_loss.backward() 			# 计算梯度
    optim.step(）			        # 反向传播
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100916215.png)

##### 8）网络模型的保存和读取

###### 保存

```python
# 方法一
	torch.save("模型名", "路径")
    """
    陷阱
    必须在load的时候将模型的类放在load的代码中
    from model_save import *
    """
    
# 方法二（官方推荐）
	torch.save(模型名.state_dict(), "路径")
```

###### 加载

```python
# 方法一（配合保存方法一）
	model = torch.load("路径")
    
# 方法二
	vgg16 = torchvision.models.vgg16(pretrained=False) # 是否提前训练模型
    vgg16.load_state_dict(torch.load("地址"))
```



