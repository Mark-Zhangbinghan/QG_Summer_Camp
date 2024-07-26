## mindspore 基础语法学习

[TOC]

昇思MindSpore总体架构：

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407170927272.png)

```python
# 基本框架
import mindspore
from mindspore import nn
from mindspore import ops  # 主要进行张量计算
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
```

### 一、处理数据集

#### 1.1 Dataset

##### 1.1.1 内置数据集

```python
import numpy as np
from mindspore.dataset import vision
# 内置数据集
from mindspore.dataset import MnistDataset, GeneratorDataset
import matplotlib.pyplot as plt
```

MindSpore的dataset提供了大量的内置数据接口和自定义数据接口。其使用数据处理流水线（Data Processing Pipeline），需指定map、batch、shuffle等操作。

```python
# 提供数据和你想要分组的大小
def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0)
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)
    
    # map()给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transforms, 'label')
    
    # 将连续的数据分为若干批（batch）
    dataset = dataset.batch(batch_size)
    return dataset

# mindspore.dataset内置的数据集可直接用shuffle=True的方法随机数据集
# 外部数据可用
dataset = dataset.shuffle(buffer_size=64)
```

对数据集进行迭代访问采用create_tuple_iterator()或create_dict_iterator()

```python
# 提取所有类型
for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break
    
# 存储一个字典类型参数
for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break
```

##### 1.1.2 自定义数据集

通过`GeneratorDataset`接口实现自定义方式的数据集加载，其支持通过可随机访问数据集对象、可迭代数据集对象和生成器(generator)构造自定义数据集，下面分别对其进行介绍

```python
dataset = GeneratorDataset(source=loader, column_names=["data"])
```

#### 1.2 Transforms

##### 1.2.1 对图像数据

> `mindspore.dataset.vision`模块提供一系列针对图像数据的Transforms。在Mnist数据处理过程中，使用了`Rescale`、`Normalize`和`HWC2CHW`变换。

```python
from mindspore.dataset import transforms, vision
from mindspore.dataset import GeneratorDataset, MnistDataset
```

这里通过Compose函数，对数据处理器进行一个集成，再用map函数对数据进行处理

```python
composed = transforms.Compose(
    [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
)

train_dataset = train_dataset.map(composed, 'image')
```

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407171125727.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407171128136.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407171129464.png)

##### 1.2.2 对文本数据

```python
from mindspore.dataset import transforms, text
```

`mindspore.dataset.text`模块提供一系列针对文本数据的Transforms。与图像数据不同，文本数据需要有分词（Tokenize）、构建词表、Token转Index等操作

```python
texts = ['Welcome to Beijing']
test_dataset = GeneratorDataset(texts, 'text')

# 分词（Tokenize）操作是文本数据的基础处理方法
# 这里我们选择基础的PythonTokenizer举例，此Tokenizer允许用户自由实现分词策略。随后我们利用map操作将此分词器应用到输入的文本中，对其进行分词。
def my_tokenizer(content):
    return content.split()
test_dataset = test_dataset.map(text.PythonTokenizer(my_tokenizer))

# Lookup为词表映射变换，用来将Token转换为Index。
# 在使用Lookup前，需要构造词表，一般可以加载已有的词表，或使用Vocab生成词表。这里我们选择使用Vocab.from_dataset方法从数据集中生成词表。
vocab = text.Vocab.from_dataset(test_dataset)
test_dataset = test_dataset.map(text.Lookup(vocab))
```

#### 1.3 Tensor

```python
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor
```

##### 1.3.1 创建张量

```python
# 根据数据框架直接生成
data = [1, 0, 1, 0]
x_data = Tensor(data)

# 改变numpy数据
np_array = np.array(data)
x_np = Tensor(np_array)

# 使用init直接初始化（在正常情况下不建议使用init对参数进行初始化）
from mindspore.common.initializer import One, Normal
	# 创建全是1的张量
	tensor1 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
	# 创建标准化分布的张量
	tensor2 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())
    
# 继承另一个张量的属性
from mindspore import ops
x_ones = ops.ones_like(x_data)
x_zeros = ops.zeros_like(x_data)
```

##### 1.3.2 张量的运算

```python
# 普通的算术运算，直接运算即可

# 利用concat从指定维度连接
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

# 利用stack则视为直接创造一个新的维度使两张量连接
output = ops.stack([data1, data2])
```

##### 1.3.3 Tensor和Numpy的变换

```python
t  = Tensor([1, 1, 1, 1, 1, 1])
n = t.asnumpy()
t = Tensor.from_numpy(n)
```

### 二、网络构建

神经网络模型是由神经网络层和Tensor操作构成的，`mindspore.nn`提供了常见神经网络层的实现，在MindSpore中，Cell类是构建所有网络的基类，也是网络的基本单元。一个神经网络模型表示为一个`Cell`，它由不同的子`Cell`构成。使用这样的嵌套结构，可以简单地使用面向对象编程的思维，对神经网络结构进行构建和管理。

```python
import mindspore
from mindspore import nn, ops
```

定义神经网络时，可以继承`nn.Cell`类，在`__init__`方法中进行子Cell的实例化和状态管理，在`construct`方法中实现Tensor操作。

```python
class Network(nn.Cell):
    def __init__(self):
		super().__init__()
        self.flatten = nn.Flatten()
        # 和pytorch不一样，这里用的是SequentialCell()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )
    # construct相当于forward层，对数据进行处理
    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits
```

### 三、模型训练

#### 3.1 定义正向计算函数

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = model(data)  # 经过模型训练
    loss = loss_fn(logits, label)
    return loss, logits
```

#### 3.2 使用value_and_grad()通过函数变换获得梯度计算函数

```python
from mindspore import Parameter

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

#### 3.3 定义训练函数，使用set_train()设置为训练模式，执行正向计算、反向传播和参数优化。

```python
def train(model, dataset):
    size = dataset.get_dataset_size()
    # 训练，启动！
    model.set_train()
    # enumerate是一个读取器，返回序号和每个序号所对应的组
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
```

#### 3.4 设置回调函数保存网络模型和参数

```python
from mindspore.train import CheckpointConfig, ModelCheckpoint

# 设置保存模型的配置信息
config = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 实例化保存模型回调接口，定义保存路径和前缀名
ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)

# 开始训练，加载保存模型和参数回调函数
trainer.train(1, train_dataset, callbacks=[ckpt_callback])
```

### 四、设置评价指标

#### 4.1 自定义Metrics函数

自定义Metrics函数需要继承`mindspore.train.Metric`父类，并重新实现父类中的`clear`方法、`update`方法和`eval`方法。

```python
import numpy as np
import mindspore as ms

class MyMAE(ms.train.Metric):
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """初始化变量_abs_error_sum和_samples_num"""
        self._abs_error_sum = 0  # 保存误差和
        self._samples_num = 0    # 累计数据量

    def update(self, *inputs):
        """更新_abs_error_sum和_samples_num"""
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        # 计算预测值与真实值的绝对误差
        abs_error_sum = np.abs(y - y_pred)
        self._abs_error_sum += abs_error_sum.sum()

        # 样本的总数
        self._samples_num += y.shape[0]

    def eval(self):
        """计算最终评估结果"""
        return self._abs_error_sum / self._samples_num
    
error = MyMAE()			# 模型赋值
error.clear()  			# 清空误差
error.update(y_pred, y)	# 更新误差
result = error.eval()	# 得到评价指标
```

