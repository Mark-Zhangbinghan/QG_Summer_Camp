## RNN神经网络 (Recurrent Neural Network)

[TOC]

#### 编解码

将输入的一句话，通过语义投射到一个潜空间中，将高维空间的对象投射到低维空间(embedding嵌入)

利用将语义数字化后的数值，来体现语义之间的相对关系，如下图：

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131418883.png)

##### 分词器缺陷：

1）将所有的TOKEN投射到一维的空间，导致空间内的信息过于密集，很难表达出复杂的语义和词语之间相互的关系
2）假设苹果和香蕉TOKEN分别为1和2，当我们想把它们同时应用时，但是(1+2=)3这个TOKEN已经被梨给占用了

##### one-hot(独热编码)缺陷：

1）所有的TOKEN都是一个独立的维度，故所有的TOKEN空间关系都是相同的，也不能表达出其语义关系，并没有充分的把空间的长度利用起来

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131504809.png)

#### 一、介绍

RNN用来预测序列数据，一个序列后面的输出与前面的输出也有联系。具体应用在于网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的数据是有连接的，并且隐藏层的数据不仅包括输入层的输出还包括上一时刻隐藏层的输出。理论上，RNN能对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关。

RNN的工作原理可以通过以下数学方程表示：

- 输入到隐藏层的转换：$h_t = \tanh(W_{ih} \cdot x_t + b_{ih} + W_{hh} \cdot h_{t-1} + b_{hh}) $
- 隐藏层到输出层的转换：$y_t = W_{ho} \cdot h_t + b_o $

**RNN的重要特点：每一步的参数共享（每一步使用的参数U、W、b都是一样的）**

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101945778.png)

> RNN的多结构

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131501957.png)

#### 二、关键特点与挑战

##### 1）参数共享

RNN在不同的时间步中共享参数，所以其学习到的权重和偏差在处理每个输入时都是相同的。与前馈神经网络不同的是，后者在每一层使用不同的参数。这种参数共享不仅降低了模型的复杂性和计算需求，也帮助模型在处理任何长度的序列时都保持有效。

##### 2）长期依赖问题

在理想状态下，RNN能够使用其“记忆”来连接序列中相隔较远的信息。然而，在实际应用中，RNN往往难以学习这些长距离依赖。这主要是因为梯度消失和梯度爆炸所导致的问题，即在训练过程中，用于更新网络权重的梯度可能变的特别小(消失)、大(爆炸)。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407111024254.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407130839547.png)

##### 3）门控机制（LSTM和GRU）

用长短期记忆网络(LSTM)和门控循环单元(GRU)，来解决长期依赖问题。这些结构通过精心设计的门控系统来控制信息的流入和流出，这有助于网络保留长期信息，同时避免梯度消失和爆炸。



![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407111108843.png)



## LSTM（长短时记忆）

LSTM（长短时记忆网络）是一种常用于处理序列数据的深度学习模型，与传统的 RNN（循环神经网络）相比，LSTM引入了三个门（ 输入门、遗忘门、输出门，如下图所示）和一个 细胞状态（cell state），这些机制使得LSTM能够更好地处理序列中的长期依赖关系。注意：小蝌蚪形状表示的是sigmoid激活函数

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407130842967.png)

> Ct是细胞状态（记忆状态）， **![img](https://img-blog.csdnimg.cn/img_convert/50c2c363af76413783b8c0f2ad5e69c9.png)**是输入的信息， ![img](https://img-blog.csdnimg.cn/img_convert/07e0f8c769dd486c8f60cc916ce9b09c.png)是隐藏状态（基于 ![img](https://img-blog.csdnimg.cn/img_convert/add5dc8f58844291b1dbf5c2b208c607.png)得到的）

#### 实现

祥见  ->  "C:\Users\24468\Desktop\QG人工智能组\Python学习\实战\LSTM.py"



