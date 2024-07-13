## Transformer机制

[TOC]

### 一、Attention机制

Attention机制的三大特点：

1. 参数少

   模型复杂度跟 [CNN](https://link.zhihu.com/?target=https%3A//easyai.tech/ai-definition/cnn/)、[RNN](https://link.zhihu.com/?target=https%3A//easyai.tech/ai-definition/rnn/) 相比，复杂度更小，参数也更少。所以对算力的要求也就更小。

2. 速度快

   Attention 解决了 RNN 不能并行计算的问题。Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。

3. 效果好

   在 Attention 机制引入之前，有一个问题大家一直很苦恼：长距离的信息会被弱化，就好像记忆能力弱的人，记不住过去的事情是一样的。Attention 是挑重点，就算文本比较长，也能从中间抓住重点，不丢失重要的信息。（类似于LSTM）

#### 1.1 介绍

Attention是一种用于提升==**基于RNN（LSTM或GRU）**==的Encoder + Decoder模型的效果的的机制（Mechanism），一般称为Attention Mechanism。

##### 应用领域：广泛应用于**机器翻译**、**语音识别**、**图像标注（Image Caption）**等很多领域

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131626676.png)

#### 1.2 原理

​	要介绍Attention Mechanism结构和原理，首先需要介绍下Seq2Seq模型的结构。基于RNN的Seq2Seq模型主要由两篇论文介绍，只是采用了不同的RNN模型。两篇文章（分别基于LSTM和GRU）所提出的Seq2Seq模型，想要解决的主要问题是，如何把机器翻译中，变长的输入X映射到一个变长输出Y的问题，其主要结构如图1所示。

##### 1）Encoder-Decoder解析

Encoder-Decoder 这个框架很好的诠释了机器学习的核心思路：将现实问题转化为数学问题，通过求解数学问题，从而解决现实问题。

Encoder 又称作编码器。它的作用就是「将现实问题转化为数学问题」
Decoder 又称作解码器，他的作用是「求解数学问题，并转化为现实世界的解决方案」

缺陷：**当输入信息太长时，会丢失掉一些信息。**

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131234406.png)

##### 2）什么是Seq2Seq

Seq2Seq（是 Sequence-to-sequence 的缩写），就如字面意思，输入一个序列，输出另一个序列。这种结构最重要的地方在于输入序列和输出序列的长度是可变的。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131129964.png)

​																	图1 传统的Seq2Seq结构

​	传统的Seq2Seq模型对输入序列X缺乏区分度，因此，2015年，Kyunghyun Cho等人在论文《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中，引入了Attention Mechanism来解决这个问题，他们提出的模型结构如图2所示。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131133794.png)

​																	图2 Attention Mechanism模块图解

### 二、Transformer机制

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131531487.png)

##### 交叉注意力

编码器与解码器并行训练，并在过程中将参数混合，每次得到一个最大可能性的字符所对应的数值作为输出

##### Multi-head Attention（多头注意力机制）

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131724372.png)

##### Positional Encoding（位置编码）

位置编码，防止transformer将所有的TOKEN一起放到模型里面，而错失了语义的前后关系

##### Masked Multi-Head （掩码)

在推理时，解码器上是一个词一个词的生成，只能使该生成词受到之前词的影响（即进行归一化和规范化）

##### Feed Forward（前馈神经网络）

也就是全连接神经网络 与CNN类似 将前面的数据连接起来

##### 残差

残差值为输入与输出合 残差还可以避免梯度消失 使注意力机制中学到的是变化的程度而不是结果

