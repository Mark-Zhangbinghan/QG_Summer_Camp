## 卷积神经网络及其变种

> 在深度学习领域，卷积神经网络（Convolutional Neural Networks, CNNs）作为一种经典的神经网络结构，以其在图像处理、计算机视觉和语音识别等领域的卓越性能而闻名。CNNs在图像处理中引领了一系列革命性的变革，例如图像分类、物体检测、语义分割等任务，取得了显著的成果。随着深度学习的快速发展，各种卷积神经网络的变种也应运而生，进一步提升了模型的性能。本篇博客将深入探讨卷积神经网络及其变种的原理，并通过实际案例和代码演示，展示其强大的能力和广泛的应用。

#### 一、LeNet-5 分析手写数字

LeNet-5是Yann Lecun等人于1998年提出的卷积神经网络模型，被广泛应用于手写数字识别等任务。==LeNet-5的结构相对简单，包含了两个卷积层和三个全连接层。卷积层使用了5x5的卷积核，并通过sigmoid激活函数引入了非线性。池化层则使用了2x2的最大池化操作，降低了特征图的尺寸。==LeNet-5通过在卷积层和全连接层之间交替使用卷积和池化操作，从而实现了对输入图像的特征提取和分类。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100906857.png)

#### 二、AlexNet （更多卷积层和全连接层）

AlexNet是由Alex Krizhevsky等人于2012年提出的卷积神经网络模型，是第一个在ImageNet图像识别比赛中取得优胜的模型。==AlexNet相比LeNet-5更加深层，并包含了多个卷积层和全连接层。AlexNet使用了ReLU激活函数，并引入了Dropout和数据增强等技术，从而进一步提高了模型的性能和鲁棒性。==
![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100926578.png)

#### 三、VGGNet（加深版的AlexNet）

VGGNet是由Karen Simonyan和Andrew Zisserman于2014年提出的卷积神经网络模型，以其深度和简洁的结构而闻名。VGGNet通过多次堆叠3x3的卷积层和2x2的最大池化层来提取特征，从而增加了网络的深度。VGGNet的结构相对一致，使得其更加容易理解和扩展

VGGNet可以看成是加深版的AlexNet，把网络分成了5段，每段都把多个尺寸为3×3的卷积核串联在一起，每段卷积接一个尺寸2×2的最大池化层，最后面接3个全连接层和一个softmax层，所有隐层的激活单元都采用ReLU函数。

VGGNet包含很多级别的网络，深度从11层到19层不等。为了解决初始化（权重初始化）等问题，==VGG采用的是一种Pre-training的方式，先训练浅层的的简单网络VGG11，再复用VGG11的权重初始化VGG13，如此反复训练并初始化VGG19，能够使训练时收敛的速度更快。==比较常用的是VGGNet-16和VGGNet-19。VGGNet-16的网络结构如下图所示：

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101016012.png)

#### 四、GoogLeNet（利用1×1的卷积核进行降维）

GoogLeNet是由Google团队于2014年提出的卷积神经网络模型，以其高效的网络结构而闻名。GoogLeNet引入了"Inception"模块，==通过并行使用不同尺寸的卷积核和池化操作，并在网络中进行了混合，从而在保持网络深度较浅的同时，提高了网络的感受野和特征提取能力==。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101110835.png)

#### 五、ResNet（使用[Batch Normalization](https://blog.csdn.net/weixin_44023658/article/details/105844861)加速训练）

ResNet是由Kaiming He等人于2015年提出的卷积神经网络模型，==以其深层的结构和"残差"连接而闻名。ResNet通过引入"跳跃连接"，允许网络中的信息在跳过一些层时保持不变，从而解决了深层网络中的梯度消失和网络退化问题==。ResNet的结构更加深层，从而可以进一步提高网络的性能。


![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101121546.png)

> （太深层了...）