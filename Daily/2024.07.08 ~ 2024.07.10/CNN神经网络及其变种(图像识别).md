## 卷积神经网络及其变种

> 在深度学习领域，卷积神经网络（Convolutional Neural Networks, CNNs）作为一种经典的神经网络结构，以其在图像处理、计算机视觉和语音识别等领域的卓越性能而闻名。CNNs在图像处理中引领了一系列革命性的变革，例如图像分类、物体检测、语义分割等任务，取得了显著的成果。随着深度学习的快速发展，各种卷积神经网络的变种也应运而生，进一步提升了模型的性能。本篇博客将深入探讨卷积神经网络及其变种的原理，并通过实际案例和代码演示，展示其强大的能力和广泛的应用。

[TOC]

#### 拓展：深度和宽度的区别：

 ##### 1.1 深度（毕竟是深度学习）

**更深的模型, 意味着更好的非线性表达能力, 可以学习更加复杂的变化, 从而可以拟合更加复杂的输入。**网络加深带来的两个主要的好处, **更强大的表达能力和逐层的特征学习。**每一层需要学习的复杂程度就越小，但是越容易发生梯度消失。

##### 1.2 宽度

**让每一层学习到更加丰富的特征, 比如不同方向, 不同频率的纹理特征。** 比如颜色的地区, 以及颜色变化的情况等。**太窄的网络, 每一层能捕获的模式有限, 此时网络再深都不可能提取到足够的信息往下传递。**

#### 一、LeNet-5 （手写数字识别）

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407132103228.png)

LeNet-5是Yann Lecun等人于1998年提出的卷积神经网络模型，被广泛应用于手写数字识别等任务。==LeNet-5的结构相对简单，包含了两个卷积层和三个全连接层。卷积层使用了5x5的卷积核，并通过sigmoid激活函数引入了非线性。池化层则使用了2x2的最大池化操作，降低了特征图的尺寸。==LeNet-5通过在卷积层和全连接层之间交替使用卷积和池化操作，从而实现了对输入图像的特征提取和分类。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131455863.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100906857.png)

#### 二、AlexNet （更多卷积层和全连接层  图像识别）

AlexNet是由Alex Krizhevsky等人于2012年提出的卷积神经网络模型，是第一个在ImageNet图像识别比赛中取得优胜的模型。==AlexNet相比LeNet-5更加深层，并包含了多个卷积层和全连接层。AlexNet使用了ReLU激活函数，并引入了Dropout和数据增强等技术，从而进一步提高了模型的性能和鲁棒性。==
![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407100926578.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407132109289.png)

#### 三、VGGNet（加深版的AlexNet  图像识别）

VGGNet是由Karen Simonyan和Andrew Zisserman于2014年提出的卷积神经网络模型，以其深度和简洁的结构而闻名。VGGNet通过多次堆叠3x3的卷积层和2x2的最大池化层来提取特征，从而增加了网络的深度。VGGNet的结构相对一致，使得其更加容易理解和扩展

VGGNet可以看成是加深版的AlexNet，把网络分成了5段，每段都把多个尺寸为3×3的卷积核串联在一起，每段卷积接一个尺寸2×2的最大池化层，最后面接3个全连接层和一个softmax层，所有隐层的激活单元都采用ReLU函数。

VGGNet包含很多级别的网络，深度从11层到19层不等。为了解决初始化（权重初始化）等问题，==VGG采用的是一种Pre-training的方式，先训练浅层的的简单网络VGG11，再复用VGG11的权重初始化VGG13，如此反复训练并初始化VGG19，能够使训练时收敛的速度更快。==比较常用的是VGGNet-16和VGGNet-19。VGGNet-16的网络结构如下图所示：

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101016012.png)

#### 四、GoogLeNet（利用1×1的卷积核进行降维  图像识别）

GoogLeNet是由Google团队于2014年提出的卷积神经网络模型，以其高效的网络结构而闻名。GoogLeNet引入了"Inception"模块，==通过并行使用不同尺寸的卷积核和池化操作，并在网络中进行了混合，从而在保持网络深度较浅的同时，提高了网络的感受野和特征提取能力==。将用不同卷积后的结果与池化的结果结合来达到输出，这样达到融合不同尺度的特征，最后的特征图包含不同感受野的信息。

###### 感受野 - 就是指输出feature map上某个元素受输入图像上影响的区域

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101110835.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407131509319.png)

通过1×1的卷积核对通道进行降维，并且通过卷积核来为其提供语义。当然也可进行升维，看卷积数量

#### 五、ResNet（残差网络  使用[Batch Normalization](https://blog.csdn.net/weixin_44023658/article/details/105844861)加速训练  图像识别、图像分割）在分类识别定位等各个赛道碾压之前的各种网络

ResNet是由Kaiming He等人于2015年提出的卷积神经网络模型，==以其深层的结构和"残差"连接而闻名。ResNet通过引入"跳跃连接"，允许网络中的信息在跳过一些层时保持不变，从而解决了深层网络中的梯度消失和网络退化问题==。ResNet的结构更加深层，从而可以进一步提高网络的性能。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407101121546.png)

当层数达到10^3的数量级时，退化的原因不再是因为结构优化的问题了，而是因为层级太深太牛逼，而作者跑的数据集太小太弱，不够这种残差结构打，导致了过拟合。

> （太深层了...）

#### 六、DenseNet（稠密连接网络  使输出层连接  图像识别、图像分割）

DenseNet（密集卷积网络）的核心思想（创新之处）是密集连接，使得每一层都与所有之前的层直接连接，即某层的输入除了包含前一层的输出外还包含前面所有层的输出。（学到LSTM后感觉类似）

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407132042405.png)

#### 七、MobileNets（移动和嵌入式视觉应用）

使用深度可分离卷积来构建轻量级的深度神经网络。这种结构显著减少了模型的大小和计算复杂度，非常适合计算资源有限的设备。

#### 八、EfficientNet（图像识别)

通过系统地研究不同维度（深度、宽度和分辨率）的缩放，提出了一种新的缩放方法，使得网络可以在保持合理计算资源的情况下达到更好的性能。



#### CNN反向传播理解（[详见这位博主的文章](https://blog.csdn.net/qq_45912037/article/details/128073903)）

###### 1.1 前向传播

在理解反向传播之前咱们先看看卷积是如何前向传播的

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407140934402.png)

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407140932803.png)

###### 2.1 卷积层中的反向传播

基于前向传播公式，我们可以利用链式法则进行反向传播计算。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407141020694.png)

![1720922967851](C:\Users\24468\AppData\Roaming\Typora\typora-user-images\1720922967851.png)

由于损失函数不唯一，故在这里并不花大篇幅来说明损失函数**∂L/∂O**结果为多少，详细可查[每个损失函数求偏导结果](https://blog.csdn.net/weixin_57643648/article/details/122704657)。常用的损失函数为MSELoss和CrossEntropyLoss。

如上所示，我们可以找到相对于输出O的局部梯度∂O/∂X和∂O/∂F利用前一层的损失梯度——∂L/∂O,使用链式法则，我们就可以计算出**∂L/∂X**和**∂L/∂F**。（这两项最简单的理解就是每一次卷积所对应的卷积核和感受野的数值）

$∂O/∂X$:

![1720922741464](C:\Users\24468\AppData\Roaming\Typora\typora-user-images\1720922741464.png)

$∂O/∂F$:

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407141005012.png)

在这里padding设置为0，故并不能使每个X都可以被卷积覆盖同样的次数，求导出来的值比较分散，这边列举几项举例

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407141028085.png)

其实这里可以用卷积计算的方式来理解它。我们先将F旋转 180 度，这可以通过先垂直翻转然后水平翻转来完成。

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407141033823.png)

在完成旋转操作后利用Full mode的卷积操作（将padding设置为(卷积核宽度-1)/2）即可求出偏导

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407141038686.png)

同理，也可以将可以将**∂L/∂F**视作输入层与**∂L/∂O**（化作张量）的卷积运算
![1720923469207](C:\Users\24468\AppData\Roaming\Typora\typora-user-images\1720923469207.png)

###### 2.2 池化层中的反向传播

**Max Pooling的反向传播为对于非最大值没有梯度。**
**对于最大值梯度为1**

