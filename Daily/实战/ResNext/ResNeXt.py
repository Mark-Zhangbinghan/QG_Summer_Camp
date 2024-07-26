# 建立层数为18和34时的残差基本结构块
import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.downsample(downsample)
        )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.left(x)
        out += identity        # resnet的核心
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=32):
        super(Bottleneck, self).__init__()
        width = int((width_per_group / 64.) * out_channel) * groups
        self.conv1 = nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self,
                 block,             # 表示block的类型
                 blocks_num,        # 表示每一层block的数量
                 num_classes=5,     # 表示类别
                 include_top=True,  # 表示是否有分类层
                 groups=1,          # 表示组卷积的个数
                 width_per_group=64):
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四组残差块创建
        # 第一组
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 64 -> 128
        # 第二组
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 128 -> 256
        # 第三组
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 256 -> 512
        # 第四组
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 512 -> 1024
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channels = channel * block.expansion

        for i in range(1, block_num):
            layers.append(block(self.in_channels,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def ResNet34(num_classes=5, include_top=True):
    return ResNeXt(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet50(num_classes=5, include_top=True):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet101(num_classes=5, include_top=True):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


# 论文中的ResNeXt50_32x4d
def ResNeXt50_32x4d(num_classes=5, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNeXt(Bottleneck, [3, 4, 6, 3],
                   num_classes=num_classes,
                   include_top=include_top,
                   groups=groups,
                   width_per_group=width_per_group)


def ResNeXt101_32x8d(num_classes=5, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNeXt(Bottleneck, [3, 4, 23, 3],
                   num_classes=num_classes,
                   include_top=include_top,
                   groups=groups,
                   width_per_group=width_per_group)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNeXt50_32x4d().to(device)
    summary(model, input_size=(3, 224, 224))

