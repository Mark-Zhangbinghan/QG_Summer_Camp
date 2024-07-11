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


dataset = torchvision.datasets.CIFAR10("../CIFAR_data", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)
mark = Mark()
for data in dataloader:
    imgs, targets = data
    output = mark(imgs)
    print(output)
