import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dataset = datasets.CIFAR10("../CIFAR_data", train=False,
                           transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)


class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
mark = Mark()
optim = torch.optim.SGD(mark.parameters(), lr=0.01)
for data in dataloader:
    imgs, targets = data
    outputs = mark(imgs)
    result_loss = loss(outputs)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
