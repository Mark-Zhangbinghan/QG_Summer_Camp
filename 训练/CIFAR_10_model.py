import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()
        """
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.max_pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)
        """
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


mark = Mark()
input = torch.ones((64, 3, 32, 32))
output = mark(input)

writer = SummaryWriter("../logs")
writer.add_graph(mark, input)
writer.close()
