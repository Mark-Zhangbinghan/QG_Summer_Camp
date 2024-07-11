import torch
from torch import nn


class Mark(nn.Module):
    def __init__(self):
        super(Mark, self).__init__()

    def forward(self, input):
        return input + 1

mark = Mark()
x = torch.tensor(1.0)
output = mark(x)
print(output)
