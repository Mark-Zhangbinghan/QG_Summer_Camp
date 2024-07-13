import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss1 = nn.L1Loss()
loss2 = nn.MSELoss()
loss3 = nn.CrossEntropyLoss()

result_L1 = loss1(inputs, targets)
result_MSE = loss2(inputs, targets)
print(result_L1)
print(result_MSE)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
result_Cross = loss3(x, y)
print(result_Cross)
