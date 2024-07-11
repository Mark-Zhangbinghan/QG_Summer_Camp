import torchvision
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     # 卷积和池化层1
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.conv2 = nn.Sequential(     # 卷积和池化层2
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc1 = nn.Sequential(   # 全连接操作
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(   # 全连接操作
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)    # 高斯连接操作

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


trans_tensor = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='pytorch_MNIST',
                                        train=True,
                                        download=True,
                                        transform=trans_tensor)
test_data = torchvision.datasets.MNIST(root='pytorch_MNIST',
                                       train=False,
                                       download=True,
                                       transform=trans_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
loss_fuc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
Epoch = 8
for epoch in range(Epoch):
    sum_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fuc(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0

    correct = 0
    total = 0
    for data in test_loader:
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs = net(test_inputs)
        _, predicted = torch.max(test_outputs, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum()
    print('第{}个epoch的识别准确率为：{}%'.format(epoch + 1, 100 * correct.item() / total))
torch.save(net.state_dict(), 'MNIST.mdl')
