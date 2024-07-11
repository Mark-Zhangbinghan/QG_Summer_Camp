import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

tensor_trans = transforms.Compose([
    transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./CIFAR_data', train=True, transform=tensor_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR_data', train=False, transform=tensor_trans, download=True)

writer = SummaryWriter('p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
