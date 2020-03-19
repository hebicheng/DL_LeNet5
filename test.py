from torchvision import datasets
from torchvision import transforms
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)