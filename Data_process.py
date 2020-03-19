from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

def data_prapare():

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor())

    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader,test_loader