
import torch
from torchvision import datasets
from torchvision import transforms
from LetNet import LeNet
def validate(testloader, device):
    # 加载训练好的model
    net = LeNet()
    net = net.to(device)
    net.load_state_dict(torch.load('./model/LeNet.pkl'))

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print("Correct rate: {:.3f}%".format(correct*100/total))
