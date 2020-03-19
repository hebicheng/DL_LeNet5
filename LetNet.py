import torch
from torch import nn, optim

class LeNet(nn.Module):
    def __init__(self):
       super(LeNet, self).__init__() 
       self.conv1 = nn.Sequential(
           # 与LeCun的数据不同，MNIST的图小大小为28*28
           # 输入为28*28*1的图像数据
           # 卷积核大小为5*5,单通道，深度为6，步长为1
           # 输出为6个24*24的特征图，参数量为24*24*6
           nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1),

           # 平均池化
           # 输出12*12*6
           nn.AvgPool2d(kernel_size=2,stride=2)
       ) 
       self.conv2 = nn.Sequential(
           # 卷积核大小为5*5，通道数6，深度为16，步长为1
           # 输出为16个8*8的特征图，参数量8*8*16
           nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5, stride=1),
           
           # 平均池化
           # 输出4*4*16
           nn.AvgPool2d(kernel_size=2,stride=2)
       )

        # 全连接层
       self.fc1 = nn.Sequential(
           nn.Linear(in_features=4*4*16,out_features=120)
       )
       self.fc2 = nn.Sequential(
           nn.Linear(in_features=120,out_features=84)
       )
       self.fc3 = nn.Sequential(
           nn.Linear(in_features=84,out_features=10)
       )

    # 前向传播
    def forward(self, input):
        conv1_output=self.conv1(input)
        conv2_output = self.conv2(conv1_output)  
        conv2_output=conv2_output.view(-1,4*4*16) # 将16张特征图转为一维
        fc1_output=self.fc1(conv2_output)
        fc2_output=self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)
        return fc3_output