import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt



def print_loss_grp(epoch, loss_list):
    plt.figure()
    plt.plot(epoch,loss_list,"r-")
    plt.show()

def train(train_loader,model,device,lr=0.001,epoches=100):
    # 多分类问题 一般使用交叉熵损失函数
    loss_fc=nn.CrossEntropyLoss() 
    optimizer=optim.SGD(params=model.parameters(),lr=lr) #采用随机梯度下降SGD
    loss_list=[]#记录每次的损失值

    for epoch in range(epoches): # 迭代
        sum_loss = 0.0
        tmp_sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = model(inputs)
            loss = loss_fc(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            tmp_sum_loss += loss.item()
            if i % 100 == 99:
                print('[{}, {}}] loss: {:.3f}}'
                        .format(epoch + 1, i + 1, tmp_sum_loss / 100))
                tmp_sum_loss = 0.0
        loss_list.append(sum_loss)

    #保存模型参数
    torch.save(model.state_dict(),"./model/LeNet.pkl")
    print('model saved.')
    #展示迭代次数与loss的图
    print_loss_grp(list(range(epoches)), loss_list)


