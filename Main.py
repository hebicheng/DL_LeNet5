import torch
from LetNet import LeNet 
from Data_process import data_prapare
from Train import train
from Test import validate
if __name__ == "__main__":
    # 是否使用cuda
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print("Using {} training..".format(device))   
    
    # 初始化模型
    model=LeNet().to(device)
    # 加载数据
    train_loader,test_loader = data_prapare()
    # 训练
    train(train_loader,model,device)
    # 验证
    validate(test_loader, device)