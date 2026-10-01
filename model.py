import torch
from torch import nn
# 神经网络的框架
from torchsummary import summary


# 神经网络的各种参数

class LeNet(nn.Module):
    # 先做一个初始化
    def __init__(self):
        super(LeNet, self).__init__()
        # 这里是所有网络约定俗成的东西
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                            padding=2)  # out_channels 输出通道，这个就是卷积核的数量 ，kernel_size是卷积核的大小
        self.sigmoid = nn.Sigmoid()
        # 池化层
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 平展层
        self.flatten = nn.Flatten()

        # 线性全连接层
        self.f5 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.sigmoid(self.c1(x))  # 128 6 28 28
        x = self.s2(x)  # 128 6 14 14
        x = self.sigmoid(self.c3(x))  # 128 16 10 10
        x = self.s4(x)  # 128 16 5 5
        x = self.flatten(x)  # 128 400
        x = self.f5(x)  # 128 120
        x = self.f6(x)  # 128 84
        x = self.f7(x)  # 128 10
        return x


# 主函数
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 判断cuda能否用
    print(device)

    model = LeNet().to(device)
    # 把模型放到设备里面
    print(summary(model, (1, 28, 28)))
# 查看模型的相关参数
