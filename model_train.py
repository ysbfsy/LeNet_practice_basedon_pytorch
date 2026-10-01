from sympy.printing.pytorch import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
#transforms用来处理数据集
import torch.utils.data as Data

import torch.nn as nn

import copy

import time

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
#导入模型
from model import LeNet

#处理训练集和验证集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              # 调整尺寸并归一化
                              download=True
                              )  # 导入训练集
    train_data,val_data = Data.random_split(train_data,[round(0.8 * len(train_data)),round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)

    return train_dataloader, val_dataloader

#模型训练
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #优化器:SGD或Adam （不过本质都是梯度下降法）
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    #损失函数:交叉熵损失函数（一般用于分类）
    criterion = nn.CrossEntropyLoss()
    #将模型放到训练设备中
    model  = model.to(device)
    #赋值当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高准确度
    best_acc = 0.0
    #训练集损失函数列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    #训练集精度列表
    train_acc_all = []
    # 训练集精度列表
    val_acc_all = []
    #当前时间
    since = time.time()
    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch, num_epochs-1))#因为这个训练是训练0-99（num_epochs=100）次
        print("-"*10)

        #初始化参数
        #初始化一些值 损失值和准确度
        train_loss = 0.0
        train_corrects = 0

        #验证集损失函数 和准确度
        val_loss = 0.0
        val_corrects = 0
        #训练集的样本数量
        train_num = 0
        # 验证集的样本数量
        val_num = 0

        #
        #- `b_x` = batch_x： ** 一个批次的输入图像张量（网络的输入数据） **
        #- `b_y` = batch_y： ** 一个批次对应的标签张量（真实答案） **

        for step,(b_x,b_y) in enumerate(train_dataloader):
            #将特征放入到设备中
            b_x = b_x.to(device)
            #将标签放入道设备中
            b_y = b_y.to(device)


            #- **特征 X（也就是你的 b_x）：图片全部像素**
            # 一张 224×224 灰度图，几千上万个像素数值。
            # > 就是图像本身，这是**输入**，机器依靠这些像素信息来做判断。
            # - **标签 Y（也就是你的 b_y）：这张图片是什么东西（数字编号 0‑9）**
            # 比如：`9 → 短靴，0 → T恤`
            # 设置模型为训练模式

            model.train()

            #前向传播。输入为一个batch ,输出为一个batch 中对应的预测
            output = model(b_x)

            # 变量output：神经网络前向传播之后的输出结果（未经过softmax的logits分数）
            # output的张量形状为 [batch_size, 10]
            # 第0维batch_size：代表当前批次里面一共有多少张图片（例如64）
            # 第1维10：代表Fashion‑MNIST数据集一共10个分类，对每一张图像给出10个类别的打分
            # 分数只是网络输出值，分数越高，神经网络判断属于该类别的可能性就越大

            # torch.argmax()函数：作用是返回一组数据中【最大值所在位置的下标（索引）】
            # dim = 1：指定搜索的维度；沿着第1维也就是10个类别这一维度去找最大值
            # 不要写成dim=0，dim=0会沿着样本batch方向查找，逻辑直接出错

            # pre_lab（predict label 预测标签）：保存运算之后的结果
            # pre_lab输出张量形状：[batch_size]，一维张量
            # pre_lab里面每一个整数，就是神经网络自己猜测出来的这张图片所属类别编号
            #查找每一行中最大值对应的行标（其实简单来讲就是预测结果）
            pre_lab = torch.argmax(output,dim = 1)

            loss = criterion(output,b_y)

            # criterion：（应该实在第47行 ）代表我们提前定义好的损失函数（比如交叉熵损失CrossEntropyLoss）
            # output：神经网络前向传播得到的输出logits，张量形状 [batch_size,10]
            #         每一张图片对应10个类别的打分值，还不是0~1的概率
            # b_y：当前批次样本的真实标签(ground‑truth)，形状 [batch_size]
            #         是数据集给的标准答案，保存每张图片真实所属类别的编号

            # loss = criterion(output,b_y)
            # 调用损失函数，把网络预测输出output 和真实标签b_y传入进去
            # 损失函数会计算【预测结果和标准答案之间的差距】
            # loss 得到一个标量(单个数字)，数值越大，说明这一批样本整体预测效果越差；
            # 数值越小，代表网络的预测结果越贴近真实答案。






            #--------------------------------------------------------------------------------
            # optimizer.zero_grad()：梯度清零
            # 原理：PyTorch里面，梯度是【累加】的，不是自动覆盖！
            # loss.backward()：反向传播，计算本次batch的梯度，然后累加到网络参数的.grad属性上面

            # 如果不清零：
            # 第1个batch做完backward → 参数保存这一批算出的梯度grad1
            # 第2个batch再backward()，PyTorch不会自动删掉旧梯度！
            # grad = grad1 + grad2 ，新旧梯度叠加在一起。
            # 相当于把前后多个批次的梯度混在一起，梯度值就错了，参数更新方向完全跑偏，模型训练不会收敛！

            # 训练循环每一个step（每一个batch）的标准顺序：
            # 1.前向传播，算出loss
            # 2.optimizer.zero_grad()   先把上一轮遗留下来的梯度全部清零！
            # 3.loss.backward()         反向传播，**只计算当前这一个batch样本产生的梯度**，存进.grad
            # 4.optimizer.step()        优化器利用这一轮算好的梯度，更新神经网络权重w和b



            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算 (用于计算梯度)
            loss.backward()

            #
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)

            #每个批次中正确预测的数量的
            train_corrects += torch.sum(pre_lab == b_y)

            train_num += b_x.size(0)


        #验证代码的撰写

        #训练：前向→算loss→反向传播→更新参数 ✅改网络权重
        #验证：前向→算loss、准确率 ❌不反向、❌不改任何权重 (不参与训练)

        for step,(b_x,b_y) in enumerate(val_dataloader):
            #将特征放入训练设备中
            b_x = b_x.to(device)
            #将标签放入训练设备中
            b_y = b_y.to(device)
            #开启验证模式（设置模型为评估模式）
            #`model.eval()`：BN、dropout 切换到推理模式
            model.eval()


            #前向传播过程，输入为一个batch,输出为一个batch中的预测
            output = model(b_x)

            # 查找每一行中最大值对应的行标（其实简单来讲就是预测结果）
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)


            val_loss += loss.item() * b_x.size(0)

            # 每个批次中正确预测的数量的
            val_corrects += torch.sum(pre_lab == b_y)

            val_num += b_x.size(0)


        #计算并保存每一次迭代的成本函数和准确率
        #计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        #计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        #计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        #计算并保存验证的准确率（item 是把 张量 变为标量）
        val_acc_all.append(val_corrects.double().item() / val_num)

        #append就是插入最后一位

        # -1就是取列表当中最后一个：
        print('{} Train Loss : {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss : {:.4f} Val Acc: {:.4f}'.format(epoch,val_loss_all[-1], val_acc_all[-1]))

    #寻找最高准确度 的权重参数
    if val_acc_all[-1] > best_acc:
        #保存当前的最高准确度
        best_acc = val_acc_all[-1]
        #保存最好模型的参数
        best_model_wts = copy.deepcopy(model.state_dict())

    #计算一下耗时
    time_use = time.time() - since
    print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))


    #选择最优参数模型
    #加载最高准确率下的模型参数
    torch.save(best_model_wts,'C:/LeNet/best_model.pth')
    torch.save(model.state_dict(), "./best_model_weights.pth")

    # ✅ ONNX导出放在这里！！只导出一次（训练结束之后）
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        r"C:/LeNet/lenet_netron.onnx",
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    #dataframe
    train_process = pd.DataFrame(data = {"epoch":range(num_epochs),
                                         "train_loss_all":train_loss_all,
                                         "val_loss_all":val_loss_all,
                                         "train_acc_all":train_acc_all,
                                         "val_acc_all":val_acc_all})
    return train_process


#loss 和acc图的绘制
def matplot_acc_loss(train_process):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)#一行两列的第一张图
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label = 'train loss')
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bs-',label = 'val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')


    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    #将模型实例化
    LeNet = LeNet()
    train_dataloader ,val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,num_epochs=30)
    matplot_acc_loss(train_process)
































