from torchvision.datasets import FashionMNIST
from torchvision import transforms
#transforms用来处理数据集

import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt



train_data  = FashionMNIST(root='./data',
                           train=True,
                           transform = transforms.Compose([transforms.Resize(size = 224),transforms.ToTensor()]) ,#调整尺寸并归一化
                           download = True
                           )#导入训练集


train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,#批处理
                               shuffle=True,#打乱
                               num_workers=0,
                               )
#获得一个batch的数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy() #将四维张量中大小等于1的维度删掉，并转换成Numpy数组
batch_y = b_y.numpy() # 将张量转换成Numpy 数组
class_label = train_data.classes #训练集的标签
print(class_label)


# ============下面补充可视化绘图================
plt.figure(figsize=(12,12))
for i in range(64):
    ax = plt.subplot(8,8,i+1)
    #取第i张图片：去掉通道那一维，转为numpy数组
    img_np = b_x[i,0,:,:].numpy()
    plt.imshow(img_np, cmap="gray")
    plt.title(class_label[batch_y[i]], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()