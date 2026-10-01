import torch
import matplotlib.pyplot as plt
from model import LeNet

# 加载模型与训练好的权重
model = LeNet()
state_dict = torch.load(r"C:\LeNet\best_model_weights.pth", weights_only=False)
model.load_state_dict(state_dict)

#取出第一层卷积权重 c1.weight  shape:[6,1,5,5]
conv1_kernel = model.c1.weight.detach().cpu()

#绘图：6个卷积核，画2行3列
plt.figure(figsize=(6,4))
for idx in range(6):
    plt.subplot(2,3,idx+1)
    # 取第idx个卷积核，单通道5*5
    one_kernel = conv1_kernel[idx,0,:,:]
    plt.imshow(one_kernel, cmap='coolwarm')   # 修改这里颜色映射！
    plt.title(f"kernel {idx+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 第二层的16个卷积核
conv2_kernel = model.c3.weight.detach().cpu()
plt.figure(figsize=(8,6))
for idx in range(16):
    plt.subplot(4,4,idx+1)
    #第二层输入通道是6，这里我们只取其中第0输入通道可视化
    one_kernel = conv2_kernel[idx,0,:,:]
    plt.imshow(one_kernel, cmap='coolwarm')
    plt.title(f"kernel2 {idx+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
