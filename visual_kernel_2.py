import torch
import matplotlib.pyplot as plt
from model import LeNet

model = LeNet()
state_dict = torch.load(r"C:\LeNet\best_model_weights.pth", weights_only=False)
model.load_state_dict(state_dict)

# ！！！一定要加上 .numpy()
conv1_kernel = model.c1.weight.detach().cpu().numpy()

plt.figure(figsize=(8,6))
for idx in range(6):
    plt.subplot(2,3,idx+1)
    one_kernel = conv1_kernel[idx,0,:,:]
    plt.imshow(one_kernel, cmap='coolwarm')
    plt.title(f"kernel {idx+1}",fontsize=11)
    # 在每一格上面写权重数字，保留2位小数
    for i in range(5):
        for j in range(5):
            val = one_kernel[i,j]
            # 根据背景深浅自动选文字颜色
            txt_color = 'white' if abs(val)>0.25 else 'black'
            plt.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=7,color=txt_color)
    plt.axis('off')

plt.tight_layout()
plt.show()
