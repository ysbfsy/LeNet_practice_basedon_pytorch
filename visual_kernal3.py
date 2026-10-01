import torch
import matplotlib.pyplot as plt
from model import LeNet

model = LeNet()
state_dict = torch.load(r"C:\LeNet\best_model_weights.pth", weights_only=False)
model.load_state_dict(state_dict)

# 第二层卷积 c3 weight shape [16,6,5,5]，取输入通道0
conv2_kernel = model.c3.weight.detach().cpu().numpy()

# 增大画布尺寸，提高整体绘图空间，4行4列子图
plt.figure(figsize=(12, 10), dpi=150)   # dpi=150提升输出像素清晰度
for idx in range(16):
    plt.subplot(4,4,idx+1)
    one_kernel = conv2_kernel[idx,0,:,:]
    plt.imshow(one_kernel, cmap='coolwarm')
    plt.title(f"kernel2 {idx+1}", fontsize=8)
    # 遍历5×5网格打印权重数值，字号缩小
    for i in range(5):
        for j in range(5):
            val = one_kernel[i,j]
            txt_color = 'white' if abs(val) > 0.25 else 'black'
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color=txt_color)
    plt.axis("off")

plt.tight_layout()
plt.savefig("conv2_kernel_with_num.png", dpi=300)   # 保存图片的时候dpi=300，保存高分辨率图用于报告
plt.show()
