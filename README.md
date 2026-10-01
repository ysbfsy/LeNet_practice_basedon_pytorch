# LeNet Fashion-MNIST 实战项目

这是一个使用 PyTorch 手写 LeNet 风格卷积神经网络的入门项目。项目在 Fashion-MNIST 数据集上完成服饰图像分类，并提供模型训练、测试、网络结构查看和卷积核可视化脚本。

## 项目特点

- 使用单通道 `28 x 28` 灰度图作为输入。
- 使用两层卷积、平均池化和三层全连接层构成 LeNet 网络。
- 使用 Adam 优化器和交叉熵损失函数训练模型。
- 自动划分训练集和验证集，并绘制 loss、accuracy 曲线。
- 支持保存 PyTorch 权重和导出 ONNX 模型。
- 支持查看训练后的卷积核及其数值。

## 网络结构

```text
输入图像: 1 x 28 x 28
    ↓
Conv2d(1 → 6, 5 x 5, padding=2) + Sigmoid
    ↓
AvgPool2d(2 x 2, stride=2)
    ↓
Conv2d(6 → 16, 5 x 5) + Sigmoid
    ↓
AvgPool2d(2 x 2, stride=2)
    ↓
Flatten(16 x 5 x 5 = 400)
    ↓
Linear(400 → 120) → Linear(120 → 84) → Linear(84 → 10)
```

模型最后输出 10 个类别的 logits，类别对应 Fashion-MNIST 的 10 个标签。

## 目录说明

| 文件或目录 | 作用 |
| --- | --- |
| `model.py` | 定义 LeNet 网络，并打印模型结构摘要 |
| `model_train.py` | 下载数据、划分训练/验证集、训练模型、绘制训练曲线并导出 ONNX |
| `model_test.py` | 加载权重，在测试集上推理并打印预测类别与真实类别 |
| `plot.py` | 随机显示 Fashion-MNIST 图片及其类别 |
| `visual_kernel.py` | 可视化第一层和第二层卷积核 |
| `visual_kernel_2.py` | 可视化第一层卷积核，并标注权重数值 |
| `visual_kernal3.py` | 可视化第二层卷积核，并标注权重数值 |
| `data/FashionMNIST` | Fashion-MNIST 数据集缓存目录 |
| `best_model.pth` | PyTorch 模型参数文件 |
| `best_model_weights.pth` | 训练后保存的模型参数文件 |
| `lenet_netron.onnx` | 导出的 ONNX 模型 |
| `Pictures` | 项目运行过程中保存的图片和可视化结果 |

## 环境安装

建议使用 Python 3.9 或更高版本，并在项目目录中执行：

```bash
cd C:\LeNet
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision torchsummary sympy pandas numpy matplotlib onnx
```

如果已经安装了 CUDA 版本的 PyTorch，请根据本机 CUDA 版本从 PyTorch 官网选择对应的安装命令。没有 GPU 时，代码会自动使用 CPU。

## 快速开始

### 1. 查看模型结构

```bash
python model.py
```

程序会打印当前设备以及 LeNet 的参数摘要。

### 2. 训练模型

```bash
python model_train.py
```

首次运行时会自动下载 Fashion-MNIST。训练脚本默认使用 batch size `128`、学习率 `0.001` 和 `30` 个 epoch。训练完成后会保存：

- `best_model.pth`
- `best_model_weights.pth`
- `lenet_netron.onnx`

同时会显示训练集和验证集的 loss、accuracy 曲线。

### 3. 测试模型

```bash
python model_test.py
```

脚本会加载 `best_model.pth`，遍历测试集并输出每张图片的预测类别和真实类别。文件中的 `test_model_process()` 函数可用于计算整个测试集的准确率；如需使用，可在 `model_test.py` 的主程序中调用该函数。

### 4. 查看数据集样本

```bash
python plot.py
```

程序会显示一个批次的 Fashion-MNIST 图片及类别名称。

### 5. 可视化卷积核

```bash
python visual_kernel.py
python visual_kernel_2.py
python visual_kernal3.py
```

运行前请确认项目目录中存在 `best_model_weights.pth`。`visual_kernal3.py` 还会生成高分辨率图片 `conv2_kernel_with_num.png`。

## Fashion-MNIST 类别

| 标签 | 类别 |
| ---: | --- |
| 0 | T-shirt/top（T恤/上衣） |
| 1 | Trouser（裤子） |
| 2 | Pullover（套衫） |
| 3 | Dress（连衣裙） |
| 4 | Coat（外套） |
| 5 | Sandal（凉鞋） |
| 6 | Shirt（衬衫） |
| 7 | Sneaker（运动鞋） |
| 8 | Bag（包） |
| 9 | Ankle boot（短靴） |

## 使用 Netron 查看 ONNX 模型

可以使用 [Netron](https://netron.app/) 打开 `lenet_netron.onnx`，查看网络层级、输入输出形状和计算图。

## 路径注意事项

当前训练和测试脚本中包含 `C:/LeNet/` 绝对路径，例如保存权重、加载权重和导出 ONNX 文件。如果将项目移动到其他目录，请将这些路径改为新的项目路径，或改成相对路径后再运行。

## 后续可以尝试的改进

- 使用 `transforms.Normalize` 对输入进行标准化。
- 增加随机种子，使训练结果更容易复现。
- 将测试流程改为输出整个测试集的准确率和混淆矩阵。
- 将保存路径、epoch、batch size 和学习率改为命令行参数。
- 对比 Sigmoid、ReLU、SGD 和 Adam 对训练效果的影响。

## 许可证

本项目主要用于学习和实验，代码可按个人需要修改和使用。
