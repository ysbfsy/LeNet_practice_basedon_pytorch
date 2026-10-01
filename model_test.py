import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import LeNet

def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              # 调整尺寸并归一化
                              download=True
                              )  # 导入训练集

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)



    return test_dataloader

test_dataloader = test_data_process()
def test_model_process(model,test_dataloader):
    #定义设备：
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #将模型放入设备中
    model = model.to(device)
    #初始化一些参数
    test_corrects = 0.0
    test_num = 0.0

    #模型推理：梯度置为0
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:

            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            #将模型设为评估模式
            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)

            test_corrects += (pre_lab == test_data_y).sum()
            #将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

if __name__ == '__main__':
    model = LeNet()
    model.load_state_dict(torch.load('C:/LeNet/best_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    #test_model_process(model,test_dataloader)

    #设定测试所用设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for b_x , b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #模型设为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            result = pre_lab.item()

            label = b_y.item()

            print("预测值：",classes[result],"-------------","真实值",classes[label])



















