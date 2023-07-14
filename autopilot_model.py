import torch
import torchvision
from efficientnet_pytorch import EfficientNet
from torch2trt import TRTModule
from torch2trt import torch2trt

OUTPUT_SIZE = 2     # 输出类别的数量
DROPOUT_PROB = 0.5  # Dropout 的概率

class AutopilotModel(torch.nn.Module):
    #初始化方法，接受一个参数 pretrained，用于指定是否加载预训练的 ResNet-18 模型作为基础网络
    def __init__(self, pretrained):
        #调用父类 torch.nn.Module 的初始化方法
        super(AutopilotModel, self).__init__()

        # 从 torchvision 加载预训练的 ResNet-18 模型
        # 创建一个 ResNet-18 模型，并将其赋值给 self.network。
        self.network = torchvision.models.resnet18(pretrained=pretrained)
        # 将 ResNet-18 的全连接层替换为自定义的序列模块
        # 替换 self.network 的最后一个全连接层，使用一个由多个线性层和丢弃层组成的序列
        self.network.fc = torch.nn.Sequential(
            
            torch.nn.Dropout(p=DROPOUT_PROB),                                           # 使用指定的概率进行 Dropout
            torch.nn.Linear(in_features=self.network.fc.in_features, out_features=128), # 添加一个线性层
            torch.nn.Dropout(p=DROPOUT_PROB),                                           # 再次应用 Dropout
            torch.nn.Linear(in_features=128, out_features=64),                          # 添加另一个线性层
            torch.nn.Dropout(p=DROPOUT_PROB),                                           # 再次应用 Dropout
            torch.nn.Linear(in_features=64, out_features=OUTPUT_SIZE)                   # 最后一个具有输出类别的线性层

        )
        self.network.cuda()    # 将模型移到 GPU 上以加速计算

    def forward(self, x):
        y = self.network(x) # 模型的前向传播
        return y
    
    def save_to_path(self, path):
        torch.save(self.state_dict(), path)     # 将模型的状态字典保存到指定路径
        
    def load_from_path(self, path):
        self.load_state_dict(torch.load(path))   # 从指定路径加载模型的状态字典   
        
