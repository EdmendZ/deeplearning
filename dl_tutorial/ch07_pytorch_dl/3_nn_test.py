import torch
import torch.nn as nn

from torchsummary import summary   # 统计模型结构和参数信息

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实现自定义神经网络类
class Model(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        # 定义全连接层，并做参数初始化
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2)
    # 前向传播
    def forward(self, x):
        # 隐藏层1
        x = self.linear1(x)
        x = torch.tanh(x)
        # 隐藏层2
        x = self.linear2(x)
        x = torch.relu(x)
        # 输出层
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x

# 定义输入数据
x = torch.randn(10, 3, device=device)

# 创建模型（三层神经网络）
model = Model().to(device=device)

# 前向传播
output = model(x)

print(output)

# 查看模型参数
# 1. 遍历 named_parameters 查看各层参数
for name, param in model.named_parameters():
    print(name, param)
    print()

# 2. 查看状态字典 state_dict
print("模型状态字典：\n", model.state_dict())

# 3. 统计信息
summary(model, input_size=(3,), batch_size=10, device=device)