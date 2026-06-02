import torch
import torch.nn as nn
from torchsummary import summary

# 1. 定义输入数据
x = torch.randn(10, 3)

# 2. 创建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1),
)

# 3. 参数初始化
def init_weights(m):
    # 判断，对所有的全连接层做初始化
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)

model.apply(init_weights)

# 4. 前向传播
output = model(x)

print(output)

summary(model, input_size=(3,), batch_size=10, device='cpu')