import torch
from torch import nn

# 1. 构造输入数据
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),

    nn.Linear(4, 4),
    nn.ReLU(),

    nn.Linear(4, 2),
    nn.Softmax(dim=-1)
)

# 前向传播
#y = model(x)

# 保存模型: 保存模型的参数
#torch.save(model.state_dict(), 'model.pth')

# 加载模型
state_dic = torch.load('model.pth')
model.load_state_dict(state_dic)

y = model(x)
print(y)


