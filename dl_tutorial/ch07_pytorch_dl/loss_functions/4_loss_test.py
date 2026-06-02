import torch
from torch import nn, optim

# 自定义模型类
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=5)
        self.linear.weight.data = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5],
             [0.6, 0.7, 0.8, 0.9, 1.0],
             [1.1, 1.2, 1.3, 1.4, 1.5]]
        ).T
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    def forward(self, x):
        x = self.linear(x)
        return x

# 1. 准备数据
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
t = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)

# 2. 创建模型
model = Model()

# 3. 前向传播，计算预测值
y = model(x)

# 4. 计算损失
loss = nn.MSELoss()
loss_value = loss(y, t)

# 5. 反向传播
loss_value.backward()

# 6. 更新参数（迭代一次）
optimizer = optim.SGD(model.parameters(), lr=1)
optimizer.step()

optimizer.zero_grad()

# 打印参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])