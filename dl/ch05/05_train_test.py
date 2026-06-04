import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
epochs = 100
lr = 0.01


# 1. 准备数据  y=wx+b
x = torch.randn(100, 1)
y = 2.5 * x + 5.0 + torch.randn(100, 1) * 0.2

# 2. 构建数据集和数据加载器
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False)

# 3. 定义模式
model = nn.Linear(1, 1)

# 4. 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化器: 用来更新参数和清零梯度
opti = optim.SGD(model.parameters(), lr=lr)

loss_list = []
# 5. 训练模型
for epoch in range(epochs):
    for input, target in loader:
        # 5.1 先做一次前向传播
        y_pred = model(input)

        # 5.2 计算损失
        loss = loss_fn(y_pred, target)

        # 5.3 反向传播计算梯度
        loss.backward()

        # 5.4 更新参数
        opti.step()

        # 5.5 清零梯度(叶子节点)
        opti.zero_grad()

        loss_list.append(loss.item())


print(model.weight)
print(model.bias)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(loss_list)

ax[1].scatter(x, y)
ax[1].plot(x, model(x).detach().numpy(), color='red')

plt.show()
