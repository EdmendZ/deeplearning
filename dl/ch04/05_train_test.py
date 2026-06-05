import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

epochs = 500
lr = 0.001

# 1. 准备数据
x = torch.randn(100, 1)
y = x * 2.5 + 5 + torch.randn(100, 1) * 0.3

# 2. 数据集和数据加载器
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False)

# 3. 定义模型
model = nn.Linear(1, 1)

# 4. 定义损失函数和优化器
loss_fn = nn.MSELoss()
# 作用: 更新参数和梯度清零
optimizer = optim.SGD(model.parameters(), lr=lr)

# 5. 训练模型
loss_list = []
for epoch in range(epochs):
    loss_total = 0
    for input, target in loader:
        # 5.1 一次向前传播
        y_pred = model(input)

        # 5.2 计算损失
        loss = loss_fn(y_pred, target)

        # 5.3 反向传播
        loss.backward()

        # 5.4 更新参数
        optimizer.step()

        # 5.5 梯度清零(清除叶子节点的梯度)
        optimizer.zero_grad()

        loss_total += loss.item() * input.shape[0]

    loss_list.append(loss_total / len(ds))  # 每个epoch的平均损失

    # 总损失 / 数据数
    # 批次的总损失 / 批次数

print(model.weight, model.bias)

import matplotlib.pyplot as plt
fig, ax =  plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(loss_list)

# 散点图
ax[1].scatter(x, y)
# 拟合曲线
ax[1].plot(x, model(x).detach(), c='r')

plt.show()









