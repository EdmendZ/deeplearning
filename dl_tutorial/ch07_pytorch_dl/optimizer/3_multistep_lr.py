import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler

# 定义二元函数 f(x1, x2) = 0.05 x1^2 + x2^2
def f(X):
    return w.dot(X**2)


# 1. 定义数据
# X 初始值 (-7, 2)
X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
w = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)

# 2. 定义超参数
lr = 0.9   # 学习率
n_iter = 500    # 迭代次数

# 3. 定义优化器
optimizer = torch.optim.SGD([X], lr=lr)

# 4. 定义学习率策略
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 200], gamma=0.7)

# 5. 迭代更新 X
X_arr = X.detach().numpy().copy()
lr_list =[]     # 记录学习率变化

for iter in range(n_iter):
    y = f(X)    # 前向传播，计算输出值（损失值）
    y.backward()    # 反向传播
    optimizer.step()    # 更新参数 X
    optimizer.zero_grad()   # 梯度清零

    X_arr = np.vstack((X_arr, X.detach().numpy()))
    lr_list.append(optimizer.param_groups[0]['lr'])
    # 执行学习率衰减
    scheduler.step()

# 画图
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# 画子图1：等高线和梯度下降过程
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
ax[0].contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
ax[0].plot(X_arr[:, 0], X_arr[:, 1], color='red')
ax[0].set_title('梯度下降过程')

# 画子图2：学习率下降曲线
ax[1].plot(lr_list, 'k')
ax[1].set_title('学习率衰减')

plt.show()