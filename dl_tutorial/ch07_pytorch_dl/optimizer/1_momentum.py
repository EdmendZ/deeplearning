import torch
import numpy as np
import matplotlib.pyplot as plt


# 定义二元函数 f(x1, x2) = 0.05 x1^2 + x2^2
def f(X):
    return w.dot(X**2)

# 定义函数：梯度下降方式迭代，更新参数 X，并保存 X 的变化列表返回
def grad_desc(X, optimizer, n_iters):
    X_arr = X.detach().numpy().copy()

    for iter in range(n_iters):
        y = f(X)    # 前向传播，计算输出值
        y.backward()    # 反向传播
        optimizer.step()    # 更新参数
        optimizer.zero_grad()   # 梯度清零

        X_arr = np.vstack((X_arr, X.detach().numpy()))

    return X_arr

# 定义函数：手动实现动量法迭代过程
def momentum(X, lr, momentum, n_iters):
    X_arr = X.detach().numpy().copy()
    V = torch.zeros_like(X)   # 定义历史梯度累积 V

    for iter in range(n_iters):
        # 计算梯度
        grad = 2 * X * w
        # 代入迭代公式
        V = momentum * V - lr * grad
        X.data += V

        X_arr = np.vstack((X_arr, X.detach().numpy()))

    return X_arr

# 1. 定义数据
# X 初始值 (-7, 2)
X = torch.tensor([-7, 2], dtype=torch.float, requires_grad=True)
w = torch.tensor([0.05, 1.0], dtype=torch.float, requires_grad=True)


# 2. 定义超参数
lr = 0.01   # 学习率
n_iter = 500    # 迭代次数

# 3. 梯度下降寻找最小值
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
X_arr1 = grad_desc(X_clone, optimizer, n_iter)
plt.plot(X_arr1[:, 0], X_arr1[:, 1], color='r')

# 3.2 对比：动量法
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], momentum=0.9, lr=lr)
X_arr2 = grad_desc(X_clone, optimizer, n_iter)
plt.plot(X_arr2[:, 0], X_arr2[:, 1], color='b')

# 3.3 对比：手动实现动量法
X_clone = X.clone().detach().requires_grad_(True)
X_arr3 = momentum(X_clone, lr=lr, momentum=0.9, n_iters=n_iter)
plt.plot(X_arr3[:, 0], X_arr3[:, 1], color='orange',linestyle='--')

# 绘制等高线图
x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, colors='gray', levels=30)
plt.legend(['SGD','Momentum','Manual Momentum'])

plt.show()