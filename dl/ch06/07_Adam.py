import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


# 模拟损失函数: 目标函数
def f(w):
    return (w ** 2).dot(torch.tensor([0.05, 1]))


def desc(w, optimizer, iter_num):
    w0_list = []
    w1_list = []
    for i in range(iter_num):
        w0_list.append(w[0].item())
        w1_list.append(w[1].item())

        # 计算损失
        loss = f(w)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

    return w0_list, w1_list


if __name__ == '__main__':
    # 初始化参数
    w = torch.tensor([-7, 2.0])

    # 定义超参数
    lr = 0.1
    iter_num = 1000

    # 定义优化器
    ## sgd
    w_clone = w.clone().requires_grad_(True)
    sgd_optimizer = optim.SGD([w_clone], lr=lr)
    w0_list, w1_list = desc(w_clone, sgd_optimizer, iter_num)
    print(w0_list, w1_list)
    plt.plot(w0_list, w1_list, label="SGD")

    ## adagrad
    w_clone = w.clone().requires_grad_(True)
    sgd_optimizer = optim.Adam([w_clone], lr=lr, betas=(0.9, 0.999))
    w0_list, w1_list = desc(w_clone, sgd_optimizer, iter_num)

    plt.plot(w0_list, w1_list, label="Adam", color="red")

    # 绘制等高线
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    plt.contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()
