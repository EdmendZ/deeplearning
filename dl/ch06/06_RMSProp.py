"""自适应学习率优化器（RMSProp）演示：对比普通 SGD 与 RMSProp 的优化轨迹。

RMSProp 是对 AdaGrad 的改进：用梯度平方的指数加权移动平均(alpha 控制衰减)
代替无限累积，避免学习率随训练单调衰减到过小，更适合长时间训练。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


# 模拟损失函数: 目标函数
def f(w):
    return (w ** 2).dot(torch.tensor([0.05, 1]))


# 执行梯度下降，记录参数轨迹
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
    ## sgd（作为对照基线）
    w_clone = w.clone().requires_grad_(True)
    sgd_optimizer = optim.SGD([w_clone], lr=lr)
    w0_list, w1_list = desc(w_clone, sgd_optimizer, iter_num)
    print(w0_list, w1_list)
    plt.plot(w0_list, w1_list, label="SGD")

    ## RMSprop（alpha 为梯度平方移动平均的衰减系数）
    w_clone = w.clone().requires_grad_(True)
    sgd_optimizer = optim.RMSprop([w_clone], lr=lr, alpha=0.99)
    w0_list, w1_list = desc(w_clone, sgd_optimizer, iter_num)

    plt.plot(w0_list, w1_list, label="RMSprop", color="red")

    # 绘制等高线
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    plt.contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()

    # 绘制等高线
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    plt.contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()
