"""学习率调度（StepLR）演示：每隔固定步数将学习率乘以衰减系数 gamma。

StepLR 是最常用的等间隔学习率衰减策略：训练初期用较大学习率快速收敛，
随后周期性地降低学习率，使参数在最优解附近更精细地调整。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR


# 模拟损失函数: 目标函数
def f(w):
    return (w ** 2).dot(torch.tensor([0.05, 1]))


# 执行梯度下降，同时记录参数轨迹和每一步的学习率变化
def desc(w, optimizer, lr_scheduler, iter_num):
    w0_list = []
    w1_list = []

    lr_list = []
    for i in range(iter_num):
        w0_list.append(w[0].item())
        w1_list.append(w[1].item())

        # 记录当前学习率（从优化器的参数组中读取）
        lr_list.append(optimizer.param_groups[0]["lr"])

        # 计算损失
        loss = f(w)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 更新学习率（按调度器策略调整）
        lr_scheduler.step()

    return w0_list, w1_list, lr_list


if __name__ == '__main__':
    # 初始化参数
    w = torch.tensor([-7, 2.0])

    # 定义超参数
    lr = 0.9
    iter_num = 500

    # 定义优化器
    ## sgd
    w_clone = w.clone().requires_grad_(True)
    sgd_optimizer = optim.SGD([w_clone], lr=lr)
    # StepLR：每 step_size 步，学习率乘以 gamma（0.9 -> 0.9*0.7 -> ...）
    lr_scheduler = StepLR(sgd_optimizer, step_size=20, gamma=0.7)

    w0_list, w1_list, lr_list = desc(w_clone, sgd_optimizer, lr_scheduler, iter_num)

    # 左图：参数优化轨迹；右图：学习率随迭代的变化曲线（呈阶梯状下降）
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(w0_list, w1_list, label="SGD")

    ax[1].plot(lr_list)

    # 绘制等高线
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    ax[0].contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()
