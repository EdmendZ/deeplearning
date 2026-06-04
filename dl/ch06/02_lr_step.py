import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR


# 模拟损失函数: 目标函数
def f(w):
    return (w ** 2).dot(torch.tensor([0.05, 1]))


def desc(w, optimizer, lr_scheduler, iter_num):
    w0_list = []
    w1_list = []

    lr_list = []
    for i in range(iter_num):
        w0_list.append(w[0].item())
        w1_list.append(w[1].item())

        lr_list.append(optimizer.param_groups[0]["lr"])

        # 计算损失
        loss = f(w)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 更新学习率
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
    lr_scheduler = StepLR(sgd_optimizer, step_size=20, gamma=0.7)

    w0_list, w1_list, lr_list = desc(w_clone, sgd_optimizer, lr_scheduler, iter_num)

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
