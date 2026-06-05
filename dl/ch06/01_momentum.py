"""动量法(Momentum)演示：对比普通 SGD 与带动量的 SGD 的优化轨迹。

通过一个各向异性的二次函数（w0 方向平缓、w1 方向陡峭），可以直观看到：
普通 SGD 在陡峭方向来回震荡，而动量法能累积历史梯度、加速收敛并减小震荡。
"""

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt


# 模拟损失函数：f(w) = 0.05*w0^2 + 1*w1^2（w1 方向梯度更陡，更容易震荡）
def f(w):
    return (w ** 2).dot(torch.tensor([0.05, 1]))


# 执行梯度下降，记录每一步参数 w0、w1 的取值，便于后续绘制优化轨迹
def grad_desc(w, optimizer, iter_num):
    w0_list = []
    w1_list = []

    for i in range(iter_num):
        # 记录当前参数位置
        w0_list.append(w[0].item())
        w1_list.append(w[1].item())

        # 计算损失
        loss = f(w)
        # 反向传播求梯度
        loss.backward()
        # 根据梯度更新参数
        optimizer.step()
        # 梯度清零，避免梯度累加
        optimizer.zero_grad()

    return w0_list, w1_list


if __name__ == '__main__':
    # 1. 初始化参数（起点远离最优解，便于观察收敛过程）
    w = torch.tensor([-7.0, 2.0])
    # 2. 定义超参数
    iter_num = 500     # 迭代次数
    lr = 0.9           # 学习率
    # 3. 定义优化器

    ## 3.1 普通sgd
    # clone().detach() 复制一份独立参数，requires_grad_(True) 开启梯度追踪
    w_clone = w.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([w_clone], lr)
    w0_list, w1_list = grad_desc(w_clone, optimizer, iter_num)
    plt.plot(w0_list, w1_list, color='red', label='SGD')

    ## 3.2 动量法（momentum 累积历史梯度，加速收敛、抑制震荡）
    w_clone = w.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([w_clone], lr, momentum=0.1)
    w0_list, w1_list = grad_desc(w_clone, optimizer, iter_num)
    plt.plot(w0_list, w1_list, color='blue', label='Momentum')



    # 绘制等高线（展示损失函数的形状，越密集表示越陡峭）
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    plt.contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()


    # meshgrid 用法演示：根据两个一维坐标轴生成二维网格坐标矩阵
    a, b = np.meshgrid([1,2], [3,4, 5, 6])
    print(a)
    print(b)
