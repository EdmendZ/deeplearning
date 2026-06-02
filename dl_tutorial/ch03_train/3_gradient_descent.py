import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_gradient    # 数值微分计算梯度

# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())   # 保存当前点到列表
        grad = numerical_gradient(f, x)   # 计算梯度
        x = x - lr * grad    # 更新点
    return x, np.array(x_history)

# 定义目标函数：f(x1, x2) = x1 ^ 2 + x2 ^ 2
def f(x):
    return x[0] ** 2 + x[1] ** 2

# 主流程
if __name__ == '__main__':
    # 1. 定义初始点
    init_x = np.array([-3.0, 4.0])

    # 2. 定义超参数
    lr = 0.1
    step_num = 20

    # 3. 梯度下降法计算最小值点
    x, x_history = gradient_descent(f, init_x, lr, step_num)

    # 画图
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.xlim([-4, 4])
    plt.ylim([-4.5, 4.5])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()