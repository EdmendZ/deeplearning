import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *      # 优化器
from collections import OrderedDict

# 求一个函数的最小值点
# 定义函数
def f(x, y):
    return x**2 / 20.0 + y**2

# 定义梯度函数
def fgrad(x, y):
    return x / 10.0, 2.0 * y

# 定义初始坐标（参数初始值）
init_point = (-7.0, 2.0)

# 定义参数字典和梯度字典
params = {}
params['x'], params['y'] = init_point[0], init_point[1]
grads = {}
grads['x'], grads['y'] = 0, 0

# 定义一组优化器，放在一个有序字典中
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr=0.9)
optimizers['Momentum'] = Momentum(lr=0.1, momentum=0.85)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

iter_num = 30
idx = 1

# 遍历优化器，调用update方法逐次迭代
for key in optimizers:
    optimizer = optimizers[key]

    # 保存参数（x、y）更新历史
    x_history, y_history = [], []
    params['x'], params['y'] = init_point[0], init_point[1]

    # 迭代参数更新
    for i in range(iter_num):
        # 保存当前点的位置（参数）
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 计算梯度
        grads['x'], grads['y'] = fgrad(params['x'], params['y'])
        # 调用update方法更新参数
        optimizer.update(params, grads)

    # 画图，2×2子图
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color='red', markersize=2, label=key)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.plot(0, 0, '+')     # 目标最小值点
    # 画等高线
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0
    plt.contour(X, Y, Z)
    plt.legend(loc='best')

plt.show()