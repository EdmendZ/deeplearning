import numpy as np

# 数值微分，传入函数f和自变量x
def numerical_diff0(f, x):
    h = 1e-4
    return (f(x+h) - f(x)) / h

# 中心差分实现数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

# 利用数值微分计算梯度
# f: 多元函数，x: 向量 [x1, x2, ..., xn]
def _numerical_gradient(f, x):
    # 定义梯度向量, 初始值为 0
    grad = np.zeros_like(x)
    # 定义微小量
    h = 1e-4
    # 遍历x中的每个变元 xi
    for i in range(x.size):
        tmp = x[i]    # 临时保存xi，后面要更改
        x[i] = tmp + h
        fxh1 = f(x)    # f(x+h)
        x[i] = tmp - h
        fxh2 = f(x)    # f(x-h)

        grad[i] = (fxh1 - fxh2) / (2 * h)    # 计算对 xi 的偏微分
        x[i] = tmp
    return grad

# 数值微分计算梯度，扩展到二维矩阵形式
def numerical_gradient(f, X):
    # 分一维和二维情况讨论
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历 X 中的每一行，分别求梯度向量
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)

        return grad