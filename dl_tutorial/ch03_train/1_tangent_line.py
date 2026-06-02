from common.gradient import numerical_diff
import matplotlib.pyplot as plt
import numpy as np

# 目标函数：f(x) = 0.01x^2 + 0.1x
def f(x):
    return 0.01 * x ** 2 + 0.1 * x

# 切线方程：tf(x) = ax + b
# 传入f和x，计算斜率和截距，返回表示切线方程的函数
def tangent_function(f, x):
    # 计算斜率a，就是x处的数值微分
    a = numerical_diff(f, x)
    print("切线斜率：", a)
    # 计算截距，切线过(x, f(x))一点
    b = f(x) - a * x
    # 返回函数，表示直线方程
    return lambda x: a * x + b

# 定义x，计算 y=f(x)
x = np.arange(0.0, 20.0, 0.1)
y = f(x)

# 指定 x = 5 处，计算切线对应的 y = ax + b
tf = tangent_function(f, x=5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("x")
plt.ylabel("f(x)")

plt.show()