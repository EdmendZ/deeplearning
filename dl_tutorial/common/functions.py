import numpy as np

# 一、激活函数
# 1. 阶跃函数
# x传入标量
def step_function0(x):
    if x >= 0:
        return 1
    else:
        return 0

# x传入向量或矩阵
def step_function(x):
    return np.array(x >= 0, dtype=int)

# 2. Sigmoid 函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 3. ReLU 函数
def relu(x):
    return np.maximum(0, x)

# 4. Softmax函数
# 输入x为向量
def softmax0(x):
    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 输入x为矩阵 N×C
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)    # 减去每条数据中xi的最大值
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 5. 恒等函数
def identity(x):
    return x

# 二、损失函数
# 1. MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 2. 交叉熵误差
def cross_entropy_error(y, t):
    # 对于一维情况，直接转换为二维
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # t 是独热编码表示，转换为正确类别标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n

# 测试
if __name__ == '__main__':
    x = np.array([[0, 1, 2], [3, 4, 5], [-1, -2, -3], [-6, -4, -5]])

    print(softmax(x))