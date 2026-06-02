import numpy as np
from common.functions import *

# ReLU层
class Relu:
    def __init__(self):
        # 记录哪些 x <= 0
        self.mask = None

    # 前向传播
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    # 反向传播
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

# Sigmoid层
class Sigmoid:
    def __init__(self):
        self.y = None

    # 前向传播
    def forward(self, x):
        y = sigmoid(x)
        self.y = y
        return y

    # 反向传播
    def backward(self, dout):
        dx = dout * self.y * (1.0 - self.y)
        return dx

# 仿射层
class Affine:
    def __init__(self, W, b):
        # 保存权重和偏置参数
        self.W = W
        self.b = b
        # 保存本层输入的 X
        self.X = None
        self.X_original_shape = None   # 原始 X 张量的形状
        # 保存参数的梯度（偏导数）
        self.dW = None
        self.db = None
    # 前向传播
    def forward(self, X):
        self.X_original_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        # 计算输出
        Y = np.dot(self.X, self.W) + self.b
        return Y
    # 反向传播
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        # 计算参数的梯度
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX.reshape(*self.X_original_shape)

# 输出层Softmax+Loss
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None    # 损失值
    # 前向传播
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    # 反向传播
    def backward(self, dout=1):
        n = self.t.shape[0]
        # 标签是独热编码
        if self.t.size == self.y.size:
            dx = (self.y - self.t)
        # 标签是分类编号，就将预测值对应索引号的元素减1
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1

        return dx / n
