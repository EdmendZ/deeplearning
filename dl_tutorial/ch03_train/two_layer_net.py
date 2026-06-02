import numpy as np
from common.gradient import numerical_gradient
from common.functions import *

class TwoLayerNet:
    # 初始化
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

    # 前向传播（预测）
    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # 计算准确度
    def accuracy(self, x, t):
        # 预测
        y = self.forward(x)
        # 将概率转换为类别号
        y = np.argmax(y, axis=1)
        # 计算准确度
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    # 损失函数
    def loss(self, x, t):
        # 前向传播得到预测值
        y = self.forward(x)
        # 直接调交叉熵损失函数
        return cross_entropy_error(y, t)

    # 计算梯度
    def numerical_gradient(self, x, t):
        # 目标损失函数
        loss = lambda w: self.loss(x, t)
        # 用字典保存所有参数对应的梯度
        grads = {}
        grads['W1'] = numerical_gradient(loss, self.params['W1'])
        grads['b1'] = numerical_gradient(loss, self.params['b1'])
        grads['W2'] = numerical_gradient(loss, self.params['W2'])
        grads['b2'] = numerical_gradient(loss, self.params['b2'])

        return grads