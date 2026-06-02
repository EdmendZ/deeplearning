import numpy as np
from common.gradient import numerical_gradient
from common.functions import *
from common.layers import *
from collections import OrderedDict    # 有序字典

class TwoLayerNet:
    # 初始化
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)
        # 按照顺序生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    # 前向传播（预测）
    def forward(self, x):
        # 遍历每个层，调用forward方法
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

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
        # 直接调最后一层的forward方法
        return self.lastLayer.forward(y, t)

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

    # 利用反向传播计算梯度
    def gradient(self, x, t):
        # 前向传播，计算所有的中间值（直接调loss）
        self.loss(x, t)
        # 反向传播
        # 单独计算最后一层
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 反向遍历之前的每一层，调用backward方法
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 提取各Affine层的参数梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads