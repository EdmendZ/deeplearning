import numpy as np
from common.gradient import numerical_gradient    # 数值微分计算梯度
from common.functions import softmax, cross_entropy_error    # 分类问题的输出层激活函数和损失函数

# 定义一个类：单层简单网络
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    # 前向传输（预测）
    def forward(self, X):
        a = np.dot(X, self.W)
        y = softmax(a)
        return y

    # 定义损失函数
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

# 主流程
if __name__ == '__main__':
    # 生成数据和标签
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    # 定义神经网络（模型）
    net = SimpleNet()

    # 定义损失函数
    loss = lambda w: net.loss(x, t)

    # 计算权重矩阵W的梯度
    dW = numerical_gradient(loss, net.W)

    print(dW)