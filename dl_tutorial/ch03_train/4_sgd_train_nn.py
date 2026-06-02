import math

import numpy as np
from two_layer_net import TwoLayerNet
from common.load_data import get_data    # 获取数据

# 1. 加载数据
x_train, x_test, t_train, t_test = get_data()

# print('x_train shape:', x_train.shape)
# print('t_train shape:', t_train.shape)
# print('x_test shape:', x_test.shape)
# print('t_test shape:', t_test.shape)

# 2. 创建二层神经网络模型
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3. 设置超参数
train_size = x_train.shape[0]    # 训练集样本总数
batch_size = 100    # 随机选择的小批次量
iter_per_epoch = math.ceil( train_size / batch_size)    # 每轮包含的迭代次数

iters_num = 1000    # 迭代总次数
learning_rate = 0.1

train_loss_list = []    # 训练误差
train_acc_list = []    # 训练准确度
test_acc_list = []    # 测试准确度

# 4. 随机梯度下降法，迭代训练模型，计算参数
for i in range(iters_num):
    # 4.1 随机选取 batch_size 个训练数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 4.2 计算当前参数下的梯度值
    grads = network.numerical_gradient(x_batch, t_batch)

    # 4.3 遍历梯度字典，更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    # 4.4 计算当前训练损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(f"iter {i}, loss {loss}")

    # 4.5 每完成一个轮次epoch，就打印一次准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Train loss: {loss}, Train acc: {train_acc}, Test acc: {test_acc}')