import numpy as np
import pandas as pd
import joblib
from docutils.nodes import Sequential
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.preprocessing import MinMaxScaler    # 归一化

from common.functions import sigmoid, softmax    # 激活函数

# 读取数据
def get_data():
    # 1. 读取数据集
    data = pd.read_csv('../data/train.csv')
    # 2. 划分训练集和测试集
    X = data.drop('label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_test, y_test

# 初始化神经网络（加载参数）
def init_network():
    network = joblib.load('../data/nn_sample')
    return network

# 前向传播（预测）
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 主流程
# 1. 获取测试数据
x, y = get_data()
print(x.shape)

# 2. 创建模型
network = init_network()

# 3. 预测，得到分类概率
y_probs = forward(network, x)

# 4. 根据概率得到分类标签
y_pred = np.argmax(y_probs, axis=1)

# 5. 计算准确率
n = x.shape[0]
accuracy_cnt = np.sum(y == y_pred)

print("准确率：", accuracy_cnt / n)