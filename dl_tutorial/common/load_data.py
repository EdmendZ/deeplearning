import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.preprocessing import MinMaxScaler    # 归一化


def get_data():
    # 1. 读取数据集
    data = pd.read_csv('../data/train.csv')
    # 2. 划分训练集和测试集
    X = data.drop('label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 3. 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    return x_train, x_test, y_train.values, y_test.values