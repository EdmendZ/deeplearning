import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_digital_data():
    # 1. 读取数据
    data = pd.read_csv('../data/train.csv')

    # 2. 区分特征和标签
    x = data.drop("label", axis=1)
    y = data["label"]

    # 3. 划分数据集: 训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 归一化
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # print(type(train_x), type(test_x), type(train_y), type(test_y))

    # 5. 统一转成tensor返回
    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()

    train_y = torch.tensor(train_y.to_numpy())
    test_y = torch.tensor(test_y.to_numpy())
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = load_digital_data()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
