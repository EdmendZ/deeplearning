import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 首先要调用父类的__init__函数
        # 定义模型各层
        self.linear_1 = nn.Linear(3, 4)
        self.linear_2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    # 向前传播
    def forward(self, x):
        x = self.linear_1(x)
        x = nn.Tanh()(x)

        x = self.linear_2(x)
        x = nn.ReLU()(x)

        x = self.out(x)
        y = nn.Softmax(dim=1)(x)
        return y

if __name__ == '__main__':
    # 1. 定义输入数据
    x = torch.randn(10, 3)

    # 2. 创建模型
    model = MyModel()

    # 3. 向前传播
    y = model(x)

    print(y)
