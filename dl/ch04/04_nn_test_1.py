
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # 首先要调用父类的__init__函数
        # 定义模型各层
        self.linear_1 = nn.Linear(3, 4)
        self.activation_1 = nn.Tanh()

        self.linear_2 = nn.Linear(4, 4)
        self.activation_2 = nn.ReLU()

        self.out = nn.Linear(4, 2)
        self.activation_3 = nn.Softmax(dim=1)

    # 向前传播
    def forward(self, x):
        for m in self.modules():
            if not isinstance(m, MyModel):
                x = m(x)
        return x

if __name__ == '__main__':
    # 1. 定义输入数据
    x = torch.randn(10, 3)

    # 2. 创建模型
    model = MyModel()

    # 3. 向前传播
    y = model(x)

    print(y)

    print(model.linear_1.weight)
    print(model.linear_1.bias)

    print("------遍历参数-----")
    for param in model.parameters():
        print(param)
    print("------遍历参数: 带名字-----")
    for name, param in model.named_parameters():
        print(name, param)

    print("------状态字典------")
    state_dict = model.state_dict()
    print(state_dict)

    print("=---- 模型的结构--- pip install torchsummay  --")
    from torchsummary import summary
    summary(model, input_size=(3,), batch_size=10, device='cpu')

