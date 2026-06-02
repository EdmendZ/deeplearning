import torch
import torch.nn as nn

from common.load_data import load_digital_data

# 1. 读取数据
train_x, test_x, train_y, test_y = load_digital_data()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(28 * 28, 50),
    nn.ReLU(),

    nn.Linear(50, 100),
    nn.ReLU(),

    nn.Linear(100, 10),
    # nn.Softmax(dim=-1)  # 训练时, softmax和损失函数(交叉熵) 整合在一起
)

# 3. 加载模型参数
state_dict = torch.load("../data/nn_example.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# 4. 预测
y_pred = model(test_x)
y_pred_class = torch.argmax(y_pred, dim=-1)

# 5. 计算准确率:  分子:预测准确的个数  分母:总的参数预测的数量
# acc_count = (y_pred_class == test_y).sum()
# acc_count = (y_pred_class.eq(test_y)).sum()
acc_count = torch.sum(y_pred_class == test_y)

acc = acc_count.item() / len(test_y)
print(acc)

print(model(test_x[33:34]), test_y[33:34])

c = 0
for i in y_pred_class == test_y:
    if not i:
        print(c)
    c += 1
