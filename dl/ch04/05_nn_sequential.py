import torch
import torch.nn as nn

# 1. 定义输入数据
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()

# 2. 创建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),

    nn.Linear(4, 4),
    nn.ReLU(),

    nn.Linear(4, 2),
    nn.Softmax(dim=-1)
)
# 假设模型训练完毕

# 保存模型(保存模型的参数)
#torch.save(model.state_dict(), 'model.pth')

# 加载参数
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

# 3. 向前传播
y = model(x)

print(y)
