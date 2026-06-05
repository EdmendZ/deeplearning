import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from common.load_data import load_digtial_data
from common.utils import get_device

batch_size = 64
lr = 0.1
epochs = 50
device = get_device()


# 1. 读取数据
train_x, test_x, train_y, test_y = load_digtial_data()

# 2. 数据集和数据加载器
train_ds = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

val_ds = TensorDataset(test_x, test_y)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=False)

# 3. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50),
    nn.ReLU(),

    nn.Linear(50, 100),
    nn.ReLU(),

    nn.Linear(100, 10),
    # nn.Softmax(dim=-1),
)

model.to(device)
print(device)
# 4. 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 5. 训练
for epoch in range(epochs):

    train_loss = 0
    train_acc_num = 0
    model.train()  # 设置为训练模式
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred, target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        train_loss += loss.item()
        y_class = model(input).argmax(dim=-1)
        train_acc_num += (y_class == target).sum().item()

    train_loss /= len(train_loader)  # 当前epoch的平均损失
    train_acc = train_acc_num / len(train_ds)


    val_loss = 0
    val_acc_num = 0
    model.eval()  # 设置为验证模式
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)
        loss = loss_fn(y_pred, target)

        val_loss += loss.item()
        y_class = model(input).argmax(dim=-1)
        val_acc_num += (y_class == target).sum().item()

    val_loss /= len(val_loader)  # 当前epoch的平均损失
    val_acc = val_acc_num / len(val_ds)

    print(
        f'第{epoch + 1}轮, 训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.6f}, 验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.6f}')
