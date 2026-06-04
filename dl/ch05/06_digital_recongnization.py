import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import load_digtial_data, get_device

batch_size = 256
lr = 0.05
epochs = 50

device = get_device()
print(device)

# 1. 先读取数据
x_train, x_test, y_train, y_test = load_digtial_data()

# 2. 创建数据集和数据加载器
train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# 3. 定义模型
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),

    nn.Linear(100, 50),
    nn.ReLU(),

    nn.Linear(50, 25),
    nn.ReLU(),

    nn.Linear(25, 20),
    nn.ReLU(),

    nn.Linear(20, 60),
    nn.ReLU(),

    nn.Linear(60, 10),
    # nn.Softmax(dim=-1),
)
model.to(device)  # 设置设备

# 5. 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 6. 定义优化器
optimizer = optim.SGD(model.parameters(), lr=lr)

# 7. 训练模型
for epoch in range(epochs):

    train_loss_total = 0
    train_acc_num = 0

    model.train()  # 设置为训练模型
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        # 1. 前向传播
        y_pred = model(input)
        # 2. 计算损失
        loss = loss_fn(y_pred, target)
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        # 6. 清零梯度
        optimizer.zero_grad()

        train_loss_total += loss.item()

        train_acc_num += (model(input).argmax(dim=-1) == target).sum().item()


    # 训练集平均损失
    train_loss = train_loss_total / len(train_loader)
    # 训练集合准确率
    train_acc = train_acc_num / len(train_ds)

    val_loss_total = 0
    val_acc_num = 0
    model.eval()  # 设置为验证模式
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        # 1. 前向传播
        y_pred = model(input)
        # 2. 计算损失
        loss = loss_fn(y_pred, target)

        val_loss_total += loss.item()
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(val_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(val_ds)

    print(
        f'第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')



