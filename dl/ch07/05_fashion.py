import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from common.load_data import get_fashion_data, get_device

batch_size = 256
lr = 0.01
epochs = 50
device = get_device()

# 1. 读取数据
x_train, x_test, y_train, y_test = get_fashion_data()

# 2. 定义数据集和数据加载器
train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

test_ds = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

# 3. 定于模型
# model = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
#
#     nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
#
#     nn.Flatten(),
#
#     nn.Linear(16*5*5, 120),
#     nn.ReLU(),
#
#     nn.Linear(120, 84),
#     nn.ReLU(),
#
#     nn.Linear(84, 10),
# )
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

# x = x_train
# for m in model:
#     x = m(x)
#     print(f"{m.__class__.__name__}: {x.shape}")

# 4. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5. 定义损失函数
loss_fn = nn.CrossEntropyLoss()

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
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)

        # 1. 前向传播
        y_pred = model(input)
        # 2. 计算损失
        loss = loss_fn(y_pred, target)

        val_loss_total += loss.item()
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(test_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(test_ds)

    print(
        f'第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
