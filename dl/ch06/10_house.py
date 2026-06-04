import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_house_data, get_device

batch_size = 64
lr = 0.3
epochs = 100
device = get_device()

# 1. 读取数据
x_train, x_test, y_train, y_test = get_house_data()

# 2. 构建数据集和数据加载器
train_ds = TensorDataset(x_train, y_train)
trian_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

# 3. 定义模型
in_features = x_train.shape[1]
model = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(64, 1),
)
model.to(device)


# 4. 定义损失函数
def loss_fn(y_pred, y_target):
    mse = nn.MSELoss()
    y_pred = torch.clamp(y_pred, min=1, max=float("inf")).squeeze()
    return torch.sqrt(mse(torch.log(y_pred), torch.log(y_target)))


# 5. 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

# 6. 训练
for epoch in range(epochs):

    train_loss_total = 0

    model.train()
    for input, target in trian_loader:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred, target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        train_loss_total += loss.item()

    train_loss = train_loss_total / len(trian_loader)

    test_loss_total = 0
    model.eval()
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred, target)

        test_loss_total += loss.item()

    test_loss = test_loss_total / len(test_loader)

    print(f"{epoch + 1} train_loss:{train_loss:.6f}, test_loss:{test_loss:.6f}")

print(model(x_test[:5].to(device)), y_test[:5])