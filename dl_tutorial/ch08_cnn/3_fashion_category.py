import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 准别数据
# 1.1 加载数据
data_train = pd.read_csv('../data/fashion-mnist_train.csv')
data_test = pd.read_csv('../data/fashion-mnist_test.csv')

# 1.2 将数据划分特征和目标，并转换成张量 （N，C，H，W）
X_train = torch.tensor(data_train.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_train = torch.tensor(data_train.iloc[:, 0].values, dtype=torch.int64)
print(X_train.shape)
print(y_train.shape)
X_test = torch.tensor(data_test.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_test = torch.tensor(data_test.iloc[:, 0].values, dtype=torch.int64)
print(X_test.shape)
print(y_test.shape)

# 找一张图片测试效果
# plt.imshow(X_train[12345, 0, :, :], cmap='gray')
# plt.show()
# print("图片真实分类标签：", y_train[12345])

# 1.3 创建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 2. 创建神经网络模型
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(400, 120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.Sigmoid(),

    nn.Linear(84, 10),
)

# 查看模型中各个层的形状
X = torch.rand((1, 1, 28, 28), dtype=torch.float)
for layer in model:
    print(f"{layer.__class__.__name__:<12}input size: {X.shape}")
    X = layer(X)
    print(f"{layer.__class__.__name__:<12}output size: {X.shape}")

# 3. 训练模型和测试
def train_test(model, train_dataset, test_dataset, lr, n_epochs, batch_size, device):
    # 参数初始化函数
    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_normal_(layer.weight)
    # 3.1 初始化相关操作
    model.apply(init_weights)
    model.to(device)
    loss = nn.CrossEntropyLoss()    # 损失函数：交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 3.2 一个epoch，进行训练和测试
    for epoch in range(n_epochs):
        # 训练过程
        model.train()
        train_loss = 0      # 训练误差
        train_correct_count = 0     # 训练预测准确个数
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch_count, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)   # 前向传播
            loss_value = loss(y_pred, y)    # 计算损失
            loss_value.backward()   # 反向传播
            optimizer.step()    # 更新参数
            optimizer.zero_grad()   # 梯度清零

            train_loss += loss_value.item() * X.shape[0]
            pred = y_pred.argmax(dim=1)
            train_correct_count += pred.eq(y).sum()

            print(f"\repoch:{epoch:0>3}[{'=' * (int((batch_count + 1) / len(train_loader) * 50)):<50}]", end="")

        # 计算平均损失
        this_loss = train_loss / len(train_dataset)
        # 计算准确率
        this_train_acc = (train_correct_count / len(train_dataset))

        # 测试过程
        model.eval()
        test_correct_count = 0  # 测试预测准确个数
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                pred = y_pred.argmax(dim=1)
                test_correct_count += pred.eq(y).sum()

        # 计算准确率
        this_test_acc = (test_correct_count / len(test_dataset))

        print(f"train loss: {this_loss:.6f}, train acc: {this_train_acc:.6f}, test acc: {this_test_acc:.6f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_test(model, train_dataset, test_dataset, lr=0.9, n_epochs=20, batch_size=256, device=device)

# 选取一个测试数据，进行验证
plt.imshow(X_test[666, 0, :, :], cmap='gray')
plt.show()
print("图片真实分类标签：", y_test[666])

# 用模型进行预测分类
ouptut = model(X_test[666].unsqueeze(0).to(device))
y_pred = ouptut.argmax(dim=1)
print("图片预测分类标签：", y_pred)