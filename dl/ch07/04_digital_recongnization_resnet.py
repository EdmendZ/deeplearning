import torch
import torch.nn as nn  # 神经网络模块，提供各种网络层
import torch.optim as optim  # 优化器模块，提供 SGD、Adam 等参数更新算法
from torch.utils.data import TensorDataset, DataLoader  # 数据集封装与批量加载工具
from torchvision import models  # torchvision 内置的经典网络结构（含 ResNet18 等）

from common.load_data import load_digtial_data, get_device  # 自定义工具：加载手写数字数据、获取计算设备

# ============ 超参数 ============
batch_size = 512  # 每个批次的样本数；ResNet 较深，用更大 batch 提升 GPU 利用率
lr = 0.05         # 学习率，控制每次参数更新的步长
epochs = 50       # 训练轮数

device = get_device()  # 获取计算设备（优先 GPU/cuda，没有则用 CPU）
print(device)

# 1. 先读取数据  (N, x) => (N, c, h, w)
#    load_digtial_data 返回扁平化数据，形状 (样本数N, 784)
x_train, x_test, y_train, y_test = load_digtial_data()

# 重塑为四维张量 (N, C, H, W)：1 通道（灰度），28×28 高宽
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# 2. 创建数据集和数据加载器
train_ds = TensorDataset(x_train, y_train)  # 打包训练特征与标签
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # 分批+打乱

val_ds = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# 3. 定义模型
#    直接使用 torchvision 的 ResNet18 残差网络；pretrained=False 表示不加载预训练权重，从零训练
model = models.resnet18(pretrained=False)

# ResNet18 原本针对 ImageNet 三通道彩色大图设计，这里需做三处改造以适配 MNIST 灰度小图：
# (1) 第一层卷积改为接收 1 通道输入（灰度图），其余参数沿用原结构
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
# (2) 去掉最开始的最大池化层（用 Identity 恒等替换），避免 28×28 小图被过早下采样丢失信息
model.maxpool = nn.Identity()
# (3) 替换最后的全连接分类层，输出维度改为 10（对应 0~9 共 10 个数字类别）
model.fc = nn.Linear(model.fc.in_features, 10)

model.to(device)  # 设置设备，把模型迁移到 GPU/CPU

# 5. 定义损失函数
#    交叉熵损失，适用于多分类，内部已包含 softmax
loss_fn = nn.CrossEntropyLoss()

# 6. 定义优化器
#    SGD 随机梯度下降，更新模型全部参数
optimizer = optim.SGD(model.parameters(), lr=lr)

# 7. 训练模型
for epoch in range(epochs):

    train_loss_total = 0  # 累计本轮训练损失
    train_acc_num = 0     # 累计本轮训练预测正确的样本数

    model.train()  # 设置为训练模型（开启 BatchNorm 的滑动统计、dropout 等训练行为）
    for input, target in train_loader:
        # 把当前批次数据迁移到与模型相同的设备
        input, target = input.to(device), target.to(device)

        # 1. 前向传播：得到 10 个类别的预测分数
        y_pred = model(input)
        # 2. 计算损失：预测与真实标签的差距
        loss = loss_fn(y_pred, target)
        # 3. 反向传播：自动求梯度
        loss.backward()
        # 4. 更新参数：按梯度调整权重
        optimizer.step()
        # 6. 清零梯度：防止梯度累加到下一批次
        optimizer.zero_grad()

        train_loss_total += loss.item()  # 取出标量损失值累加

        # 统计预测正确数：argmax 取最大分数对应类别，与真实标签比较
        train_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 训练集平均损失（总损失 / 批次数）
    train_loss = train_loss_total / len(train_loader)
    # 训练集合准确率（正确数 / 样本总数）
    train_acc = train_acc_num / len(train_ds)

    val_loss_total = 0  # 累计本轮验证损失
    val_acc_num = 0     # 累计本轮验证预测正确的样本数
    model.eval()  # 设置为验证模式（BatchNorm 使用累积统计，关闭 dropout）
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        # 1. 前向传播（验证阶段不更新参数）
        y_pred = model(input)
        # 2. 计算损失，观察模型在未见数据上的表现
        loss = loss_fn(y_pred, target)

        val_loss_total += loss.item()
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(val_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(val_ds)

    # 打印本轮训练/验证的损失与准确率，观察收敛情况
    print(
        f'第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
