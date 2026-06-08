import torch
import torch.nn as nn  # 神经网络模块，提供各种网络层（卷积、线性、激活等）
import torch.optim as optim  # 优化器模块，提供 SGD、Adam 等参数更新算法
from torch.utils.data import TensorDataset, DataLoader  # 数据集封装与批量加载工具


from common.load_data import load_digtial_data, get_device  # 自定义工具：加载手写数字数据、获取计算设备

# ============ 超参数 ============
batch_size = 128  # 每个批次（batch）的样本数量，一次喂入网络的图片张数
lr = 0.05         # 学习率（learning rate），控制每次参数更新的步长大小
epochs = 50       # 训练轮数，整个训练集被完整遍历的次数

device = get_device()  # 获取计算设备（优先 GPU/cuda，没有则用 CPU）
print(device)

# 1. 先读取数据  (N, x) => (N, c, h, w)
#    load_digtial_data 返回扁平化的数据，形状为 (样本数N, 784)
x_train, x_test, y_train, y_test = load_digtial_data()

# 卷积网络要求输入为四维张量 (N, C, H, W)：
#   -1 表示样本数自动推断；1 = 通道数（灰度图单通道）；28×28 = 图片高宽
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# 2. 创建数据集和数据加载器
#    TensorDataset 把特征和标签打包成 (input, target) 对
train_ds = TensorDataset(x_train, y_train)
# DataLoader 负责按 batch_size 分批，shuffle=True 表示每轮打乱顺序（防止学习到样本排列规律）
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = TensorDataset(x_test, y_test)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# 3. 定义模型
#    使用 Sequential 顺序容器堆叠各层，数据从上到下依次流过
model = nn.Sequential(
    # 第一组卷积：输入 1 通道 -> 输出 8 通道；kernel=3 卷积核，padding=1 保持尺寸不变
    # 输出形状: (8, 28, 28)
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),  # 激活函数，引入非线性，把负值置 0
    nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，尺寸减半 -> (8, 14, 14)

    # 第二组卷积：8 通道 -> 16 通道，padding=1 保持空间尺寸
    # 输出形状: (16, 14, 14)
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 再次减半 -> (16, 7, 7)

    # 第三组卷积：16 通道 -> 32 通道，padding=1 保持尺寸
    # 输出形状: (32, 7, 7)
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),  # (c, h, w)

    nn.Flatten(),  # 展平：把 (32, 7, 7) 拉成一维向量 32*7*7=1568，便于接全连接层

    # 全连接层：1568 维 -> 128 维，做特征压缩
    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),

    # 输出层：128 维 -> 10 维，对应 0~9 共 10 个数字类别
    nn.Linear(128, 10),
    nn.ReLU(),
)
model.to(device)  # 设置设备，把模型参数迁移到 GPU/CPU

# 5. 定义损失函数
#    CrossEntropyLoss 交叉熵损失，适用于多分类，内部已包含 softmax
loss_fn = nn.CrossEntropyLoss()

# 6. 定义优化器
#    SGD 随机梯度下降，根据梯度和学习率更新 model 的全部参数
optimizer = optim.SGD(model.parameters(), lr=lr)

# 7. 训练模型
for epoch in range(epochs):

    train_loss_total = 0  # 累计本轮训练损失
    train_acc_num = 0     # 累计本轮训练预测正确的样本数

    model.train()  # 设置为训练模型（开启 dropout/BN 等训练行为）
    for input, target in train_loader:
        # 把当前批次数据迁移到与模型相同的设备上
        input, target = input.to(device), target.to(device)

        # 1. 前向传播：输入图片，得到 10 个类别的预测分数
        y_pred = model(input)
        # 2. 计算损失：比较预测与真实标签的差距
        loss = loss_fn(y_pred, target)
        # 3. 反向传播：自动求各参数的梯度
        loss.backward()
        # 4. 更新参数：优化器按梯度调整权重
        optimizer.step()
        # 6. 清零梯度：防止梯度在下一批次中累加
        optimizer.zero_grad()

        train_loss_total += loss.item()  # .item() 取出标量数值

        # 统计预测正确数：argmax 取最大分数对应的类别，与真实标签比较
        train_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 训练集平均损失（总损失 / 批次数量）
    train_loss = train_loss_total / len(train_loader)
    # 训练集合准确率（正确数 / 样本总数）
    train_acc = train_acc_num / len(train_ds)

    val_loss_total = 0  # 累计本轮验证损失
    val_acc_num = 0     # 累计本轮验证预测正确的样本数
    model.eval()  # 设置为验证模式（关闭 dropout/BN 的训练行为）
    for input, target in val_loader:
        input, target = input.to(device), target.to(device)

        # 1. 前向传播（验证阶段不需要反向传播和参数更新）
        y_pred = model(input)
        # 2. 计算损失，仅用于观察模型在未见数据上的表现
        loss = loss_fn(y_pred, target)

        val_loss_total += loss.item()
        val_acc_num += (model(input).argmax(dim=-1) == target).sum().item()

    # 验证集平均损失
    val_loss = val_loss_total / len(val_loader)
    # 验证集准确率
    val_acc = val_acc_num / len(val_ds)

    # 打印本轮训练/验证的损失与准确率，观察模型收敛情况
    print(
        f'第{epoch + 1}轮, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
