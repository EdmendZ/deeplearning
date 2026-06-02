import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split     # 划分数据集
from sklearn.preprocessing import StandardScaler, OneHotEncoder    # 针对数值型和类别型特征的处理操作
from sklearn.compose import ColumnTransformer    # 列转换器
from sklearn.pipeline import Pipeline    # 管道操作
from sklearn.impute import SimpleImputer    # 处理缺失值

from torch.utils.data import Dataset, DataLoader, TensorDataset  # 数据集和数据加载器

# 读取数据，返回数据集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv('../data/house_prices.csv')

    # 2. 去除无关特征（特征选择）
    data.drop(['Id'], axis=1, inplace=True)

    # 3. 划分特征和目标值
    X = data.drop(['SalePrice'], axis=1)
    y = data['SalePrice']

    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 5. 特征预处理（特征转换）
    # 5.1 筛选两类特征：数值型 和 类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    # 5.2 对两种类型的特征，分别定义不同的转换操作
    numerical_transformer = Pipeline(steps=[
        ('fillNA', SimpleImputer(strategy='mean')),     # 用均值填充缺失值
        ('scaler', StandardScaler())   # 标准化
    ])
    categorical_transformer = Pipeline(steps=[
        ('fillNA', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # 5.3 构建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    # 5.4 做列转换，并生成新的DataFrame
    x_train = pd.DataFrame(preprocessor.fit_transform(x_train).toarray(), columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(preprocessor.transform(x_test).toarray(), columns=preprocessor.get_feature_names_out())
    # 5.5 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())

    return train_dataset, test_dataset, x_train.shape[1]    # 返回训练集、测试集和特征维度

# 主流程
# 1. 得到数据集
train_dataset, test_dataset, n_features = create_dataset()
# print(test_dataset)
print(n_features)

# 2. 搭建神经网络模型
model = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.BatchNorm1d(128),        # 批量标准化 BN 层
    nn.ReLU(),
    nn.Dropout(0.2),            # 随机失活 Dropout 层
    nn.Linear(128, 1)
)

# 3. 定义损失函数
def log_rmse(y_pred, y_true):
    y_pred.squeeze_()
    y_pred = torch.clamp(y_pred, 1, float('inf'))
    mse = nn.MSELoss()
    return torch.sqrt( mse(torch.log(y_pred), torch.log(y_true)) )

# 4. 模型训练和测试
def train_test(model, train_dataset, test_dataset, lr, n_epochs, batch_size, device):
    # 单独定义权重初始化的函数
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
    # 1. 初始化
    model.apply(init_weights)
    model.to(device)

    # 2. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []    # 记录每个epoch训练误差
    test_loss_list = []     # 记录每个epoch测试误差

    # 3. 按照epoch进行训练和测试
    for epoch in range(n_epochs):
        # 3.1 训练
        model.train()
        train_loss_total = 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 通过loader遍历每个mini-batch数据
        for batch_count, (X, target) in enumerate(train_loader):
            X, target = X.to(device), target.to(device)
            y_pred = model(X)       # 前向传播，得到预测值
            loss_value = log_rmse(y_pred, target)   # 计算损失
            loss_value.backward()   # 反向传播，计算梯度
            optimizer.step()        # 更新参数
            optimizer.zero_grad()   # 梯度清零
            train_loss_total += loss_value.item()   # 累加损失

            print(f"\repoch:{epoch:0>3}[{'=' * (int((batch_count + 1) / len(train_loader) * 50)):<50}]", end="")

        # 每轮（epoch）训练结束，计算平均训练误差
        train_loss_avg = train_loss_total / len(train_loader)
        train_loss_list.append(train_loss_avg)

        # 3.2 测试
        model.eval()

        test_loss_total = 0
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for X, target in test_loader:
                X, target = X.to(device), target.to(device)
                y_pred = model(X)       # 前向传播，得到预测值
                loss_value = log_rmse(y_pred, target)   # 计算损失

                test_loss_total += loss_value.item()   # 累加损失

        # 每轮（epoch）训练结束，计算平均训练误差
        test_loss_avg = test_loss_total / len(test_loader)
        test_loss_list.append(test_loss_avg)

        # 打印输出
        print(f"train loss: {train_loss_avg:.6f}, test loss: {test_loss_avg:.6f}")

    # 所有epoch训练结束，返回误差列表
    return train_loss_list, test_loss_list

# 判断是否支持cuda，如果支持则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr=0.1, n_epochs=200, batch_size=64, device=device)

# 5. 画图
plt.plot(train_loss_list, 'r-', label='train loss', linewidth=3)
plt.plot(test_loss_list, 'k--', label='test loss', linewidth=2)
plt.legend(loc='best')
plt.show()