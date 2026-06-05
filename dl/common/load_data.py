import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# 设备检测统一放在 utils.py，这里 re-export（重新导出）以保持向后兼容：
# 既有 `from common.load_data import get_device` 的代码无需改动，
# 同时避免在两个文件里维护两份相同实现。
from common.utils import get_device


def load_digtial_data():
    """加载手写数字识别（MNIST 风格）数据并完成预处理。

    流程：读 train.csv -> 拆特征/标签 -> 划分训练测试集
          -> MinMax 归一化到 [0,1] -> 转 Tensor。

    Returns:
        (x_train, x_test, y_train, y_test):
            x 为 float 型像素特征 Tensor，y 为整型标签 Tensor。
    """
    data = pd.read_csv("../data/train.csv")

    x = data.drop("label", axis=1)  # 特征：所有像素列
    y = data["label"]               # 标签：数字 0~9

    # 划分数据集：80% 训练 / 20% 测试，固定随机种子保证可复现
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 归一化：用训练集 fit，再对训练/测试集 transform，避免测试集信息泄漏
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 转成tensor统一返回
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    # 标签保持整型（分类任务的 CrossEntropyLoss 要求 long 类型类别索引）
    y_train = torch.tensor(y_train.to_numpy())
    y_test = torch.tensor(y_test.to_numpy())
    return x_train, x_test, y_train, y_test


def get_house_data():
    """加载房价预测（回归任务）数据，含完整特征工程流水线。

    与数字识别不同，房价数据同时包含数值列和类别列、且有缺失值，
    因此用 ColumnTransformer 对两类特征分别处理。

    Returns:
        (x_train, x_test, y_train, y_test): 全部为 float Tensor，
        y 是连续的房价（回归目标）。
    """
    # 1.加载数据
    data = pd.read_csv("../data/house_prices.csv")
    data.drop("Id", axis=1, inplace=True)  # Id 只是行号，对预测无意义，丢弃
    # 2. 区分特征和标签
    x = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]               # 标签：房屋售价

    # 3. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 特征工程
    ## 识别数字列和类别类
    num_features = x.select_dtypes(exclude="object").columns  # 数值型特征
    cat_features = x.select_dtypes(include="object").columns  # 类别型（字符串）特征

    # 数值特征流水线：均值填补缺失 -> 标准化（均值0方差1）
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),  # 用均值填充缺失的值
        ("scaler", StandardScaler())
    ])
    # 类别特征流水线：用常量填补缺失 -> One-Hot 独热编码
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),
        # drop="first" 去掉一列防共线性；handle_unknown 处理测试集出现的新类别；
        # sparse_output=False 输出稠密矩阵，方便转 Tensor
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    # 把两条流水线按列分配，组合成一个整体转换器
    ct = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # 同样：训练集 fit_transform，测试集仅 transform
    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    # 回归任务的标签是连续值，需用 float
    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()

    return x_train, x_test, y_train, y_test


def get_fashion_data():
    """加载 Fashion-MNIST 服饰图像数据，输出适配 CNN 的四维张量。

    train/test 已经是两个独立 CSV，无需再切分。第 0 列是标签，
    其余 784 列是 28x28 的灰度像素。

    Returns:
        (x_train, x_test, y_train, y_test): x 形状为
        (N, 1, 28, 28)（通道数 1），可直接喂给卷积网络。
    """
    train_data = pd.read_csv("../data/fashion-mnist_train.csv")
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")

    x_train = train_data.iloc[:, 1:]  # 第 1 列起为像素特征
    x_test = test_data.iloc[:, 1:]

    y_train = train_data.iloc[:, 0].values  # 第 0 列为类别标签
    y_test = test_data.iloc[:, 0].values

    # 归一化到 [0,1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # reshape 成 (样本数, 通道=1, 高=28, 宽=28) 的图像张量
    x_train = torch.tensor(x_train).reshape(-1, 1, 28, 28).float()
    x_test = torch.tensor(x_test).reshape(-1, 1, 28, 28).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # 直接运行本文件时的快速自测入口
    get_fashion_data()
