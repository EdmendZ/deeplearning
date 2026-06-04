import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def load_digtial_data():
    data = pd.read_csv("../data/train.csv")

    x = data.drop("label", axis=1)
    y = data["label"]

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 转成tensor统一返回
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train.to_numpy())
    y_test = torch.tensor(y_test.to_numpy())
    return x_train, x_test, y_train, y_test


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_house_data():
    # 1.加载数据
    data = pd.read_csv("../data/house_prices.csv")
    data.drop("Id", axis=1, inplace=True)
    # 2. 区分特征和标签
    x = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # 3. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 4. 特征工程
    ## 识别数字列和类别类
    num_features = x.select_dtypes(exclude="object").columns
    cat_features = x.select_dtypes(include="object").columns

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),  # 用均值填充缺失的值
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="NaN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    ct = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train.values).float()
    y_test = torch.tensor(y_test.values).float()

    return x_train, x_test, y_train, y_test


def get_fashion_data():
    train_data = pd.read_csv("../data/fashion-mnist_train.csv")
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")

    x_train = train_data.iloc[:, 1:]
    x_test = test_data.iloc[:, 1:]

    y_train = train_data.iloc[:, 0].values
    y_test = test_data.iloc[:, 0].values

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    x_train = torch.tensor(x_train).reshape(-1, 1, 28, 28).float()
    x_test = torch.tensor(x_test).reshape(-1, 1, 28, 28).float()

    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    get_fashion_data()
