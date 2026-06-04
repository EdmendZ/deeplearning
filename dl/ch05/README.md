# Ch05 — 损失函数与数据加载（Dataset / DataLoader）

本章分两大块：

1. **损失函数（Loss Function）**：损失函数衡量「模型预测」和「真实标签」之间的差距，是训练时反向传播的优化目标。按任务类型分为**回归损失**和**分类损失**两类。
2. **数据加载**：用 `Dataset` 封装数据、用 `DataLoader` 按批次（batch）迭代，是 PyTorch 训练循环里喂数据的标准方式。

| 文件 | 主题 |
| --- | --- |
| `01_regression_loss_fn.ipynb` | 回归损失：L1 / MSE / SmoothL1 |
| `02_class_loss_fn.ipynb` | 分类损失：BCE / BCEWithLogits / CrossEntropy |
| `03_Dataset_DataLoader.ipynb` | 自定义 Dataset、TensorDataset、DataLoader 批次迭代 |

> 核心概念：损失函数在 PyTorch 里也是 `nn.Module` 的子类。用法统一为「先实例化，再像函数一样调用」：
> ```python
> loss_fn = nn.MSELoss()       # 1. 实例化损失函数
> loss = loss_fn(output, target)  # 2. 传入(预测, 真实)得到标量损失
> ```

---

## 1. 回归损失函数（`01_regression_loss_fn.ipynb`）

回归任务预测的是**连续值**，损失衡量预测值与真实值的数值差距。

```python
import torch
import torch.nn as nn

output = torch.randn(5)   # 模拟神经网络的输出（预测值）
target = torch.randn(5)   # 真实目标值
```

| 损失 | 公式（单样本） | 特点 | API |
| --- | --- | --- | --- |
| **MAE / L1** | `|y - ŷ|` | 对异常值不敏感、梯度恒定 | `nn.L1Loss()` |
| **MSE / L2** | `(y - ŷ)²` | 对大误差惩罚更重、对异常值敏感 | `nn.MSELoss()` |
| **Smooth L1** | 误差小用 L2、误差大用 L1 | 兼顾两者，目标框回归常用 | `nn.SmoothL1Loss()` |

```python
nn.L1Loss()(output, target)        # MAE：平均绝对误差
nn.MSELoss()(output, target)       # MSE：均方误差
nn.SmoothL1Loss()(output, target)  # Smooth L1：小误差平滑、大误差线性
```

> **怎么选？** 数据干净、想严惩大误差 → MSE；数据有异常值/噪声 → L1 或 Smooth L1。Smooth L1 在误差接近 0 时像 MSE（平滑、梯度小），误差大时像 L1（梯度不爆炸），是个折中。

---

## 2. 分类损失函数（`02_class_loss_fn.ipynb`）

分类任务预测的是**类别**，核心是交叉熵（Cross Entropy）思想。

### 2.1 二分类：BCE（二元交叉熵）

二分类时模型输出一个分数，需先用 `sigmoid` 映射成「属于正类的概率」，再算 BCE。

```python
output = torch.randn(5)              # 神经网络原始输出（logits）
proba = torch.sigmoid(output)        # sigmoid 映射成 (0,1) 概率
target = torch.tensor([0, 1, 1, 0, 1]).float()  # 真实标签，需为 float

loss_fn = nn.BCELoss()               # 输入必须是「概率」
loss = loss_fn(proba, target)
```

更推荐用 `BCEWithLogitsLoss`，它把 **sigmoid + BCE 合并**，直接吃原始 logits，数值更稳定：

```python
loss_fn = nn.BCEWithLogitsLoss()     # 内部自带 sigmoid，传 logits 即可
loss = loss_fn(output, target)       # 结果与上面 BCELoss 完全一致
```

> **为什么合并版更稳？** 分开算时 `sigmoid` 可能产生极接近 0/1 的值，再取 log 容易溢出；合并版用 log-sum-exp 技巧规避了这个数值问题。**实战一律优先 `BCEWithLogitsLoss`**，模型最后一层就不用再加 sigmoid。

### 2.2 多分类：CrossEntropyLoss（交叉熵）

```python
output = torch.randn(10, 6)   # 10 条样本，6 个类别的原始分数（logits）
target = torch.tensor([1, 2, 1, 4, 5, 3, 0, 3, 4, 3])  # 标签编码：每个元素是类别下标

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, target)
```

`CrossEntropyLoss` 的两个关键点：

1. **输入是 logits，不是概率**：它内部 = `LogSoftmax + NLLLoss`，已经包含了 softmax，所以**模型最后一层不要再加 `nn.Softmax`**。
2. **target 两种形式都支持**：
   - 标签编码（类别下标，`shape=(N,)`，整型）—— 最常用；
   - 独热编码（one-hot，`shape=(N, C)`，浮点）—— 二者算出的 loss 相同。

```python
# 标签编码 → 独热编码，二者 loss 相同
target_one = torch.zeros(10, 6)
target_one[torch.arange(10), target] = 1
nn.CrossEntropyLoss()(output, target_one)   # 与标签编码版结果一致
```

> 对比记忆：**二分类用 BCE（配 sigmoid），多分类用 CrossEntropy（自带 softmax）**。两者本质都是交叉熵，区别只在输出维度和归一化方式。

---

## 3. Dataset 与 DataLoader（`03_Dataset_DataLoader.ipynb`）

训练时数据通常很大，无法一次性喂进模型。PyTorch 用两层抽象解决：

- **`Dataset`**：定义「单条数据怎么取」——回答 `len(ds)`（共多少条）和 `ds[i]`（第 i 条是什么）。
- **`DataLoader`**：在 Dataset 之上做**分批（batch）、打乱（shuffle）、丢弃尾批**等，并支持迭代。

### 3.1 自定义 Dataset

继承 `Dataset`，实现三个方法即可：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):     # 接收/保存数据
        self.data = data
    def __len__(self):            # 返回样本总数 → len(ds)
        return len(self.data)
    def __getitem__(self, idx):   # 按下标取样本 → ds[idx]
        return self.data[idx]

ds = MyDataset(torch.arange(1, 16))
len(ds)    # 15
ds[0:3]    # tensor([1, 2, 3])
```

### 3.2 内置 `TensorDataset`

当数据已经是张量时，不必手写类，直接用 `TensorDataset` 把**特征和标签打包**，它会按第一维对齐切片：

```python
from torch.utils.data import TensorDataset

x = torch.randn(10, 3)   # 10 条样本，每条 3 个特征
y = torch.randn(10)      # 10 个标签
ds = TensorDataset(x, y) # 打包；ds[i] 返回 (x[i], y[i]) 元组
len(ds)    # 10
ds[0:3]    # (前3条特征, 前3个标签)
```

### 3.3 DataLoader 批次迭代

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=3,    # 每批 3 条
    shuffle=True,    # 每个 epoch 打乱顺序（训练集建议开）
    drop_last=True,  # 总数除不尽 batch_size 时，丢掉最后不满的一批
)

for input, target in loader:   # 每次迭代拿到一个 batch
    ...                        # 训练循环里：前向 → 算 loss → 反向 → 更新
```

| 参数 | 作用 | 常用设置 |
| --- | --- | --- |
| `batch_size` | 每批样本数 | 16 / 32 / 64 … |
| `shuffle` | 是否每个 epoch 打乱 | 训练集 `True`，验证/测试集 `False` |
| `drop_last` | 是否丢弃最后不满的 batch | 对 batch 大小敏感时设 `True` |

> 10 条数据、`batch_size=3`、`drop_last=True` → 只产出 3 个完整批次（9 条），剩 1 条被丢弃；若 `drop_last=False` 则会多一个只含 1 条的批次。

---

## 全章脉络总结

```
任务类型 ─┬─ 回归 ──► L1 / MSE / SmoothL1Loss
          └─ 分类 ─┬─ 二分类 ──► BCEWithLogitsLoss（自带 sigmoid）
                   └─ 多分类 ──► CrossEntropyLoss（自带 softmax）

数据喂入：Dataset（定义取数）──► DataLoader（分批/打乱/迭代）──► 训练循环
```

| 知识点 | API |
| --- | --- |
| 平均绝对误差 | `nn.L1Loss()` |
| 均方误差 | `nn.MSELoss()` |
| 平滑 L1 | `nn.SmoothL1Loss()` |
| 二元交叉熵 | `nn.BCELoss()`（吃概率）/ `nn.BCEWithLogitsLoss()`（吃 logits，推荐） |
| 多分类交叉熵 | `nn.CrossEntropyLoss()`（吃 logits，自带 softmax） |
| 自定义数据集 | 继承 `Dataset`，实现 `__init__`/`__len__`/`__getitem__` |
| 张量数据集 | `TensorDataset(x, y)` |
| 批次加载 | `DataLoader(ds, batch_size, shuffle, drop_last)` |