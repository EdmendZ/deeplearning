# ch05 知识点总结：损失函数 · 数据集 · 自动求导 · 训练流程

> 本文档系统总结 `dl/ch05` 各文件涉及的深度学习知识点，
> 每个方法都给出**英文全称**与**缩写**，并在最后做对比总结。
>
> | 文件 | 主题 |
> |------|------|
> | `01_regression_loss_fn.ipynb` | 回归任务损失函数 |
> | `02_class_loss_fn.ipynb` | 分类任务损失函数 |
> | `03_Dataset_DataLoader.ipynb` | 数据集与数据加载器 |
> | `04_backward_test.ipynb` | 自动求导（反向传播）机制 |
> | `05_train_test.py` | 完整训练流程（线性回归） |
> | `06_digital_recongnization.py` | 手写数字识别（多分类实战） |

---

## 一、回归损失函数（Regression Loss Functions）

> 回归任务：预测**连续值**（如房价、温度）。损失函数衡量预测值与真实值的"数值差距"。

### 1.1 MAE —— 平均绝对误差

- **英文全称**：Mean Absolute Error
- **缩写**：MAE，也叫 **L1 Loss**
- **PyTorch API**：`nn.L1Loss()`

**公式：**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i \rvert$$

```python
loss_fn = nn.L1Loss()
loss = loss_fn(output, target)   # 取预测与目标差的绝对值再求平均
```

**特点：**
- 对**异常值(outlier)不敏感**（误差不平方，不会被放大）。
- 在 0 点处不可导，梯度恒定（为 ±1），接近最优时收敛较慢。

### 1.2 MSE —— 均方误差

- **英文全称**：Mean Squared Error
- **缩写**：MSE，也叫 **L2 Loss**
- **PyTorch API**：`nn.MSELoss()`

**公式：**

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```python
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)   # 取差的平方再求平均
```

**特点：**
- 误差被**平方放大**，对**异常值非常敏感**（大误差惩罚很重）。
- 处处可导，梯度随误差减小而减小，收敛平滑。**回归任务最常用**。

### 1.3 Smooth L1 —— 平滑 L1 损失（Huber Loss）

- **英文全称**：Smooth L1 Loss（即 Huber Loss 的特例）
- **缩写**：Smooth L1
- **PyTorch API**：`nn.SmoothL1Loss()`

**公式（分段函数）：**

$$
\text{SmoothL1}(x)=
\begin{cases}
0.5\,x^2 & \text{if } \lvert x \rvert < 1 \\
\lvert x \rvert - 0.5 & \text{otherwise}
\end{cases}
\quad (x = y - \hat{y})
$$

```python
loss_fn = nn.SmoothL1Loss()
loss = loss_fn(output, target)
```

**特点（结合了 MSE 和 MAE 的优点）：**
- **误差小时**像 MSE（平方），梯度平滑、收敛快。
- **误差大时**像 MAE（线性），对异常值不敏感、更稳健。
- 常用于目标检测（如 Faster R-CNN 的边框回归）。

---

## 二、分类损失函数（Classification Loss Functions）

> 分类任务：预测**离散类别**。核心思想是先把网络输出变成"概率"，再用交叉熵衡量预测概率分布与真实分布的差距。

### 2.1 BCE —— 二元交叉熵

- **英文全称**：Binary Cross Entropy
- **缩写**：BCE
- **PyTorch API**：`nn.BCELoss()`

**用途：二分类**（输出"是/否"、"0/1"的概率）。

**公式：**

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\big]$$

```python
output = torch.randn(5)
proba = torch.sigmoid(output)      # ★必须先用 Sigmoid 把输出压到 (0,1) 区间当作概率
target = torch.tensor([0,1,1,0,1]).float()

loss_fn = nn.BCELoss()
loss = loss_fn(proba, target)      # 输入是"概率"
```

**关键点：`nn.BCELoss()` 的输入必须是已经过 Sigmoid 的概率值。**

### 2.2 BCEWithLogits —— 带 Logits 的二元交叉熵

- **英文全称**：Binary Cross Entropy With Logits
- **缩写**：BCEWithLogits
- **PyTorch API**：`nn.BCEWithLogitsLoss()`

```python
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(output, target)     # ★直接传网络原始输出(logits)，内部自动做 Sigmoid
```

**关键点：把 `Sigmoid + BCE` 整合到一起。**
- 输入是**原始输出 logits**，无需手动 Sigmoid。
- 内部用 LogSumExp 技巧，**数值更稳定**（避免 log(0) 溢出）。
- 官方推荐用它代替 `Sigmoid + BCELoss` 的组合。

> 代码里两种写法算出的损失相同（都是 0.6115），印证了它们等价。

> **Logits 含义**：神经网络最后一层未经激活的原始输出（取值范围 ±∞），
> 还不是概率，需要经 Sigmoid（二分类）或 Softmax（多分类）才变成概率。

### 2.3 CE —— 交叉熵损失

- **英文全称**：Cross Entropy Loss
- **缩写**：CE
- **PyTorch API**：`nn.CrossEntropyLoss()`

**用途：多分类**（如手写数字 0~9 共 10 类）。

```python
output = torch.randn(10, 6)        # 10 个样本，6 个类别的原始 logits
target = torch.tensor([1,2,1,4,5,3,0,3,4,3])   # 标签编码：每个样本的正确类别索引

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, target)     # ★直接传 logits，内部自动 Softmax
```

**关键点（极其重要）：**

$$\text{CrossEntropyLoss} = \text{Softmax} + \log + \text{NLLLoss}$$

- **NLLLoss** = Negative Log Likelihood Loss（负对数似然损失）。
- 输入是**原始 logits**，**不要**自己先做 Softmax（内部已包含）。
- 目标 `target` 可以是两种形式：
  1. **标签编码（类别索引）**：`[1, 2, 1, ...]`，最常用。
  2. **独热编码（One-Hot）**：`[[0,1,0,0,0,0], ...]`。

> 代码里两种 target 形式算出的损失相同（都是 2.0400），印证它们等价。

**Softmax —— 归一化指数函数**
- **英文全称**：Softmax (Normalized Exponential Function)
- 作用：把多个 logits 转成**和为 1 的概率分布**，最大的 logit 对应最大概率。

---

## 三、数据集与数据加载器（Dataset & DataLoader）

> 把原始数据"包装"成模型能高效读取的格式：分批(batch)、打乱(shuffle)、迭代。

### 3.1 自定义数据集（Custom Dataset）

- **基类**：`torch.utils.data.Dataset`
- 必须实现三个方法：

```python
class MyDataset(Dataset):
    def __init__(self, data):       # 初始化，保存数据
        self.data = data

    def __len__(self):              # 返回数据集大小 -> len(ds)
        return len(self.data)

    def __getitem__(self, idx):     # 按索引取样本 -> ds[i]
        return self.data[idx]
```

| 方法 | 作用 | 触发方式 |
|------|------|---------|
| `__init__` | 加载/保存数据 | 创建对象时 |
| `__len__` | 返回样本总数 | `len(ds)` |
| `__getitem__` | 取出第 idx 个样本 | `ds[idx]` |

### 3.2 内置数据集 TensorDataset

- **英文全称**：Tensor Dataset
- **PyTorch API**：`torch.utils.data.TensorDataset`

```python
x = torch.randn(10, 3)   # 特征
y = torch.randn(10)      # 标签
ds = TensorDataset(x, y) # 把特征和标签按行"打包配对"
ds[0:3]                  # 返回 (前3行特征, 前3个标签)
```

**特点：** 把多个张量按**第一维（样本维）对齐打包**，`ds[i]` 自动返回一个元组 `(x[i], y[i])`。适合数据已经是张量的场景，无需自定义。

### 3.3 DataLoader —— 数据加载器

- **英文全称**：Data Loader
- **PyTorch API**：`torch.utils.data.DataLoader`

```python
loader = DataLoader(ds, batch_size=3, shuffle=True, drop_last=False)
for input, target in loader:   # 每次迭代吐出一个 batch
    ...
```

**核心参数：**

| 参数 | 含义 |
|------|------|
| `batch_size` | 每批样本数（如 3 表示每次取 3 条） |
| `shuffle` | 是否在每个 epoch 打乱数据顺序（训练集设 True，防止顺序偏差） |
| `drop_last` | 最后一批数量不足 batch_size 时是否丢弃 |

> 例：10 个样本、batch_size=3、drop_last=False → 分成 4 批（3+3+3+1）。
> 若 drop_last=True → 只有 3 批（最后那 1 个被丢弃）。

---

## 四、自动求导 / 反向传播（Autograd / Backpropagation）

> PyTorch 的核心机制：自动构建计算图、自动求梯度。`04_backward_test.ipynb` 演示了它的细节。

### 4.1 计算流程

```python
# 1. 准备数据
x = torch.tensor([10.0], requires_grad=True)
y = torch.tensor([20.0], requires_grad=True)
# 2. 定义参数（叶子节点）
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
# 3. 前向传播（构建计算图）
z = w * x + b
# 4. 计算损失
loss_fn = nn.MSELoss()
loss = loss_fn(z, y)
# 5. 反向传播（自动求梯度）
loss.backward()
# 梯度存到各张量的 .grad 属性
print(w.grad, b.grad)
```

### 4.2 关键概念

**① `requires_grad=True`**
- 开启梯度追踪。只有设了它，该张量才会进入计算图、`backward()` 才会算它的梯度。

**② 叶子节点（Leaf Tensor）—— `is_leaf`**
- **英文**：Leaf Tensor
- **定义**：用户直接创建、不是由其他张量运算得到的张量（如 `x, y, w, b`）。
- 运算得到的张量（如 `z, loss`）是**非叶子节点**，`is_leaf` 为 `False`。

```python
x.is_leaf      # True  （手动创建）
z.is_leaf      # False （由 w*x+b 计算而来）
loss.is_leaf   # False
```

| 张量 | is_leaf |
|------|---------|
| x, y, w, b | True（叶子） |
| z, loss | False（非叶子） |

**③ `backward()` —— 反向传播**
- **必须在标量(scalar)上调用**（loss 是单个数值）。
- 沿计算图反向，用链式法则(Chain Rule)自动算出每个 `requires_grad=True` 张量的梯度。

**④ `retain_grad()` —— 保留非叶子节点的梯度**
- 默认情况下，**非叶子节点的 `.grad` 不会被保存**（算完即丢，省内存）。
- 想查看中间变量（如 `z`、`loss`）的梯度，需先调用 `z.retain_grad()`。
- 代码第二个 cell 没对 z 用 retain_grad，访问 `loss.grad` 就报了 UserWarning 并返回 None。

**⑤ 梯度累加（Gradient Accumulation）**
- PyTorch 梯度**默认累加**！代码里连续两次 `backward()` 不清零，
  `w.grad` 从 -180 变成 -360（翻倍），正是累加的证据。
- 所以训练循环里每步都要 `optimizer.zero_grad()` 清零。

---

## 五、完整训练流程（Training Loop）

> `05_train_test.py`（线性回归）和 `06_digital_recongnization.py`（手写数字多分类）
> 把前面所有知识点串成了完整的训练流程。

### 5.1 训练五步法（通用模板）

```python
for epoch in range(epochs):           # 外层：遍历所有轮次
    for input, target in loader:      # 内层：遍历每个 batch
        y_pred = model(input)         # ① 前向传播：算预测
        loss = loss_fn(y_pred, target)# ② 计算损失
        loss.backward()               # ③ 反向传播：算梯度
        optimizer.step()              # ④ 更新参数
        optimizer.zero_grad()         # ⑤ 梯度清零
```

| 步骤 | 代码 | 作用 |
|------|------|------|
| ① 前向传播 | `model(input)` | 用当前参数算出预测值 |
| ② 计算损失 | `loss_fn(...)` | 衡量预测与真实的差距 |
| ③ 反向传播 | `loss.backward()` | 自动求梯度 |
| ④ 更新参数 | `optimizer.step()` | 沿梯度反方向更新权重 |
| ⑤ 梯度清零 | `optimizer.zero_grad()` | 防止梯度累加 |

### 5.2 关键名词

**Epoch（轮次）**：把**整个训练集**完整过一遍叫一个 epoch。
**Batch（批次）**：一次喂给模型的一小撮样本。
**Iteration（迭代）**：处理完一个 batch 叫一次迭代。

> 关系：1 个 epoch = (数据总量 / batch_size) 次 iteration。

**SGD —— 随机梯度下降**
- **英文全称**：Stochastic Gradient Descent
- **PyTorch API**：`optim.SGD(model.parameters(), lr=lr)`
- 作用：根据梯度更新参数，`lr`(learning rate) 是学习率（步长）。

### 5.3 06 多分类实战的额外要点

**① 模型结构（多层感知机 MLP）**
- **英文全称**：Multi-Layer Perceptron
- **缩写**：MLP

```python
model = nn.Sequential(
    nn.Linear(784, 50), nn.ReLU(),   # 输入 784=28×28 像素
    nn.Linear(50, 100), nn.ReLU(),
    nn.Linear(100, 10),              # 输出 10 类(0~9)，不加 Softmax
)
```

- **ReLU**：Rectified Linear Unit（修正线性单元），激活函数，引入非线性。
- 最后一层**不加 Softmax**，因为 `CrossEntropyLoss` 内部已含 Softmax。

**② 训练 / 评估模式切换**

```python
model.train()   # 训练模式：启用 Dropout、BatchNorm 用批统计量
model.eval()    # 评估模式：关闭 Dropout、BatchNorm 用全局统计量
```

**③ 准确率(Accuracy)计算**

```python
y_class = model(input).argmax(dim=-1)        # 取概率最大的类别索引
acc_num += (y_class == target).sum().item()  # 统计预测正确的个数
acc = acc_num / len(ds)                       # 正确数 / 总数
```

- `argmax(dim=-1)`：在类别维上取最大值的索引，即模型预测的类别。

**④ GPU 加速**
```python
device = get_device()                          # 自动选 GPU/CPU
input, target = input.to(device), target.to(device)  # 数据搬到设备
model.to(device)                               # 模型搬到设备
```
> 注意：模型和数据必须在**同一设备**上才能计算。

---

## 六、总结与对比

### 6.1 损失函数选择速查表

| 任务类型 | 损失函数 | 全称 | API | 输入要求 |
|----------|---------|------|-----|---------|
| 回归 | MAE / L1 | Mean Absolute Error | `nn.L1Loss` | 预测值 |
| 回归 | MSE / L2 | Mean Squared Error | `nn.MSELoss` | 预测值 |
| 回归 | Smooth L1 | Smooth L1 / Huber Loss | `nn.SmoothL1Loss` | 预测值 |
| 二分类 | BCE | Binary Cross Entropy | `nn.BCELoss` | **概率**(需先 Sigmoid) |
| 二分类 | BCEWithLogits | BCE With Logits | `nn.BCEWithLogitsLoss` | **logits**(原始输出) |
| 多分类 | CE | Cross Entropy | `nn.CrossEntropyLoss` | **logits**(原始输出) |

**选择口诀：**
- 回归默认用 **MSE**；数据有较多异常值用 **MAE** 或 **Smooth L1**。
- 二分类用 **BCEWithLogits**（比 Sigmoid+BCE 更稳）。
- 多分类用 **CrossEntropy**（自带 Softmax，最后一层别加 Softmax）。

### 6.2 缩写全称对照表

| 缩写 | 英文全称 | 中文 |
|------|---------|------|
| MAE | Mean Absolute Error | 平均绝对误差 |
| MSE | Mean Squared Error | 均方误差 |
| L1 / L2 | L1 / L2 Loss | L1/L2 范数损失 |
| BCE | Binary Cross Entropy | 二元交叉熵 |
| CE | Cross Entropy | 交叉熵 |
| NLL | Negative Log Likelihood | 负对数似然 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| ReLU | Rectified Linear Unit | 修正线性单元 |
| lr | learning rate | 学习率 |

### 6.3 整章知识脉络

```
数据 → 损失 → 求导 → 训练
 │       │      │       │
 ├ Dataset/DataLoader（怎么喂数据）          [03]
 ├ 损失函数（怎么衡量好坏）                   [01 回归 / 02 分类]
 ├ Autograd 反向传播（怎么算梯度）           [04]
 └ 训练五步法（怎么迭代优化）                 [05 回归 / 06 分类]
```

**一条完整的训练链路：**

```
原始数据
  → TensorDataset 打包
  → DataLoader 分批
  → model 前向传播得预测
  → loss_fn 计算损失
  → loss.backward() 反向求梯度
  → optimizer.step() 更新参数
  → optimizer.zero_grad() 清零
  → 循环往复直到收敛
```

> 这一章是深度学习训练的**地基**：搞懂了损失函数怎么选、数据怎么加载、
> 梯度怎么自动求、训练循环怎么写，后面所有模型（CNN、RNN、Transformer）都是在这套框架上扩展。
