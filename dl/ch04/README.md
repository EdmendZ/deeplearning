# Ch04 — 用 PyTorch 搭建神经网络

本章从神经网络最基础的「全连接层 + 激活函数」开始，一步步演示用 `torch.nn` **构建模型、查看参数、保存/加载模型**，最后用一个手写数字识别（MNIST/Fashion 风格）的完整推理示例收尾。

| 文件 | 主题 |
| --- | --- |
| `01_linear_test.ipynb` | 全连接层 `nn.Linear` 与前向传播 |
| `02_activate_fun.ipynb` | 各类激活函数 |
| `03_nn_test.py` | 用 `nn.Module` 自定义模型（函数式调用激活） |
| `04_nn_test_1.py` | 把层注册为成员 + 遍历参数 / 状态字典 / 模型摘要 |
| `05_nn_sequential.py` | 用 `nn.Sequential` 快速堆叠 + 模型保存/加载 |
| `06_digital_recongnization.py` | 手写数字识别完整推理 + 准确率计算 |

> 核心概念：在 PyTorch 里，**网络层和激活函数都是 `nn.Module` 的子类**。每个 Module 实现了 `__call__`，内部会自动调用你写的 `forward`，所以 `model(x)` 等价于 `model.forward(x)`（且 `model(x)` 还会触发 hook 等机制，**推荐用 `model(x)` 而不是直接调 `forward`**）。

---

## 1. 全连接层 `nn.Linear`（`01_linear_test.ipynb`）

全连接层（线性层）做的就是 `y = x @ Wᵀ + b`。

```python
import torch.nn as nn
linear = nn.Linear(in_features=2, out_features=3, bias=True)

print(linear.weight)  # 形状 (out_features, in_features) = (3, 2)
print(linear.bias)    # 形状 (out_features,) = (3,)
```

- `weight` 和 `bias` 都是 `Parameter`，`requires_grad=True`，即**会被自动求导、参与训练**。
- 权重在创建时已被随机初始化。

### 前向传播

```python
x = torch.randn(10, 2)    # 10 条样本，每条 2 个特征

y = linear.forward(x)     # 写法一：显式调用 forward
y = linear(x)             # 写法二（推荐）：__call__ 内部会调用 forward
# 输出 shape: (10, 3)
```

- 输入形状 `(batch, in_features)`，输出 `(batch, out_features)`。
- 输出带 `grad_fn=<AddmmBackward0>`，说明 PyTorch 已记录计算图，可用于反向传播。

---

## 2. 激活函数（`02_activate_fun.ipynb`）

激活函数给网络引入**非线性**，否则多层线性层叠加仍等价于一层线性。每个激活函数都有三种等价调用方式：

```python
x = torch.tensor([0.7, -1.2, 2.5, 0, -3.1])

torch.sigmoid(x)     # 1) torch 函数
x.sigmoid()          # 2) tensor 方法
nn.Sigmoid()(x)      # 3) 当作 Module（类）来用 —— 搭网络时常用这种
```

常见激活函数：

| 函数 | 特点 / 取值范围 | 调用 |
| --- | --- | --- |
| **Sigmoid** | 压到 `(0,1)`，常用于二分类输出 | `torch.sigmoid` |
| **Tanh** | 压到 `(-1,1)`，零中心 | `torch.tanh` |
| **ReLU** | `max(0,x)`，最常用的隐藏层激活 | `torch.relu` |
| **Softmax** | 沿指定维归一化成概率（和为 1），多分类输出 | `torch.softmax(x, dim=-1)` |
| LeakyReLU / PReLU | ReLU 变体，负区间有小斜率，缓解「神经元死亡」 | `nn.LeakyReLU()` |
| ELU / Softplus / GELU | 平滑的 ReLU 类变体（GELU 在 Transformer 里常用） | `nn.ELU()` 等 |

```python
torch.softmax(x, dim=-1)   # -> [0.1296, 0.0194, 0.7838, 0.0643, 0.0029]，和为 1
```

> **Softmax 的 `dim` 很关键**：在哪个维度上归一化，那个维度的值之和就是 1。多分类一般对「类别维」做 softmax。

---

## 3. 自定义模型：继承 `nn.Module`（`03_nn_test.py`）

标准做法是继承 `nn.Module`，在 `__init__` 里定义层，在 `forward` 里串联前向逻辑。

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()            # 必须先调用父类 __init__
        self.linear_1 = nn.Linear(3, 4)
        self.linear_2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.Tanh()(x)              # 这里把激活当函数临时用
        x = self.linear_2(x)
        x = nn.ReLU()(x)
        x = self.out(x)
        y = nn.Softmax(dim=1)(x)
        return y

x = torch.randn(10, 3)   # 10 条样本，3 个特征
model = MyModel()
y = model(x)             # (10, 2)，每行是一个概率分布
```

- `super().__init__()` 必须调用，否则 `nn.Module` 的参数注册机制不会生效。
- 网络结构：`3 → 4 (Tanh) → 4 (ReLU) → 2 (Softmax)`。
- 注意这里激活函数是在 `forward` 里**临时 `nn.Tanh()(x)`** 创建的——能跑，但激活层不会作为模型成员被记录（对比下一节）。

---

## 4. 注册层 + 查看参数（`04_nn_test_1.py`）

把激活函数也作为 `self.xxx` 成员保存，这样它们会被 `nn.Module` **自动注册**，可被遍历、保存、统计。

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(3, 4)
        self.activation_1 = nn.Tanh()
        self.linear_2 = nn.Linear(4, 4)
        self.activation_2 = nn.ReLU()
        self.out = nn.Linear(4, 2)
        self.activation_3 = nn.Softmax(dim=1)

    def forward(self, x):
        for m in self.modules():          # 遍历所有子模块依次前向
            if not isinstance(m, MyModel): # 跳过模型自身
                x = m(x)
        return x
```

- `self.modules()` 会按定义顺序返回**包括模型自身在内**的所有子模块，所以要 `if not isinstance(m, MyModel)` 跳过自己，避免无限递归。
- 这种「遍历 modules 自动前向」的写法依赖**定义顺序与前向顺序一致**，本质上和 `nn.Sequential`（第 5 节）思路相同。

### 查看参数的几种方式

```python
model.linear_1.weight        # 直接访问某层权重 / 偏置
model.linear_1.bias

for param in model.parameters():            # 遍历所有可训练参数
    ...
for name, param in model.named_parameters(): # 带名字遍历，调试更直观
    print(name, param)

state_dict = model.state_dict()   # 状态字典：{参数名: 张量}，用于保存/加载
```

### 打印模型结构摘要

```python
from torchsummary import summary          # pip install torchsummary
summary(model, input_size=(3,), batch_size=10, device='cpu')
```

- `state_dict()` 是 PyTorch 保存模型的标准载体（只存参数，不存代码）。
- `torchsummary.summary` 打印每层输出形状和参数量，类似 Keras 的 `model.summary()`。

---

## 5. 快速堆叠 `nn.Sequential` + 保存/加载（`05_nn_sequential.py`）

当网络只是「层按顺序往下走」时，不必写类，直接用 `nn.Sequential`：

```python
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=-1),
)
```

### 模型的保存与加载（推荐：只存参数）

```python
# 保存：推荐保存 state_dict（参数字典），而不是整个模型对象
torch.save(model.state_dict(), 'model.pth')

# 加载：先用相同结构建好 model，再把参数灌进去
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

y = model(x)
```

> **为什么只存 `state_dict`？** 它与代码解耦、可移植、不依赖具体类定义；加载时只需先构造同结构的模型再 `load_state_dict`。直接 `torch.save(model)` 会把类路径一起序列化，换环境容易出问题。

> 运行提示：脚本里 `torch.save(...)` 被注释掉了，但 `torch.load('model.pth')` 仍会执行，因此**直接运行需要先有 `model.pth` 文件**（取消保存那行的注释先跑一次生成即可）。

---

## 6. 手写数字识别完整推理（`06_digital_recongnization.py`）

把前面所有知识串起来：加载数据 → 定义模型 → 加载训练好的参数 → 预测 → 算准确率。

### 数据加载（`common/load_data.py`）

```python
def load_digital_data():
    data = pd.read_csv('../data/train.csv')      # 1. 读 CSV
    x = data.drop("label", axis=1)               # 2. 特征/标签分离
    y = data["label"]
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42)    # 3. 8:2 划分训练/测试
    scaler = MinMaxScaler()                       # 4. 归一化到 [0,1]
    train_x = scaler.fit_transform(train_x)       #    训练集 fit_transform
    test_x = scaler.transform(test_x)             #    测试集只 transform（不能再 fit）
    # 5. 转成张量；特征用 float，标签用整型
    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    train_y = torch.tensor(train_y.to_numpy())
    test_y = torch.tensor(test_y.to_numpy())
    return train_x, test_x, train_y, test_y
```

> **关键点**：测试集只能 `transform`，不能 `fit_transform`——归一化的参数（min/max）必须来自训练集，否则会发生「数据泄漏」。

### 模型定义与推理

```python
model = nn.Sequential(
    nn.Linear(28*28, 50),   # 输入是 28×28=784 维展平的图片
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10),     # 输出 10 类（数字 0~9）
    # nn.Softmax(dim=-1)    # 训练时 softmax 与交叉熵损失合并，这里推理不需要
)

state_dict = torch.load("../data/nn_example.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

y_pred = model(test_x)
y_pred_class = torch.argmax(y_pred, dim=-1)   # 取每行最大值下标 = 预测类别
```

> **为什么输出层注释掉了 Softmax？** 训练时通常用 `nn.CrossEntropyLoss`，它**内部已经包含 Softmax**（log-softmax + NLL），所以模型最后一层只输出原始分数（logits）。推理时只关心「哪个分数最大」，`argmax` 对 logits 和 softmax 结果是一样的，故可省略。
> `map_location=cpu` 保证在没有 GPU 的机器上也能加载（即便参数是在 GPU 上训练保存的）。

### 计算准确率

```python
# 三种等价写法：预测正确的数量
acc_count = (y_pred_class == test_y).sum()
acc_count = (y_pred_class.eq(test_y)).sum()
acc_count = torch.sum(y_pred_class == test_y)

acc = acc_count.item() / len(test_y)   # 正确数 / 总数
print(acc)
```

最后还遍历找出所有预测错误的样本下标，便于做错误分析：

```python
c = 0
for i in y_pred_class == test_y:
    if not i:
        print(c)   # 打印预测错误的样本序号
    c += 1
```

---

## 全章脉络总结

```
nn.Linear  ─┐
            ├─►  组合成模型  ─┬─ 写法A: 继承 nn.Module（灵活，可写复杂逻辑）
激活函数   ─┘                 └─ 写法B: nn.Sequential（简单顺序堆叠）
                                        │
                            训练好后 → state_dict 保存/加载
                                        │
                              加载参数 → model(x) 推理
                                        │
                          argmax 取类别 → 比对标签算准确率
```

| 知识点 | API |
| --- | --- |
| 全连接层 | `nn.Linear(in, out)` |
| 调用模型 | `model(x)`（优于 `model.forward(x)`） |
| 激活函数 | `nn.ReLU()` / `nn.Softmax(dim=...)` 等 |
| 自定义模型 | 继承 `nn.Module`，写 `__init__` + `forward`，先 `super().__init__()` |
| 顺序模型 | `nn.Sequential(...)` |
| 查看参数 | `parameters()` / `named_parameters()` / `state_dict()` |
| 保存/加载 | `torch.save(model.state_dict(), p)` + `load_state_dict(torch.load(p))` |
| 模型摘要 | `torchsummary.summary` |
| 多分类预测 | `torch.argmax(y, dim=-1)` |
| 准确率 | `(pred == label).sum().item() / len(label)` |