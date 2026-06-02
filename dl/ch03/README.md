# Ch03 — PyTorch Tensor 基础

本章围绕 PyTorch 的核心数据结构 **Tensor（张量）** 展开，覆盖从创建、类型转换、运算、索引到形状变换的完整基本功。张量是深度学习里一切数据（输入、参数、梯度）的载体，可以理解为「带 GPU 加速和自动求导能力的多维数组」。

| Notebook | 主题 |
| --- | --- |
| `01_tensor_create.ipynb` | 张量的多种创建方式 |
| `02_tensor_conversation.ipynb` | 张量与 ndarray / 标量的互相转换 |
| `03_tensor_calc.ipynb` | 张量运算（元素级、矩阵乘、统计） |
| `04_tensor_index.ipynb` | 张量索引（简单、范围、列表、布尔） |
| `05_tensor_shape.ipynb` | 张量形状变换（转置、reshape、增删维度、拼接） |

---

## 1. 张量的创建（`01_tensor_create.ipynb`）

### 1.1 基于「内容」创建 —— 小写 `torch.tensor`

`torch.tensor(data)` 把已有数据（标量 / 列表 / numpy 数组）包成张量，**数据类型由传入内容自动推断**。

```python
torch.tensor(10)                       # 标量      shape: torch.Size([])
torch.tensor([10, 20, 30])             # 向量      shape: torch.Size([3])
torch.tensor([[10, 20, 30],
              [100, 200, 300]])        # 矩阵      shape: torch.Size([2, 3])
torch.tensor(np.array([[10, 20, 30]])) # 来自 numpy
```

- `.shape` 查看形状，`.dtype` 查看类型。整数列表会推断为 `torch.int64`。

### 1.2 基于「形状」创建 —— 大写 `torch.Tensor`

`torch.Tensor(2, 3)` 按给定形状申请内存（值未初始化），**默认类型为 `float32`**。

```python
torch.Tensor(2, 3)   # 2×3，dtype=torch.float32
```

> **小结（两个最易混点）**
> 1. 小写 `tensor` = 基于内容，类型由数据决定。
> 2. 大写 `Tensor` = 基于形状，类型默认 `float32`。

### 1.3 指定类型

三种等价写法把张量转成指定类型：

```python
torch.tensor([1, 3], dtype=torch.float)  # 创建时指定
torch.tensor([1, 3]).float()             # 方法转换
torch.tensor([1, 3]).to(torch.float)     # to 转换
```

也可用类型化构造器直接按形状建：`LongTensor`(int64) / `IntTensor`(int32) / `ShortTensor`(int16) / `ByteTensor`(uint8) / `FloatTensor`(float32) / `DoubleTensor`(float64) / `HalfTensor`(float16) / `BoolTensor`(bool)。

### 1.4 基于区间创建

```python
torch.arange(1, 10, 2)              # [1,10) 步长2  -> [1,3,5,7,9]
torch.arange(10)                   # [0,10)
torch.linspace(1, 9, 10)           # [1,9] 均匀取 10 个点（含端点）
torch.logspace(1, 3, 5, base=e)    # 指数等比，base^start ... base^end
```

- `linspace` 的步长 = `(end-start)/(steps-1)`，**两端都包含**；`arange` 左闭右开。

### 1.5 基于填充创建

```python
torch.ones(3, 4)      torch.ones_like(t)    # 全 1 / 形状跟随 t
torch.zeros(3, 4)     torch.zeros_like(t)   # 全 0
torch.empty(2, 3)                            # 未初始化（值是内存残留）
torch.full((2, 3), 10)                       # 全部填 10
torch.eye(5)                                 # 5×5 单位矩阵
```

### 1.6 随机创建

```python
torch.rand(2, 3)                      # 均匀分布 [0,1)
torch.randint(1, 10, (2, 3))          # 均匀分布整数 [1,10)
torch.randn(2, 3)                     # 标准正态 N(0,1)
torch.normal(mean=0, std=1, size=(2,3))  # 指定均值/标准差的正态
torch.randperm(100)                   # 0~99 的随机排列（洗牌）
```

可复现性：`torch.manual_seed(42)` 固定随机数种子，`torch.initial_seed()` 查看当前种子。**做实验时固定种子，结果才能复现。**

---

## 2. 张量转换（`02_tensor_conversation.ipynb`）

### 2.1 类型转换

```python
t1.type(torch.float32)   # type 方法
t1.to(torch.float32)     # to 方法
t1.float()               # 快捷方法（注意：返回新张量，不改原张量）
```

> 注意 notebook 中 `t2 = t1.float(); print(t2.dtype)` 打印出 `int64`，是因为 `.float()` **不是原地操作**——它返回新张量，原 `t1` 仍是 int64。要保留结果必须接收返回值。

### 2.2 Tensor ↔ ndarray（关键：内存共享）

```python
n1 = t1.numpy()           # tensor -> ndarray，二者【共享内存】
n2 = t1.numpy().copy()    # .copy() 切断共享

t1 = torch.from_numpy(n1)         # ndarray -> tensor，二者【共享内存】
t1 = torch.from_numpy(n1).clone() # .clone() 切断共享
```

- 共享内存意味着改一个另一个也变。需要独立副本时用 `.copy()` / `.clone()`。

### 2.3 Tensor → 标量

```python
t1[0].item()   # 取出单个元素为 Python 数字
```

- `.item()` 要求张量**只含一个元素**（无论它嵌套多少层 `[[[10]]]` 都行）。

---

## 3. 张量运算（`03_tensor_calc.ipynb`）

### 3.1 基本运算与原地操作

```python
t1 + 10        # 运算符
t1.add(10)     # 方法（返回新张量）
t1.add_(10)    # 带下划线 = 原地修改 t1
```

> **约定：方法名带尾部下划线 `_` 表示原地（in-place）操作**，会直接改变调用者本身（`add_`、`mul_` …）。

### 3.2 元素级运算与广播

```python
t1 = torch.tensor([[1,2,3],[4,5,6]])   # (2,3)
t2 = torch.tensor([[10],[60]])         # (2,1) 可广播到 (2,3)
t1 + t2        # 广播相加
t1 * t2        # 对位相乘 = 哈达玛积（Hadamard product）
```

- **广播（broadcasting）**：维度为 1 的轴会被自动「拉伸」对齐另一个张量，省去手写循环。

### 3.3 矩阵乘法

```python
t1 @ t2          # 推荐写法
t1.mm(t2)        # 仅限二维矩阵
t1.matmul(t2)    # 支持高维（批量矩阵乘）
```

- 高维相乘规则：**最后两个维度满足矩阵相乘条件**，其余维度保持一致或满足广播。
- `mm` 只能用于二维矩阵，对 3 维张量会报 `RuntimeError: self must be a matrix`（notebook 中演示了该报错）。

### 3.4 统计运算

```python
t1.sum()              # 全局求和
t1.sum(dim=0)         # 沿指定维度求和（该维度被压掉）
t1.max()  t1.argmax() # 最大值 / 最大值的下标
t1.max(dim=1)         # 返回 (values, indices) 具名元组
t1.float().mean(dim=0)    # 均值（mean/std/var 需浮点类型）
t1.float().std()  .var()  # 标准差 / 方差
t1.float().unique()       # 去重
t1.float().sort(dim=0)    # 排序，返回 (values, indices)
```

- 理解 `dim` 的关键：**`dim=k` 表示沿第 k 个维度做聚合，运算后该维度消失**（除非加 `keepdim=True`）。
- `mean/std/var` 不能作用于整型，需先 `.float()`。

---

## 4. 张量索引（`04_tensor_index.ipynb`）

以 `t` 形状为 `(2, 5, 4)` 为例。

### 4.1 简单索引

```python
t[0][0][0]    # 等价于
t[0, 0, 0]    # 推荐：逗号分隔多维下标
t[1, 1]       # 省略后面维度 = 取整个子张量
```

### 4.2 范围（切片）索引

```python
t[1, 1:]        # 第二块的第 1 行到末尾
t[1, 1:-2:2]    # start:stop:step，与 Python 切片一致
```

### 4.3 列表（花式）索引

用多个列表在各维度上「配对」取元素：

```python
t[[0,1], [1,2], [2,2]]   # 取 (0,1,2) 和 (1,2,2) 两个标量
t[[0,1], [1,2]]          # 维度不足时，剩余维度整体保留
```

经典应用 —— **构造独热编码（one-hot）**：

```python
labels = [0,1,3,4,5,2]
onehot = torch.zeros(6, 6)
onehot[[0,1,2,3,4,5], labels] = 1   # 每行在 label 列置 1
```

### 4.4 布尔索引

用一个布尔掩码（mask）筛选元素：

```python
mask = t > 5
t[mask]              # 取出所有 >5 的值，结果是一维

mask = t[:, :, 0] > 5    # 每行首元素 >5 的行
t[mask]

mask = t[:, 0, :] > 5    # 每列首元素 >5，配合 .mT 转置筛列
t.mT[mask].mT
```

---

## 5. 张量形状变换（`05_tensor_shape.ipynb`）

以 `t1` 形状为 `(3, 2, 6)` 为例。

### 5.1 维度变换（转置 / 重排）

```python
t1.T                  # 全维度逆序转置 (3,2,6) -> (6,2,3)
t1.mT                 # 只转最后两个维度（matrix transpose）
t1.transpose(0, 1)    # 交换任意两个维度
t1.permute(2, 1, 0)   # 重排所有维度（最灵活）
```

- **转置的本质是「索引交换」**，并不会真正搬动内存，只是改变访问顺序 → 导致内存不连续。

### 5.2 reshape / view（改变形状）

```python
t1.reshape(-1)          # 拉平成一维（-1 自动推断）
t1.reshape(-1, 9)       # 自动算行数
t1.reshape(3, 1, 2, -1) # 升维
t1.view(3, 1, 2, -1)    # 与 reshape 类似，但要求内存连续
```

**`view` vs `reshape`：**

```python
t1.is_contiguous()              # True
t1.mT.is_contiguous()           # False（转置后不连续）
t1.mT.contiguous().view(-1)     # 先 .contiguous() 才能 view
```

- `view` **要求底层内存连续**，否则报错；转置/permute 之后需先 `.contiguous()`。
- `reshape` 更安全：连续时等价于 view，不连续时自动复制。

### 5.3 增 / 删维度

```python
t1.unsqueeze(dim=-1)   # 在指定位置插入一个大小为 1 的维度
t2.squeeze()           # 删除所有大小为 1 的维度
```

- 常用于对齐维度（如给数据补 batch / channel 维）。

### 5.4 张量拼接

```python
torch.cat((t1, t2), dim=1)    # 沿已有维度拼接，其他维度必须匹配
torch.stack((t1, t2), dim=0)  # 先 unsqueeze 再 cat，会【新增】一个维度
torch.vstack((t1, t2))        # 垂直堆叠（沿第 0 维）
```

- `cat` 与 `stack` 的本质区别：**`cat` 在已有维度上接长，`stack` 会多出一个新维度**。

---

## 速查表

| 需求 | 用法 |
| --- | --- |
| 按内容建张量（类型自动） | `torch.tensor(data)` |
| 按形状建张量（float32） | `torch.Tensor(2, 3)` |
| 转类型 | `.to(dtype)` / `.float()` / `.type()` |
| Tensor↔numpy（共享内存） | `.numpy()` / `torch.from_numpy()` |
| 取标量 | `.item()` |
| 原地运算 | 方法名加 `_`（`add_`） |
| 矩阵乘 | `@` / `matmul`（高维）/ `mm`（仅二维） |
| 沿维度聚合 | `sum/mean/max(dim=k)` |
| 布尔筛选 | `t[t > 5]` |
| 拉平/改形状 | `reshape(-1)` / `view(...)` |
| 转置 | `.T` / `.mT` / `transpose` / `permute` |
| 拼接 | `cat`（接长）/ `stack`（升维） |
| 固定随机种子 | `torch.manual_seed(42)` |