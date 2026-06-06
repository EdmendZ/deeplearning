# ch06 知识点总结：优化器 · 学习率调度 · 参数初始化 · 正则化

> 本文档系统总结 `dl/ch06` 各文件涉及的深度学习知识点，
> 每个方法都给出**英文全称**与**缩写**，并在最后做对比总结。
>
> | 文件 | 主题 |
> |------|------|
> | `01_momentum.py` | 动量法优化器 Momentum |
> | `02_lr_step.py` | 学习率调度 StepLR |
> | `03_lr_multi.py` | 学习率调度 MultiStepLR |
> | `04_lr_exp.py` | 学习率调度 ExponentialLR |
> | `05_adagrad.py` | 自适应优化器 AdaGrad |
> | `06_RMSProp.py` | 自适应优化器 RMSProp |
> | `07_Adam.py` | 自适应优化器 Adam |
> | `08_param_init.ipynb` | 参数初始化 |
> | `09_reg.ipynb` | 正则化（BN / 权值衰减 / Dropout） |
> | `10_house.py` | 房价预测综合实战 |
>
> **本章主线**：上一章（ch05）解决"怎么训练"，本章解决"怎么训练得更快更好"——
> 即如何选优化器、调学习率、初始化权重、防止过拟合。

---

## 一、优化器（Optimizers）

> 优化器决定了"拿到梯度后，参数该怎么更新"。从最朴素的 SGD 一路改进到 Adam，
> 主要在解决两个问题：**① 更新方向**（动量）和 **② 各参数的学习率**（自适应）。

### 0. 实验背景：病态损失面

01~07 都用同一个二次函数做演示：

$$f(w_0, w_1) = 0.05\,w_0^2 + 1\,w_1^2$$

`w1` 方向系数是 `w0` 的 20 倍 → **各向异性
/ 病态(ill-conditioned)** 损失面。
普通 SGD 在陡峭的 w1 方向来回震荡、收敛慢，正好用来凸显各优化器的改进效果。

### 1.1 SGD —— 随机梯度下降（基线）

- **英文全称**：Stochastic Gradient Descent
- **缩写**：SGD
- **PyTorch API**：`optim.SGD(params, lr)`

**更新公式：**

$$w \leftarrow w - \eta \cdot g$$

（`η` 是学习率 lr，`g` 是当前梯度）

**缺点：** 所有参数共用一个固定学习率，在病态损失面上陡峭方向震荡、平缓方向慢。

### 1.2 Momentum —— 动量法

- **英文全称**：Momentum（Gradient Descent with Momentum）
- **缩写**：Momentum
- **PyTorch API**：`optim.SGD(params, lr, momentum=0.1)`

**更新公式（引入"速度" v）：**

$$v \leftarrow \beta v + g \qquad w \leftarrow w - \eta v$$

（`β` 即 `momentum` 系数，累积历史梯度）

```python
optimizer = optim.SGD([w_clone], lr, momentum=0.1)
```

**核心思想：** 模拟物理惯性，累积历史梯度。
- 方向一致的平缓方向（w0）→ 速度累积，**加速前进**。
- 来回摆动的陡峭方向（w1）→ 正负梯度抵消，**抑制震荡**。

### 1.3 AdaGrad —— 自适应梯度

- **英文全称**：Adaptive Gradient
- **缩写**：AdaGrad
- **PyTorch API**：`optim.Adagrad(params, lr)`

**更新公式（累积梯度平方）：**

$$s \leftarrow s + g^2 \qquad w \leftarrow w - \frac{\eta}{\sqrt{s}+\epsilon}\,g$$

```python
optimizer = optim.Adagrad([w_clone], lr=lr)
```

**核心思想：自适应学习率** —— 为每个参数维护历史梯度平方和 `s`，据此缩放步长。
- 梯度大的方向（w1）→ `s` 大 → 学习率自动变小。
- 梯度小的方向（w0）→ `s` 小 → 学习率相对较大。

**缺点：** `s` 单调累加、只增不减 → 学习率会**越来越小**，训练后期可能过早停滞。

### 1.4 RMSProp —— 均方根传播

- **英文全称**：Root Mean Square Propagation
- **缩写**：RMSProp
- **PyTorch API**：`optim.RMSprop(params, lr, alpha=0.99)`

**更新公式（梯度平方的指数加权移动平均）：**

$$s \leftarrow \alpha s + (1-\alpha)g^2 \qquad w \leftarrow w - \frac{\eta}{\sqrt{s}+\epsilon}\,g$$

```python
optimizer = optim.RMSprop([w_clone], lr=lr, alpha=0.99)
```

**核心思想：对 AdaGrad 的改进。** 用**指数加权移动平均(EWMA)** 代替无限累加：
- `alpha`（衰减系数）让旧梯度的影响逐渐衰减，`s` 不会无限增大。
- 学习率不会单调衰减到 0，**适合长时间训练**。

> **EWMA**：Exponentially Weighted Moving Average，指数加权移动平均。

### 1.5 Adam —— 自适应矩估计

- **英文全称**：Adaptive Moment Estimation
- **缩写**：Adam
- **PyTorch API**：`optim.Adam(params, lr, betas=(0.9, 0.999))`

**核心思想：Momentum + RMSProp 的结合体**

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1-\beta_1)g &&\text{(一阶矩：梯度均值，类似动量)}\\
v &\leftarrow \beta_2 v + (1-\beta_2)g^2 &&\text{(二阶矩：梯度平方均值，类似 RMSProp)}\\
\hat{m} &= \frac{m}{1-\beta_1^t},\quad \hat{v}=\frac{v}{1-\beta_2^t} &&\text{(偏差校正 Bias Correction)}\\
w &\leftarrow w - \frac{\eta}{\sqrt{\hat{v}}+\epsilon}\hat{m}
\end{aligned}
$$

```python
optimizer = optim.Adam([w_clone], lr=lr, betas=(0.9, 0.999))
```

- `betas[0]=0.9`：一阶矩衰减率（控制动量）。
- `betas[1]=0.999`：二阶矩衰减率（控制自适应学习率）。
- **偏差校正**：解决训练初期 m、v 接近 0 导致估计偏小的问题。

**特点：** 既有动量的加速抗震荡，又有 RMSProp 的自适应学习率，**收敛快、鲁棒、超参好调**，是目前**最常用、适用面最广**的优化器。

### 优化器演进总结

```
SGD（固定方向+固定学习率）
 ├─[改方向]→ Momentum（累积历史梯度，加速抗震荡）
 └─[改学习率]→ AdaGrad（累积梯度平方，自适应）
                 └─[改进]→ RMSProp（指数移动平均，不让lr衰减到0）
Momentum + RMSProp ──合体──→ Adam（兼具两者优点）★最常用
```

---

## 二、学习率调度（Learning Rate Scheduling）

> 优化器解决"怎么更新"，调度器解决"学习率随训练怎么变"。
> 通用做法：**前期大学习率快速收敛，后期小学习率精细微调**。
> 三个调度器都通过 `lr_scheduler.step()`（放在 `optimizer.step()` 之后）来更新学习率。

### 2.1 StepLR —— 等间隔衰减

- **英文全称**：Step Learning Rate
- **缩写**：StepLR
- **PyTorch API**：`torch.optim.lr_scheduler.StepLR`

```python
lr_scheduler = StepLR(sgd_optimizer, step_size=20, gamma=0.7)
```

- `step_size=20`：每 20 步衰减一次。
- `gamma=0.7`：每次学习率 × 0.7。
- 学习率呈**阶梯状**下降：0.9 → 0.63 → 0.441 → …

### 2.2 MultiStepLR —— 里程碑式衰减

- **英文全称**：Multi-Step Learning Rate
- **缩写**：MultiStepLR
- **PyTorch API**：`torch.optim.lr_scheduler.MultiStepLR`

```python
lr_scheduler = MultiStepLR(sgd_optimizer, [20, 50, 100], gamma=0.7)
```

- `milestones=[20,50,100]`：只在第 20、50、100 步衰减。
- 比 StepLR **更灵活**，可自定义衰减时机。

### 2.3 ExponentialLR —— 指数衰减

- **英文全称**：Exponential Learning Rate
- **缩写**：ExponentialLR
- **PyTorch API**：`torch.optim.lr_scheduler.ExponentialLR`

```python
lr_scheduler = ExponentialLR(sgd_optimizer, gamma=0.99)
```

- `gamma=0.99`：**每一步**学习率 × 0.99。
- 学习率呈**平滑的指数衰减**，没有阶梯式突变。

**读取当前学习率：**
```python
optimizer.param_groups[0]["lr"]   # 从优化器参数组读取当前 lr
```

### 调度器对比

| 调度器 | 衰减时机 | 曲线形状 | 关键参数 |
|--------|---------|---------|---------|
| StepLR | 每固定步数 | 阶梯（等间隔） | `step_size`, `gamma` |
| MultiStepLR | 指定里程碑 | 阶梯（不等间隔） | `milestones`, `gamma` |
| ExponentialLR | 每一步 | 平滑指数曲线 | `gamma` |

> ⚠️ **关键顺序**：`lr_scheduler.step()` 必须放在 `optimizer.step()` **之后**调用。

---

## 三、参数初始化（Parameter Initialization）

> 好的初始权重能加速收敛、避免梯度消失/爆炸。`08_param_init.ipynb` 演示了 `torch.nn.init` 的各种方法。

### 3.1 常数初始化（Constant Initialization）

```python
nn.init.zeros_(w)            # 全 0
nn.init.ones_(w)             # 全 1
nn.init.constant_(w, 5)      # 全部置为指定常数
```
> ⚠️ 隐藏层权重**不能全部初始化为同一常数**，否则所有神经元对称、学到相同特征（对称性问题）。

### 3.2 单位矩阵初始化（Identity Initialization）

```python
nn.init.eye_(w)              # 主对角线为 1，其余为 0
```

### 3.3 随机分布初始化

```python
nn.init.normal_(w, mean=0, std=1)   # 正态(高斯)分布
nn.init.uniform_(w, a=1, b=2)       # 均匀分布 [a, b]
```

### 3.4 Xavier 初始化（Glorot Initialization）

- **英文全称**：Xavier / Glorot Initialization
- **PyTorch API**：`nn.init.xavier_uniform_` / `nn.init.xavier_normal_`

```python
nn.init.xavier_uniform_(w)   # 均匀分布版本
nn.init.xavier_normal_(w)    # 正态分布版本
```

**核心思想：** 根据输入维度(fan_in)和输出维度(fan_out)自动设定方差，使每层输出方差稳定。**适合 Sigmoid / Tanh 等激活函数。**

### 3.5 He 初始化（Kaiming Initialization）

- **英文全称**：He / Kaiming Initialization
- **PyTorch API**：`nn.init.kaiming_uniform_` / `nn.init.kaiming_normal_`

```python
nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
```

- **专为 ReLU 系列激活函数设计**（考虑了 ReLU 会丢掉一半信号）。
- `mode`：`fan_in`（按输入维度，保持前向方差）或 `fan_out`（按输出维度）。
- `nonlinearity`：指定后接的激活函数。

### 初始化方法选择

| 激活函数 | 推荐初始化 |
|----------|-----------|
| Sigmoid / Tanh | **Xavier (Glorot)** |
| ReLU / LeakyReLU | **He (Kaiming)** |

---

## 四、正则化（Regularization）

> 正则化用于**缓解过拟合(overfitting)**，提升模型泛化能力。`09_reg.ipynb` 演示三种手段。

### 4.1 BatchNorm —— 批归一化

- **英文全称**：Batch Normalization
- **缩写**：BN
- **PyTorch API**：`nn.BatchNorm1d(num_features)`

```python
bn = nn.BatchNorm1d(num_features=3)   # num_features = 特征数(列数)
y = bn(x)                             # 按列(每个特征)归一化为零均值、单位方差
```

**核心思想：** 对每一批数据**按特征做标准化**（减均值除标准差），再做可学习的缩放平移。

**作用：**
- 稳定数据分布，**加速收敛**、允许更大学习率。
- 缓解梯度消失/爆炸，有轻微正则化效果。

> 训练时用当前批的统计量，评估时（`model.eval()`）用训练阶段累积的全局统计量。

### 4.2 Weight Decay —— 权值衰减（L2 正则）

- **英文全称**：Weight Decay（L2 Regularization）
- **缩写**：Weight Decay / L2
- **PyTorch API**：优化器的 `weight_decay` 参数

```python
optim.SGD(params, lr=lr, weight_decay=0.2)
```

**核心思想：** 在损失上加权重的 L2 惩罚项，等价于每步更新时让权重额外缩小一点：

$$L_{\text{total}} = L_{\text{原始}} + \lambda \sum w^2$$

**作用：** 抑制权重过大，让模型更平滑，**缓解过拟合**。`λ` 即 `weight_decay`。

### 4.3 Dropout —— 随机失活

- **英文全称**：Dropout
- **缩写**：Dropout
- **PyTorch API**：`nn.Dropout(p)`

```python
dp = nn.Dropout(p=0.5)   # 训练时以概率 p 随机置 0
y = dp(x)
```

**核心思想：** 训练时**以概率 p 随机将神经元输出置 0**，其余元素放大 `1/(1-p)` 倍（保持期望不变）。

**作用：**
- 防止神经元过度依赖彼此（协同适应），强迫网络学到更鲁棒的特征。
- 相当于训练了多个子网络的集成，**有效缓解过拟合**。

> 仅在训练时生效（`model.train()`）；评估时（`model.eval()`）自动关闭，使用全部神经元。

---

## 五、综合实战：房价预测（`10_house.py`）

> 把本章的优化器、初始化思想、正则化技巧综合用到一个真实回归任务上。

### 5.1 模型结构（含正则化的 MLP）

```python
model = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.BatchNorm1d(128),   # 批归一化：稳定+加速
    nn.ReLU(),
    nn.Dropout(p=0.3),     # 随机失活：防过拟合

    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(64, 1),      # 回归输出单个房价
)
```

**经典组合套路：`Linear → BatchNorm → ReLU → Dropout`**

### 5.2 损失函数：RMSLE

- **英文全称**：Root Mean Squared Logarithmic Error
- **缩写**：RMSLE

```python
def loss_fn(y_pred, y_target):
    mse = nn.MSELoss()
    y_pred = torch.clamp(y_pred, min=1, max=float("inf")).squeeze()  # 防止 log 出现非正数
    return torch.sqrt(mse(torch.log(y_pred), torch.log(y_target)))   # 在对数空间算 MSE 再开方
```

**为什么用 RMSLE：** 房价跨量级很大（几十万到几千万），在**对数空间**计算误差能让模型关注**相对误差**而非绝对误差，对大小房价一视同仁。

### 5.3 优化器与训练/评估模式

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

model.train()   # 训练模式：启用 Dropout、BN 用批统计量
# ... 训练循环 ...
model.eval()    # 评估模式：关闭 Dropout、BN 用全局统计量
```

> ⚠️ 用了 BN 和 Dropout 后，**必须正确切换 train/eval 模式**，否则评估结果会出错。

---

## 六、总结与对比

### 6.1 缩写全称对照表

| 缩写 | 英文全称 | 中文 |
|------|---------|------|
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| Momentum | Gradient Descent with Momentum | 动量法 |
| AdaGrad | Adaptive Gradient | 自适应梯度 |
| RMSProp | Root Mean Square Propagation | 均方根传播 |
| Adam | Adaptive Moment Estimation | 自适应矩估计 |
| EWMA | Exponentially Weighted Moving Average | 指数加权移动平均 |
| StepLR | Step Learning Rate | 等间隔学习率衰减 |
| MultiStepLR | Multi-Step Learning Rate | 里程碑学习率衰减 |
| ExponentialLR | Exponential Learning Rate | 指数学习率衰减 |
| Xavier | Xavier / Glorot Initialization | Xavier 初始化 |
| He | He / Kaiming Initialization | He(凯明) 初始化 |
| BN | Batch Normalization | 批归一化 |
| Dropout | Dropout | 随机失活 |
| L2 | L2 Regularization / Weight Decay | 权值衰减 |
| RMSLE | Root Mean Squared Logarithmic Error | 均方根对数误差 |
| MLP | Multi-Layer Perceptron | 多层感知机 |

### 6.2 优化器选择速查

| 场景 | 推荐 | 理由 |
|------|------|------|
| 通用首选 | **Adam** | 收敛快、超参好调、鲁棒 |
| 追求最终精度/泛化 | **SGD + Momentum** | 调好后泛化常优于 Adam |
| 稀疏特征 | AdaGrad | 自适应稀疏梯度 |
| RNN/长训练 | RMSProp | 学习率不会衰减到 0 |

### 6.3 防止过拟合工具箱

| 手段 | API | 作用 |
|------|-----|------|
| 权值衰减 | `weight_decay=` | 惩罚大权重 |
| Dropout | `nn.Dropout(p)` | 随机失活神经元 |
| BatchNorm | `nn.BatchNorm1d` | 稳定分布 + 轻微正则 |
| 学习率调度 | `lr_scheduler` | 后期精细微调 |

### 6.4 整章知识脉络

```
怎么训练得更快更好？
 │
 ├─ 优化器：怎么更新参数          [01 Momentum / 05 AdaGrad / 06 RMSProp / 07 Adam]
 │    SGD → Momentum / AdaGrad → RMSProp → Adam
 │
 ├─ 学习率调度：怎么调步长        [02 StepLR / 03 MultiStepLR / 04 ExponentialLR]
 │    前期快收敛，后期细微调
 │
 ├─ 参数初始化：怎么开局          [08]
 │    Sigmoid/Tanh→Xavier，ReLU→He
 │
 └─ 正则化：怎么防过拟合          [09]
      BatchNorm + Dropout + Weight Decay
 │
 └─ 综合实战                      [10 房价预测]
      Linear→BN→ReLU→Dropout + Adam + RMSLE
```

> **本章定位**：ch05 教会"训练流程怎么搭"，ch06 则是**训练调优大全**——
> 选对优化器、调好学习率、合理初始化、加上正则化，
> 这套"组合拳"是把模型从"能跑"训到"跑得好"的关键。
