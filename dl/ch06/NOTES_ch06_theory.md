# ch06 理论精要：优化器 · 学习率调度 · 参数初始化 · 正则化

> 
本文档**重理论、轻代码**，系统梳理 `dl/ch06` 涉及的核心方法。
> 每个方法给出**英文全称 / 缩写**、**数学原理**、**直觉解释**与**适用场景**，最后做横向对比。
> 配套可运行代码与逐行注释见同目录 `NOTES_ch06_summary.md` 及各 `.py` / `.ipynb` 文件。

| 主题 | 涉及文件 | 关键词 |
|------|----------|--------|
| 优化器 | `01_momentum.py` · `05_adagrad.py` · `06_RMSProp.py` · `07_Adam.py` | Momentum / AdaGrad / RMSProp / Adam |
| 学习率调度 | `02`～`04_lr_*.py` | StepLR / MultiStepLR / ExponentialLR |
| 参数初始化 | `08_param_init.ipynb` | Xavier / He(Kaiming) |
| 正则化与训练稳定 | `09_reg.ipynb` · `10_house.py` | BatchNorm / Weight Decay / Dropout |

本章贯穿一个**各向异性二次损失面** $f(w)=0.05\,w_0^2 + 1\cdot w_1^2$ 作为演示：$w_1$ 方向陡峭、$w_0$ 方向平缓。这是观察"震荡 / 收敛 / 自适应"行为的经典玩具问题。

---

## 一、优化器（Optimizers）

> 优化器决定**如何用梯度更新参数**。所有方法都围绕一个核心痛点：
> 在各向异性损失面上，单一学习率要么在陡峭方向震荡，要么在平缓方向爬行太慢。

记号：参数 $\theta$，梯度 $g_t=\nabla_\theta L$，学习率 $\eta$，小常数 $\epsilon$（防除零）。

### 1.0 SGD —— 随机梯度下降（基线）

- **英文全称**：Stochastic Gradient Descent
- **更新式**：$\theta_{t+1} = \theta_t - \eta\, g_t$

**问题**：所有方向共用同一个 $\eta$。陡峭方向（$w_1$）步子过大来回横跳，平缓方向（$w_0$）步子过小，整体收敛慢。后续所有优化器都是为缓解这一点而生。

### 1.1 Momentum —— 动量法

- **英文全称**：Momentum / SGD with Momentum
- **核心思想**：模拟物理惯性，累积历史梯度的指数加权平均，沿一致方向加速、对反复变号的方向抑制。

**更新式**（PyTorch 实现）：

$$v_{t} = \mu\, v_{t-1} + g_t,\qquad \theta_{t+1} = \theta_t - \eta\, v_t$$

- $\mu$ 为**动量系数**（`momentum`，常取 0.9）。$v_t$ 是"速度"。
- **直觉**：陡峭方向梯度正负交替，累加后相互抵消 → 震荡减小；平缓方向梯度方向一致，累加后增大 → 加速前进。

**适用**：几乎所有任务都可在 SGD 上叠加动量，是 SGD 的默认增强。

### 1.2 AdaGrad —— 自适应梯度

- **英文全称**：Adaptive Gradient
- **核心思想**：为**每个参数维度**维护一个学习率，按其历史梯度平方和缩放。

**更新式**：

$$r_t = r_{t-1} + g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t}+\epsilon}\, g_t$$

- **直觉**：梯度一直很大的方向（陡峭）$r_t$ 累积快 → 有效学习率自动变小；梯度小的方向步长相对保留。天然适配各向异性损失面。
- **致命缺陷**：$r_t$ **单调递增且永不衰减**，训练后期分母过大 → 学习率趋近于 0，**过早停止学习**。

**适用**：稀疏特征、短期训练；长程训练不推荐（被 RMSProp 取代）。

### 1.3 RMSProp —— 均方根传播

- **英文全称**：Root Mean Square Propagation
- **核心思想**：把 AdaGrad 的"无限累加"换成**指数加权移动平均（EWMA）**，让旧梯度逐渐遗忘。

**更新式**：

$$r_t = \alpha\, r_{t-1} + (1-\alpha)\, g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t}+\epsilon}\, g_t$$

- $\alpha$ 为**衰减率**（`alpha`，常取 0.99）。
- **直觉**：$r_t$ 只反映"最近一段"的梯度规模，不再无限增长 → 学习率不会单调衰减到 0，**适合长时间训练**。

**适用**：RNN 等长序列、非平稳目标；是 AdaGrad 的直接改进版。

### 1.4 Adam —— 自适应矩估计

- **英文全称**：Adaptive Moment Estimation
- **核心思想**：**Momentum（一阶矩）+ RMSProp（二阶矩）二合一**，并做偏差校正。是目前最通用的默认优化器。

**更新式**：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad\text{(一阶矩，类动量)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad\text{(二阶矩，类 RMSProp)}$$

**偏差校正**（修正初期 $m,v$ 偏向 0 的问题）：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^{t}},\qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}}$$

**参数更新**：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\, \hat{m}_t$$

- `betas=(β₁, β₂)` 默认 `(0.9, 0.999)`。$\beta_1$ 控制动量、$\beta_2$ 控制自适应缩放。
- **直觉**：既有动量的"方向加速 + 抑震荡"，又有 RMSProp 的"逐维自适应步长"，开箱即用、对超参不敏感。

**适用**：绝大多数深度学习任务的首选默认值。

### 优化器横向对比

| 优化器 | 一阶矩(动量) | 二阶矩(自适应) | 学习率会衰减到 0？ | 一句话定位 |
|--------|:---:|:---:|:---:|------|
| SGD | ✗ | ✗ | — | 基线，方向单一 |
| Momentum | ✓ | ✗ | — | 加速 + 抗震荡 |
| AdaGrad | ✗ | ✓(累加) | **会**（缺陷） | 逐维自适应，后期熄火 |
| RMSProp | ✗ | ✓(EWMA) | 否 | AdaGrad 的长程修复版 |
| **Adam** | ✓ | ✓(EWMA) | 否 | 动量 + RMSProp，万能默认 |

---

## 二、学习率调度（Learning Rate Scheduling）

> 调度器**不改变更新规则**，只在训练过程中按策略**逐步降低学习率**：
> 早期大步快速逼近，后期小步精细收敛。PyTorch 中通过 `lr_scheduler.step()` 触发。

| 调度器 | 英文全称 | 衰减规则 | 曲线形状 |
|--------|----------|----------|----------|
| **StepLR** | Step Learning Rate | 每隔 `step_size` 步：$\eta \leftarrow \eta\cdot\gamma$ | 等间隔阶梯 |
| **MultiStepLR** | Multi-Step LR | 在指定里程碑 `milestones` 处：$\eta \leftarrow \eta\cdot\gamma$ | 自定义阶梯 |
| **ExponentialLR** | Exponential LR | 每一步：$\eta_t = \eta_0\cdot\gamma^{t}$ | 平滑指数衰减 |

- **StepLR**：最常用，固定周期衰减（如每 20 步 ×0.7）。
- **MultiStepLR**：比 StepLR 灵活，可在 `[20, 50, 100]` 等任意节点衰减，适配分阶段训练。
- **ExponentialLR**：每步都乘 $\gamma$（如 0.99），**连续平滑**无阶梯突变。

**对比记忆**：StepLR/MultiStepLR 是"台阶式"（离散下降），ExponentialLR 是"滑梯式"（连续下降）。三者都用 $\gamma<1$ 控制衰减速度。

---

## 三、参数初始化（Parameter Initialization）

> 初始权重的分布直接影响**梯度能否稳定地正向 / 反向流动**。
> 不当初始化会导致梯度消失 / 爆炸，使深层网络无法训练。PyTorch 接口在 `torch.nn.init`。

### 3.1 朴素初始化（演示用，实战慎用）

| 方法 | API | 说明 |
|------|-----|------|
| 常数 | `constant_` / `zeros_` / `ones_` | 全部置同一值。**全相同会导致神经元对称、无法区分**，仅作演示 |
| 单位阵 | `eye_` | 主对角线为 1，其余 0 |
| 正态分布 | `normal_(mean, std)` | 从高斯分布采样 |
| 均匀分布 | `uniform_(a, b)` | 从 $[a,b]$ 均匀采样 |

### 3.2 Xavier 初始化（Glorot 初始化）⭐

- **英文全称**：Xavier / Glorot Initialization
- **目标**：让**前向激活值**与**反向梯度**的方差在层间保持一致。
- **原理**：方差由扇入 $n_{in}$ 与扇出 $n_{out}$ 共同决定：

$$\text{Var}(W) = \frac{2}{n_{in}+n_{out}}$$

  - 均匀版 `xavier_uniform_`：$W\sim U\!\left[-\sqrt{\tfrac{6}{n_{in}+n_{out}}},\ \sqrt{\tfrac{6}{n_{in}+n_{out}}}\right]$
  - 正态版 `xavier_normal_`
- **适用**：**Sigmoid / Tanh** 等关于 0 对称、近似线性的激活函数。

### 3.3 He 初始化（Kaiming 初始化）⭐

- **英文全称**：He / Kaiming Initialization
- **动机**：ReLU 会把约一半输入置 0，砍掉一半方差，Xavier 在此场景偏小。He 据此**翻倍方差补偿**：

$$\text{Var}(W) = \frac{2}{n_{in}}$$

- API：`kaiming_uniform_` / `kaiming_normal_`，参数 `mode='fan_in'`（按扇入，保前向方差）或 `'fan_out'`（按扇出，保反向方差），`nonlinearity='relu'`。
- **适用**：**ReLU / LeakyReLU** 系列激活函数 —— 现代 CNN/MLP 的默认选择。

**选型口诀**：**Tanh/Sigmoid → Xavier；ReLU → He(Kaiming)**。

---

## 四、正则化与训练稳定（Regularization）

> 三种最常用手段，分别从**激活分布**、**权重大小**、**网络结构**三个角度抑制过拟合、稳定训练。

### 4.1 BatchNorm —— 批归一化 ⭐

- **英文全称**：Batch Normalization
- **API**：`nn.BatchNorm1d(num_features)`（对每个特征通道独立归一化）
- **原理**：对一个 mini-batch，按特征维计算均值 $\mu_B$、方差 $\sigma_B^2$，标准化后再用可学习的 $\gamma,\beta$ 缩放平移：

$$\hat{x} = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y = \gamma\hat{x} + \beta$$

- **作用**：缓解内部协变量偏移，使各层输入分布稳定 → **加速收敛、允许更大学习率、轻微正则化效果**。
- **训练 / 评估差异（关键）**：训练用**当前批统计量**，评估用训练期累积的**全局滑动平均**。故必须正确切换 `model.train()` / `model.eval()`。

### 4.2 Weight Decay —— 权值衰减（L2 正则）

- **英文全称**：Weight Decay（等价 L2 Regularization）
- **原理**：在损失上附加权重平方惩罚 $\frac{\lambda}{2}\lVert\theta\rVert^2$，等效于每步更新对权重额外乘以收缩因子：

$$\theta_{t+1} = \theta_t - \eta\,(g_t + \lambda\,\theta_t)$$

- **作用**：惩罚过大权重 → 模型更平滑、降低过拟合。
- **用法**：作为优化器参数传入即可，如 `SGD(..., weight_decay=0.2)`，无需手写惩罚项。

### 4.3 Dropout —— 随机失活 ⭐

- **英文全称**：Dropout
- **API**：`nn.Dropout(p)`
- **原理**：训练时以概率 $p$ 随机将神经元输出置 0，保留的神经元放大 $\frac{1}{1-p}$（**inverted dropout**，保证期望不变）。
- **作用**：阻止神经元间的"共适应"，相当于训练时集成大量子网络 → 强力抑制过拟合。
- **训练 / 评估差异（关键）**：仅在训练时丢弃；评估时**关闭 Dropout**、使用完整网络。同样依赖 `train()` / `eval()` 切换。

### 三者对比

| 方法 | 作用对象 | 主要收益 | 是否依赖 train/eval 切换 |
|------|----------|----------|:---:|
| BatchNorm | 层激活分布 | 稳定 + 加速 + 轻正则 | **是** |
| Weight Decay | 权重大小 | 抑制过拟合（平滑） | 否 |
| Dropout | 网络连接 | 强抑制过拟合 | **是** |

---

## 五、综合实战：房价预测（`10_house.py`）

把本章方法串成一条完整流水线，是知识点的集大成演示：

- **网络结构**：每个隐藏层 `Linear → BatchNorm1d → ReLU → Dropout`
  —— 归一化稳定 + 非线性表达 + 随机失活防过拟合。
- **优化器**：`Adam(betas=(0.9, 0.999))` —— 万能默认。
- **损失函数**：**RMSLE**（Root Mean Squared Logarithmic Error，均方根对数误差）

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_i\big(\log \hat{y}_i - \log y_i\big)^2}$$

  在对数空间度量误差，对房价这类**跨数量级**的目标更稳健，关注相对误差而非绝对误差（需先 `clamp(min=1)` 避免 $\log$ 非正）。
- **训练 / 评估规范**：训练循环用 `model.train()`，评估用 `model.eval()` + `torch.no_grad()` —— 正确触发 BatchNorm / Dropout 的两套行为并节省显存。

---

## 六、全章速记总结

```
更新方向        SGD → +惯性 → Momentum
逐维步长        SGD → +历史梯度² → AdaGrad → +遗忘(EWMA) → RMSProp
集大成          Momentum + RMSProp + 偏差校正 → Adam ⭐(默认)

学习率衰减      StepLR(等间隔) / MultiStepLR(里程碑) / ExponentialLR(平滑)

初始化          Tanh/Sigmoid → Xavier ；ReLU → He(Kaiming)

正则化          BatchNorm(稳分布) + WeightDecay(L2 缩权) + Dropout(随机失活)
                ↑ BatchNorm 与 Dropout 必须区分 train()/eval()
```

> **一句话**：本章解决"**怎么把网络又快又稳地训练好**"——
> 优化器和调度器管"**走得快**"，初始化管"**起步稳**"，正则化管"**不过拟合**"。
