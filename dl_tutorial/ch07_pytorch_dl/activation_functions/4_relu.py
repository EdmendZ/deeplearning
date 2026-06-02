import torch
import matplotlib.pyplot as plt

# 定义x和y
x = torch.linspace(-5, 5, 1000, requires_grad=True)
y = torch.relu(x)

# 创建子图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# fig.set_size_inches(12, 4)

# 1. relu函数图像
ax[0].plot(x.detach(), y.detach(), color='purple')
ax[0].set_title('relu(x)')

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_position('zero')
ax[0].spines['bottom'].set_position('zero')

# 反向传播计算梯度，需要先得到一个标量
y.sum().backward()

# 2. 导函数图像
ax[1].plot(x.data, x.grad, color='purple')
ax[1].set_title("relu'(x)")

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_position('zero')
ax[1].spines['bottom'].set_position('zero')

plt.show()
