import torch
import torch.nn as nn

# 定义数据
x = torch.tensor([10.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 定义参数和偏执
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 定义损失函数
loss_fn = nn.MSELoss()

# 正向传播
z = w * x + b
#z.retain_grad()

# 计算损失
loss = loss_fn(z, y)
#loss.retain_grad()

# 反向传播: 计算梯度
loss.backward()

print(f"loss: {loss.grad}")
print(f"z: {z.grad}")
print(f"w: {w.grad}")
print(f"b: {b.grad}")
print(f"x: {x.grad}")


z = w * x + b
loss = loss_fn(z, y)
loss.backward()
print("-------------------")
print(f"loss: {loss.grad}")
print(f"z: {z.grad}")
print(f"w: {w.grad}")
print(f"b: {b.grad}")
print(f"x: {x.grad}")

print(w.is_leaf)
print(b.is_leaf)
print(x.is_leaf)
print(z.is_leaf)
print(loss.is_leaf)

