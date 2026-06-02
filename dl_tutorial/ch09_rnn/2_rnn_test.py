import torch
import torch.nn as nn

# 定义RNN层
rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=2)

# 定义输入数据，形状 (L=1, N=3, H=8)
input = torch.rand(1, 3, 8)
hx = torch.randn(2, 3, 16)
print("input: ", input)
print("hx: ", hx)

# 前向传播
output, hidden = rnn(input, hx)

print(output.size())
print(hidden.size())
print("output: ", output)
print("hidden: ", hidden)
