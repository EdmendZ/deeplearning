import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 读取图片
img = plt.imread("../data/duck.jpg")
print(img.shape)    # (1080, 1080, 3)

# 2. 对图片数据进行转换，得到输入特征图
input = torch.tensor(img).permute(2,0,1).float()
print("输入特征图的维度：", input.shape)

# 3. 定义卷积层，输入输出通道数 3，卷积核维度 9×9，步幅 S = 3，填充 P = 0
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride=3, padding=0)
# print(conv.weight)
# print(conv.weight.shape)

# 4. 前向传播：对输入进行卷积操作
output = conv(input)
print("输出特征图的维度：", output.shape)

# 5. 将输出特征图转换为图片数据
# output = torch.clamp(output.int(), 0, 255)
output = (output - torch.min(output)) / (torch.max(output) - torch.min(output)) * 255
out_img = output.int().permute(1,2,0).detach().numpy()

# 画图
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(out_img)
plt.show()