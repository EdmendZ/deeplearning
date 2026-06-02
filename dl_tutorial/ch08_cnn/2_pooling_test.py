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

# 4. 前向传播：卷积
output1 = conv(input)
print("卷积后输出特征图的维度：", output1.shape)

# 5. 定义池化层
pool = nn.MaxPool2d(kernel_size=6, stride=6, padding=1)
# 6. 前向传播：池化
output2 = pool(output1)
print("池化后输出特征图的维度：", output2.shape)

# 7. 将输出特征图转换为图片数据
output1 = (output1 - torch.min(output1)) / (torch.max(output1) - torch.min(output1)) * 255
output1_img = output1.int().permute(1,2,0).detach().numpy()
output2 = (output2 - torch.min(output2)) / (torch.max(output2) - torch.min(output2)) * 255
output2_img = output2.int().permute(1,2,0).detach().numpy()

# 画图
fig, ax = plt.subplots(1, 3, figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(output1_img)
ax[2].imshow(output2_img)
plt.show()