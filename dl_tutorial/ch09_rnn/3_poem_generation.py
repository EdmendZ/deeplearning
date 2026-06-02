import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re       # 引入正则

# 1. 数据预处理
def preprocess(file_path):
    poem_list = []
    char_set = set()

    # 1.1 从文件按行读取每一首诗
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 逐行处理，先用正则去除标点符号
            line = re.sub(r'[，。、？！：]', '', line).strip()
            # 将原诗保存到 poem_list 中
            poem_list.append(list(line))
            # 按字去重，保存到 char_set 中
            char_set.update(list(line))
    # 1.2 构建词表（id2word 列表，和 word2id 字典）
    vocab = list(char_set) + ["<UNK>"]
    word2id = {word:id for id, word in enumerate(vocab)}

    # 1.3 将原始诗转换为id的序列
    id_squences = []
    for poem in poem_list:
        id_seq = [word2id.get(word) for word in poem]
        id_squences.append(id_seq)
    return id_squences, vocab, word2id

id_squences, vocab, word2id = preprocess('../data/poems.txt')
print(len(id_squences))
print(len(vocab))
print(len(word2id))
# print(id_squences)

# 2. 定义训练数据集DataSet
class PoetryDataset(Dataset):
    def __init__(self, id_squences, seq_len):
        self.data = []
        self.seq_len = seq_len
        # 遍历诗的id列表，截取长度为L的序列x和“后续”序列y
        for seq in id_squences:
            # 遍历当前诗（seq）的每一个字的id
            for i in range(0, len(seq) - self.seq_len):
                self.data.append( (seq[i:i+self.seq_len], seq[i+1:i+self.seq_len+1]) )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从data中获取序列数据，转换成张量
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y

dataset = PoetryDataset(id_squences, 24)
print(len(dataset))

# 3. 搭建模型
class PoetryRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=1):
        super().__init__()
        # 定义模型中的层：嵌入层，RNN，全连接层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, input, hx=None):
        embedded = self.embedding(input)
        output, hn = self.rnn(embedded, hx)
        output = self.linear(output)
        return output, hn

# 定义模型
model = PoetryRNNLM(len(vocab), embedding_dim=256, hidden_size=512, num_layers=2)

# 4. 模型训练
def train(model, dataset, lr, epoch_num, batch_size, device):
    # 4.1 初始化相关操作
    model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss()    # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)   # 优化器

    # 4.2 迭代训练
    for epoch in range(epoch_num):
        loss_total = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 小批量梯度下降更新参数
        for batch_count, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # print("x: ", x.shape)
            # print("y: ", y.shape)
            output, _ = model(x)   # 前向传播
            # print("output: ", output.shape)
            loss_value = loss(output.transpose(1, 2), y)    # 计算损失
            loss_value.backward()   # 反向传播
            optimizer.step()    # 更新参数
            optimizer.zero_grad()   # 梯度清零

            loss_total += loss_value.item() * x.shape[0]

            print(f"\repoch:{epoch:0>3}[{'=' * (int((batch_count + 1) / len(dataloader) * 50)):<50}]", end="")
        # 计算训练平均损失
        print(f"loss: {loss_total / len(dataset):.6f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, dataset, lr=1e-3, epoch_num=20, batch_size=32, device=device)

# 5. 生成文本
def generate_poem(model, vocab, word2id, start_token, line_num=4, line_length=7):
    # 设置验证模式
    model.eval()
    poem = []
    current_len = line_length   # 用变量保存每行剩余字数
    # 判断起始字，放入poem中
    start_token_id = word2id.get(start_token, word2id['<UNK>'])
    if start_token_id != word2id['<UNK>']:
        poem.append(vocab[start_token_id])
        current_len -= 1

    # 定义神经网络的输入数据，形状(1, 1)
    input = torch.LongTensor([[start_token_id]]).to(device)

    # 逐行逐字生成诗句
    with torch.no_grad():
        for i in range(line_num):
            for interpunction in ["，", "。\n"]:
                # 生成每句诗的 line_length个字
                while current_len > 0:
                    # 前向传播
                    output, _ = model(input)
                    prob = torch.softmax(output[0,0], dim=-1)
                    # 按概率进行随机选取
                    next_token = torch.multinomial(prob, num_samples=1)
                    # 将下一个字的id转成汉字，保存
                    poem.append(vocab[next_token.item()])
                    # 更新迭代
                    input = next_token.unsqueeze(0)
                    current_len -= 1
                # 当前诗句已生成，追加标点
                poem.append(interpunction)
                current_len = line_length
    return "".join(poem)

print(generate_poem(model, vocab, word2id, start_token="一", line_num=4, line_length=7))