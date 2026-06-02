import torch
import torch.nn as nn
import jieba

# torch.manual_seed(42)
text = "自然语言是由文字构成的，而语言的含义是由单词构成的。即单词是含义的最小单位。因此为了让计算机理解自然语言，首先要让它理解单词含义。"

# 1. 调用jieba库分词
original_words = jieba.lcut(text)
print(original_words)
print(len(original_words))

# 2. 过滤词汇，构建词表
stopwords = {"的", "是", "而", "由", "，", "。"}
# 2.1 过滤
words = [word for word in original_words if word not in stopwords]
print(words)
print(len(words))
# 2.2 构建id2word
vocab = list(set(words))    # id2word 词表，利用set去重
print(vocab)
print(len(vocab))
# 2.3 构建word2id字典
word2id = dict()
for id, word in enumerate(vocab):
    word2id[word] = id
print(word2id)

# 3. 定义词嵌入层
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=5)
print(embedding.weight)

# 4. 前向传播，得到词向量
for id, word in enumerate(vocab):
    word_vec = embedding(torch.tensor(id))
    print(f"id:{id:>2}, word:{word:8},\t word_vec:{word_vec.detach().numpy()}")