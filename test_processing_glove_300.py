import os
import torch
import numpy as np
import gensim
import gensim.downloader
from gensim.models import Word2Vec
import jieba
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utilz import *

# 首先加载数据，使用jieba进行中文分词，然后训练Word2Vec模型，创建词向量表示，然后保存。
# 接着，它还加载了预训练的GloVe模型，并计算了两个单词的相似度。

# 1.加载数据
video_ids, clip_ids, texts, annotations, modes = load_data('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/label.csv')
texts[:5]

# 2.打印前5条文本数据
print(texts[:5])

# 3.jieba.lcut中文分词工具
example_seg = list(jieba.lcut(texts[0]))
print(example_seg)

# Construct a vocabulary and train a word2vec
# 构建词汇表vocabs,并找出最大句子长度max_len
vocabs = []
max_len = 0
for s in texts:
    tokens = list(jieba.lcut(s))
    vocabs.append(tokens)
    if len(tokens) > max_len:
        max_len = len(tokens)

# 打印词汇表长度和最大句子长度
print(len(vocabs))
print(max_len)

# 打印前10个分词后的句子
vocabs[:10]
print(vocabs[:10])

# 进行模型训练，适用的是Word2Vec模型
model_own = Word2Vec(sentences=vocabs, vector_size=100, sg=1, min_count=1)
model_own.save("word2vec.model")
model_own.train(vocabs, total_examples=24162, epochs=10)

#创建文本的词向量表示
word_embs = {'train':[], 'valid':[], 'test':[]}

for s, mode in zip(vocabs, modes):
    tmp_embs = []
    for w in s:
        tmp_embs.append(model_own.wv[w])
    if len(tmp_embs) < max_len:
        tmp_embs = np.concatenate([tmp_embs, np.zeros((max_len - len(tmp_embs), 100))], axis=0)
    word_embs[mode].append(tmp_embs[:max_len])

# 保存词向量表示
save_features(word_embs, '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_wav2vec.pkl')

# 打印训练集词向量表示的shape
np.shape(word_embs['train'])
print(np.shape(word_embs['train']))

# 获取gensim可以下载的预训练模型的列表，并打印
gensim.downloader.info()['models'].keys()
print(gensim.downloader.info()['models'].keys())

# 加载预训练的Glove模型
glove_300 = gensim.downloader.load('glove-wiki-gigaword-300')

# 进行’friend‘和’enemy‘在GloVe模型中的相似度，并打印
glove_300.similarity('friend', 'enemy')
print(glove_300.similarity('friend', 'enemy'))