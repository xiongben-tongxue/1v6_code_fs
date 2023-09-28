import numpy as np
import torch
import jieba
# 从transformers库导入AutoTokenizer和AutoModelForMaskedLM类，用于处理和加载BERT模型。
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utilz import *

# 处理一个文本数据集，使用BERT模型提取文本特征，并保存这些特征以供后续使用。
# 加载预训练的BERT中文模型的tokenizer，用于文本的分词和编码。
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# 加载预训练的BERT中文模型，并设置output_hidden_states=True以输出模型的隐藏状态。
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese",
                                             output_hidden_states = True)
# 调用load_data函数（可能定义在utilz模块中）加载数据。
video_ids, clip_ids, texts, annotations, modes = load_data('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/label.csv')
# 显示texts列表的前5项，但是该行没有任何输出操作，可能是用于调试。
texts[:5]
# 初始化一个字典，用于存储训练、验证和测试数据的词嵌入。
word_embs = {'train':[], 'valid':[], 'test':[]}
# 初始化空列表和变量，然后遍历texts，使用jieba分词，将分词结果添加到vocabs列表，并更新max_len为最长的文本长度。
vocabs = []
max_len = 0
for s in texts:
    tokens = list(jieba.lcut(s))
    vocabs.append(tokens)
    if len(tokens) > max_len:
        max_len = len(tokens)

print(len(vocabs))
print(max_len)

# 遍历texts和modes，为每个文本生成BERT词嵌入。
# 在循环中，它首先添加特殊标记"[CLS]"和"[SEP]"
# 然后使用tokenizer进行分词和编码
# 接着将tokens和segment ids转换为torch张量
# 并传递给BERT模型以获取词嵌入。
# 如果词嵌入的长度小于max_len，则用零填充至max_len。
for text, mode in zip(texts, modes):
    # print(text)
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        tmp_embs = outputs[-1][0].cpu().detach().numpy()[0]
    if len(tmp_embs) < max_len:
        tmp_embs = np.concatenate([tmp_embs, np.zeros((max_len-len(tmp_embs), 768))],
                                  axis=0)
    word_embs[mode].append(tmp_embs[:max_len])
# 调用save_features函数（可能定义在utilz模块中）将word embeddings保存到文件。
save_features(word_embs, '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_bert.pkl')

# 获取训练数据的word embeddings的形状，但是该行没有任何输出操作，可能是用于调试。
np.shape(word_embs['train'])

# 打印训练数据的word embeddings的形状。
print(np.shape(word_embs['train']))
