import os
import torch
import numpy as np
import librosa
# import opensmile
# import transformers as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from utilz import *


# Chinese  加载预训练的Wav2Vec2处理器和模型，用于后续的特征提取。此处是专门针对中文的
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

# 启用梯度检查点,这个有助于减少模型训练时候的内存使用情况
model.gradient_checkpointing_enable()

# 该函数用于提取音频文件的特征。
# 该函数接受三个参数：文件路径、提取模式和最大长度。
# 根据提取模式的不同，使用不同的方法来提取特征，并将提取到的特征截断或填充到指定的最大长度。
def get_speech_feature(file, mode, max_len=128):
    if mode == 'mel':
        y, sr = librosa.load(file)
        mel_spec = librosa.feature.melspectrogram(y, sr, n_mels=128).T  # (time_steps, 128) (100,128)
        if len(mel_spec) < max_len:
            mel_spec = np.concatenate([mel_spec, np.zeros((max_len - len(mel_spec), 128))], axis=0)
        return mel_spec[:max_len]  # (128,128)

    # elif mode == 'opensmile':
    #     smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
    #                             feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    #     sml_fs = smile.process_file(file)
    #     if len(sml_fs) < max_len * 2:
    #         sml_fs = np.concatenate([sml_fs, np.zeros((max_len * 2 - len(sml_fs), 65))], axis=0)
    #     return sml_fs[:max_len * 2]  # (256,65)

    elif mode == 'wav2vec':
        y, sr = librosa.load(file, sr=16000)
        inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model.wav2vec2(**inputs)
        wav2vec_fs = outputs.extract_features[0].cpu().detach().numpy()  # for CH
        print(type(wav2vec_fs))
        if len(wav2vec_fs) < max_len:
            wav2vec_fs = np.concatenate([wav2vec_fs, np.zeros((max_len - len(wav2vec_fs), 512))], axis=0)
        return wav2vec_fs[:max_len]  # (128,512)


MODE = 'wav2vec'

#下面这两个文件，必须要跑完整，因为后面要用，如果跑不完整，结果就不会很好
# video_path = '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/'
video_path = 'C:/Test/MSA Datasets/SIMS/Raw/'
# video_ids, clip_ids, texts, annotations, modes = load_data('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/label.csv')
video_ids, clip_ids, texts, annotations, modes = load_data('C:/Test/MSA Datasets/SIMS/label.csv')

# 初始化一些字典和列表，用于存储提取到的特征和标签
acoustic = {'train': [], 'valid': [], 'test': []}
labels = {'train': [], 'valid': [], 'test': []}
label_dict = {'Negative': 0, 'Positive': 1, 'Neutral': 2}

# 遍历每个音频文件，调用get_speech_feature函数提取特征，并将提取到的特征和标签分别存储到相应的字典和列表中。
for video_id, clip_id, annotation, mode in zip(video_ids, clip_ids, annotations, modes):
    clip_id_ = '000' + str(clip_id)
    file_path = video_path + str(video_id) + '/' + clip_id_[-4:] + '.mp4'

    print(file_path, mode)

    acoustic[mode].append(get_speech_feature(file_path, mode=MODE))
    labels[mode].append(label_dict[annotation])

# 我这里因为数据没有跑完整，导致在后面的训练集中数据的模型不太好，现在将windows系统中跑完的数据集拿过来用
save_features(acoustic, 'C:/Test/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
save_features(labels, 'C:/Test/MSA Datasets/SIMS/data/labels.pkl')