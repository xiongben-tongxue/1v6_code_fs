import os
import torch
import numpy as np
import librosa
import opensmile
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from utilz import *

# Chinese
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

def get_speech_feature_fun(file, mode, max_len=128):
    if mode == 'mel':
        y, sr = librosa.load(file)
        mel_spec = librosa.feature.melspectrogram(y, sr, n_mels=128).T  # (time_steps, 128) (100,128)
        if len(mel_spec) < max_len:
            mel_spec = np.concatenate([mel_spec, np.zeros((max_len - len(mel_spec), 128))], axis=0)
        return mel_spec[:max_len]  # (128,128)

    elif mode == 'opensmile':
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        sml_fs = smile.process_file(file)
        if len(sml_fs) < max_len * 2:
            sml_fs = np.concatenate([sml_fs, np.zeros((max_len * 2 - len(sml_fs), 65))], axis=0)
        return sml_fs[:max_len * 2]  # (256,65)

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

video_path = 'C:/Test/MSA Datasets/SIMS/Raw/'
video_ids, clip_ids, texts, annotations, modes = load_data('C:/Test/MSA Datasets/SIMS/label.csv')

acoustic = {'train': [], 'valid': [], 'test': []}
labels = {'train': [], 'valid': [], 'test': []}
label_dict = {'Negative': 0, 'Positive': 1, 'Neutral': 2}

for video_id, clip_id, annotation, mode in zip(video_ids, clip_ids, annotations, modes):
    clip_id_ = '000' + str(clip_id)
    file_path = video_path + str(video_id) + '/' + clip_id_[-4:] + '.mp4'

    print(file_path, mode)

    acoustic[mode].append(get_speech_feature_fun(file_path, mode=MODE))
    labels[mode].append(label_dict[annotation])

save_features(acoustic, 'C:/Test/MSA Datasets/data/acoustic_wav2vec.pkl')
save_features(labels, 'C:/Test/MSA Datasets/data/labels.pkl')