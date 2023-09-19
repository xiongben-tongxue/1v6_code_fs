import torch
import librosa
import opensmile

# import torch
# x = torch.rand(5, 3)
# print(x)

# 加载文件
y, sr = librosa.load('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/video_0001/0001.mp4', sr=16000) # ffmpeg

# librosa
mel_spec = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128).T
print(mel_spec.shape)  #(40, 128)

# opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

sml_fs = smile.process_file('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/video_0001/0001.mp4')
print(sml_fs.shape)  #(124, 65)

