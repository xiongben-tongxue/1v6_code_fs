import torch
import librosa
# import torch
# x = torch.rand(5, 3)
# print(x)

# 加载文件
y, sr = librosa.load('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/video_0001/0001.mp4', sr=16000) # ffmpeg

# librosa
mel_spec = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128).T
print(mel_spec.shape)