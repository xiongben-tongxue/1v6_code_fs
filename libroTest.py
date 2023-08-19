# Beat tracking example

import librosa
import project_function

# 使用函数提取音频
video_path = "C:/Test/0001.mp4"
audio_output_path = "C:/Test/0001.wav"
#project_function.extract_audio_from_video(video_path, audio_output_path)


filename = video_path
y, sr = librosa.load(filename, sr=None)

mel_spec = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128).T
print(mel_spec.shape) # 25ms
