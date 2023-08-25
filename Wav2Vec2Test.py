import librosa
import torch
import project_function
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

# 使用函数提取音频
video_path = "C:/Test/0001.mp4"
audio_output_path = "C:/Test/0001.wav"
#project_function.extract_audio_from_video(video_path, audio_output_path)

filename = video_path
y, sr = librosa.load(filename, sr=None)

# English
# MELD
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Chinese
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

# inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

inputs = processor(y, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    # outputs = model(**inputs) # for EN
    outputs = model.wav2vec2(**inputs)

# last_hidden_states = outputs.last_hidden_state  # for EN
last_hidden_states = outputs.extract_features  # for CH
list(last_hidden_states.shape)