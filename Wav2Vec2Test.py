import librosa   #用于音频处理
import torch     #是pyTorch库，用于深度学习和张量运算
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

# 这段代码从一个视频文件中加载音频数据，使用预训练的 Wav2Vec2 模型提取特征，并检查特征的形状
# 使用函数提取视频
video_path = "/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/video_0001/0001.mp4"

filename = video_path
# 读取视频文件中的音频，并将其加载到‘y’变量中，sr是音频的采样率
y, sr = librosa.load(filename, sr=None)


# Chinese  加载预训练的Wav2Vec2处理器和模型，用于后续的特征提取。此处是专门针对中文的
processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

# 启用梯度检查点
model.gradient_checkpointing_enable()

# 重新采样音频到16000Hz
y = librosa.resample(y, orig_sr=sr, target_sr=16000)
sr = 16000

# 你使用处理器来准备音频数据，使其可以被模型接受。
inputs = processor(y, sampling_rate=sr, return_tensors="pt")
# 你在torch.no_grad()上下文中运行模型来避免在推理时计算梯度
with torch.no_grad():
    outputs = model.wav2vec2(**inputs)

last_hidden_states = outputs.extract_features  # for CH
print(list(last_hidden_states.shape))