import subprocess

def extract_audio_from_video(video_path, audio_output_path):
    command = [
        'ffmpeg',
        '-i', video_path,       # 输入视频文件的路径
        '-vn',                  # 只保留音频流
        '-acodec', 'pcm_s16le', # 设置音频编解码器为 pcm_s16le
        '-ar', '44100',         # 设置音频采样率为 44100
        '-ac', '2',             # 设置音频通道为 2 (立体声)
        audio_output_path      # 输出音频文件的路径
    ]
    subprocess.run(command, check=True)

