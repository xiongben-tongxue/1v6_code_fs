import numpy as np
from utilz import *
import cv2
import clip
import torch
from PIL import Image
import logging

# 设置日志记录的基本配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 定义视频路径
video_path = 'C:/Test/MSA Datasets/SIMS/Raw/'
logging.info(f"Set video path to {video_path}")

video_ids, clip_ids, texts, annotations, modes = load_data('C:/Test/MSA Datasets/SIMS/label.csv')
logging.info(f"Loaded data for {len(video_ids)} videos")

# https://github.com/openai/CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

max_len = 10
visual = {'train': [], 'valid': [], 'test': []}
logging.info(f"Set visual to {visual} and max_len to {max_len}")

with torch.no_grad():
    for video_id, clip_id, mode in zip(video_ids, clip_ids, modes):
        # 定义视频文件路径
        clip_id_ = '000' + str(clip_id)
        file = video_path + str(video_id) + '/' + clip_id_[-4:] + '.mp4'
        logging.info(f"Processing video file: {file}")

        cap = cv2.VideoCapture(file)
        image_features = []
        while cap.isOpened():
            success, image = cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_ = preprocess(Image.fromarray(np.uint8(image))).unsqueeze(0).to(device)
                image_ft = model.encode_image(image_).cpu().detach().numpy()[0]
                image_features.append(image_ft)
            else:
                break

        if len(image_features) > max_len:
            image_features = image_features[::int(len(image_features) / max_len)]
        if len(image_features) < max_len:
            image_features = np.concatenate([image_features, np.zeros((max_len - len(image_features), 512))], axis=0)

        visual[mode].append(image_features[:max_len])


save_path = 'C:/Test/MSA Datasets/SIMS/data/visual_clip.pkl'
save_features(visual, save_path)
logging.info(f"Saved features to {save_path}")

np.shape(visual['train'])