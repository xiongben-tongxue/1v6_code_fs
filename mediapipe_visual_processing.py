# 导入需要的库
import cv2
import mediapipe as mp
import numpy as np
import logging
from utilz import *

# 设置日志记录的基本配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 定义视频路径
video_path = '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/'
logging.info(f"Set video path to {video_path}")

video_ids, clip_ids, texts, annotations, modes = load_data('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/label.csv')
logging.info(f"Loaded data for {len(video_ids)} videos")

# 设置模式和最大长度
MODE = 'ori'
# Face会报错
# MODE = 'face'
max_len = 10
logging.info(f"Set MODE to {MODE} and max_len to {max_len}")

# mediapipe
visual = {'train':[], 'valid':[], 'test':[]}
for video_id, clip_id, mode in zip(video_ids, clip_ids, modes):
    # 定义视频文件路径
    clip_id_ = '000' + str(clip_id)
    file = video_path + str(video_id) + '/' + clip_id_[-4:] + '.mp4'
    logging.info(f"Processing video file: {file}")

    # 如果模式为ori，处理视频为灰度图像
    if MODE == 'ori':
        cap = cv2.VideoCapture(file)
        images = []
        while cap.isOpened():
            success, image = cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(cv2.resize(image, (64, 64)))
            else:
                break
        if len(images) > max_len:
            images = images[::int(len(images) / max_len)]
        if len(images) < max_len:
            images = np.concatenate([images, np.zeros((max_len - len(images), 64, 64))], axis=0)
        visual[mode].append(images[:max_len])

    # 如果模式为face，处理视频中的人脸，在处理人脸的过程中，一直会报错误，解决不了
    #[ERROR:0@343.396] global cap_ffmpeg_impl.hpp:1278 open Could not open codec h264, error: -35
    #[ERROR:0@343.396] global cap_ffmpeg_impl.hpp:1286 open VIDEOIO/FFMPEG: Failed to initialize VideoCapture
    elif MODE == 'face':
        mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        # mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(file)
        # faces = []  问题的根本原因在于，你的faces列表可能没有被初始化为一个Numpy数组，因此当你尝试连接时，它被当作一个一维列表。
        # 为了解决这个问题，你需要确保faces是一个三维的Numpy数组，即使它是空的。这样，当你尝试使用np.concatenate将新的人脸图片添加到它时，它们的维度将匹配。
        faces = np.array([]).reshape(0, 28, 28)
        while cap.isOpened():
            success, image = cap.read()
            if success:
                img_h, img_w, img_c = image.shape
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = mp_face_detection.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if results.detections:
                    for detection in results.detections:
                        xmin = int(detection.location_data.relative_bounding_box.xmin * img_w)
                        ymin = int(detection.location_data.relative_bounding_box.ymin * img_h)
                        width = int(detection.location_data.relative_bounding_box.width * img_w)
                        height = int(detection.location_data.relative_bounding_box.height * img_w)
                        ymin = max(0, ymin)
                        xmin = max(0, xmin)
                        face = cv2.resize(image[ymin:ymin + width, xmin:xmin + width], (28, 28))
            else:
                break
        if len(faces) > max_len:
            faces = faces[::int(len(faces) / max_len)]
        if len(faces) < max_len:
            try:
                faces = np.concatenate([faces, np.zeros((max_len - len(faces), 28, 28))], axis=0)
            except Exception as e:
                logging.error(f"Error while processing faces for video {file}: {e}")
                faces = np.zeros((max_len, 28, 28))
        visual[mode].append(faces[:max_len])

# 保存处理后的数据
save_path = '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/visual_mpori.pkl'
# 下面这个会报错
#save_path = '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/visual_mp_face.pkl'
save_features(visual, save_path)
logging.info(f"Saved features to {save_path}")
