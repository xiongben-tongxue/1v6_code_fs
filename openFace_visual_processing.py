from utilz import *

# 这个OpenFace太难安装了，一直安装不上，算了，不用这个了。
video_path = '/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/'
video_ids, clip_ids, texts, annotations, modes = load_data('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/label.csv')


for video_id, clip_id, mode in zip(video_ids, clip_ids, modes):
    clip_id_ = '000' + str(clip_id)
    file = video_path + str(video_id) + '/' + clip_id_[-4:] + '.mp4'
    os.system('/Users/wangdong/WorkSpace/OpenFace/build/bin/FaceLandmarkVidMulti -f {} -out_dir ./data/OFprocessed/{}/{}/'.format(file, str(video_id), str(clip_id)))