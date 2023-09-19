import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

sml_fs = smile.process_file('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/Raw/video_0001/0001.mp4')
print(sml_fs.shape)