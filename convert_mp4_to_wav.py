import os

path_dataset = './MELD.Raw/'
dicts = os.listdir(path_dataset)
print(dicts)

for d in dicts[:]:
    dict_path = path_dataset+d+'/'
    if os.path.isdir(dict_path):
        files = os.listdir(dict_path)
        for f in files:
            if f.endswith('.mp4'):
                filename = dict_path+f
                actual_filename = filename[:-4]
                os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, actual_filename))