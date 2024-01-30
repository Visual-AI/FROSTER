import os
train_list_path = 'val_1.csv'
dst_path = 'val.csv'

dataset_path = '/root/paddlejob/workspace/env_run/output/xiaohu/FROSTER/hmdb51_test'

f = open(train_list_path)
f_w = open(dst_path, 'w')

vid2path = {}
class_list = os.listdir(dataset_path)
for cls in class_list:
    cls_path = os.path.join(dataset_path, cls)
    videos = os.listdir(cls_path)
    for vid in videos:
        vid2path[vid] = os.path.join(cls, vid)

for line in f.readlines():
    line = line.strip()
    video_name, cls_idx = line.split(' ')
    if video_name in vid2path:
        f_w.write(vid2path[video_name]+','+cls_idx+'\n')
    else:
        video_name = video_name[:-4] + '.mp4'
        if video_name in vid2path:
            f_w.write(vid2path[video_name]+','+cls_idx+'\n')
        else:
            print[video_name]