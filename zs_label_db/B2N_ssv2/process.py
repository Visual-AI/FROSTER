import os
train_list_path = 'test_1.csv'
dst_path = 'test.csv'

dataset_path = '$ROOT/ssv2'

f = open(train_list_path)
f_w = open(dst_path, 'w')

vid2path = {}
video_list = os.listdir(dataset_path)
for vid in video_list:
    vid2path[vid] = os.path.join(vid)

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