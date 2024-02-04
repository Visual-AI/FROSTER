import os
train_list_path = 'train_s3.csv'
dst_path = 'train_3.csv'

f = open(train_list_path)
f_w = open(dst_path, 'w')

vid2path = {}
dataset_path = '$ROOT/k400/train'

class_list = os.listdir(dataset_path)
for cls in class_list:
    cls_path = os.path.join(dataset_path, cls)
    videos = os.listdir(cls_path)
    for vid in videos:
        vid2path[vid] = os.path.join(cls, vid)

# dataset_path_1 = '$ROOT/k400/val'
# class_list = os.listdir(dataset_path_1)
# for cls in class_list:
#     cls_path = os.path.join(dataset_path_1, cls)
#     videos = os.listdir(cls_path)
#     for vid in videos:
#         vid2path[vid] = os.path.join(cls, vid)

count = 0
for line in f.readlines():
    line = line.strip()
    video_name, cls_idx = line.split(' ')
    video_name = video_name.split('/')[1][:-4]
    # video_name = video_name.split('_')[:-2]
    # video_name = '_'.join(video_name)
    after = ['.mp4', '.mp4.mkv', '.mkv', '.mp4.webm']
    flag = 0
    for a in after:
        video_name_ = video_name + a
        if video_name_ in vid2path:
            f_w.write('train/' + vid2path[video_name_]+','+cls_idx+'\n')
            break
        else:
            flag += 1
    if flag == 4:
        count += 1
        print(video_name)

print(count)