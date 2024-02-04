import os
train_list_path = 'val_1.csv'
dst_path = 'val.csv'

dataset_path = '$ROOT/ucf101'

f = open(train_list_path)
f_w = open(dst_path, 'w')

for line in f.readlines():
    line = line.strip()
    video_name, cls_idx = line.split(' ')
    class_name = video_name.split('_')[1]
    if class_name == 'HandStandPushups':
        class_name = 'HandstandPushups'
    video_path = os.path.join(dataset_path, class_name, video_name)
    if os.path.exists(video_path):
        new_line = os.path.join(class_name, video_name) + ',' + str(cls_idx) + '\n'
        f_w.write(new_line)
    else:
        print(video_path)