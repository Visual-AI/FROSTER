import os
import json

input_path = 'train_idx2cls.csv'
output_path = 'raw_train_idx2cls.json'

f = open(input_path, 'r')
f_w = open(output_path, 'w')

out_dict = {}
for line in f.readlines():
    line = line.strip('\n').split(',')
    idx = line[0]
    cls = line[1]
    out_dict[idx] = cls

json.dump(out_dict, f_w)