import json
original_rephrase_path = '$ROOT/label_rephrase/k400_rephrased_classes.json'
input_path = 'train_idx2cls.csv'
output_path = 'train_idx2cls.json'

f = open(input_path)
f_rephrase = open(original_rephrase_path)
f_out = open(output_path, 'w')

rephrase = json.load(f_rephrase)
rephrase_new = {}

for k, v in rephrase.items():
    cls_name = v.split(':')[0]
    rephrase_new[cls_name.lower()] = v

out_dict = {}
for line in f.readlines():
    line = line.strip()
    idx, cls_name = line.split(',')
    try:
        txt = rephrase_new[cls_name.lower()]
        out_dict[idx] = txt
    except:
        print(cls_name)

json.dump(out_dict, f_out)