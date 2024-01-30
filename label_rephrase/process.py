import json

input_rephrased_path = 'ucf101_rephrased.txt'
# k400idx2cls_path = 'k400_index2cls.json'

rephrased = open(input_rephrased_path, 'r')
# k400idx2cls = open(k400idx2cls_path, 'r')
# k400idx2cls = json.load(k400idx2cls)

rephrased_lines = rephrased.readlines()

count = 0
output_dict = {}

for line in rephrased_lines:
    line = line.strip('\n')
    if line == '':
        continue
    index = line.split('.')[0]
    cls_txt = line.split('.')[1][1:]
    cls_txt = cls_txt.replace('\"', '')
    print(cls_txt)
    output_dict[index] = cls_txt
    # try:
    #     assert cls.lower() == k400idx2cls[index].lower()
    # except:
    #     print(cls.lower(), k400idx2cls[index].lower())
json.dump(output_dict, open('ucf101_rephrased_classes.json', 'w'))