import json
input_path = 'test_rephrase.txt'
output_path = 'test_idx2cls.json'

f_in = open(input_path)
f_out = open(output_path, 'w')

input_lines = f_in.readlines()

out_dict = {}
for line in input_lines:
    line = line.strip()
    if line == '':
        continue
    idx, txt = line.split('.')[0], line.split('.')[1][1:]
    out_dict[idx] = txt

json.dump(out_dict, f_out)