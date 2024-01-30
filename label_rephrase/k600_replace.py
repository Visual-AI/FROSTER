import json
original_path = 'k600_split2_classes.json'
rephrased_path = 'k600_rephrased_classes.json'
output_path = 'k600_split2_rephrased_classes.json'

f_rephrased = open(rephrased_path, 'r')
f_original = open(original_path, 'r')
f_output = open(output_path, 'w')

rephrased = json.load(f_rephrased)
original = json.load(f_original)
# output = json.load(f_output)

rephrased_cls2txt = {}
for k, v in rephrased.items():
    cls = v.split(':')[0].lower()
    rephrased_cls2txt[cls] = v

rephrased_cls2txt_split = {}
for k, v in original.items():
    try:
        txt = rephrased_cls2txt[v.lower()]
        rephrased_cls2txt_split[k] = txt
    except:
        print(k,v)

json.dump(rephrased_cls2txt_split, f_output)