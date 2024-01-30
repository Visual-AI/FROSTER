import json
import copy
k600_classes_path = 'k600-index2cls.json'
k400_rephrased_path = 'k400_rephrased_classes.json'

k600_classes = open(k600_classes_path, 'r')
k600_classes = json.load(k600_classes)

k400_rephrased = open(k400_rephrased_path, 'r')
k400_rephrased = json.load(k400_rephrased)

k600_idx, k600_cls = [], []
for k600, v600 in k600_classes.items():
    k600_idx.append(k600)
    k600_cls.append(v600.lower())
k600_rephrased = copy.deepcopy(k600_cls)

missing_cls = []
k400_idx, k400_cls = [], []
for k400, v400 in k400_rephrased.items():
    v400_name = v400.split(":")[0]
    try:
        idx = k600_cls.index(v400_name.lower())
        k600_rephrased[idx] = v400
    except:
        missing_cls.append(v400_name)

k600_rephrased_path = 'k600_rephrased_classes.txt'
f = open(k600_rephrased_path, 'w')

for i in range(len(k600_idx)):
    f.write(k600_idx[i] + '. ' + k600_rephrased[i]+'\n')