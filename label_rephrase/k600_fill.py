k600_rephrased_path = 'k600_rephrased_classes.txt'
output_path = 'k600_missing_classes.txt'

k600_rephrased = open(k600_rephrased_path)

k600_rephrased_classes_lines = k600_rephrased.readlines()
f = open(output_path, 'w')

for line in k600_rephrased_classes_lines:
    if ':' not in line:
        f.write(line)
        