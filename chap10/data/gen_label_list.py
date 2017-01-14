import os
import sys

dataset = sys.argv[1].rstrip(os.sep)

class_dirs = os.listdir(dataset)

with open('{}.txt'.format(dataset), 'w') as f:
    for class_dir in class_dirs:
        class_path = os.sep.join([dataset, class_dir])
        label = int(class_dir)
        lines = ['{}/{} {}'.format(class_path, x, label) for x in os.listdir(class_path)]
        f.write('\n'.join(lines) + '\n')
