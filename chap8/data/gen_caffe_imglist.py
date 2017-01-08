import os
import sys

input_path = sys.argv[1].rstrip(os.sep)
output_path = sys.argv[2]

filenames = os.listdir(input_path)

with open(output_path, 'w') as f:
    for filename in filenames:
        filepath = os.sep.join([input_path, filename])
        label = filename[:filename.rfind('.')].split('_')[1]
        line = '{} {}\n'.format(filepath, label)
        f.write(line)

