import os
import sys

mnist_path = 'mnist'
data_sets = ['train', 'val']

for data_set in data_sets:
    odd_list = '{}_odd.txt'.format(data_set)
    even_list = '{}_even.txt'.format(data_set)
    all_list = '{}_all.txt'.format(data_set)
    root = os.sep.join([mnist_path, data_set])
    filenames = os.listdir(root)
    with open(odd_list, 'w') as f_odd, open(even_list, 'w') as f_even, open(all_list, 'w') as f_all:
        for filename in filenames:
            filepath = os.sep.join([root, filename])
            label = int(filename[:filename.rfind('.')].split('_')[1])
            line = '{} {}\n'.format(filepath, label)
            f_all.write(line)

            line = '{} {}\n'.format(filepath, int(label/2))
            if label % 2:
                f_odd.write(line)
            else:
                f_even.write(line)
