import os
import random
import re

train_dir = 'mnist/train'
val_dir = 'mnist/val'
n_train = 100000
n_val = 10000

pattern = re.compile('\d+_(\d)\.jpg')

for img_dir, n_pairs in zip([train_dir, val_dir], [n_train, n_val]):
    imglist = os.listdir(img_dir)
    n_samples = len(imglist)
    dataset = img_dir[img_dir.rfind(os.sep)+1:]
    with open('{}.txt'.format(dataset), 'w') as f, \
            open('{}_p.txt'.format(dataset), 'w') as f_p:
        for i in range(n_pairs):
            filename = imglist[random.randint(0, n_samples-1)]
            digit = pattern.findall(filename)[0]
            filepath = os.sep.join([img_dir, filename])

            filename_p = imglist[random.randint(0, n_samples-1)]
            digit_p = pattern.findall(filename_p)[0]
            filepath_p = os.sep.join([img_dir, filename_p])

            label = 1 if digit == digit_p else 0

            f.write('{} {}\n'.format(filepath, label))
            f_p.write('{} {}\n'.format(filepath_p, label))
