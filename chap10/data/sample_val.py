import os
import random

N = 300

os.system('mkdir -p val')
class_dirs = os.listdir('train')

for class_dir in class_dirs:
    os.system('mkdir -p val/{}'.format(class_dir))
    root = 'train/{}'.format(class_dir)
    print('Sampling validation set with {} images from {} ...'.format(N, root))
    filenames = os.listdir(root)
    random.shuffle(filenames)
    val_filenames = filenames[:N]
    for filename in val_filenames:
        src_filepath = os.sep.join([root, filename])
        dst_filepath = os.sep.join(['val', class_dir, filename])
        cmd = 'mv {} {}'.format(src_filepath, dst_filepath)
        os.system(cmd)
