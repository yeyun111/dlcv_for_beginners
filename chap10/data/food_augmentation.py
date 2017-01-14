import os

n_total = 3000

class_dirs = os.listdir('train')

for class_dir in class_dirs:
    src_path = 'train/{}'.format(class_dir)
    n_samples = len(os.listdir(src_path))
    n_aug = n_total - n_samples
    cmd = 'python run_augmentation.py {} temp {}'.format(src_path, n_aug)
    os.system(cmd)
    cmd = 'mv temp/* {}'.format(src_path)
    os.system(cmd)

os.system('rm -r temp')
