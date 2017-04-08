import os
import pickle, gzip
from matplotlib import pyplot

# Load the dataset
print('Loading data from mnist.pkl.gz ...')
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

imgs_dir = 'mnist'
os.system('mkdir -p {}'.format(imgs_dir))
datasets = {'train': train_set, 'val': valid_set, 'test': test_set}
for dataname, dataset in datasets.items():
    print('Converting {} dataset ...'.format(dataname))
    data_dir = os.sep.join([imgs_dir, dataname])
    os.system('mkdir -p {}'.format(data_dir))
    for i, (img, label) in enumerate(zip(*dataset)):
        filename = '{:0>6d}_{}.jpg'.format(i, label)
        filepath = os.sep.join([data_dir, filename])
        img = img.reshape((28, 28))
        pyplot.imsave(filepath, img, cmap='gray')
        if (i+1) % 10000 == 0:
            print('{} images converted!'.format(i+1))

