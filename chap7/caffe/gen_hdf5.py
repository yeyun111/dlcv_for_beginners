import pickle
import numpy as np
import h5py

with open('../data.pkl', 'rb') as f:
    samples, labels = pickle.load(f)
sample_size = len(labels)

samples = np.array(samples).reshape((sample_size, 2))
labels = np.array(labels).reshape((sample_size, 1))

h5_filename = 'data.h5'
with h5py.File(h5_filename, 'w') as h:
    h.create_dataset('data', data=samples)
    h.create_dataset('label', data=labels)

with open('data_h5.txt', 'w') as f:
    f.write(h5_filename)
