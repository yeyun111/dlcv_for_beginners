import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('/path/to/caffe/python')
import caffe

ZOOM_IN_SIZE = 50
PAD_SIZE = 4

WEIGHTS_FILE = 'freq_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'

net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
kernels = net.params['conv1'][0].data

kernels -= kernels.min()
kernels /= kernels.max()

zoomed_in_kernels = []
for kernel in kernels:
    zoomed_in_kernels.append(cv2.resize(kernel[0], (ZOOM_IN_SIZE, ZOOM_IN_SIZE), interpolation=cv2.INTER_NEAREST))

# plot 12*8 squares kernels
half_pad = PAD_SIZE / 2
padded_size = ZOOM_IN_SIZE+PAD_SIZE
padding = ((0, 0), (half_pad, half_pad), (half_pad, half_pad))

padded_kernels = np.pad(zoomed_in_kernels, padding, 'constant', constant_values=1)
padded_kernels = padded_kernels.reshape(8, 12, padded_size, padded_size).transpose(0, 2, 1, 3)
kernels_img = padded_kernels.reshape((8*padded_size, 12*padded_size))[half_pad:-half_pad, half_pad: -half_pad]

plt.imshow(kernels_img, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()
