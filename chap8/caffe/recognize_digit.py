import sys
sys.path.append('/path/to/caffe/python')
import numpy as np
import cv2
import caffe

MEAN = 128
SCALE = 0.00390625

imglist = sys.argv[1]

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('lenet.prototxt', 'mnist_lenet_iter_36000.caffemodel', caffe.TEST)
net.blobs['data'].reshape(1, 1, 28, 28)

with open(imglist, 'r') as f:
    line = f.readline()
    while line:
        imgpath, label = line.split()
        line = f.readline()
        image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE).astype(np.float) - MEAN
        image *= SCALE
        net.blobs['data'].data[...] = image
        output = net.forward()
        pred_label = np.argmax(output['prob'][0])
        print('Predicted digit for {} is {}'.format(imgpath, pred_label))
