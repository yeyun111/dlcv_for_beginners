import os
import sys
sys.path.append('/path/to/caffe/python')
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
import caffe

WEIGHTS_FILE = 'mnist_siamese_iter_20000.caffemodel'
DEPLOY_FILE = 'mnist_siamese.prototxt'
IMG_DIR = 'mnist/test'
MEAN = 128
SCALE = 0.00390625

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

pattern = re.compile('\d+_(\d)\.jpg')

image_list = os.listdir(IMG_DIR)
n_imgs = len(image_list)

net.blobs['data'].reshape(n_imgs, 1, 28, 28)

labels = []
for i, filename in enumerate(image_list):
    digit = int(pattern.findall(filename)[0])
    labels.append(digit)
    filepath = os.sep.join([IMG_DIR, filename])
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(np.float) - MEAN
    image *= SCALE
    net.blobs['data'].data[i, ...] = image

labels = np.array(labels)

output = net.forward()
feat = output['feat']

colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
          '#ff00ff', '#990000', '#999900', '#009900', '#009999']
legend = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure('feat')
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(),
             feat[labels==i,1].flatten(),
             '.', c=colors[i])
plt.legend(legend)

plt.figure('ip2')
ip2_feat = net.blobs['ip2'].data
model = TSNE(n_components=2)

ip2_vis_feat = model.fit_transform(ip2_feat)
for i in range(10):
    plt.plot(ip2_vis_feat[labels==i,0].flatten(),
             ip2_vis_feat[labels==i,1].flatten(),
             '.', c=colors[i])
plt.legend(legend)

plt.show()
