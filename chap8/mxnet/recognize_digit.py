import sys
import os
import cv2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
import numpy as np
import mxnet as mx

input_path = sys.argv[1].rstrip(os.sep)

mod = mx.mod.Module.load('mnist_lenet', 35, context=mx.gpu(2))
mod.bind(
    data_shapes=[('data', (1, 1, 28, 28))], 
    for_training=False)

filenames = os.listdir(input_path)
for filename in filenames:
    filepath = os.sep.join([input_path, filename])
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = (img.astype(np.float)-128) * 0.00390625
    img = img.reshape((1, 1)+img.shape)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    pred_label = np.argmax(prob)
    print('Predicted digit for {} is {}'.format(filepath, pred_label))
