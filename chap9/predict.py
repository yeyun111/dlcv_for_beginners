import sys
import numpy
sys.path.append('/opt/caffe/python')
import caffe

WEIGHTS_FILE = 'chap9_example_iter_20000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'
IMAGE_SIZE = (100, 100)
MEAN_VALUE = 128

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
net.blobs['data'].reshape(1, 1, *IMAGE_SIZE)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', numpy.array([MEAN_VALUE]))
transformer.set_raw_scale('data', 255)

image_list = sys.argv[1]

with open(image_list, 'r') as f:
    for line in f.readlines():
        filename = line[:-1]
        image = caffe.io.load_image(filename, False)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        fx, fy = output['pred'][0]

        print('The predicted frequence score for {} is {}, {}'.format(filename, fx, fy))
