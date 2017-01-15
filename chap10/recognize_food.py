import sys
import numpy as np
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'food_resnet-10_iter_10000.caffemodel'
DEPLOY_FILE = 'food_resnet_10_cvgj_deploy.prototxt'

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

image_list = sys.argv[1]
result_list = '{}_results.txt'.format(image_list[:image_list.rfind('.')])

foods = open('/path/to/keywords.txt', 'rb').read().split()
with open(image_list, 'r') as f, open(result_list, 'w') as f_ret:
    for line in f.readlines():
        filepath, label = line.split()
        label = int(label)
        image = caffe.io.load_image(filepath)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        probs = output['prob'][0]
        pred = np.argmax(probs)

        print('{}, predicted: {}, true: {}'.format(filepath, foods[pred], foods[label]))
        result_line = '{} {} {} {}\n'.format(filepath, label, pred, ' '.join([str(x) for x in probs]))
        f_ret.write(result_line)
