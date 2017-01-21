import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'food_resnet-10_iter_10000.caffemodel'
DEPLOY_FILE = 'food_resnet_10_cvgj_deploy.prototxt'
FEATURE_MAPS = 'layer_512_1_sum'
FC_LAYER = 'fc_food'

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

image_list = sys.argv[1]

cmap = plt.get_cmap('jet')
with open(image_list, 'r') as f:
    for line in f.readlines():
        filepath = line.split()[0]
        image = caffe.io.load_image(filepath)
        # uncomment the following 2 lines to forward with
        # original image size and corresponding activation maps
        #transformer.inputs['data'] = (1, 3, image.shape[0], image.shape[1])
        #net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        pred = np.argmax(output['prob'][0])

        feature_maps = net.blobs[FEATURE_MAPS].data[0]
        fc_params = net.params[FC_LAYER]
        fc_w = fc_params[0].data[pred]
        #fc_b = fc_params[1].data[pred]

        activation_map = np.zeros_like(feature_maps[0])
        for feature_map, w in zip(feature_maps, fc_w):
            activation_map += feature_map * w
        #activation_map += fc_b

        # Visualize as
        # left: original image
        # middle: activation map
        # right: original image overlaid with activation map in 'jet' colormap
        image = np.round(image*255).astype(np.uint8)
        h, w = image.shape[:2]
        activation_map = cv2.resize(activation_map, (w, h), interpolation=cv2.INTER_CUBIC)
        activation_map -= activation_map.min()
        activation_map /= activation_map.max()
        activation_color_map = np.round(cmap(activation_map)[:, :, :3]*255).astype(np.uint8)
        activation_map = np.stack(np.round([activation_map*255]*3).astype(np.uint8))
        activation_map = activation_map.transpose(1, 2, 0)
        overlay_img = image/2 + activation_color_map/2
        vis_img = np.hstack([image, activation_map, overlay_img])
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Activation Map Visualization', vis_img)
        cv2.waitKey()
