import sys
import os
import numpy as np
import cv2
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'open_nsfw/nsfw_model/resnet_50_1by2_nsfw.caffemodel'
DEPLOY_FILE = 'deploy_global_pooling.prototxt'
FEATURE_MAPS = 'eltwise_stage3_block2'
FC_LAYER = 'fc_nsfw'

SHORT_EDGE = 320
MOSAIC_RANGE = [5, 15]

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)
input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.system('mkdir -p {}'.format(output_dir))

porno = 1
mask_th = 0.5

filenames = os.listdir(input_dir)
for i, filename in enumerate(filenames):
    filepath = os.sep.join([input_dir, filename])

    image = cv2.imread(filepath)[:, :, :3]
    height, width = image.shape[:2]

    short_edge_image = min(image.shape[:2])
    scale_ratio = float(SHORT_EDGE) / float(short_edge_image)
    if scale_ratio < 1:
        transformed_image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
    else:
        transformed_image = np.copy(image)
    transformed_image = transformed_image.astype(np.float32)
    transformed_image -= np.array([104., 117., 123.])
    transformed_image = np.transpose(transformed_image, (2, 0, 1))

    net.blobs['data'].reshape(1, 3, transformed_image.shape[1], transformed_image.shape[2])
    net.blobs['data'].data[...] = transformed_image

    mosaic_size = np.random.random_integers(MOSAIC_RANGE[0], MOSAIC_RANGE[1]+1, 1)
    scale_mosaic = 1 / float(mosaic_size)
    mosaic_image = cv2.resize(image, (0, 0), fx=scale_mosaic, fy=scale_mosaic)
    mosaic_image = cv2.resize(mosaic_image, (width, height), interpolation=cv2.INTER_NEAREST)

    net.forward()
    feature_maps = net.blobs[FEATURE_MAPS].data[0]
    fc_params = net.params[FC_LAYER]
    fc_w = fc_params[0].data[porno]

    activation_map = np.zeros_like(feature_maps[0])
    for feature_map, w in zip(feature_maps, fc_w):
        activation_map += feature_map * w

    activation_map = cv2.resize(activation_map, (width, height), interpolation=cv2.INTER_CUBIC)
    activation_map -= activation_map.min()
    activation_map /= activation_map.max()
    mask = np.zeros(activation_map.shape)
    mask[activation_map > mask_th] = 1
    image_with_mosaic = np.copy(image)
    image_with_mosaic[mask > mask_th] = mosaic_image[mask > mask_th]

    output_filepath = os.sep.join([output_dir, filename])
    cv2.imwrite(output_filepath, image_with_mosaic)

    if (i+1) % 100 == 0:
        print('{} images processed!'.format(i+1))
        
    # uncomment the following for visualization
    #vis_img = np.hstack([image, image_with_mosaic])
    #cv2.imshow('Mosaic Visualization', vis_img)
    #cv2.waitKey()

print('Done!')

