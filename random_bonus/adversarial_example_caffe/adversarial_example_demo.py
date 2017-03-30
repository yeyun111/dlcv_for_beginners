import sys
import os
from operator import itemgetter
import numpy
import cv2
sys.path.append('/opt/caffe/python')
import caffe
import skimage
import logging


def make_n_test_adversarial_example(img, net, transformer, epsilon, data_blob='data', prob_blob='prob', label_index=None, top_k=5):

    # Load image & forward
    transformed_img = transformer.preprocess(data_blob, img)
    net.blobs[data_blob].data[0] = transformed_img
    net.forward()
    probs = [x for x in enumerate(net.blobs[prob_blob].data.flatten())]
    num_classes = len(probs)
    sorted_probs = sorted(probs, key=itemgetter(1), reverse=True)
    top_preds = sorted_probs[:top_k]
    pred = sorted_probs[0][0]

    # if label_index is set,
    # generate a adversarial example toward the label,
    # else
    # reduce the probability of predicted label
    if type(label_index) is int and 0 <= label_index < num_classes:
        net.blobs[prob_blob].diff[0][label_index] = 1.
    else:
        net.blobs[prob_blob].diff[0][pred] = -1.

    # generate attack image with fast gradient sign method
    diffs = net.backward()
    diff_sign_mat = numpy.sign(diffs[data_blob])
    adversarial_noise = epsilon * diff_sign_mat

    # clip exceeded values
    attack_hwc = transformer.deprocess(data_blob, transformed_img + adversarial_noise[0])
    attack_hwc[attack_hwc > 1] = 1.
    attack_hwc[attack_hwc < 0] = 0.
    attack_img = transformer.preprocess(data_blob, attack_hwc)

    net.blobs[data_blob].data[...] = attack_img
    net.forward()
    probs = [x for x in enumerate(net.blobs[prob_blob].data.flatten())]
    sorted_probs = sorted(probs, key=itemgetter(1), reverse=True)
    top_attacked_preds = sorted_probs[:top_k]

    return attack_hwc, top_preds, top_attacked_preds


if __name__ == '__main__':
    # path to test image
    image_path = sys.argv[1]

    # model to attack
    model_definition = 'squeezenet-v1.0-deploy-with-force-backward.prototxt'
    model_weights = 'squeezenet_v1.0.caffemodel'
    channel_means = numpy.array([104., 117., 123.])

    # initialize net
    net = caffe.Net(model_definition, model_weights, caffe.TEST)
    n_channels, height, width = net.blobs['data'].shape[-3:]
    net.blobs['data'].reshape(1, n_channels, height, width)

    # initialize transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', channel_means)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    # load labels from imagenet synset words
    with open('synset_words.txt', 'r') as f:
        labels = [x.rstrip()[x.find(' '):].split(',')[0] for x in f.readlines()]

    # load image
    img = caffe.io.load_image(image_path)

    # make adversarial example to reduce the predicted probability
    make_n_test_adversarial_example(img, net, transformer, 1.0)

    # make adversarial example to increase the probability of a specified label
    make_n_test_adversarial_example(img, net, transformer, 1.0, label_index=39)



'''

def main(data_root, data_list, output_path, model, weights, mean, method, noise_range, gpu, labels):
    # init
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    cf = CaffeForward(model, weights, mean)

    # get mean
    mean_r, mean_g, mean_b = cf.transformer_.mean['data'].flatten()

    # make output path ready
    os.system('mkdir -p {}'.format(output_path))
    output_label_paths = []
    for label in labels:
        output_label_path = os.sep.join([output_path, label])
        output_label_paths.append(output_label_path)
        os.system('mkdir -p {}'.format(output_label_path))
    num_labels = len(labels)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='{}.log'.format(output_path),
                        filemode='w')

    # get noise range
    noise_lb, noise_ub = noise_range

    with open(data_list, 'r') as f:
        count = 0
        valid_count = 0
        line = f.readline()
        while line:

            try:
                # load image and label
                if DEBUG:
                    logging.debug('WTF | ========================')
                filepath, label_index = line.split()
                filename = filepath.split(os.sep)[-1]
                filepath = os.sep.join([data_root, filepath])
                label_index = int(label_index)
                label = labels[label_index]
                output_label_path = output_label_paths[label_index]
                wrong_labels = range(num_labels)
                wrong_labels.pop(label_index)
                wrong_label = numpy.random.choice(wrong_labels)

                # calculate diff
                img = caffe.io.load_image(filepath, True)
                cf.net_.blobs['data'].data[...] = cf.transformer_.preprocess('data', img)
                cf.net_.forward()
                if DEBUG:
                    logging.debug(cf.net_.blobs['prob'].data[0])
                cf.net_.blobs['prob'].diff[0][wrong_label] = 1.
                diffs = cf.net_.backward()

                # generate attack image
                ch_img = cf.net_.blobs['data'].data
                noise_coeff = (noise_ub - noise_lb) * numpy.random.random() + noise_lb
                diff_sign_mat = numpy.sign(diffs['data'])
                noise_coeff *= numpy.round(numpy.mean(numpy.abs(diff_sign_mat)))

                if noise_coeff < noise_lb:
                    count += 1
                    line = f.readline()
                    if count % 1000 == 0:
                        logging.info('{:.3f}k samples generated! out of {}k samples'.format(float(valid_count) / 1000.,
                                                                                            int(count / 1000)))
                    continue

                noise = noise_coeff * diff_sign_mat

                # handle exceeding
                attack_ch_img = numpy.round(7 * noise + ch_img)
                attack_ch_img[0][0][attack_ch_img[0][0] + mean_r > 255] = 255 - mean_r
                attack_ch_img[0][1][attack_ch_img[0][1] + mean_g > 255] = 255 - mean_g
                attack_ch_img[0][2][attack_ch_img[0][2] + mean_b > 255] = 255 - mean_b
                attack_ch_img[0][0][attack_ch_img[0][0] + mean_r < 0] = -mean_r
                attack_ch_img[0][1][attack_ch_img[0][1] + mean_g < 0] = -mean_g
                attack_ch_img[0][2][attack_ch_img[0][2] + mean_b < 0] = -mean_b

                # export attack image
                attack_img = attack_ch_img.transpose(0, 2, 3, 1)[0] + numpy.array([mean_r, mean_g, mean_b])
                attack_img = attack_img.astype(numpy.uint8)
                attacked_imagename = filename[:filename.rfind('.')] + '_{:.2f}_{}'.format(noise_coeff, labels[
                    wrong_label]) + filename[filename.rfind('.'):]
                attacked_filepath = os.sep.join([output_label_path, attacked_imagename])

                cv2.cvtColor(attack_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(attacked_filepath, attack_img)

                if DEBUG:
                    cf.net_.blobs['data'].data[...] = attack_ch_img
                    cf.net_.forward()
                    logging.debug(cf.net_.blobs['prob'].data[0])

                    img = caffe.io.load_image(attacked_filepath, True)
                    cf.net_.blobs['data'].data[...] = cf.transformer_.preprocess('data', img)
                    cf.net_.forward()
                    logging.debug(cf.net_.blobs['prob'].data[0])
                    logging.debug('WTF | {}'.format(attacked_filepath))

                count += 1
                valid_count += 1
                if count % 1000 == 0:
                    logging.info('{:.3f}k samples generated! out of {}k samples'.format(float(valid_count) / 1000.,
                                                                                        int(count / 1000)))
            except Exception as e:
                logging.warning('WTF happend!')
                logging.warning(line)
                logging.warning(e)

            line = f.readline()







class CaffeForward:

    def __init__(self, net_def, weights, mean_file):
        self.net_ = caffe.Net(net_def, weights, caffe.TEST)
        n_channels, height, width = [x for x in self.net_.blobs['data'].shape][-3:]
        self.net_.blobs['data'].reshape(1, n_channels, height, width)

        self.transformer_ = caffe.io.Transformer({'data': self.net_.blobs['data'].data.shape})
        self.transformer_.set_transpose('data', (2, 0, 1))
        if mean_file.endswith('npy'):
            channels_mean = numpy.load(mean_file)[0].mean(1).mean(1)
            #print(channels_mean, channels_mean.dtype, type(channels_mean))
        else:
            with open(mean_file, 'r') as f:
                channels_mean = numpy.array([float(x) for x in f])
                #print(channels_mean, channels_mean.dtype, type(channels_mean))
        self.transformer_.set_mean('data', channels_mean)
        self.transformer_.set_raw_scale('data', 255)
        self.transformer_.set_channel_swap('data', (2, 1, 0))

    def forward(self, img_path):
        img = caffe.io.load_image(img_path, True)
        self.net_.blobs['data'].data[...] = self.transformer_.preprocess('data', img)
        self.net_.forward()

    def forward_img(self, img):
        self.net_.blobs['data'].data[...] = self.transformer_.preprocess('data', img.astype(numpy.float)/255)
        self.net_.forward()

    def get_blob(self, blob_name):
        return self.net_.blobs[blob_name].data

    def get_img_feature(self, img_path, blob_name):
        self.forward(img_path)
        return self.get_feature(blob_name)

    def get_params(self, layer_name):
        return self.net_.params[layer_name]

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print('Use: python cnn_fe.py [model_file] [model_weights] [model_mean] [image_list]')
        sys.exit(0)

    model_file = sys.argv[1]
    model_weights = sys.argv[2]
    model_mean = sys.argv[3]
    img_list = sys.argv[4]
    gpu_id = int(sys.argv[5])

    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    cf = CaffeForward(model_file, model_weights, model_mean)

    with open(img_list, 'r') as f:
        line = f.readline().rstrip()
        while line:
            img_path = line.split()[0]
            try:
                cf.forward(img_path)
                probs = cf.get_blob('prob')
                print(' '.join([line]+[str(x) for x in probs[0]]))
                line = f.readline().rstrip()
            except Exception as e:
                line = f.readline().rstrip()
                continue
'''
