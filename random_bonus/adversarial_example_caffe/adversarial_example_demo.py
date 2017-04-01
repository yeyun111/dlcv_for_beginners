import sys
from operator import itemgetter
import numpy
from matplotlib import pyplot
sys.path.append('/path/to/caffe/python')
import caffe


def make_n_test_adversarial_example(
        img, net, transformer, epsilon,
        data_blob='data', prob_blob='prob',
        label_index=None, top_k=5):

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
    net.blobs[prob_blob].diff[...] = 0
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

    net.blobs[data_blob].data[0] = attack_img
    net.forward()
    probs = [x for x in enumerate(net.blobs[prob_blob].data.flatten())]
    sorted_probs = sorted(probs, key=itemgetter(1), reverse=True)
    top_attacked_preds = sorted_probs[:top_k]

    return attack_hwc, top_preds, top_attacked_preds


def visualize_attack(title, original_img, attack_img, original_preds, attacked_preds, labels):
    pred = original_preds[0][0]
    attacked_pred = attacked_preds[0][0]
    k = len(original_preds)
    fig_name = '{}: {} to {}'.format(title, labels[pred], labels[attacked_pred])

    pyplot.figure(fig_name)
    for img, plt0, plt1, preds in [
        (original_img, 231, 234, original_preds),
        (attack_img, 233, 236, attacked_preds)
    ]:
        pyplot.subplot(plt0)
        pyplot.axis('off')
        pyplot.imshow(img)
        ax = pyplot.subplot(plt1)
        pyplot.axis('off')
        ax.set_xlim([0, 2])
        bars = ax.barh(range(k-1, -1, -1), [x[1] for x in preds])
        for i, bar in enumerate(bars):
            x_loc = bar.get_x() + bar.get_width()
            y_loc = k - i - 1
            label = labels[preds[i][0]]
            ax.text(x_loc, y_loc, '{}: {:.2f}%'.format(label, preds[i][1]*100))

    pyplot.subplot(232)
    pyplot.axis('off')
    noise = attack_img - original_img
    pyplot.imshow(255 * noise)


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

    examples = [
        (None, 1.0),    # make adversarial example to reduce the predicted probability
        (296, 1.0),     # make adversarial example toward ice bear(296)
        (9, 1.0),       # make adversarial example toward ostrich(9)
        (9, 2.0),       # make adversarial example toward ostrich(9) with stronger noise
        (9, 6.0),       # make adversarial example toward ostrich(9) with very strong noise
        (9, 18.0),      # make adversarial example toward ostrich(9) with too strong noise
        (752, 1.0),     # make adversarial example toward racket(752)
        (752, 2.0),     # make adversarial example toward racket(752) with stronger noise
        (752, 6.0),     # make adversarial example toward racket(752) with very strong noise
        (752, 18.0),    # make adversarial example toward racket(752) with too strong noise
    ]

    for i, (label_index, epsilon) in enumerate(examples):
        attack_img, original_preds, attacked_preds = \
            make_n_test_adversarial_example(img, net, transformer, epsilon, label_index=label_index)
        visualize_attack('example{}'.format(i), img, attack_img, original_preds, attacked_preds, labels)

    # try to make adversarial example toward racket(752) with epsilon=0.1, iterate 10 times
    attack_img, original_preds, attacked_preds = \
        make_n_test_adversarial_example(img, net, transformer, 0.1, label_index=752)
    for i in range(9):
        attack_img, _, attacked_preds = \
            make_n_test_adversarial_example(attack_img, net, transformer, 0.1, label_index=752)
    visualize_attack('racket_iterative'.format(i), img, attack_img, original_preds, attacked_preds, labels)

    pyplot.show()
