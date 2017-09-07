import os
import argparse
import torch.optim as optim

OPTIMIZERS = {
    'adadelta': optim.Adadelta,
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple Demo of Image Segmentation with U-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general options
    parser.add_argument('mode',
                        help='train/test')
    parser.add_argument('dataroot',
                        help='Directory containing training images in "images" and "segmentations" or test images')
    parser.add_argument('--cpu',
                        help='Set to CPU mode', action='store_true')
    parser.add_argument('--color_labels',
                        help='Colors of labels in segmentation image',
                        type=str, default='(0,0,0),(255,255,255)')

    # training options
    parser.add_argument('--img-dir',
                        help='Directory under [dataroot] containing images',
                        type=str, default='images')
    parser.add_argument('--seg-dir',
                        help='Directory under [dataroot] containing segmentations',
                        type=str, default='segmentations')
    parser.add_argument('--epochs',
                        help='Num of training epochs',
                        type=int, default=20)
    parser.add_argument('--batch-size',
                        help='Batch size',
                        type=int, default=4)
    parser.add_argument('--optimizer',
                        help='Optimizer: Adadelta/Adam/RMSprop/SGD',
                        type=str, default='SGD')
    parser.add_argument('--lr',
                        help='Learning rate, for Adadelta it is the base learning rate',
                        type=float, default=0.01)
    parser.add_argument('--lr-policy',
                        help='Learning rate policy, example:"5:0.0005,10:0.0001,18:1e-5"',
                        type=str, default='')
    parser.add_argument('--no-batchnorm',
                        help='Do NOT use batch normalization', action='store_true')
    parser.add_argument('--print-interval',
                        help='Print info after each specified iterations',
                        type=int, default=20)

    # test options
    parser.add_argument('--model',
                        help='Path to pre-trained model',
                        type=str, default='')
    parser.add_argument('--output-dir',
                        help='Directory for output results',
                        type=str, default='')

    args = parser.parse_args()
    args.dataroot = args.dataroot.rstrip(os.sep)
    args.color_labels = eval('[{}]'.format(args.color_labels))
    args.optimizer = OPTIMIZERS[args.optimizer.lower()]
    args.lr_policy = eval('{{{}}}'.format(args.lr_policy)) if args.lr_policy else {}

    return args
