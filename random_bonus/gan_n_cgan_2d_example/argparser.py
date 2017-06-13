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
        description='A Simple Demo of Generative Adversarial Networks with 2D Samples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_path',
                        help='Image or directory containing images to define distribution')

    parser.add_argument('--z_dim',
                        help='Dimensionality of latent space',
                        type=int, default=2)
    parser.add_argument('--iterations',
                        help='Num of training iterations',
                        type=int, default=2000)
    parser.add_argument('--batch_size',
                        help='Batch size of each kind',
                        type=int, default=2000)
    parser.add_argument('--optimizer',
                        help='Optimizer: Adadelta/Adam/RMSprop/SGD',
                        type=str, default='Adadelta')
    parser.add_argument('--d_lr',
                        help='Learning rate of discriminator, for Adadelta it is the base learning rate',
                        type=float, default=1)
    parser.add_argument('--g_lr',
                        help='Learning rate of generator, for Adadelta it is the base learning rate',
                        type=float, default=1)
    parser.add_argument('--d_steps',
                        help='Steps of discriminators in each iteration',
                        type=int, default=3)
    parser.add_argument('--g_steps',
                        help='Steps of generator in each iteration',
                        type=int, default=1)
    parser.add_argument('--d_hidden_size',
                        help='Num of hidden units in discriminator',
                        type=int, default=100)
    parser.add_argument('--g_hidden_size',
                        help='Num of hidden units in generator',
                        type=int, default=50)
    parser.add_argument('--display_interval',
                        help='Interval of iterations to display/export images',
                        type=int, default=10)
    parser.add_argument('--no_display',
                        help='Show plots during training', action='store_true')
    parser.add_argument('--export',
                        help='Export images', action='store_true')
    parser.add_argument('--cpu',
                        help='Set to CPU mode', action='store_true')

    args = parser.parse_args()
    args.input_path = args.input_path.rstrip(os.sep)
    args.optimizer = OPTIMIZERS[args.optimizer.lower()]

    return args
