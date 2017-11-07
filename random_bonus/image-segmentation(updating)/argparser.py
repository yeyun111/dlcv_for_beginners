import os
import argparse


def parse_param_file(filepath):
    with open(filepath, 'r') as f:
        kw_exprs = [x.strip() for x in f.readlines() if x.strip()]
    return eval('dict({})'.format(','.join(kw_exprs)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple Demo of Image Segmentation with U-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general options
    parser.add_argument('mode',
                        help='train/test')
    parser.add_argument('dataroot',
                        help='Directory containing training images in "images" and "segmentations" or test images')
    parser.add_argument('config',
                        help='Path to config file')
    parser.add_argument('--cpu',
                        help='Set to CPU mode', action='store_true')
    parser.add_argument('--output-dir',
                        help='Directory of output for both train/test',
                        type=str, default='')

    # test options
    parser.add_argument('--model',
                        help='Path to pre-trained model',
                        type=str, default='')

    args = parser.parse_args()

    params = {
        # general params
        'network': 'triangle',
        'layers': [32, 64, 128, 256, 512],
        'groups': 1, 
        'color_labels': [], 
        'image_width': None,
        'image_height': None
    }

    kwargs = parse_param_file(args.config)
    
    # other params specified in config file
    if args.mode == 'train':

        # default: no augmentation, with batch-norm

        train_params = {
            # training params
            'optimizer': 'SGD',
            'lr_policy': {0: 1e-4},
            'momentum': 0.9,
            'nesterov': True,
            'batch_norm': True,
            'batch_size': 4,
            'val_batch_size': None,
            'epochs': 24,
            'print_interval': 50,
            'validation_interval': 1000,
            'checkpoint_interval': 10000,
            'random_horizontal_flip': False,
            'random_square_crop': False,
            'random_crop': None,  # example: (0.81, 0.1), use 0.81 as area ratio, & 0.1 as the hw ratio variation
            'random_rotation': 0,
            'img_dir': 'images',
            'seg_dir': 'segmentations',
            'regression': False,
        }

        params.update(train_params)
        if params['val_batch_size'] is None:
            params['val_batch_size'] = params['batch_size']

    # update params from config
    for k, v in kwargs.items():
        if k in params:
            params[k] = v

    # set params to args
    for k, v in params.items():
        setattr(args, k, v)

    args.dataroot = args.dataroot.rstrip(os.sep)

    return args
