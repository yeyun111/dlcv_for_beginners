import os
import argparse
import random
import math
from multiprocessing import cpu_count, Pool
from functools import partial

import cv2

import image_augmentation as ia

def parse_args():
    parser = argparse.ArgumentParser(
        description='A Simple Image Data Augmentation Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_dir',
                        help='Directory containing images')
    parser.add_argument('output_dir',
                        help='Directory for augmented images')
    parser.add_argument('num',
                        help='Number of images to be augmented',
                        type=int)

    parser.add_argument('--num_procs',
                        help='Number of processes for paralleled augmentation',
                        type=int, default=cpu_count())

    parser.add_argument('--p_mirror',
                        help='Ratio to mirror an image',
                        type=float, default=0.5)

    parser.add_argument('--p_crop',
                        help='Ratio to randomly crop an image',
                        type=float, default=1.0)
    parser.add_argument('--crop_size',
                        help='The ratio of cropped image size to original image size, in area',
                        type=float, default=0.8)
    parser.add_argument('--crop_hw_vari',
                        help='Variation of h/w ratio',
                        type=float, default=0.1)

    parser.add_argument('--p_rotate',
                        help='Ratio to randomly rotate an image',
                        type=float, default=1.0)
    parser.add_argument('--p_rotate_crop',
                        help='Ratio to crop out the empty part in a rotated image',
                        type=float, default=1.0)
    parser.add_argument('--rotate_angle_vari',
                        help='Variation range of rotate angle',
                        type=float, default=10.0)

    parser.add_argument('--p_hsv',
                        help='Ratio to randomly change gamma of an image',
                        type=float, default=1.0)
    parser.add_argument('--hue_vari',
                        help='Variation of hue',
                        type=int, default=10)
    parser.add_argument('--sat_vari',
                        help='Variation of saturation',
                        type=float, default=0.1)
    parser.add_argument('--val_vari',
                        help='Variation of value',
                        type=float, default=0.1)

    parser.add_argument('--p_gamma',
                        help='Ratio to randomly change gamma of an image',
                        type=float, default=1.0)
    parser.add_argument('--gamma_vari',
                        help='Variation of gamma',
                        type=float, default=2.0)

    args = parser.parse_args()
    args.input_dir = args.input_dir.rstrip('/')
    args.output_dir = args.output_dir.rstrip('/')

    return args

def generate_image_list(args):
    filenames = os.listdir(args.input_dir)
    num_imgs = len(filenames)

    num_ave_aug = int(math.floor(args.num/num_imgs))
    rem = args.num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

    img_list = [
        (os.sep.join([args.input_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]

    random.shuffle(img_list)  # in case the file size are not uniformly distributed
    return img_list

def augment_image(image_num_pair, args):
    filepath, n = image_num_pair
    img = cv2.imread(filepath)
    filename = filepath.split(os.sep)[-1]
    dot_pos = filename.rfind('.')
    imgname = filename[:dot_pos]
    ext = filename[dot_pos:]

    print('Augmenting {} ...'.format(filename))
    for i in range(n):
        img_varied = img.copy()
        varied_imgname = '{}_{:0>3d}_'.format(imgname, i)
        if random.random() < args.p_mirror:
            img_varied = cv2.flip(img_varied, 1)
            varied_imgname += 'm'
        if random.random() < args.p_crop:
            img_varied = ia.random_crop(
                img_varied,
                args.crop_size,
                args.crop_hw_vari)
            varied_imgname += 'c'
        if random.random() < args.p_rotate:
            img_varied = ia.random_rotate(
                img_varied,
                args.rotate_angle_vari,
                args.p_rotate_crop)
            varied_imgname += 'r'
        if random.random() < args.p_hsv:
            img_varied = ia.random_hsv_transform(
                img_varied,
                args.hue_vari,
                args.sat_vari,
                args.val_vari)
            varied_imgname += 'h'
        if random.random() < args.p_gamma:
            img_varied = ia.random_gamma_transform(
                img_varied,
                args.gamma_vari)
            varied_imgname += 'g'
        output_filepath = os.sep.join([
            args.output_dir,
            '{}{}'.format(varied_imgname, ext)])
        cv2.imwrite(output_filepath, img_varied)

def main():
    args = parse_args()
    params_str = str(args)[10:-1]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('Starting image data augmentation for {}\n'
          'with\n{}\n'.format(args.input_dir, params_str))

    image_list = generate_image_list(args)
    aug_img = partial(augment_image, args=args)
    pool = Pool(args.num_procs)
    pool.map(aug_img, image_list)

    print('\nDone!')

if __name__ == '__main__':
    main()
