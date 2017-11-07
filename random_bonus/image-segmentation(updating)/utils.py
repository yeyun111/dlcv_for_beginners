from datetime import datetime
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets.folder import *
from torch.optim import SGD, Adadelta, Adam, Adagrad, RMSprop, ASGD
import cv2

OPTIMIZERS = {
    'sgd': SGD,
    'adadelta': Adadelta,
    'adam': Adam,
    'adagrad': Adagrad,
    'rmsprop': RMSprop,
    'asgd': ASGD
}


class SegmentationImageFolder(ImageFolder):
    """A simplified segmentation data loader where the images are arranged in this way: ::

        root/images/001.png
        root/images/002.png
        root/images/003.png

        root/segmentations/001.png
        root/segmentations/002.png
        root/segmentations/003.png

        images in the two folder should be corresponding, sorting by name

    Args:
        please refer to
        https://github.com/frombeijingwithlove/dlcv_for_beginners/blob/master/chap6/data_augmentation/image_augmentation.py
    """

    def __init__(self, root,
                 image_folder='images', segmentation_folder='segmentations',
                 labels=[(0, 0, 0), (255, 255, 255)],
                 image_size=None,
                 random_horizontal_flip=False,
                 random_rotation=0,
                 random_crop=None,
                 random_square_crop=False,
                 loader=default_loader,
                 label_regr=False,
                 multi_scale=0):
        super(SegmentationImageFolder, self).__init__(root, loader=loader)
        pair_len = len(self.imgs) // 2
        assert image_folder in self.classes and segmentation_folder in self.classes
        if image_folder < segmentation_folder:
            self.imgs = [(self.imgs[i][0], self.imgs[i+pair_len][0]) for i in range(pair_len)]
        else:
            self.imgs = [(self.imgs[i+pair_len][0], self.imgs[i][0]) for i in range(pair_len)]
        self.img_folder = image_folder
        self.seg_folder = segmentation_folder
        self.labels = [numpy.array(x, dtype=numpy.uint8) for x in labels]
        self.image_size = image_size
        self.flip_lr = random_horizontal_flip
        self.random_rotation = random_rotation
        self.random_crop = random_crop
        self.random_square_crop = random_square_crop
        self.label_regr=label_regr
        self.multi_scale=multi_scale

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        imgpath, segpath = self.imgs[index]
        img = self.loader(imgpath)
        seg = self.loader(segpath)

        # manually transform to incorporate horizontal flip & one-hot coding for segmentation labels
        if self.random_rotation:
            w, h = img.size
            angle = self.random_rotation % 360
            img = img.rotate(angle)
            seg = seg.rotate(angle)

            angle_crop = angle % 180
            if angle_crop > 90:
                angle_crop = 180 - angle_crop
            theta = angle_crop * numpy.pi / 180.0
            hw_ratio = float(h) / float(w)
            tan_theta = numpy.tan(theta)
            numerator = numpy.cos(theta) + numpy.sin(theta) * tan_theta
            r = hw_ratio if h > w else 1 / hw_ratio
            denominator = r * tan_theta + 1
            crop_mult = numerator / denominator
            w_crop = int(round(crop_mult * w))
            h_crop = int(round(crop_mult * h))
            x0 = int((w - w_crop) / 2)
            y0 = int((h - h_crop) / 2)

            img = img.crop((x0, y0, x0+w_crop, y0+h_crop))
            seg = seg.crop((x0, y0, x0+w_crop, y0+h_crop))

        if self.random_crop:
            area_ratio, hw_vari = self.random_crop
            w, h = img.size
            hw_delta = numpy.random.uniform(-hw_vari, hw_vari)
            hw_mult = 1 + hw_delta
            w_crop = int(round(w * numpy.sqrt(area_ratio * hw_mult)))
            if w_crop > w - 2:
                w_crop = w - 2
            h_crop = int(round(h * numpy.sqrt(area_ratio / hw_mult)))
            if h_crop > h - 2:
                h_crop = h - 2
            x0 = numpy.random.randint(0, w - w_crop - 1)
            y0 = numpy.random.randint(0, h - h_crop - 1)
            img = img.crop((x0, y0, x0+w_crop, y0+h_crop))
            seg = seg.crop((x0, y0, x0+w_crop, y0+h_crop))

        if self.random_square_crop:
            w, h = img.size
            if w > h:
                x0 = random.randint(0, w-h-1)
                img = img.crop((x0, 0, x0+h, h))
                seg = seg.crop((x0, 0, x0+h, h))
            elif w < h:
                y0 = random.randint(0, h-w-1)
                img = img.crop((0, y0, w, y0+w))
                seg = seg.crop((0, y0, w, y0+w))

        if self.image_size:
            img = img.resize(self.image_size)
            seg = seg.resize(self.image_size, Image.NEAREST)

        # random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        # one-hot coding for segmentation labels
        seg_arr = numpy.array(seg)
        seg = numpy.zeros(seg_arr.shape[:2], dtype=numpy.int64)
        for i, label_color in enumerate(self.labels):
            label_indices = numpy.where(seg_arr == label_color)[:2]
            seg[label_indices[0], label_indices[1]] = i

        if self.multi_scale:
            h, w = seg.shape
            seg = [seg] + [cv2.resize(seg, (w//(2**i), h//(2**i)), interpolation=cv2.INTER_NEAREST).astype(numpy.int64) for i in range(1, self.multi_scale)]

        # to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(img)
        if self.multi_scale:
            seg = [torch.Tensor(x) if self.label_regr else torch.LongTensor(x) for x in seg]
        else:
            seg = torch.Tensor(seg) if self.label_regr else torch.LongTensor(seg)

        return img, seg

    def __len__(self):
        return len(self.imgs)


class CrossEntropyLoss2D(nn.Module):
    def __init__(self, size_average=True):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss_2d = nn.NLLLoss2d(size_average=size_average)

    def forward(self, outputs, targets):
        return self.nll_loss_2d(F.log_softmax(outputs), targets)


class MSCrossEntropyLoss2D(nn.Module):
    def __init__(self, weights, size_average=True):
        super(MSCrossEntropyLoss2D, self).__init__()
        self.nll_loss_2d = nn.NLLLoss2d(size_average=size_average)
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.weights[0] * self.nll_loss_2d(F.log_softmax(outputs[0]), targets[0])
        for i in range(len(self.weights)-1):
            loss += self.weights[i+1] * self.nll_loss_2d(F.log_softmax(outputs[i+1]), targets[i])
        return loss


def get_datetime_string():
    datetime_now = datetime.now()
    return '{}-{}-{}-{}-{}-{}'.format(
        datetime_now.year,
        datetime_now.month,
        datetime_now.day,
        datetime_now.hour,
        datetime_now.minute,
        datetime_now.second
    )


# borrowed from 
# https://github.com/pytorch/examples/tree/master/imagenet
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(name, model_params, **kwargs):
    name = name.lower()
    if name == 'sgd':
        optimizer = OPTIMIZERS[name](
            model_params,
            lr=kwargs['lr'],
            momentum=kwargs['momentum'],
            nesterov=kwargs['nesterov']
        )
    elif name in ['adadelta', 'adam', 'adagrad', 'asgd']:
        optimizer = OPTIMIZERS[name](model_params, lr=kwargs['lr'])
    elif name == 'rmsprop':
        optimizer = OPTIMIZERS[name](
            model_params,
            lr=kwargs['lr'],
            momentum=kwargs['momentum'],
        )
    else:
        raise Exception('Not supported optimizer!')

    return optimizer
