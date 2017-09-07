import logging
import os
import sys
import numpy
from PIL import Image
import torch
import torchvision
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from argparser import parse_args
import utils
import networks


def train(args):
    # set logger
    logging_dir = args.output_dir if args.output_dir else 'train-{}'.format(utils.get_datetime_string())
    os.mkdir('{}'.format(logging_dir))
    logging.basicConfig(
        level=logging.INFO,
        filename='{}/log.txt'.format(logging_dir),
        format='%(asctime)s %(message)s',
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # initialize loader
    if args.no_data_aug:
        train_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'train']),
                                                  image_folder=args.img_dir,
                                                  segmentation_folder=args.seg_dir,
                                                  labels=args.color_labels,
                                                  image_size=(args.image_width, args.image_height))
    else:
        train_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'train']),
                                                  image_folder=args.img_dir,
                                                  segmentation_folder=args.seg_dir,
                                                  labels=args.color_labels,
                                                  image_size=(args.image_width, args.image_height),
                                                  random_horizontal_flip=True,
                                                  random_rotation=6,
                                                  random_crop=(0.85, 0.1))
    val_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'val']),
                                            image_folder=args.img_dir,
                                            segmentation_folder=args.seg_dir,
                                            labels=args.color_labels,
                                            image_size=(args.image_width, args.image_height))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # initialize model, input channels need to be calculated by hand
    model = networks.UNet([32, 64, 128, 256, 512], 3, 2)
    if not args.cpu:
        model.cuda()

    criterion = utils.CrossEntropyLoss2D()

    # optimizer & lr policy
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    logging.info('| Learning Rate\t| Initialized learning rate: {}'.format(lr))

    # train
    for epoch in range(args.epochs):
        model.train()
        # update lr if lr_policy is defined
        if epoch in args.lr_policy:
            lr = args.lr_policy[epoch]
            optimizer = optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
            logging.info('| Learning Rate\t| Epoch: {}\t| Change learning rate to {}'.format(epoch, lr))

        # iterate all samples
        losses = utils.AverageMeter()
        for i_batch, (img, seg) in enumerate(train_loader):

            img = Variable(img)
            seg = Variable(seg)

            if not args.cpu:
                img = img.cuda()
                seg = seg.cuda()

            # compute output
            output = model(img)
            loss = criterion(output, seg)
            losses.update(loss.data[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % args.print_interval == 0:
                logging.info(
                    '| Epoch: {}/{}\t'
                    '| Iteration: {}/{}\t'
                    '| Training loss: {}'.format(
                        epoch, args.epochs,
                        i_batch+1, len(train_loader),
                        losses.avg
                    )
                )
                losses = utils.AverageMeter()

        logging.info(
            '| Epoch: {}/{}\t'
            '| Iteration: {}/{}\t'
            '| Training loss: {}'.format(
                epoch, args.epochs,
                i_batch+1, len(train_loader),
                losses.avg
            )
        )

        model.eval()
        losses = utils.AverageMeter()
        for i_batch, (img, seg) in enumerate(val_loader):

            img = Variable(img)
            seg = Variable(seg)

            if not args.cpu:
                img = img.cuda()
                seg = seg.cuda()

            # compute output
            output = model(img)
            loss = criterion(output, seg)
            losses.update(loss.data[0], float(img.size(0))/float(args.batch_size))

        logging.info(
            '| Epoch: {}/{}\t'
            '| Validation loss: {}'.format(
                epoch, args.epochs,
                losses.avg
            )
        )

        model_weights_path = '{}/epoch-{}.pth'.format(logging_dir, epoch+1)
        torch.save(model.state_dict(), model_weights_path)
        logging.info('| Checkpoint\t| {} is saved for epoch {}'.format(model_weights_path, epoch+1))


def test(args):
    if not args.model:
        print('Need a pretrained model!')
        return

    # check if output dir exists
    output_dir = args.output_dir if args.output_dir else 'test-{}'.format(utils.get_datetime_string())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load model
    model = networks.UNet([32, 64, 128, 256, 512], 3, 2)
    model.load_state_dict(torch.load(args.model))
    model = model.eval()

    if not args.cpu:
        model.cuda()

    # iterate all images with one by one
    transform = torchvision.transforms.ToTensor()
    for filename in [x for x in os.listdir(args.dataroot)]:
        filepath = os.sep.join([args.dataroot, filename])
        with open(filepath, 'r') as f:
            img = Image.open(f)
            img = img.resize((128, 256))
            img = transform(img)
            img = img.view(1, *img.shape)
            img = Variable(img)
        if not args.cpu:
            img = img.cuda()
        output = model(img)
        _, c, h, w = output.data.shape
        output_argmax = numpy.argmax(output.data.numpy()[0], axis=0)
        out_img = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        for i, color in enumerate(args.color_labels):
            out_img[output_argmax == i] = numpy.array(args.color_labels[i], dtype=numpy.uint8)
        out_img = Image.fromarray(out_img)
        seg_filepath = os.sep.join([output_dir, filename[:filename.rfind('.')]+'.png'])
        out_img.save(seg_filepath)
        print('{} is exported!'.format(seg_filepath))


if __name__ == '__main__':

    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print('Wrong input!')
