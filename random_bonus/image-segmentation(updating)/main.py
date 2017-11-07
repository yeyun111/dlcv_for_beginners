import logging
import os
import numpy
from PIL import Image
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.functional as F
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

    logging.info('=========== Taks {} started! ==========='.format(args.output_dir))
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))
    logging.info('========================================')

    # initialize loader
    multi_scale = len(args.layers) if args.network != 'unet' else 0
    train_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'train']),
                                              image_folder=args.img_dir,
                                              segmentation_folder=args.seg_dir,
                                              labels=args.color_labels,
                                              image_size=(args.image_width, args.image_height),
                                              random_horizontal_flip=args.random_horizontal_flip,
                                              random_rotation=args.random_rotation,
                                              random_crop=args.random_crop,
                                              random_square_crop=args.random_square_crop,
                                              label_regr=args.regression,
                                              multi_scale=multi_scale)
    val_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'val']),
                                            image_folder=args.img_dir,
                                            segmentation_folder=args.seg_dir,
                                            labels=args.color_labels,
                                            image_size=(args.image_width, args.image_height),
                                            random_square_crop=args.random_square_crop,
                                            label_regr=args.regression)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size)

    # initialize model, input channels need to be calculated by hand
    n_classes = len(args.color_labels)

    if args.network == 'unet':
        network = networks.UNet
        criterion = nn.MSELoss() if args.regression else utils.CrossEntropyLoss2D()
    elif args.network == 'triangle':
        network = networks.TriangleNet
        criterion = utils.MSCrossEntropyLoss2D([0.15]+[0.85/float(multi_scale)]*multi_scale)
    else:
        pass
    val_criterion = utils.CrossEntropyLoss2D()

    if args.regression:
        model = network(args.layers, 3, 1, groups=args.groups)
    else:
        model = network(args.layers, 3, n_classes, groups=args.groups)
    if not args.cpu:
        model.cuda()

    # train
    iterations = 0
    for epoch in range(args.epochs):
        model.train()
        # update lr according to lr policy
        if epoch in args.lr_policy:
            lr = args.lr_policy[epoch]
            optimizer = utils.get_optimizer(args.optimizer, model.parameters(),
                                            lr=lr, momentum=args.momentum, nesterov=args.nesterov)
            if epoch > 0:
                logging.info('| Learning Rate | Epoch: {: >3d} | Change learning rate to {}'.format(epoch+1, lr))
            else:
                logging.info('| Learning Rate | Initial learning rate: {}'.format(lr))

        # iterate all samples
        losses = utils.AverageMeter()
        for i_batch, (img, seg) in enumerate(train_loader):

            img = Variable(img)
            seg = Variable(seg) if not multi_scale else [Variable(x) for x in seg]

            if not args.cpu:
                img = img.cuda()
                seg = seg.cuda() if not multi_scale else [x.cuda() for x in seg]

            # compute output
            output = model(img)
            loss = criterion(output, seg)
            losses.update(loss.data[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging training curve
            if iterations % args.print_interval == 0:
                logging.info(
                    '| Iterations: {: >6d} '
                    '| Epoch: {: >3d}/{: >3d} '
                    '| Batch: {: >4d}/{: >4d} '
                    '| Training loss: {:.6f}'.format(
                        iterations, 
                        epoch+1, args.epochs,
                        i_batch, len(train_loader)-1,
                        losses.avg
                    )
                )
                losses = utils.AverageMeter()

            # validation on all val samples
            if iterations % args.validation_interval == 0:
                model.eval()
                val_losses = utils.AverageMeter()
                gt_pixel_count = [0] * n_classes
                pred_pixel_count = [0] * n_classes
                intersection_pixel_count = [0] * n_classes
                union_pixel_count = [0] * n_classes

                for img, seg in val_loader:

                    img = Variable(img)
                    seg = Variable(seg)

                    if not args.cpu:
                        img = img.cuda()
                        seg = seg.cuda()

                    # compute output
                    output = model(img)
                    loss = val_criterion(output, seg)
                    val_losses.update(loss.data[0], float(img.size(0))/float(args.batch_size))
                    output_numpy = output.data.numpy() if args.cpu else output.data.cpu().numpy()
                    pred_labels = numpy.argmax(output_numpy, axis=1)
                    gt_labels = seg.data.numpy() if args.cpu else seg.data.cpu().numpy()

                    pred_labels = pred_labels.flatten()
                    gt_labels = gt_labels.flatten()

                    for i in range(n_classes):
                        pred_pixel_count[i] += (pred_labels == i).sum()
                        gt_pixel_count[i] += (gt_labels == i).sum()
                        gt_dumb = numpy.full(gt_labels.shape, -1, dtype=numpy.int)
                        pred_dumb = numpy.full(pred_labels.shape, -2, dtype=numpy.int)
                        gt_dumb[gt_labels == i] = 0
                        pred_dumb[pred_labels == i] = 0
                        intersection_pixel_count[i] += (gt_dumb == pred_dumb).sum()
                        pred_dumb[gt_labels == i] = 0
                        union_pixel_count[i] += (pred_dumb == 0).sum()

                # calculate mPA & mIOU
                mPA = 0
                mIOU = 0
                for i in range(n_classes):
                    mPA += float(intersection_pixel_count[i]) / float(gt_pixel_count[i])
                    mIOU += float(intersection_pixel_count[i]) / float(union_pixel_count[i])
                mPA /= float(n_classes)
                mIOU /= float(n_classes)

                logging.info(
                    '| Iterations: {: >6d} '
                    '| Epoch: {: >3d}/{: >3d} '
                    '| Average mPA: {:.4f} '
                    '| Average mIOU: {:.4f} '
                    '| Validation loss: {:.6f} '.format(
                        iterations, 
                        epoch+1, args.epochs,
                        mPA,
                        mIOU,
                        val_losses.avg
                    )
                )

                model.train()

            if iterations % args.checkpoint_interval == 0 and iterations > 0:
                model_weights_path = '{}/iterations-{:0>6d}-epoch-{:0>3d}.pth'.format(logging_dir, iterations, epoch+1)
                torch.save(model.state_dict(), model_weights_path)
                logging.info('| Checkpoint | {} is saved!'.format(model_weights_path))

            iterations += 1


def test(args):
    if not args.model:
        print('Need a pretrained model!')
        return

    if not args.color_labels:
        print('Need to specify color labels')
        return

    resize_img = False if args.image_width is None or args.image_height is None else True

    # check if output dir exists
    output_dir = args.output_dir if args.output_dir else 'test-{}'.format(utils.get_datetime_string())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load model
    if args.network == 'unet':
        network = networks.UNet
    elif args.network == 'triangle':
        network = networks.TriangleNet
    else:
        pass

    model = network(args.unet_layers, 3, len(args.color_labels), groups=args.groups)
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
            img = img.resize((args.image_width, args.image_height))
            img = transform(img)
            img = img.view(1, *img.shape)
            img = Variable(img)
        if not args.cpu:
            img = img.cuda()
        output = model(img)
        _, c, h, w = output.data.shape
        output_numpy = output.data.numpy()[0] if args.cpu else output.data.cpu().numpy()[0]
        output_argmax = numpy.argmax(output_numpy, axis=0)
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
        if args.image_width is None or args.image_height is None:
            print('Image size must be specified for training!')
            exit(0)
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print('Wrong input! Please specify "train" or "test"')

