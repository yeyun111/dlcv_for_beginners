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

    logging.info('=========== Taks {} started! ==========='.format(args.output_dir))
    for arg in vars(args):
        logging.info('{}: {}'.format(arg, getattr(args, arg)))
    logging.info('========================================')

    # initialize loader
    train_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'train']),
                                              image_folder=args.img_dir,
                                              segmentation_folder=args.seg_dir,
                                              labels=args.color_labels,
                                              image_size=(args.image_width, args.image_height),
                                              random_horizontal_flip=args.random_horizontal_flip,
                                              random_rotation=args.random_rotation,
                                              random_crop=args.random_crop,
                                              random_square_crop=args.random_square_crop)
    val_set = utils.SegmentationImageFolder(os.sep.join([args.dataroot, 'val']),
                                            image_folder=args.img_dir,
                                            segmentation_folder=args.seg_dir,
                                            labels=args.color_labels,
                                            image_size=(args.image_width, args.image_height),
                                            random_square_crop=args.random_square_crop)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # initialize model, input channels need to be calculated by hand
    model = networks.UNet(args.unet_layers, 3, len(args.color_labels), use_bn=args.batch_norm)
    if not args.cpu:
        model.cuda()

    criterion = utils.CrossEntropyLoss2D()

    # train
    iterations = 0
    for epoch in range(args.epochs):
        model.train()
        # update lr according to lr policy
        if epoch in args.lr_policy:
            lr = args.lr_policy[epoch]
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=args.nesterov)
            if epoch > 0:
                logging.info('| Learning Rate | Epoch: {: >3d} | Change learning rate to {}'.format(epoch+1, lr))
            else:
                logging.info('| Learning Rate | Initial learning rate: {}'.format(lr))

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

            if iterations % args.validation_interval == 0:
                model.eval()
                val_losses = utils.AverageMeter()
                for img, seg in val_loader:

                    img = Variable(img)
                    seg = Variable(seg)

                    if not args.cpu:
                        img = img.cuda()
                        seg = seg.cuda()

                    # compute output
                    output = model(img)
                    loss = criterion(output, seg)
                    val_losses.update(loss.data[0], float(img.size(0))/float(args.batch_size))

                logging.info(
                    '| Iterations: {: >6d} '
                    '| Epoch: {: >3d}/{: >3d} '
                    '| Validation loss: {:.6f}'.format(
                        iterations, 
                        epoch+1, args.epochs,
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

    # check if output dir exists
    output_dir = args.output_dir if args.output_dir else 'test-{}'.format(utils.get_datetime_string())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # load model
    model = networks.UNet(args.unet_layers, 3, len(args.color_labels))
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
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print('Wrong input! Please specify "train" or "test"')

