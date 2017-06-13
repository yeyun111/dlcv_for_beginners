#!/usr/bin/env python
# Generative Adversarial Networks (GAN) example with 2D samples in PyTorch.
import os
from skimage import io
import torch
import torch.nn as nn
from torch.autograd import Variable
from sampler import generate_lut, sample_2d
from visualizer import GANDemoVisualizer
from argparser import parse_args
from networks import SimpleMLP

DIMENSION = 2

args = parse_args()
cuda = False if args.cpu else True
bs = args.batch_size
z_dim = args.z_dim

density_img = io.imread(args.input_path, True)
lut_2d = generate_lut(density_img)

visualizer = GANDemoVisualizer('GAN 2D Example Visualization of {}'.format(args.input_path))

generator = SimpleMLP(input_size=z_dim, hidden_size=args.g_hidden_size, output_size=DIMENSION)
discriminator = SimpleMLP(input_size=DIMENSION, hidden_size=args.d_hidden_size, output_size=1)

if cuda:
    generator.cuda()
    discriminator.cuda()
criterion = nn.BCELoss()

d_optimizer = args.optimizer(discriminator.parameters(), lr=args.d_lr)
g_optimizer = args.optimizer(generator.parameters(), lr=args.d_lr)

for train_iter in range(args.iterations):
    for d_index in range(args.d_steps):
        # 1. Train D on real+fake
        discriminator.zero_grad()

        #  1A: Train D on real
        real_samples = sample_2d(lut_2d, bs)
        d_real_data = Variable(torch.Tensor(real_samples))
        if cuda:
            d_real_data = d_real_data.cuda()
        d_real_decision = discriminator(d_real_data)
        labels = Variable(torch.ones(bs))
        if cuda:
            labels = labels.cuda()
        d_real_loss = criterion(d_real_decision, labels)  # ones = true

        #  1B: Train D on fake
        latent_samples = torch.randn(bs, z_dim)
        d_gen_input = Variable(latent_samples)
        if cuda:
            d_gen_input = d_gen_input.cuda()
        d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = discriminator(d_fake_data)
        labels = Variable(torch.zeros(bs))
        if cuda:
            labels = labels.cuda()
        d_fake_loss = criterion(d_fake_decision, labels)  # zeros = fake

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(args.g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        generator.zero_grad()

        latent_samples = torch.randn(bs, z_dim)
        g_gen_input = Variable(latent_samples)
        if cuda:
            g_gen_input = g_gen_input.cuda()
        g_fake_data = generator(g_gen_input)
        g_fake_decision = discriminator(g_fake_data)
        labels = Variable(torch.ones(bs))
        if cuda:
            labels = labels.cuda()
        g_loss = criterion(g_fake_decision, labels)  # we want to fool, so pretend it's all genuine

        g_loss.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if train_iter % args.display_interval == 0:
        loss_d_real = d_real_loss.data.cpu().numpy()[0] if cuda else d_real_loss.data.numpy()[0]
        loss_d_fake = d_fake_loss.data.cpu().numpy()[0] if cuda else d_fake_loss.data.numpy()[0]
        loss_g = g_loss.data.cpu().numpy()[0] if cuda else g_loss.data.numpy()[0]

        msg = 'Iteration {}: D_loss(real/fake): {:.6g}/{:.6g} G_loss: {:.6g}'.format(train_iter, loss_d_real, loss_d_fake, loss_g)
        print(msg)

        gen_samples = g_fake_data.data.cpu().numpy() if cuda else g_fake_data.data.numpy()

        if args.no_display:
            visualizer.draw(real_samples, gen_samples, msg, show=False)
        else:
            visualizer.draw(real_samples, gen_samples, msg)

        if args.export:
            filename = args.input_path.split(os.sep)[-1]
            output_dir = 'gan_training_{}'.format(filename[:filename.rfind('.')])
            os.system('mkdir -p {}'.format(output_dir))
            export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(train_iter)])
            visualizer.savefig(export_filepath)

if not args.no_display:
    visualizer.show()
