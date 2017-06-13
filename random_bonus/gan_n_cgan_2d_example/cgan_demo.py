#!/usr/bin/env python
# Conditional Generative Adversarial Networks (GAN) example with 2D samples in PyTorch.
import os
import numpy
from skimage import io
import torch
import torch.nn as nn
from torch.autograd import Variable
from sampler import generate_lut, sample_2d
from visualizer import CGANDemoVisualizer
from argparser import parse_args
from networks import SimpleMLP

DIMENSION = 2

args = parse_args()
cuda = False if args.cpu else True
bs = args.batch_size
z_dim = args.z_dim

image_paths = [os.sep.join([args.input_path, x]) for x in os.listdir(args.input_path)]
density_imgs = [io.imread(x, True) for x in image_paths]
luts_2d = [generate_lut(x) for x in density_imgs]
# Sampling based on visual density, a too small batch size may result in failure with conditions
pix_sums = [numpy.sum(x) for x in density_imgs]
total_pix_sums = numpy.sum(pix_sums)
c_indices = [0] + [int(sum(pix_sums[:i+1])/total_pix_sums*bs+0.5) for i in range(len(pix_sums)-1)] + [bs]

c_dim = len(luts_2d)    # Dimensionality of condition labels <--> number of images

visualizer = CGANDemoVisualizer('Conditional GAN 2D Example Visualization of {}'.format(args.input_path))

generator = SimpleMLP(input_size=z_dim+c_dim, hidden_size=args.g_hidden_size, output_size=DIMENSION)
discriminator = SimpleMLP(input_size=DIMENSION+c_dim, hidden_size=args.d_hidden_size, output_size=1)

if cuda:
    generator.cuda()
    discriminator.cuda()
criterion = nn.BCELoss()

d_optimizer = args.optimizer(discriminator.parameters(), lr=args.d_lr)
g_optimizer = args.optimizer(generator.parameters(), lr=args.d_lr)

y = numpy.zeros((bs, c_dim))
for i in range(c_dim):
    y[c_indices[i]:c_indices[i + 1], i] = 1  # conditional labels, one-hot encoding
y = Variable(torch.Tensor(y))
if cuda:
    y = y.cuda()

for train_iter in range(args.iterations):
    for d_index in range(args.d_steps):
        # 1. Train D on real+fake
        discriminator.zero_grad()

        #  1A: Train D on real samples with conditions
        real_samples = numpy.zeros((bs, DIMENSION))
        for i in range(c_dim):
            real_samples[c_indices[i]:c_indices[i+1], :] = sample_2d(luts_2d[i], c_indices[i+1]-c_indices[i])

        # first c dimensions is the condition inputs, the last 2 dimensions are samples
        real_samples = Variable(torch.Tensor(real_samples))
        if cuda:
            real_samples = real_samples.cuda()
        d_real_data = torch.cat([y, real_samples], 1)
        if cuda:
            d_real_data = d_real_data.cuda()
        d_real_decision = discriminator(d_real_data)
        labels = Variable(torch.ones(bs))
        if cuda:
            labels = labels.cuda()
        d_real_loss = criterion(d_real_decision, labels)  # ones = true

        #  1B: Train D on fake
        latent_samples = Variable(torch.randn(bs, z_dim))
        if cuda:
            latent_samples = latent_samples.cuda()
        # first c dimensions is the condition inputs, the last z_dim dimensions are latent samples
        d_gen_input = torch.cat([y, latent_samples], 1)
        d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
        conditional_d_fake_data = torch.cat([y, d_fake_data], 1)
        if cuda:
            conditional_d_fake_data = conditional_d_fake_data.cuda()
        d_fake_decision = discriminator(conditional_d_fake_data)
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

        latent_samples = Variable(torch.randn(bs, z_dim))
        if cuda:
            latent_samples = latent_samples.cuda()
        g_gen_input = torch.cat([y, latent_samples], 1)
        g_fake_data = generator(g_gen_input)
        conditional_g_fake_data = torch.cat([y, g_fake_data], 1)
        g_fake_decision = discriminator(conditional_g_fake_data)
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

        real_samples_with_y = d_real_data.data.cpu().numpy() if cuda else d_real_data.data.numpy()
        gen_samples_with_y = conditional_g_fake_data.data.cpu().numpy() if cuda else conditional_g_fake_data.data.numpy()
        if args.no_display:
            visualizer.draw(real_samples_with_y, gen_samples_with_y, msg, show=False)
        else:
            visualizer.draw(real_samples_with_y, gen_samples_with_y, msg)

        if args.export:
            filename = args.input_path.split(os.sep)[-1]
            output_dir = 'cgan_training_{}'.format(filename)
            os.system('mkdir -p {}'.format(output_dir))
            export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(train_iter)])
            visualizer.savefig(export_filepath)

if not args.no_display:
    visualizer.show()
