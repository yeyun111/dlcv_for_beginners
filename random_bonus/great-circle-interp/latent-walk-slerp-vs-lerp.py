from __future__ import print_function
import argparse
import numpy
import torch.utils.data
from torch.autograd import Variable
import networks
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_samples', type=int, default=10, help='how many images')
parser.add_argument('--n_steps', type=int, default=11, help='steps for interpolation')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='netG_epoch_49.pth', help="path to netG")

opt = parser.parse_args()
print(opt)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = 3

netG = networks.NetG(ngf, nz, nc, ngpu)
netG.eval()
netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
print(netG)

n_steps = opt.n_steps
for epoch in range(opt.n_samples):
    u = numpy.random.randn(nz)
    v = numpy.random.randn(nz)
    lu = numpy.linalg.norm(u)
    lv = numpy.linalg.norm(v)
    theta = numpy.arccos(numpy.dot(u, v)/lu/lv)

    ndimgs_slerp = []
    ndimgs_lerp = []
    for i in range(n_steps+1):
        t = float(i) / float(n_steps)

        # slerp
        z_slerp = numpy.sin((1 - t) * theta) / numpy.sin(theta) * u + numpy.sin(t * theta) / numpy.sin(theta) * v

        noise_t = z_slerp.reshape((1, nz, 1, 1))
        noise_t = torch.FloatTensor(noise_t)
        noisev = Variable(noise_t)
        fake = netG(noisev)
        timg = fake[0]
        timg = timg.data

        timg.add_(1).div_(2)
        ndimg = timg.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        ndimgs_slerp.append(ndimg)

        # lerp
        z_lerp = (1 - t) * u + t * v

        noise_t = z_lerp.reshape((1, nz, 1, 1))
        noise_t = torch.FloatTensor(noise_t)
        noisev = Variable(noise_t)
        fake = netG(noisev)
        timg = fake[0]
        timg = timg.data

        timg.add_(1).div_(2)
        ndimg = timg.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        ndimgs_lerp.append(ndimg)

    print('exporting {} ...'.format(epoch))

    # export slerp result
    ndimg = numpy.hstack(ndimgs_slerp)
    im = Image.fromarray(ndimg)
    im.save('e{:0>3d}-slerp.png'.format(epoch))

    # export lerp result
    ndimg = numpy.hstack(ndimgs_lerp)
    im = Image.fromarray(ndimg)
    im.save('e{:0>3d}-lerp.png'.format(epoch))
