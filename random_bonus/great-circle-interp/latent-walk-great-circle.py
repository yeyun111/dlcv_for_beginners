from __future__ import print_function
import argparse
import os
import numpy
from scipy.stats import chi
import torch.utils.data
from torch.autograd import Variable
from networks import NetG
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=10, help='how many paths')
parser.add_argument('--n_steps', type=int, default=23, help='steps to walk')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='netG_epoch_49.pth', help="trained params for G")

opt = parser.parse_args()
output_dir = 'gcircle-walk'
os.system('mkdir -p {}'.format(output_dir))
print(opt)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = 3

netG = NetG(ngf, nz, nc, ngpu)
netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
netG.eval()
print(netG)

for j in range(opt.niter):
    # step 1
    r = chi.rvs(df=100)

    # step 2
    u = numpy.random.normal(0, 1, nz)
    w = numpy.random.normal(0, 1, nz)
    u /= numpy.linalg.norm(u)
    w /= numpy.linalg.norm(w)

    v = w - numpy.dot(u, w) * u
    v /= numpy.linalg.norm(v)

    ndimgs = []
    for i in range(opt.n_steps):
        t = float(i) / float(opt.n_steps)
        # step 3
        z = numpy.cos(t * 2 * numpy.pi) * u + numpy.sin(t * 2 * numpy.pi) * v
        z *= r

        noise_t = z.reshape((1, nz, 1, 1))
        noise_t = torch.FloatTensor(noise_t)
        noisev = Variable(noise_t)
        fake = netG(noisev)
        timg = fake[0]
        timg = timg.data

        timg.add_(1).div_(2)
        ndimg = timg.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        ndimgs.append(ndimg)

    print('exporting {} ...'.format(j))
    ndimg = numpy.hstack(ndimgs)

    im = Image.fromarray(ndimg)
    filename = os.sep.join([output_dir, 'gc-{:0>6d}.png'.format(j)])
    im.save(filename)
