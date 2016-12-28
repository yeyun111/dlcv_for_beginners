import os
import sys
import datetime
import cv2

from multiprocessing import Process

import numpy as np
from matplotlib import pyplot

H_IMG, W_IMG = 100, 100
SAMPLE_SIZE = 70000
SAMPLES_DIR = 'samples'
N_PROCS = 2

def make_noise(index):
    h = np.random.randint(1, H_IMG)
    w = np.random.randint(1, W_IMG)
    noise = np.random.random((h, w))
    noisy_img = cv2.resize(noise, (H_IMG, W_IMG), interpolation=cv2.INTER_CUBIC)
    fx = float(w) / float(W_IMG)
    fy = float(h) / float(H_IMG)
    filename = '{}/{:0>5d}_{}_{}.jpg'.format(SAMPLES_DIR, index, fx, fy)
    pyplot.imsave(filename, noisy_img, cmap='gray')

def make_noises(i0, i1):
    np.random.seed(datetime.datetime.now().microsecond)
    for i in xrange(i0, i1):
        make_noise(i)
    print('Noises from {} to {} are made!'.format(i0+1, i1))
    sys.stdout.flush()

def main():
    cmd = 'mkdir -p {}'.format(SAMPLES_DIR)
    os.system(cmd)

    length = float(SAMPLE_SIZE)/float(N_PROCS)
    indices = [int(round(i * length)) for i in range(N_PROCS + 1)]
    processes = [Process(target=make_noises, args=(indices[i], indices[i+1])) for i in range(N_PROCS)]

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

    print('Done!')

if __name__ == '__main__':
    main()

