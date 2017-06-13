from itertools import cycle
import numpy
from matplotlib import pyplot
from skimage import filters


class GANDemoVisualizer:

    def __init__(self, title, l_kde=100, bw_kde=5):
        self.title = title
        self.l_kde = l_kde
        self.resolution = 1. / self.l_kde
        self.bw_kde_ = bw_kde
        self.fig, self.axes = pyplot.subplots(ncols=3, figsize=(13.5, 4))
        self.fig.canvas.set_window_title(self.title)

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=500, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples, cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples, cmap)

        if show:
            pyplot.draw()
            pyplot.pause(pause_time)

    @staticmethod
    def draw_samples(axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        axis.plot(generated_samples[:max_sample_size, 0], generated_samples[:max_sample_size, 1], '.')
        axis.plot(real_samples[:max_sample_size, 0], real_samples[:max_sample_size, 1], 'kx')
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def draw_density_estimation(self, axis, title, samples, cmap):
        axis.clear()
        axis.set_xlabel(title)
        density_estimation = numpy.zeros((self.l_kde, self.l_kde))
        for x, y in samples:
            if 0 < x < 1 and 0 < y < 1:
                density_estimation[int((1-y) / self.resolution)][int(x / self.resolution)] += 1
        density_estimation = filters.gaussian(density_estimation, self.bw_kde_)
        axis.imshow(density_estimation, cmap=cmap)
        axis.xaxis.set_major_locator(pyplot.NullLocator())
        axis.yaxis.set_major_locator(pyplot.NullLocator())

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    @staticmethod
    def show():
        pyplot.show()


class CGANDemoVisualizer(GANDemoVisualizer):

    def __init__(self, title, l_kde=100, bw_kde=5):
        GANDemoVisualizer.__init__(self, title, l_kde, bw_kde)

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=500, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples[:, -2:], cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples[:, -2:], cmap)

        if show:
            pyplot.draw()
            pyplot.pause(pause_time)

    def draw_samples(self, axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        g_samples = numpy.copy(generated_samples)
        r_samples = numpy.copy(real_samples)
        numpy.random.shuffle(g_samples)
        numpy.random.shuffle(r_samples)
        g_samples = g_samples[:max_sample_size, :]
        r_samples = r_samples[:max_sample_size, :]
        color_iter = cycle('bgrcmy')
        for i in range(g_samples.shape[1]-2):
            c = next(color_iter)
            samples = g_samples[g_samples[:, i] > 0, :][:, -2:]
            axis.plot(samples[:, 0], samples[:, 1], c+'.', markersize=5)
            samples = r_samples[r_samples[:, i] > 0, :][:, -2:]
            axis.plot(samples[:, 0], samples[:, 1], c+'x', markersize=5)
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    @staticmethod
    def show():
        pyplot.show()
