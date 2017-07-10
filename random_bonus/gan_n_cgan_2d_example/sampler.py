from functools import partial
import numpy
from skimage import transform

EPS = 1e-66
RESOLUTION = 0.001
num_grids = int(1/RESOLUTION+0.5)

def generate_lut(img):
    """
    linear approximation of CDF & marginal
    :param density_img:
    :return: lut_y, lut_x
    """
    density_img = transform.resize(img, (num_grids, num_grids))
    x_accumlation = numpy.sum(density_img, axis=1)
    sum_xy = numpy.sum(x_accumlation)
    y_cdf_of_accumulated_x = [[0., 0.]]
    accumulated = 0
    for ir, i in enumerate(range(num_grids-1, -1, -1)):
        accumulated += x_accumlation[i]
        if accumulated == 0:
            y_cdf_of_accumulated_x[0][0] = float(ir+1)/float(num_grids)
        elif EPS < accumulated < sum_xy - EPS:
            y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), accumulated/sum_xy])
        else:
            break
    y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), 1.])
    y_cdf_of_accumulated_x = numpy.array(y_cdf_of_accumulated_x)

    x_cdfs = []
    for j in range(num_grids):
        x_freq = density_img[num_grids-j-1]
        sum_x = numpy.sum(x_freq)
        x_cdf = [[0., 0.]]
        accumulated = 0
        for i in range(num_grids):
            accumulated += x_freq[i]
            if accumulated == 0:
                x_cdf[0][0] = float(i+1) / float(num_grids)
            elif EPS < accumulated < sum_xy - EPS:
                x_cdf.append([float(i+1)/float(num_grids), accumulated/sum_x])
            else:
                break
        x_cdf.append([float(i+1)/float(num_grids), 1.])
        if accumulated > EPS:
            x_cdf = numpy.array(x_cdf)
            x_cdfs.append(x_cdf)
        else:
            x_cdfs.append(None)

    y_lut = partial(numpy.interp, xp=y_cdf_of_accumulated_x[:, 1], fp=y_cdf_of_accumulated_x[:, 0])
    x_luts = [partial(numpy.interp, xp=x_cdfs[i][:, 1], fp=x_cdfs[i][:, 0]) if x_cdfs[i] is not None else None for i in range(num_grids)]

    return y_lut, x_luts

def sample_2d(lut, N):
    y_lut, x_luts = lut
    u_rv = numpy.random.random((N, 2))
    samples = numpy.zeros(u_rv.shape)
    for i, (x, y) in enumerate(u_rv):
        ys = y_lut(y)
        x_bin = int(ys/RESOLUTION)
        xs = x_luts[x_bin](x)
        samples[i][0] = xs
        samples[i][1] = ys

    return samples

if __name__ == '__main__':
    from skimage import io
    density_img = io.imread('inputs/random.jpg', True)
    lut_2d = generate_lut(density_img)
    samples = sample_2d(lut_2d, 10000)

    from matplotlib import pyplot
    fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(9, 4))
    fig.canvas.set_window_title('Test 2D Sampling')
    ax0.imshow(density_img, cmap='gray')
    ax0.xaxis.set_major_locator(pyplot.NullLocator())
    ax0.yaxis.set_major_locator(pyplot.NullLocator())

    ax1.axis('equal')
    ax1.axis([0, 1, 0, 1])
    ax1.plot(samples[:, 0], samples[:, 1], 'k,')
    pyplot.show()
