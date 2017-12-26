import numpy
from matplotlib import pyplot


def dist_o2l(p1, p2):
    # distance from origin to the line defined by (p1, p2)
    p12 = p2 - p1
    u12 = p12 / numpy.linalg.norm(p12)
    l_pp = numpy.dot(-p1, u12)
    pp = l_pp*u12 + p1
    return numpy.linalg.norm(pp)

dim = 100
N = 100000

rvs = []
dists2l = []
for i in range(N):
    u = numpy.random.randn(dim)
    v = numpy.random.randn(dim)
    rvs.extend([u, v])
    dists2l.append(dist_o2l(u, v))

dists = [numpy.linalg.norm(x) for x in rvs]

print('Distances to samples, mean: {}, std: {}'.format(numpy.mean(dists), numpy.std(dists)))
print('Distances to lines, mean: {}, std: {}'.format(numpy.mean(dists2l), numpy.std(dists2l)))

fig, (ax0, ax1) = pyplot.subplots(ncols=2, figsize=(11, 5))
ax0.hist(dists, 100, normed=1, color='g')
ax1.hist(dists2l, 100, normed=1, color='b')
pyplot.show()
