import sys
import numpy as np

sys.path.append('/path/to/caffe/python')
import caffe

solver = caffe.SGDSolver('solver.prototxt')
solver.solve()

net = solver.net
net.blobs['data'] = np.array([[0.5, 0.5]])
output = net.forward()
print(output)
