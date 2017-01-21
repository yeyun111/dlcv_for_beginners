import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D     

np.random.seed(42)

n_grids = 51            
c = n_grids / 2         
nf = 2                  

x = np.linspace(0, 1, n_grids)
y = np.linspace(0, 1, n_grids)
X, Y = np.meshgrid(x, y)

spectrum = np.zeros((n_grids, n_grids), dtype=np.complex)
noise = [np.complex(x, y) for x, y in np.random.uniform(-1,1,((2*nf+1)**2/2, 2))]
noisy_block = np.concatenate((noise, [0j], np.conjugate(noise[::-1])))

spectrum[c-nf:c+nf+1, c-nf:c+nf+1] = noisy_block.reshape((2*nf+1, 2*nf+1))
Z = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))

fig = plt.figure('3D surface & wire')

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.7, cmap='jet', rstride=1, cstride=1, lw=0)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, lw=0.5)

plt.show()
