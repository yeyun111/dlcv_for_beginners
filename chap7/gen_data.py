import pickle
import numpy as np
import matplotlib.pyplot as plt


def cos_curve(x):
    return 0.25*np.sin(2*x*np.pi+0.5*np.pi) + 0.5

np.random.seed(123)
samples = []
labels = []

sample_density = 50
for i in range(sample_density):
    x1, x2 = np.random.random(2)
    bound = cos_curve(x1)
    if bound - 0.1 < x2 <= bound + 0.1:
        continue
    else:
        samples.append((x1, x2))
        if x2 > bound:
            labels.append(1)
        else:
            labels.append(0)

with open('data.pkl', 'wb') as f:
    pickle.dump((samples, labels), f)

for i, sample in enumerate(samples):
    plt.plot(sample[0], sample[1],
             'o' if labels[i] else '^',
             mec='r' if labels[i] else 'b',
             mfc='none',
             markersize=10)

x1 = np.linspace(0, 1)
plt.plot(x1, cos_curve(x1), 'k--')
plt.show()
