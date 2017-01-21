import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

speed_map = {
    'dog': (48, '#7199cf'),
    'cat': (45, '#4fc4aa'),
    'cheetah': (120, '#e1a7a2')
}

fig = plt.figure('Bar chart & Pie chart')

ax = fig.add_subplot(121)
ax.set_title('Running speed - bar chart')

xticks = np.arange(3)

bar_width = 0.5

animals = speed_map.keys()
speeds = [x[0] for x in speed_map.values()]
colors = [x[1] for x in speed_map.values()]
bars = ax.bar(xticks, speeds, width=bar_width, edgecolor='none')

ax.set_ylabel('Speed(km/h)')
ax.set_xticks(xticks+bar_width/2)
ax.set_xticklabels(animals)
ax.set_xlim([bar_width/2-0.5, 3-bar_width/2])
ax.set_ylim([0, 125])

for bar, color in zip(bars, colors):
    bar.set_color(color)

ax = fig.add_subplot(122)
ax.set_title('Running speed - pie chart')

labels = ['{}\n{} km/h'.format(a, s) for a, s in zip(animals, speeds)]

ax.pie(speeds, labels=labels, colors=colors)

plt.axis('equal')
plt.show()
