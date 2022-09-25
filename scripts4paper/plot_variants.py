# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.palettes import color_palette

sns.set_theme(style='whitegrid', color_codes=True)
sns.set_style('ticks')
bar_colors=sns.color_palette()

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
data = np. array([[0.4168, 0.4295, 0.4168, 0.4287],
                [0.1357, 0.0641, 0.1663, 0.0720],
                [0.1805, 0.1426, 0.2007, 0.1827],
                [0.1907, 0.1716, 0.1882, 0.1695],
                [0.0447, 0.0368, 0.0445, 0.0365],
                [1.0001, 1.0003, 1.0002, 1.0002],
                [0.0249, 0.0232, 0.0242, 0.0219],
                [0.0861, 0.0747, 0.0966, 0.0759]])


data = data / np.max(data, axis=1).reshape(-1,1)

length = len(data)

x_labels = ['abalone', 'airfoil', 'housing', 'cadata', 'ccpp', 
            'energy', 'cpu', 'superconduct']

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)

palette = iter(bar_colors[len(bar_colors):0:-1]) # reversed colors
palette = iter(bar_colors)

ax.bar(x, data[:,0], width, color=next(palette), label='LESS-C-V')
ax.bar(x + width, data[:,1], width, color=next(palette), label='LESS-C')
ax.bar(x + (2 * width), data[:,2], width, color=next(palette), label='LESS-V')
ax.bar(x + (3 * width), data[:,3], width, color=next(palette), label='LESS')

ax.set_ylabel('Scaled MSE')
# ax.set_ylim([0.0, 1.0])
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
# ax.set_xlabel('Problems')
ax.set_title('Results of LESS Variants')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

fig.tight_layout()
plt.show()
# %%