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
data = np. array([[0.4284, 0.4648, 0.4288, 0.4499],
                [0.3338, 0.2214, 0.0717, 0.0732],
                [0.2677, 0.1739, 0.1738, 0.1806],
                [0.3250, 0.2173, 0.1998, 0.1694],
                [0.0612, 0.0526, 0.0431, 0.0364],
                [1.0002, 1.0029, 1.0253, 1.1376],
                [0.0785, 0.0306, 0.0233, 0.0221],
                [0.3004, 0.1982, 0.0758, 0.0765]])

data = data / np.max(data, axis=1).reshape(-1,1)

length = len(data)

x_labels = ['abalone', 'airfoil', 'housing', 'cadata', 'ccpp', 
            'energy', 'cpu', 'superconduct']

# Set plot parameters
fig, ax = plt.subplots()
width = 0.2 # width of bar
x = np.arange(length)

# palette = iter(bar_colors[len(bar_colors):0:-1]) # reversed colors
palette = iter(bar_colors)

ax.bar(x, data[:,0], width, color=next(palette), label='LESS')
ax.bar(x + width, data[:,1], width, color=next(palette), label='LESS-lDT')
ax.bar(x + (2 * width), data[:,2], width, color=next(palette), label='LESS-gRF')
ax.bar(x + (3 * width), data[:,3], width, color=next(palette), label='LESS-lDT-gRF')

ax.set_ylabel('Scaled MSE')
# ax.set_ylim([0.0, 1.0])
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
# ax.set_xlabel('Problems')
ax.set_title('Effects of Changing Local and Global Estimators')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

fig.tight_layout()
plt.show()
# %%