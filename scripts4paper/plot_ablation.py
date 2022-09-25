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

# Superconduct failed without a global function.
# So, it is set to 1.0 for the first two cases

data = np. array([[0.6313, 0.6145, 0.4818, 0.4303],
    [0.5812, 0.5759, 0.4899, 0.2760],
    [0.4602, 0.4601, 0.2901, 0.1246],
    [0.4426, 0.4395, 0.3639, 0.2581],
    [0.0838, 0.0820, 0.0714, 0.0556],
    [1.0043, 1.0041, 1.0002, 1.0002],
    [0.4503, 0.4451, 0.2888, 0.0489],
    [1.0000, 1.0000, 0.2699, 0.2610]])

data = data / np.max(data, axis=1).reshape(-1,1)

length = len(data)

x_labels = ['abalone', 'airfoil', 'housing', 'cadata', 'ccpp', 
            'energy', 'cpu', 'superconduct*']


# Set plot parameters
fig, ax = plt.subplots()
width = 0.20 # width of bar
x = np.arange(length)

# palette = iter(bar_colors[len(bar_colors):0:-1]) # reversed colors
palette = iter(bar_colors)

ax.bar(x, data[:,0], width, color=next(palette), label='NoW-NoG')
ax.bar(x + width, data[:,1], width, color=next(palette), label='W-NoG')
ax.bar(x + (2 * width), data[:,2], width, color=next(palette), label='NoW-G')
ax.bar(x + (3 * width), data[:,3], width, color=next(palette), label='LESS')

ax.set_ylabel('Scaled MSE')
# ax.set_ylim([0.0, 1.0])
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
# ax.set_xlabel('Problems')
ax.set_title('Ablation Study Results')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

fig.tight_layout()
plt.show()
# %%
