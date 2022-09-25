# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.palettes import color_palette

sns.set_theme(style='whitegrid', color_codes=True)
sns.set_style('ticks')
bar_colors=sns.color_palette('Greens_d')

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%

# 5%
superconduct_data = np.array([7.0102, 4.8459, 2.8251, 2.3642, 2.1134, 1.9714, 2.0797])
# 1%
# superconduct_data = np.array([28.0914, 20.5280, 11.7642, 9.4408, 8.7120, 8.9835, 9.4183])

superconduct_data = np.max(superconduct_data)/superconduct_data

# 5%
msd_data = np.array([330.1488, 223.4622, 158.5780, 139.0882, 128.0560, 119.7166, 124.2996])
# 1%
# msd_data = np.array([1465.1601, 971.9309, 705.5551, 591.0014, 551.5115, 562.0950, 579.8756])
    
msd_data = np.max(msd_data)/msd_data

df = pd.DataFrame({'Number of Processors':[1, 2, 4, 8, 16, 32, 60],
                   'superconduct': superconduct_data,
                   'msd': msd_data})

dfm = df.melt('Number of Processors', var_name='Problems', value_name='Speedup')

g = sns.pointplot(x='Number of Processors', y='Speedup', hue='Problems', data=dfm)
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles, labels=labels)

plt.show()
# %%
