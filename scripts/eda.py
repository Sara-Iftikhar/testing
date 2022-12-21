"""
==============
EDA
==============
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import seaborn as sns

from ai4water.eda import EDA
from easy_mpl.utils import create_subplots

from utils import data_before_encoding, box_violin

# %%

ads_df = data_before_encoding()

# %%
print(ads_df.shape)

# %%
ads_df.head()

# %%
ads_df.tail()

# %%

ads_df['Adsorbent'].unique()
# %%

ads_df['Dye'].unique()

# %%

ads_df.pop("Adsorbent")
ads_df.pop("Dye")

eda = EDA(data = ads_df, save=False, show=False)

eda.correlation()
plt.tight_layout()
plt.show()

# %%

h_paras = ads_df.columns
fig, axes = create_subplots(ads_df.shape[1])
# fig, axes = plt.subplots(nrows=4, ncols=3,
#                             figsize=(12, 12),
#                         squeeze=False)

if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

for ax, col, label  in zip(axes.flat, ads_df, h_paras):

    sns.lineplot(ads_df[col], ax=ax,
                palette = 'Spectral',
           #show=False,
         #title=label,
         )
    ax.legend(fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

# %%

fig, axes = create_subplots(ads_df.shape[1])
for ax, col in zip(axes.flat, ads_df.columns):
    sns.boxplot(ads_df[col], ax=ax,
                fliersize=0.6,
                color='lightpink',
                orient='h',
                )
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()

# %%

fig, axes = create_subplots(ads_df.shape[1])
for ax, col in zip(axes.flat, ads_df.columns):
    box_violin(ax=ax, data=ads_df[col], palette="Set2")
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()


# %%

h_paras = ads_df.columns
fig, axes = create_subplots(ads_df.shape[1])
# fig, axes = plt.subplots(nrows=4, ncols=3,
#                             figsize=(12, 12),
#                         squeeze=False)

if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

for ax, col, label  in zip(axes.flat, ads_df, h_paras):

    sns.histplot(ads_df[col], ax=ax,
           #show=False,
         #title=label,
         )
    ax.legend(fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()
