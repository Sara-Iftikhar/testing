"""
==============
EDA
==============
"""

import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ai4water.eda import EDA
from easy_mpl import hist

from utils import data_before_encoding

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

eda = EDA(data = ads_df, save=False)

eda.correlation()

# %%

h_paras = ads_df.columns

fig, axes = plt.subplots(nrows=4, ncols=3,
                            figsize=(12, 12),
                        squeeze=False)

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
plt.show()

# %%

ax = sns.boxplot(ads_df)
ax.set_xticks(np.arange(len(ads_df.columns)))
ax.set_xticklabels(ads_df.columns, rotation=65)
plt.show()

# %%

h_paras = ads_df.columns

fig, axes = plt.subplots(nrows=4, ncols=3,
                            figsize=(12, 12),
                        squeeze=False)

if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

for ax, col, label  in zip(axes.flat, ads_df, h_paras):

    hist(ads_df[col], ax=ax,
           show=False,
         title=label,
         )
    ax.legend(fontsize=10)
plt.legend()
plt.show()
