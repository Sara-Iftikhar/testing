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
from easy_mpl import boxplot
from easy_mpl.utils import create_subplots

from utils import data_before_encoding


COLUMNS = {
    "Adsorption_time (min)": "Adsop. Time",
    "calcination_temperature": "Calc. Temp",
    "calcination (min)": "Calc. ",
    "initial concentration": "Ini. Conc.",
    "adsorbent loading": "Adsorb. Load.",
    "adsorption_temperature": "Adsorp. Temp.",
}

ads_df = data_before_encoding()

ads_df = ads_df.rename(COLUMNS, axis=1)

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

fig, axes = create_subplots(ads_df.shape[1])
for ax, col in zip(axes.flat, ads_df.columns):
    sns.boxplot(ads_df[col], ax=ax,
                fliersize=0.6,
                color='lightpink',
                orient='h',
                )
    ax.set_xlabel(col)
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
