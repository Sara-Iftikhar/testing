"""
==============
1. EDA
==============
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import seaborn as sns
import pandas as pd

from ai4water.eda import EDA
from easy_mpl import plot, boxplot, hist
from easy_mpl.utils import create_subplots

from utils import make_data, box_violin, \
    DYE_TYPES, ADSORBENT_TYPES

# %%
# Loading the original dataset

ads_df, _, _ = make_data(encode=False)

# %%
# Here, we are printing the shape of original dataset.
# The first value shows the number of samples/examples/datapoints
# and the second one shows the number of features.

print(ads_df.shape)

# %%
# The first five samples are

ads_df.head()

# %%
# The last five samples are

ads_df.tail()

# %%
# The names of different adsorbents are

ads_df['Adsorbent'].unique()

# %%
# The names of different dyes are

ads_df['Dye'].unique()

# %%
# Removing the categorical features from our dataframe

ads_df.pop("Adsorbent")
ads_df.pop("Dye")

# %%
# get statistical summary of data

pd.set_option('display.max_columns', None)

print(ads_df.describe())

# %%
# initializing an instance of EDA class from AI4Water
# in order to get some insights of the data

eda = EDA(data = ads_df, save=False, show=False)

# %%
# plot correlation between numerical features

ax = eda.correlation(figsize=(9,9))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# making a line plot for numerical features

fig, axes = create_subplots(ads_df.shape[1])

for ax, col, label  in zip(axes.flat, ads_df, ads_df.columns):

    plot(ads_df[col].values, ax=ax, ax_kws=dict(ylabel=col),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

# %%

fig, axes = create_subplots(ads_df.shape[1])
for ax, col in zip(axes.flat, ads_df.columns):
    boxplot(ads_df[col].values, ax=ax, vert=False, fill_color='lightpink',
            flierprops={"ms": 1.0}, show=False, patch_artist=True,
            widths=0.6, medianprops={"color": "gray"},
            ax_kws=dict(xlabel=col, xlabel_kws={'weight': "bold"}))
plt.tight_layout()
plt.show()

# %%
# show the box and (half) violin plots together

fig, axes = create_subplots(ads_df.shape[1])
for ax, col in zip(axes.flat, ads_df.columns):
    box_violin(ax=ax, data=ads_df[col], palette="Set2")
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()


# %%

fig, axes = create_subplots(ads_df.shape[1])

for ax, col, label  in zip(axes.flat, ads_df, ads_df.columns):

    hist(ads_df[col].values, ax=ax, bins=10,  show=False,
         grid=False,linewidth=0.5, edgecolor="k", color="khaki",
         ax_kws=dict(ylabel="Counts", xlabel=col))
plt.tight_layout()
plt.show()

# %%

df, _, _ = make_data(encode=False)
df.pop('Dye')
feature = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
df['Adsorbent'] = feature

df_gb = df.loc[df['Adsorbent']=="GB"]
df_ac = df.loc[df['Adsorbent']=="AC"]
df_bio = df.loc[df['Adsorbent']=="Biochar"]

fig, axes = create_subplots(ads_df.shape[1])

for ax, col in zip(axes.flat, ads_df.columns):

    boxplot([df_gb[col], df_ac[col], df_bio[col]],
            labels=["GB", "AC", "BC"],
                ax=ax,
                flierprops={"ms": 0.6},
                fill_color='lightpink',
                patch_artist=True,
                widths=0.5,
            medianprops={"color": "gray"},
            vert=False,
            show=False,
            ax_kws=dict(xlabel=col, xlabel_kws={'weight': 'bold'})
                )
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()

# %%

df, _, _ = make_data(encode=False)
df.pop('Adsorbent')
feature = df['Dye']
d = {k:DYE_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
df['Dye'] = feature

df_an = df.loc[df['Dye']=="Anionic"]
df_cat = df.loc[df['Dye']=="Cationic"]

fig, axes = create_subplots(ads_df.shape[1])

for ax, col in zip(axes.flat, ads_df.columns):

    boxplot([df_an[col], df[col]],
            labels=["AN", "CT"],
                ax=ax,
                flierprops={"ms": 0.6},
            medianprops={"color": "gray"},
                fill_color='lightpink',
            patch_artist=True,
                vert=False,
                widths=0.5,
            show=False,
            ax_kws=dict(xlabel=col, xlabel_kws={"weight": "bold"})
                )
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()

# %%

df, _, _ = make_data(encode=False)
feature = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
df['Adsorbent'] = feature

df_ac = df.loc[df['Adsorbent']=="AC"]
df_ac['code'] = "AC"
df_bc = df.loc[df['Adsorbent']=="Biochar"]
df_bc['code'] = "BC"
df_gb = df.loc[df['Adsorbent']=="GB"]
df_gb['code'] = "GB"


df_ac.describe()

# %%

df_ac.median()

# %%

df_bc.describe()

# %%

df_bc.median()

# %%

df_gb.describe()

# %%

df_gb.median()

#%%

COLUMNS = ['Pyrolysis Temperature', 'Pyrolysis Time (min)', 'Surface Area', 'Pore Volume']

fig, axes = create_subplots(len(COLUMNS))

for ax, col in zip(axes.flat, COLUMNS):
    df_ads_feat = pd.concat([df_ac[[col, 'code']],
                             df_bc[[col, 'code']],
                             df_gb[[col, 'code']]])

    boxplot([df_ac[col], df_bc[col], df_gb[col]],
            labels=["AC", "BC", "GB"],
            ax=ax,
            flierprops={"ms": 0.6},
            medianprops={"color": "black"},
            fill_color='lightpink',
            patch_artist=True,
            vert=False,
            widths=0.5,
            show=False,
            ax_kws=dict(xlabel=col, xlabel_kws={"weight": "bold"})
                )
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()
