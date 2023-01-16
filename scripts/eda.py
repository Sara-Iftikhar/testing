"""
==============
1. EDA
==============
"""

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import pandas as pd

from ai4water.eda import EDA
from easy_mpl import plot, boxplot, hist
from easy_mpl.utils import create_subplots

from utils import make_data, box_violin, \
    DYE_TYPES, ADSORBENT_TYPES

# %%
# Loading the original dataset

data, _, _ = make_data()

# %%
# Here, we are printing the shape of original dataset.
# The first value shows the number of samples/examples/datapoints
# and the second one shows the number of features.

print(data.shape)

# %%
# The first five samples are

data.head()

# %%
# The last five samples are

data.tail()

# %%
# The names of different adsorbents are

data['Adsorbent'].unique()

# %%
# The names of different dyes are

data['Dye'].unique()

# %%
# Removing the categorical features from our dataframe

data.pop("Adsorbent")
data.pop("Dye")

# %%
# get statistical summary of data

pd.set_option('display.max_columns', None)

print(data.describe())

# %%
# initializing an instance of EDA class from AI4Water
# in order to get some insights of the data

eda = EDA(data = data, save=False, show=False)

# %%
# plot correlation between numerical features

ax = eda.correlation(figsize=(9,9))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

# %%
# making a line plot for numerical features

fig, axes = create_subplots(data.shape[1])

for ax, col, label  in zip(axes.flat, data, data.columns):

    plot(data[col].values, ax=ax, ax_kws=dict(ylabel=col),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

# %%

fig, axes = create_subplots(data.shape[1])
for ax, col in zip(axes.flat, data.columns):
    boxplot(data[col].values, ax=ax, vert=False, fill_color='lightpink',
            flierprops={"ms": 1.0}, show=False, patch_artist=True,
            widths=0.6, medianprops={"color": "gray"},
            ax_kws=dict(xlabel=col, xlabel_kws={'weight': "bold"}))
plt.tight_layout()
plt.show()

# %%
# show the box and (half) violin plots together

fig, axes = create_subplots(data.shape[1])
for ax, col in zip(axes.flat, data.columns):
    box_violin(ax=ax, data=data[col], palette="Set2")
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()


# %%

fig, axes = create_subplots(data.shape[1])

for ax, col, label  in zip(axes.flat, data, data.columns):

    hist(data[col].values, ax=ax, bins=10,  show=False,
         grid=False,linewidth=0.5, edgecolor="k", color="khaki",
         ax_kws=dict(ylabel="Counts", xlabel=col))
plt.tight_layout()
plt.show()

# %%

data, _, _ = make_data()
data.pop('Dye')
feature = data['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
data['Adsorbent'] = feature

df_gb = data.loc[data['Adsorbent']=="GB"]
df_ac = data.loc[data['Adsorbent']=="AC"]
df_bio = data.loc[data['Adsorbent']=="Biochar"]
data.pop('Adsorbent')

fig, axes = create_subplots(data.shape[1])

for ax, col in zip(axes.flat, data.columns):

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

data, _, _ = make_data()
data.pop('Adsorbent')
feature = data['Dye']
d = {k:DYE_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
data['Dye'] = feature

df_an = data.loc[data['Dye']=="Anionic"]
df_cat = data.loc[data['Dye']=="Cationic"]
data.pop('Dye')

fig, axes = create_subplots(data.shape[1])

for ax, col in zip(axes.flat, data.columns):

    boxplot([df_an[col], df_cat[col]],
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

data, _, _ = make_data()
feature = data['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
data['Adsorbent'] = feature

df_ac = data.loc[data['Adsorbent']=="AC"]
df_ac['code'] = "AC"
df_bc = data.loc[data['Adsorbent']=="Biochar"]
df_bc['code'] = "BC"
df_gb = data.loc[data['Adsorbent']=="GB"]
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
