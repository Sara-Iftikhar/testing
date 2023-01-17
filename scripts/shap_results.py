"""
====================
7. Interpretation
====================
"""

import platform
import site
site.addsitedir("D:\\mytools\\AI4Water")

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
import pandas as pd

import shap
from shap import Explanation
from shap.plots import beeswarm, violin, heatmap, waterfall
from shap import DeepExplainer, GradientExplainer, KernelExplainer

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder

from alibi.explainers import IntegratedGradients
from alibi.utils import gen_category_map
from alibi.explainers import plot_pd_variance
from alibi.explainers import PartialDependenceVariance

from easy_mpl import imshow, pie, bar_chart
from easy_mpl.utils import create_subplots
from ai4water.postprocessing import PermutationImportance
from ai4water.postprocessing import PartialDependencePlot

from utils import get_dataset, get_fitted_model, evaluate_model, \
    box_violin, shap_scatter, DYE_TYPES, ADSORBENT_TYPES, make_data

# %%

CAT_FEATURES = ['Adsorbent', 'Dye']

print(shap.__version__)

# %%

dataset, adsorbent_enc, dye_enc = get_dataset(encoding="ohe")

X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()

# %%

feature_names = dataset.input_features[0:10] + ['Adsorbent'] + ['Dye']
# %%

model, _ = get_fitted_model(model_type='functional')

# %%

test_p = model.predict(x=X_test)

# %%

evaluate_model(y_test, test_p)

# %%

train_p = model.predict(x=X_train)
print(f"Average value on prediction on training data {train_p.mean()}")

# %%

evaluate_model(y_train, train_p)

# %%
adsorbent_ohe = X_test[:, 10:58]
adsorbent_original = adsorbent_enc.inverse_transform(adsorbent_ohe)
print(adsorbent_ohe.shape, adsorbent_original.shape)

# %%

dye_ohe = X_test[:, 58:]
dye_original = dye_enc.inverse_transform(dye_ohe)

print(dye_ohe.shape, dye_original.shape)

# %%

x_test_original = np.column_stack((X_test[:, 0:10], adsorbent_original, dye_original))

# %%
# Deep Explainer
# ==============
exp = DeepExplainer(model._model, data=X_train)

sv = exp.shap_values(X_test)[0]

dye_sv = sv[:, 58:].sum(axis=1)

adsorbent_sv = sv[:, 10:58].sum(axis=1)

shap_values = np.column_stack((sv[:, 0:10], dye_sv, adsorbent_sv))

print(shap_values.shape)

# %%

sv_df = pd.DataFrame(shap_values, columns=feature_names)
fig, axes = create_subplots(shap_values.shape[1])
for ax, col in zip(axes.flat, sv_df.columns):
    box_violin(ax=ax, data=sv_df[col], palette="Set2")
    ax.set_xlabel(col)
plt.tight_layout()
plt.show()


# %%

imshow(shap_values, aspect="auto", colorbar=True,
       xticklabels=feature_names, show=False)
plt.tight_layout()
plt.show()

# %%

shap_values_exp = Explanation(
    shap_values,
    data=x_test_original,
    feature_names=feature_names
)


# %%
# Surface Area
# ---------------
df = pd.DataFrame(x_test_original, columns=feature_names)


shap_values_dye_dec = Explanation(
    shap_values,
    data=df.values,
    feature_names=feature_names
)

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'])

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = df['Adsorption Time (min)'], cmap = 'RdBu')
# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')

# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = df['Adsorption Temperature'], cmap = 'RdBu')
# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface Area'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%
# Initial Concentration
# -----------------------

# It is the initial concentration of pollutant (aka adsorbate)

shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = df['Pyrolysis Time (min)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')

# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False
              )
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Pyrolysis Temperature
# --------------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = df['Solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = df['Adsorbent Loading'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')
# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False
              )
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Pyrolysis Time (min)
# --------------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'])
# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = df['Initial Concentration'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = df['Solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = df['Pyrolysis Temperature'], cmap = 'RdBu')
# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False
              )
plt.tight_layout()
plt.show()


# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%
# Adsorption Time (min)
# ---------------------

df_sv = pd.DataFrame(shap_values, columns=feature_names)
df_sv['adsorption_time_inp'] = x_test_original[:, 0]
df_sv = df_sv.loc[df_sv['adsorption_time_inp']<1399.0]
shap_values_ads = df_sv.iloc[:, 0:-1]

df_ads_t = pd.DataFrame(x_test_original, columns=feature_names)
df_ads_t = df_ads_t.loc[df_ads_t['Adsorption Time (min)']< 1399.0]


shap_values_exp_ads = Explanation(
    shap_values_ads.values.copy(),
    data=df_ads_t.values,
    feature_names=feature_names
)

shap_scatter(shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'])


# %%

shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
    feature_wrt = df_ads_t["Pore Volume"], cmap = 'RdBu')

# %%

shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
    feature_wrt = df_ads_t['Pyrolysis Time (min)'], cmap = 'RdBu'
)

# %%
feature_wrt = df_ads_t['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
    feature_wrt = feature_wrt.map(d),
    is_categorical=True,
    show=False
)
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df_ads_t['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Adsorbent Loading
# -------------------
# It represents the amount of material (adsorbent) in the system

shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = df['Solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = df['Pyrolysis Temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')

# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False
              )
plt.tight_layout()
plt.show()


# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Adsorbent Loading'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%
# Pore Volume
# -----------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'])

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = df['Solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = df['Pyrolysis Temperature'], cmap = 'RdBu')
# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False,
              )
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore Volume'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%
# Solution pH
# ------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = df['Surface Area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = df['Pore Volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = df['Pyrolysis Temperature'], cmap = 'RdBu')
# %%
feature_wrt = df['Dye']
d = {k:DYE_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False,
              )
plt.tight_layout()
plt.show()

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = feature_wrt,
              is_categorical=True,
             palette_name="tab20",
             show=False)
plt.tight_layout()
plt.show()

# %%

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Bar chart of shap values for three classes i.e., ‘Physical properties’,
# ‘Synthesis Conditions’ and ‘Adsorption Experimental Conditions

sv_bar = np.mean(np.abs(shap_values_exp.values), axis=0)

colors = {'Adsorption Experimental Conditions': '#60AB7B',
          'Physical Properties': '#F9B234',
          'Synthesis Conditions': '#E91B23'}

classes = ['Adsorption Experimental Conditions', 'Synthesis Conditions',
           'Synthesis Conditions', 'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions', 'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions', 'Adsorption Experimental Conditions',
           'Physical Properties', 'Physical Properties', 'Synthesis Conditions',
           'Adsorption Experimental Conditions']

df = pd.DataFrame({'features': shap_values_exp.feature_names, 'classes': classes})

ax = bar_chart(sv_bar, shap_values_exp.feature_names,
          bar_labels=sv_bar, bar_label_kws={'label_type':'edge',
                                            'fontsize': 10,
                                            'weight': 'bold'},
          show=False, sort=True, color=['#60AB7B', '#E91B23', '#E91B23',
                                        '#60AB7B', '#60AB7B', '#60AB7B',
                                        '#60AB7B', '#60AB7B', '#F9B234',
                                        '#F9B234', '#E91B23', '#60AB7B'])
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(xlabel='mean(|SHAP value|)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')

labels = df['classes'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, loc='lower right')

plt.tight_layout()
plt.show()

# %%

sv_norm = sv_bar / np.sum(sv_bar)

synthesis = np.sum(sv_norm[[2,1,10]])
physical = np.sum(sv_norm[[8,9]])
experimental = np.sum(sv_norm[[3,0,5,11,7,6,4]])

ax = pie(fractions=[experimental, physical, synthesis], labels=['','',''],
            colors=colors.values(),
         textprops={"fontsize":18, "weight":"bold"}, show=False,)
plt.tight_layout()
plt.show()

# %%

heatmap(shap_values_exp, show=False)
plt.tight_layout()
plt.show()

# %%

heatmap(shap_values_exp, feature_values=shap_values_exp.abs.max(0), show=False)
plt.tight_layout()
plt.show()

# %%

heatmap(shap_values_exp, instance_order=shap_values_exp.sum(1), show=False)
plt.tight_layout()
plt.show()

# %%

beeswarm(shap_values_exp, show=False)
plt.tight_layout()
plt.show()

# %%

violin(shap_values, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# %%
# This figure shows the learning of our model for one sample
# having maximum test prediction. ``E[f(X)]`` is average prediction
# for the training data. ``f(X)`` is the actual prediction for this sample.
# The figure tells us the negative or positive role of the input
# features in predicting the target equal to ``f(X)`` starting from
# ``E[f(X)]``.

# %%

# Default SHAP colors
default_pos_color = "#ff0051"
default_neg_color = "#008bfb"
# Custom colors
positive_color = "#ca0020"
negative_color = "#92c5de"

index = test_p.argmax()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False, max_display=20)
# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if isinstance(fcc, matplotlib.patches.FancyArrow):
            if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                fcc.set_facecolor(positive_color)
            elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                fcc.set_color(negative_color)
        elif isinstance(fcc, plt.Text):
            if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                fcc.set_color(positive_color)
            elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                fcc.set_color(negative_color)
axes = plt.gca()
axes.figure.set_size_inches((8,6))
plt.tight_layout()
plt.show()

# %%
# This is same figure as above but for a different sample i.e.,
# the minimum value of prediction.

# %%

index = test_p.argmin()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False, max_display=20)
# Change the colormap of the artists
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if isinstance(fcc, matplotlib.patches.FancyArrow):
            if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                fcc.set_facecolor(positive_color)
            elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                fcc.set_color(negative_color)
        elif isinstance(fcc, plt.Text):
            if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                fcc.set_color(positive_color)
            elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                fcc.set_color(negative_color)
axes = plt.gca()
axes.figure.set_size_inches((8,6))
plt.tight_layout()
plt.show()

# %%

tsne = TSNE(n_components=2)
sv_2D = tsne.fit_transform(shap_values)

s = plt.scatter(sv_2D[:, 0], sv_2D[:, 1], c=y_test.reshape(-1,), cmap="Spectral",
            s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(s)
plt.title('TSNE projection of shap values', fontsize=18)
plt.show()

# %%

sv_umap = UMAP(n_components=2).fit_transform(shap_values)
s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=y_test.reshape(-1,),
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Predicted Adsorption Capacity', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,0],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Adsorption Time (min)', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,1],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Pyrolysis Temperature', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,2],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Pyrolysis Time (min)', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,3],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Initial Concentration', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,4],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('Solution pH', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%
# Kernel Explainer
# =================

if platform.system()=='Windows':

    X_train_summary = shap.kmeans(X_train, 10)

    exp = KernelExplainer(model.predict, X_train_summary)

    sv = exp.shap_values(X_test)[0]

    dye_sv = sv[:, 58:].sum(axis=1)

    adsorbent_sv = sv[:, 10:58].sum(axis=1)

    shap_values = np.column_stack((sv[:, 0:10], dye_sv, adsorbent_sv))

    print(shap_values.shape)

    # %%

    shap_values_exp = Explanation(
        shap_values,
        data=x_test_original,
        feature_names=feature_names
    )

    # %%

    beeswarm(shap_values_exp, show=False)
    plt.tight_layout()
    plt.show()

# %%
# Gradient Explainer
# ===================

exp = GradientExplainer(model._model, data=[X_train])

sv = exp.shap_values(X_test)[0]

dye_sv = sv[:, 58:].sum(axis=1)

adsorbent_sv = sv[:, 10:58].sum(axis=1)

shap_values = np.column_stack((sv[:, 0:10], dye_sv, adsorbent_sv))

print(shap_values.shape)

# %%

shap_values_exp = Explanation(
    shap_values,
    data=x_test_original,
    feature_names=feature_names
)

# %%

beeswarm(shap_values_exp, show=False)
plt.tight_layout()
plt.show()


# %%
# partial dependence plots
# ============================
# Now we will calculate partial dependence plot (PDP) and Individual Component
# Elements (ICE) curves using ``PartialDependencePlot`` class of ai4water.


pdp = PartialDependencePlot(
    model.predict,
    X_train,
    num_points=20,
    feature_names=model.input_features,
    show=False,
    save=False
)

# %%
# We calculate pdp plots only once and then plot them with
# different options
feature = [f for f in model.input_features if f.startswith("Dye")]
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)

# %%

ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel("Dye")
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel("Dye")
plt.show()

# %%
# Calculate pdp and ice for Adsorbent

feature = [f for f in model.input_features if f.startswith("Adsorbent")]
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)

# %%

ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(adsorbent_enc.categories_[0])
ax.set_xlabel("Adsorbent")
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        ice=False, pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(adsorbent_enc.categories_[0])
ax.set_xlabel("Adsorbent")
plt.show()

# %%
# pdp and ice for Surface Area
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, 'Surface area')

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, 'Surface area',
                   pdp_line_kws={'color': 'darkcyan'})
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        'Surface area', ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()


# %%
# pdp and ice for Initial Concentration
feature = 'initial concentration'
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)


# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                   pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
# pdp and ice for Pyrolysis Temperature
feature = 'calcination_temperature'
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)


# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                   pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
# Accumulated Local Effects
# ===========================

from alepython import ale_plot

class Model:
    def predict(self, X):
        return model.predict(X).reshape(-1,)

ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=["Surface area"]
             )

# %%
ae_eff = ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=['calcination_temperature']
             )

# %%
ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=['initial concentration']
             )

# %%

ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=["Surface area", 'Pore volume']
             )

# %%
# Permutation importance
# =======================
# Permutation importance quantifies reduction in model performance when
# we corrupt one feature column intentionally. The corruption in one feature
# column is carried out by randomly permuting its values ``n`` number of
# times. Then the average reduction in model performance is recorded
# as permutation feature importance for the feature.

cat_map = {'Catalyst': list(range(10, 58)), 'Anions': list(range(58, 74))}

pimp = PermutationImportance(
    model.predict, X_train, y_train,
    show=False,
    save=False,
    cat_map=cat_map,
    feature_names = model.input_features,
    n_repeats=20)

pimp.plot_1d_pimp()
plt.tight_layout()
plt.show()

# %%
pimp.plot_1d_pimp("barchart")
plt.tight_layout()
plt.show()

# %%
# Integrated Gradients
# =======================

ig  = IntegratedGradients(model._model,
                          layer=None,
                          method="gausslegendre",
                          n_steps=50,
                          internal_batch_size=100)

# %%
# Training data
explanation = ig.explain(X_train,
                         baselines=None,
                         target=None)

attributions = explanation.attributions[0]

con = attributions[:, 0:10]
ads = attributions[:, 10:58]
dye = attributions[:, 58:]
attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])

imshow(attr, aspect="auto", colorbar=True,
       xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
       show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
plt.tight_layout()
plt.show()

# %%
# Test data
explanation = ig.explain(X_test,
                         baselines=None,
                         target=None)

attributions = explanation.attributions[0]

con = attributions[:, 0:10]
ads = attributions[:, 10:58]
dye = attributions[:, 58:]
attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])

imshow(attr, aspect="auto", colorbar=True,
       xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
       show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
plt.tight_layout()
plt.show()

# %%
# Partial Dependence Variance
# ============================
# We give the data to ``PartialDependenceVariance`` class label-encoded
# data. When is then one-hot-encoded inside the ``predictor_fn``. Then
# the data is given to model to make a prediction

# first get data without any encoding
data, _, _ = make_data()
ads_ohe_encoder = OneHotEncoder(sparse=False)
ads_ohe_encoder.fit(data.loc[:, 'Adsorbent'].values.reshape(-1,1))

dye_ohe_encoder = OneHotEncoder(sparse=False)
dye_ohe_encoder.fit(data.loc[:, 'Dye'].values.reshape(-1,1))

# now get the label-encoded data. This will be passed to ``PartialDependenceVariance``
# class.
data_le, ads_le_encoder, dye_le_encoder = make_data(encoding="le")

def predictor_fn(X):

    # The X is given/suggested by ``PartialDependenceVariance`` which
    # means it is label-encoded. First inverse transform
    # to get the string columns
    ads_encoded = X[:, -2]
    ads_decoded = ads_le_encoder.inverse_transform(ads_encoded.astype(np.int16))
    ads_ohe_encoded = ads_ohe_encoder.transform(ads_decoded.reshape(-1,1))
    ads_cols = [f'Adsorbent_{i}' for i in range(ads_ohe_encoded.shape[1])]

    dye_encoded = X[:, -1]
    dye_decoded = dye_le_encoder.inverse_transform(dye_encoded.astype(np.int16))
    dye_ohe_encoded = dye_ohe_encoder.transform(dye_decoded.reshape(-1,1))
    dye_cols = [f'Dye_{i}' for i in range(dye_ohe_encoded.shape[1])]

    X = pd.DataFrame(X, columns=data.columns.tolist()[0:-1])
    X.pop('Adsorbent')
    X.pop('Dye')

    X[ads_cols] = ads_ohe_encoded
    X[dye_cols] = dye_ohe_encoded

    return model.predict(X.values).reshape(-1,)


category_map = gen_category_map(data)

pd_variance = PartialDependenceVariance(predictor=predictor_fn,
                                        feature_names=data.columns.tolist()[0:-1],
                                        categorical_names=category_map,
                                        target_names=["Adsorption"])

exp_importance = pd_variance.explain(X=data_le.values[:, 0:-1], method='importance')

# %%

bar_chart(
    exp_importance.feature_importance.reshape(-1,),
    sort=True,
    labels=data.columns.tolist()[0:-1],
    ax_kws=dict(tight_layout=True)
)

# %%
f, ax = plt.subplots(3,4, figsize=(10,10))
plot_pd_variance(exp=exp_importance, summarise=False, ax=ax)
plt.subplots_adjust(hspace=0.1)
plt.show()