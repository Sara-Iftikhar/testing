"""
====================
shap
====================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import shap
import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from shap import DeepExplainer, GradientExplainer, KernelExplainer
from shap import Explanation
from shap.plots import beeswarm, violin, heatmap, waterfall
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from easy_mpl import imshow, bar_chart, pie
from umap import UMAP
from easy_mpl.utils import create_subplots, make_cols_from_cmap

from utils import get_dataset, get_fitted_model, evaluate_model, \
    box_violin, shap_interaction_all, shap_scatter, DYE_TYPES, ADSORBENT_TYPES

# %%

CAT_FEATURES = ['Adsorbent', 'Dye']

print(shap.__version__)

# %%

dataset, adsorbent_enc, dye_enc = get_dataset()


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

feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Initial Concentration'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Calcination Temperature
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
feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Temperature'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()


# %%
# Calcination (min)
# --------------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pyrolysis Time (min)'])

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

enc = LabelEncoder()
#df_ads_t['Adsorbent'] = enc.fit_transform(df_ads_t['Adsorbent'])

dye_enc_ads_t = LabelEncoder()
#df_ads_t['Dye'] = dye_enc_ads_t.fit_transform(df_ads_t['Dye'])


shap_values_exp_ads = Explanation(
    shap_values_ads.values.copy(),
    data=df_ads_t.values,
    feature_names=feature_names
)

shap_scatter(shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'])


# %%

shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption Time (min)'],
    feature_wrt = df_ads_t["Pore Volume"], cmap = 'RdBu'
)
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
feature_wrt = df['Adsorbent']
d = {k:ADSORBENT_TYPES[k] for k in feature_wrt.unique()}
shap_scatter(shap_values=shap_values_dye_dec[:, 'Solution pH'],
              feature_wrt = feature_wrt.map(d),
              is_categorical=True,
             show=False)
plt.tight_layout()
plt.show()

# %%

sv_bar = np.mean(np.abs(shap_values_exp.values), axis=0)

ax = bar_chart(sv_bar, shap_values_exp.feature_names,
          bar_labels=sv_bar, bar_label_kws={'label_type':'edge'},
          show=False, sort=True, cmap='summer_r')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(xlabel='mean(|SHAP value|)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')
plt.tight_layout()
plt.show()

# %%

sv_norm = sv_bar / np.sum(sv_bar)

synthesis = np.sum(sv_norm[[2,1,10]])
physical = np.sum(sv_norm[[8,9]])
experimental = np.sum(sv_norm[[3,0,5,11,7,6,4]])

labels = ["Synthesis \n Conditions", "Physical \n Properties", "Adsorption \n Experimental \n Conditions"]
colors = make_cols_from_cmap("PuBu", 3, low=0.3, high=0.8)
ax = pie([synthesis, physical, experimental], labels=labels, colors=colors,
         textprops={"fontsize":14, "weight":"bold"}, show=False)
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

index = test_p.argmax()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False)
plt.tight_layout()
plt.show()

# %%

index = test_p.argmin()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False)
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





