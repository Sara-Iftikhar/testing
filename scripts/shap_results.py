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
from easy_mpl import imshow
from umap import UMAP
from easy_mpl.utils import create_subplots

from utils import get_dataset, get_fitted_model, evaluate_model, \
    box_violin, shap_interaction_all, shap_scatter

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
dye_enc = LabelEncoder()
df['Dye'] = dye_enc.fit_transform(df['Dye'])

shap_values_dye_dec = Explanation(
    shap_values,
    data=df.values,
    feature_names=feature_names
)

shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'],
              feature_wrt = df['Adsorption_time (min)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'],
              feature_wrt = df['Pore volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'],
              feature_wrt = df['adsorption_temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Surface area'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False
              )
plt.tight_layout()
plt.show()


# %%
# Initial Concentration
# -----------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'initial concentration'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'initial concentration'],
              feature_wrt = df['calcination (min)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'initial concentration'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'initial concentration'],
              feature_wrt = df['Pore volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'initial concentration'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False
              )
plt.tight_layout()
plt.show()


# %%
# Calcination Temperature
# --------------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'],
              feature_wrt = df['solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'],
              feature_wrt = df['adsorbent loading'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination_temperature'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False
              )
plt.tight_layout()
plt.show()


# %%
# calcination (min)
# --------------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'])

shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['Pore volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['initial concentration'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['calcination_temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'calcination (min)'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False
              )
plt.tight_layout()
plt.show()

# %%
# Adsorption_time (min)
# ---------------------

df_sv = pd.DataFrame(shap_values, columns=feature_names)
df_sv['adsorption_time_inp'] = x_test_original[:, 0]
df_sv = df_sv.loc[df_sv['adsorption_time_inp']<1399.0]
shap_values_ads = df_sv.iloc[:, 0:-1]

df_ads_t = pd.DataFrame(x_test_original, columns=feature_names)
df_ads_t = df_ads_t.loc[df_ads_t['Adsorption_time (min)']< 1399.0]

enc = LabelEncoder()
df_ads_t['Adsorbent'] = enc.fit_transform(df_ads_t['Adsorbent'])

dye_enc_ads_t = LabelEncoder()
df_ads_t['Dye'] = dye_enc_ads_t.fit_transform(df_ads_t['Dye'])


shap_values_exp_ads = Explanation(
    shap_values_ads.values.copy(),
    data=df_ads_t.values,
    feature_names=feature_names
)

shap_scatter(shap_values=shap_values_exp_ads[:, 'Adsorption_time (min)'])


# %%

shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption_time (min)'],
    feature_wrt = df_ads_t["Pore volume"], cmap = 'RdBu'
)
# %%

shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption_time (min)'],
    feature_wrt = df_ads_t["calcination (min)"], cmap = 'RdBu'
)

# %%
shap_scatter(
    shap_values=shap_values_exp_ads[:, 'Adsorption_time (min)'],
    feature_wrt = df_ads_t['Dye'],
    feature_wrt_encoder = dye_enc_ads_t,
    is_categorical=True,
    show=False
)
plt.tight_layout()
plt.show()


# %%
# Adsorbent Loading
# -------------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['calcination_temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['Pore volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'adsorbent loading'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False
              )
plt.tight_layout()
plt.show()

# %%
# Pore Volume
# ----------

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'])

# %%

shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'],
              feature_wrt = df['solution pH'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'],
              feature_wrt = df['calcination_temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'Pore volume'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False,
              )
plt.tight_layout()
plt.show()

# %%
# Solution pH
# ------------

shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'])
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'],
              feature_wrt = df['Volume (L)'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'],
              feature_wrt = df['Surface area'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'],
              feature_wrt = df['Pore volume'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'],
              feature_wrt = df['calcination_temperature'], cmap = 'RdBu')
# %%
shap_scatter(shap_values=shap_values_dye_dec[:, 'solution pH'],
              feature_wrt = df['Dye'],
              is_categorical=True,
              feature_wrt_encoder=dye_enc,
             show=False,
              )
plt.tight_layout()
plt.show()

# %%

shap.plots.bar(shap_values_exp, show=False)
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
cbar.ax.set_ylabel('Adsorption_time (min)', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,1],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('calcination_temperature', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,2],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('calcination (min)', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,3],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('initial concentration', rotation=270)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()

# %%

s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=X_test[:,4],
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
cbar = plt.colorbar(s)
cbar.ax.set_ylabel('solution pH', rotation=270)
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





