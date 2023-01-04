"""
====================
MLP
====================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns
from ai4water.functional import Model
from ai4water.models import MLP
from ai4water.utils.utils import get_version_info
from ai4water.postprocessing import LossCurve, ProcessPredictions
from ai4water.utils.utils import dateandtime_now
from ai4water.utils import edf_plot
from easy_mpl import plot, regplot, ridge, circular_bar_plot
from SeqMetrics import RegressionMetrics

from utils import get_data, evaluate_model, get_dataset, data_before_encoding

get_version_info()

# %%

X_train, y_train, X_test, y_test = get_data()
ds ,  _, _ = get_dataset()
original_data = data_before_encoding()

# %%
# There are total 12 input features used in this study, which are listed below.
# Two of them are categorical features i.e. ``Adsorbent`` and ``Dye``. Categorical
# features have encoded using One-Hot encoder.

# %%

print(original_data.columns[:-1])

# %%
# While there is one target, which is listed below

# %%

print(original_data.columns[-1])

# %%

path = os.path.join(os.getcwd(),'results',f'mlp_{dateandtime_now()}')
os.makedirs(path)

model = Model(
    model=MLP(units=99, num_layers=4,
              activation='relu'),
    lr=0.006440897421063212,
    input_features=ds.input_features,
    output_features=ds.output_features,
    epochs=600, batch_size=48,
    verbosity=0,
    prefix=path,
)

# %%

h = model.fit(X_train,y_train,
          validation_data=(X_test, y_test))

# %%
# Training data
# --------------

train_p = model.predict(x=X_train,)

# %%

evaluate_model(y_train, train_p)

# %%

pp = ProcessPredictions(mode='regression', forecast_len=1,
                   path=path)

# %%

pp.murphy_plot(y_train,train_p, prefix="train", where=path, inputs=X_train)

# %%

metrics = RegressionMetrics(y_train, train_p)
errors = metrics.calculate_all()

for err in ['kl_sym']:
    errors.pop(err)

n_errors = {}
for k,v in errors.items():
    if 0.<v<5.0:
        n_errors[k] = v

ax = circular_bar_plot(n_errors, sort=True, show=False, figsize=(8,9))
plt.tight_layout()
plt.show()


# %%

regplot(pd.DataFrame(y_train), pd.DataFrame(train_p), 'Training',
        annotation_key='$R^2$',
        annotation_val=RegressionMetrics(y_train,train_p).r2(),
        marker_size=60,
        marker_color='snow',
        line_style='--',
        line_color='indigo',
        line_kws=dict(linewidth=3.0),
        scatter_kws=dict(linewidths=1.1, edgecolors=np.array([56, 86, 199])/255,
                         marker="8",
                         alpha=0.7
                         )
        )

# %%
# Test data
# ----------

test_p = model.predict(x=X_test,)

# %%

evaluate_model(y_test, test_p)

# %%

pp = ProcessPredictions(mode='regression', forecast_len=1, path=path)

# %%

pp.murphy_plot(y_test, test_p, prefix="test", where=path, inputs=X_test)

# %%
metrics = RegressionMetrics(y_test, test_p)
errors = metrics.calculate_all()

for err in ['kl_sym']:
    errors.pop(err)

n_errors = {}
for k,v in errors.items():
    if 0.<v<5.0:
        n_errors[k] = v

ax = circular_bar_plot(n_errors, sort=True, show=False, figsize=(8,9))
plt.tight_layout()
plt.show()

# %%

regplot(pd.DataFrame(y_test), pd.DataFrame(test_p), 'Test',
        annotation_key='$R^2$',
        annotation_val=RegressionMetrics(y_test,test_p).r2(),
        marker_size=60,
        marker_color='snow',
        line_style='--',
        line_color='indigo',
        line_kws=dict(linewidth=3.0),
        scatter_kws=dict(linewidths=1.1, edgecolors=np.array([56, 86, 199])/255,
                         marker="8",
                         alpha=0.7
                         )
        )

# %%
# combined
# --------

# %%

legend_properties = {'weight':'bold',
                     'size': 14}

ax = plot(h.history['loss'], show=False, label='Training'
                    , xlabel='Epochs', ylabel='Loss'
                    )
ax = plot(h.history['val_loss'], ax=ax, label='Test',
                xlabel='Epochs', ylabel='Loss', show=False)

ax.set_ylabel(ylabel= 'Loss', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Epochs', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.tight_layout()
plt.savefig(f'paper\\figures\\fig3a.png', dpi=400)
plt.show()

# %%
# scatter plot of prediction and errors with KDE

train_er = pd.DataFrame((y_train - train_p), columns=['Error'])
train_er['prediction'] = train_p
train_er['hue'] = 'Training'
test_er = pd.DataFrame((y_test - test_p), columns=['Error'])
test_er['prediction'] = test_p
test_er['hue'] = 'Test'

df_er = pd.concat([train_er, test_er], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}

g = sns.jointplot(data=df_er, x="prediction",
                     y="Error",
              hue='hue', palette='husl')
ax = g.ax_joint
ax.axhline(0.0)
ax.set_ylabel(ylabel= 'Residuals', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Prediction', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.tight_layout()
plt.savefig(f'paper\\figures\\fig3d.png', dpi=400)
plt.show()

# %%

legend_properties = {'weight':'bold',
                     'size': 14}
_, ax = plt.subplots(#figsize=(5,4)
                     )

edf_plot(np.abs(y_train-train_p), label='Training',
        c=np.array([200, 49, 40])/255,
         #c=np.array([234, 106, 41])/255,
         linewidth=2.5,
         show=False, ax=ax,)
edf_plot(np.abs(y_test-test_p), xlabel='Absolute error',
         c=np.array([68, 178, 205])/255, linewidth=2.5,
         label='Test', ax=ax, show=False,
         grid=True)
ax.set_ylabel(ylabel= 'Commulative Probabilty', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Absolute Error', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().round(2), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.title("Empirical Distribution Function Plot",fontweight="bold")
plt.tight_layout()
plt.savefig(f'paper\\figures\\fig3c.png', dpi=400)
plt.show()

# %%

legend_properties = {'weight':'bold',
                     'size': 14}

ax = regplot(pd.DataFrame(y_train), pd.DataFrame(train_p),
        marker_size=60,
        ci=False,
        marker_color='indigo',
        line_style='--',
        line_color='indigo',
        line_kws=dict(linewidth=3.0),
        scatter_kws=dict(linewidths=0, edgecolors='snow',
                         marker="8",
                         alpha=0.5,
                         label='Training'
                         ),
             show=False
        )

regplot(pd.DataFrame(y_test), pd.DataFrame(test_p),
        marker_size=60,
        ci=False,
        marker_color='crimson',
        line_kws=dict(linewidth=0),
        scatter_kws=dict(linewidths=0, edgecolors='crimson',
                         marker="s",
                         alpha=0.5,
                         label='Test'
                         ),
        show=False,
        ax=ax
        )
ax.set_ylabel(ylabel= 'Predicted Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Experimental Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.tight_layout()
plt.show()

# %%

legend_properties = {'weight':'bold',
                     'size': 14,}
fig, axes = plt.subplots(#figsize=(9,7)
                         )
ax = ridge([train_p.reshape(-1,), test_p.reshape(-1,)],
           color=['snow', 'snow'],
           line_color=['indigo', 'crimson'],
           line_width=3.0,
           share_axes=True,
           fill_kws={'alpha':0.05},
           show=False,
           ax=axes,
           cut=0.15
           )
ax[0].set_ylabel('Prediction Distribution', fontsize=14, weight='bold')
#ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_xlabel('Experimental Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
#ax[0].tick_params(axis='x', labelsize=15)
ax[0].set_xticklabels(ax[0].get_xticks().astype(int), size=12, weight='bold')
ax[0].set_yticklabels(ax[0].get_yticks(), size=12, weight='bold')
ax[0].set_ylim(-0, 0.004)
ax2 = ax[0].twinx()


ax2 = regplot(pd.DataFrame(y_train), pd.DataFrame(train_p),
        marker_size=60,
        ci=False,
        marker_color='indigo',
        line_style='-.',
        line_color='black',
        line_kws=dict(linewidth=3.0),
        scatter_kws=dict(linewidths=0, edgecolors='snow',
                         marker="8",
                         alpha=0.5,
                         label='Training'
                         ),
        show=False,
        ax=ax2,
        )

ax2 = regplot(pd.DataFrame(y_test), pd.DataFrame(test_p),
        marker_size=60,
        ci=False,
        marker_color='crimson',
        line_kws=dict(linewidth=0),
        scatter_kws=dict(linewidths=0, edgecolors='crimson',
                         marker="s",
                         alpha=0.5,
                         label='Test'
                         ),
        show=False,
        ax=ax2
        )
ax2.set_ylabel('Predicted Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax2.set_yticklabels(ax2.get_yticks().astype(int), size=12, weight='bold')
ax2.legend(prop=legend_properties, loc = 'upper center')
plt.tight_layout()
plt.savefig(f'paper\\figures\\fig.png', dpi=400)
plt.show()



# %%
# scatter plot of true and predicted with train and test KDE

train_df = pd.DataFrame(np.column_stack([y_train, train_p]),
                        columns=['true', 'predicted'])

train_df['hue'] = 'Training'

test_df = pd.DataFrame(np.column_stack([y_test, test_p]),
                        columns=['true', 'predicted'])

test_df['hue'] = 'Test'

df = pd.concat([train_df, test_df], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}

g = sns.jointplot(data=df, x="true",
                     y="predicted",
              hue='hue', palette='husl')

ax = g.ax_joint

ax.set_ylabel(ylabel= 'Predicted Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Experimental Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.tight_layout()
plt.savefig(f'paper\\figures\\fig3b.png', dpi=400)
plt.show()