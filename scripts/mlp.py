"""
====================
MLP
====================
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import get_version_info
from ai4water.postprocessing import LossCurve, ProcessPredictions
from ai4water.utils.utils import dateandtime_now
from ai4water.utils import edf_plot
from easy_mpl import plot, regplot, ridge
from SeqMetrics import RegressionMetrics

from utils import get_data, evaluate_model, get_dataset

get_version_info()

# %%

X_train, y_train, X_test, y_test = get_data()
ds ,  _, _ = get_dataset()

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

ax = plot(h.history['loss'], show=False, label='training'
                    , xlabel='Epochs', ylabel='Loss'
                    )
plot(h.history['val_loss'], ax=ax, label='validation',
                xlabel='Epochs', ylabel='Loss'
                )

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
pp.is_multiclass_ = False
pp.errors_plot(y_train, train_p, 'train', path)

# %%
pp.residual_plot(pd.DataFrame(y_train), pd.DataFrame(train_p), 'train', path)

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
pp.is_multiclass_ = False
pp.errors_plot(y_test, test_p, 'test', path)

# %%
pp.residual_plot(pd.DataFrame(y_test), pd.DataFrame(test_p), 'test', path)

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

_, ax = plt.subplots(figsize=(5,4))

edf_plot(np.abs(y_train-train_p), label='Training',
        c=np.array([200, 49, 40])/255,
         #c=np.array([234, 106, 41])/255,
         linewidth=2.5,
         show=False, ax=ax,)
edf_plot(np.abs(y_test-test_p), xlabel='Absolute error',
         c=np.array([68, 178, 205])/255, linewidth=2.5,
         label='Test', ax=ax, show=False)
plt.tight_layout()
plt.show()

# %%

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
ax.legend(fontsize=16, markerscale=1.5)
plt.show()


# %%

fig, axes = plt.subplots(figsize=(9,7))
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
ax[0].set_ylabel('Prediction distribution', fontsize=20)
ax[0].tick_params(axis='y', labelsize=15)
ax[0].set_xlabel('Observed', fontsize=20)
ax[0].tick_params(axis='x', labelsize=15)
ax[0].set_ylim(-0, 0.003)
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
ax2.legend(fontsize=20, markerscale=1.5, loc=9)
ax2.set_ylabel('Predicted', fontsize=20)
ax2.tick_params(axis='y', labelsize=15)
plt.tight_layout()
plt.show()

# %%

# predicted
sns.distplot(pd.DataFrame(train_p))
sns.distplot(pd.DataFrame(test_p))

plt.show()

# true
sns.distplot(pd.DataFrame(y_train), bins=1000)
sns.distplot(pd.DataFrame(y_test), bins=1000)

plt.show()

# %%

train_df = pd.DataFrame(np.column_stack([y_train, train_p]),
                        columns=['train_true', 'train_predicted'])

test_df = pd.DataFrame(np.column_stack([y_test, test_p]),
                        columns=['test_true', 'test_predicted'])

ax = sns.jointplot(data=train_df, x="train_true",
                   y="train_predicted", kind="reg")

sns.jointplot(data=test_df, x="test_true",
                   y="test_predicted", kind="reg",
              ax=ax)