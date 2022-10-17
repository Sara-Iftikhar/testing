"""
====================
MLP
====================
"""

import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import get_version_info
from ai4water.postprocessing import LossCurve, ProcessPredictions
from ai4water.utils.utils import dateandtime_now
from ai4water.utils import edf_plot
from easy_mpl import plot, regplot
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
    model=MLP(units=37, num_layers=4,
              activation='relu'),
    lr=0.004561316449575947,
    input_features=ds.input_features,
    output_features=ds.output_features,
    epochs=400, batch_size=24,
    verbosity=0
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
