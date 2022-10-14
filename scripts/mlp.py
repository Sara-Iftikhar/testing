"""
====================
MLP
====================
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import get_version_info
from ai4water.postprocessing import LossCurve, ProcessPredictions
from ai4water.utils.utils import dateandtime_now

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
)

# %%

model.fit(X_train,y_train,
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

pp.edf_plot(y_train, train_p, 'train', path)

# %%

pp.murphy_plot(y_train,train_p, prefix="train", where=path, inputs=X_train)

# %%
pp.is_multiclass_ = False
pp.errors_plot(y_train, train_p, 'train', path)

# %%
pp.residual_plot(pd.DataFrame(y_train), pd.DataFrame(train_p), 'train', path)

# %%

data_ = [pd.DataFrame(y_train, columns=['observed']), pd.DataFrame(train_p, columns=['predicted'])]
data = pd.concat(data_, axis=1)

sns.set_style("white")
gridobj = sns.lmplot(x="observed", y="predicted",
                     data=data,
                     height=7, aspect=1.3, robust=True,
                     scatter_kws=dict(s=150, linewidths=1.0, edgecolors='black',
                                      marker="3",
                                      alpha=0.5))
gridobj.ax.set_xticklabels(gridobj.ax.get_xticklabels(), fontsize=20)
gridobj.ax.set_yticklabels(gridobj.ax.get_yticklabels(), fontsize=20)
gridobj.ax.set_xlabel(gridobj.ax.get_xlabel(), fontsize=24)
gridobj.ax.set_ylabel(gridobj.ax.get_ylabel(), fontsize=24)
plt.show()

# %%
# Test data
# --------------
test_p = model.predict(x=X_test,)

# %%

evaluate_model(y_test, test_p)

# %%

pp = ProcessPredictions(mode='regression', forecast_len=1, path=path)

pp.edf_plot(y_test, test_p, 'test', path)

# %%

pp.murphy_plot(y_test, test_p, prefix="test", where=path, inputs=X_test)

# %%
pp.is_multiclass_ = False
pp.errors_plot(y_test, test_p, 'test', path)

# %%
pp.residual_plot(pd.DataFrame(y_test), pd.DataFrame(test_p), 'test', path)

# %%

data_ = [pd.DataFrame(y_test, columns=['observed']), pd.DataFrame(test_p, columns=['predicted'])]
data = pd.concat(data_, axis=1)

sns.set_style("white")
gridobj = sns.lmplot(x="observed", y="predicted",
                     data=data,
                     height=7, aspect=1.3, robust=True,
                     scatter_kws=dict(s=150, linewidths=1.0, edgecolors='black',
                                      marker="3",
                                      alpha=0.5))
gridobj.ax.set_xticklabels(gridobj.ax.get_xticklabels(), fontsize=20)
gridobj.ax.set_yticklabels(gridobj.ax.get_yticklabels(), fontsize=20)
gridobj.ax.set_xlabel(gridobj.ax.get_xlabel(), fontsize=24)
gridobj.ax.set_ylabel(gridobj.ax.get_ylabel(), fontsize=24)
plt.tight_layout()
plt.show()
