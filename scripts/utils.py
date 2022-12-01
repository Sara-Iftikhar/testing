import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from ai4water.functional import Model as f_model
from ai4water import Model
from ai4water.models import MLP
from ai4water.postprocessing import prediction_distribution_plot

from ai4water.preprocessing import DataSet
from ai4water.utils.utils import dateandtime_now

from SeqMetrics import RegressionMetrics
from easy_mpl import violin_plot


# function for OHE
def _ohe_encoder(df:pd.DataFrame, col_name:str)->tuple:
    assert isinstance(col_name, str)

    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    return df, cols_added, encoder

def data_before_encoding():

    # read excel
    ads_df = pd.read_excel('Adsorption and regeneration data_1007c.xlsx', sheet_name=0)
    dye_df = pd.read_excel('Dyes data.xlsx', sheet_name=0)

    # dropping unnecessary columns
    ads_df = ads_df.drop(columns=['final concentation', 'Volume (mL)',
                                  'Unnamed: 16', 'Unnamed: 17',
                                  'Unnamed: 18', 'Unnamed: 19',
                                  'Unnamed: 20', 'Unnamed: 21',
                                  'Unnamed: 22', 'Unnamed: 23',
                                  'Particle size'
                                  ])

    dye_df = dye_df.drop(columns=['C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C',
                                  'N/C', 'Average pore size',
                                  'rpm', 'g/L', 'Ion Concentration (M)',
                                  'Humic acid', 'wastewater type',
                                  'Adsorption type', 'Cf', 'Ref'
                                  ])

    # merging data
    data = [ads_df, dye_df]

    whole_data = pd.concat(data)
    whole_data = whole_data.dropna()

    whole_data = whole_data.reset_index()
    whole_data.pop('index')

    return whole_data


def _make_data():

    whole_data = data_before_encoding()

    #applying OHE
    whole_data_enc_, _, adsorbent_enc = _ohe_encoder(whole_data, 'Adsorbent')
    whole_data_enc__ = whole_data_enc_.drop(columns='Adsorbent')

    whole_data_enc, _, dye_encoder = _ohe_encoder(whole_data_enc__, 'Dye')
    whole_data_enc = whole_data_enc.drop(columns='Dye')

    df1 = whole_data_enc.pop('qe')
    whole_data_enc['qe'] = df1

    return whole_data_enc, adsorbent_enc, dye_encoder


def get_dataset():
    whole_data_enc, adsorbent_encoder, dye_encoder = _make_data()

    dataset = DataSet(data=whole_data_enc,
                      seed=1575,
                      val_fraction=0.0,
                      split_random=True,
                      input_features=whole_data_enc.columns.tolist()[0:-1],
                      output_features=whole_data_enc.columns.tolist()[-1:],
                      )
    return dataset, adsorbent_encoder, dye_encoder


def get_data():

    dataset, _, _ = get_dataset()

    X_train, y_train = dataset.training_data()

    X_test, y_test = dataset.test_data()
    return X_train, y_train, X_test, y_test


def make_path():
    path = os.path.join(os.getcwd(), 'results', f'mlp_{dateandtime_now()}')
    os.makedirs(path)
    return path


def get_fitted_model(return_path=False,
                     model_type=None):

    X_train, y_train, X_test, y_test = get_data()
    ds, _, _ = get_dataset()

    path = make_path()
    if model_type=='functional':
        model = f_model(
            model=MLP(units=99, num_layers=4,
                      activation='relu'),
            lr=0.006440897421063212,
            input_features=ds.input_features,
            output_features=ds.output_features,
            epochs=400, batch_size=48,
            verbosity=0
        )
    else:
        model = Model(
            model=MLP(units=37, num_layers=4,
                      activation='relu'),
            lr=0.006440897421063212,
            input_features=ds.input_features,
            output_features=ds.output_features,
            epochs=400, batch_size=48,
            verbosity=0
        )

    h = model.fit(X_train, y_train)

    if return_path:
        return model, path, h
    return model, h


def confidenc_interval(model, X_train, y_train, X_test, y_test, alpha,
                    n_splits=5):

    def generate_results_dataset(preds, _ci):
        _df = pd.DataFrame()
        _df['prediction'] = preds
        if _ci >= 0:
            _df['upper'] = preds + _ci
            _df['lower'] = preds - _ci
        else:
            _df['upper'] = preds - _ci
            _df['lower'] = preds + _ci

        return _df

    path = make_path()
    model.fit(X_train, y_train, batch_size=48, verbose=0,
              validation_data=(X_test, y_test),
              epochs=400)

    residuals = y_train - model.predict(X_train)
    ci = np.quantile(residuals, 1 - alpha)
    preds = model.predict(X_test)
    df = generate_results_dataset(preds.reshape(-1, ), ci)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = []
    estimators = []
    for train_index, test_index in kf.split(X_train):
        X_train_, X_test_ = X_train[train_index], X_train[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]

        path = make_path()
        model.fit(X_train_, y_train_, validation_data=(X_test_, y_test_),
                   verbose=0, batch_size=48, epochs=400)

        estimators.append(model)
        _pred = model.predict(X_test_)
        res.extend(list(y_test_ - _pred.reshape(-1, )))

    y_pred_multi = np.column_stack([e.predict(X_test) for e in estimators])

    ci = np.quantile(res, 1 - alpha)
    top = []
    bottom = []
    for i in range(y_pred_multi.shape[0]):
        if ci > 0:
            top.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
        else:
            top.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))

    preds = np.median(y_pred_multi, axis=1)
    df = pd.DataFrame()
    df['pred'] = preds
    df['upper'] = top
    df['lower'] = bottom

    return df


def plot_ci(df, alpha):
    # plots the confidence interval

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(np.arange(len(df)), df['upper'], df['lower'], alpha=0.5, color='C1')
    p1 = ax.plot(df['pred'], color="C1", label="Prediction")
    p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
    percent = int((1 - alpha) * 100)
    ax.legend([(p2[0], p1[0]), ], [f'{percent}% Confidence Interval'],
              fontsize=12)
    ax.set_xlabel("Test Samples", fontsize=12)
    ax.set_ylabel("Adsorption Capacity", fontsize=12)

    return ax


def evaluate_model(true, predicted):
    metrics = RegressionMetrics(true, predicted)
    for i in ['mse', 'rmse', 'r2', 'r2_score', 'mape']:
        print(i, getattr(metrics, i)())
    return


def plot_violin_(feature_name, test_p, cut,
                 grid=None,
                 show_bar=False,
                 show_violin=True):

    _, _, X_test, _ = get_data()
    dataset, _, _ = get_dataset()

    ax, df = prediction_distribution_plot(
        mode='regression',
        inputs=pd.DataFrame(X_test, columns=dataset.input_features),
        prediction=test_p,
        feature=feature_name,
        feature_name=feature_name,
        show=False,
        cust_grid_points=grid
    )

    if show_bar:
        plt.show()

    if feature_name == 'calcination_temperature':
        df.drop(3, inplace=True)
        df['display_column'] = ['[25,550)', '[550,600)', '[600,700)', '[700,800)', '[800,900)']

    elif feature_name == 'initial concentration':
        df.drop(0, inplace=True)
        df['display_column'] = ['[1.01,10)', '[10,50)', '[50,100)', '[100,200)', '[200,300)', '[300,400)', '[400,900)']

    elif feature_name == 'Volume (L)':
        df.drop(1, inplace=True)
        df['display_column'] = ['[0.02,0.04)', '[0.04,0.05)', '[0.05,0.1)', '[0.1,0.25)', '[0.25,1)']

    elif feature_name == 'adsorbent loading':
        df.drop(2, inplace=True)
        df['display_column'] = ['[0.0,0.01)', '[0.01,0.04)', '[0.04,0.1)', '[0.1,0.5)', '[0.5,2.47)', '[2.47,10)']

    preds = {}
    for interval in df['display_column']:
        st, en = interval.split(',')
        st = float(''.join(e for e in st if e not in ["]", ")", "[", "("]))
        en = float(''.join(e for e in en if e not in ["]", ")", "[", "("]))
        df1 = pd.DataFrame(X_test, columns=dataset.input_features)
        df1['target'] = test_p
        df1 = df1[[feature_name, 'target']]
        df1 = df1[(df1[feature_name] >= st) & (df1[feature_name] < en)]
        preds[interval] = df1['target'].values

    for k, v in preds.items():
        assert len(v) > 0, f"{k} has no values in it"

    plt.close('all')
    ax = violin_plot(list(preds.values()), cut=cut, show=False)
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(list(preds.keys()))
    ax.set_title(feature_name)
    ax.set_facecolor("#fbf9f4")

    if show_violin:
        plt.show()
    return ax
