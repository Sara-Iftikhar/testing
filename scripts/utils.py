
import os
import random
from typing import Union, List, Tuple, Any
from collections.abc import KeysView, ValuesView

import shap
from shap.plots import scatter as sh_scatter

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ai4water.functional import Model as f_model
from ai4water import Model
from ai4water.models import MLP
from ai4water.postprocessing import prediction_distribution_plot

from ai4water.preprocessing import DataSet
from ai4water.utils.utils import dateandtime_now

from SeqMetrics import RegressionMetrics

from easy_mpl import violin_plot, scatter
from easy_mpl.utils import is_rgb
from easy_mpl.utils import BAR_CMAPS
from easy_mpl.utils import process_axes
from easy_mpl.utils import create_subplots
from easy_mpl.utils import to_1d_array, make_cols_from_cmap


ADSORBENT_TYPES = {
    "GIC": "GB",
    "Exfoliated GIC": "GB",
    "PAC": "AC",
    "APAC": "AC",
    "CS": "AC",
    "AC600": "AC",
    "AC700": "AC",
    "AC800": "AC",
    "AC900": "AC",
    "CMCAC": "AC",
    "CS-AC-KOH": "AC",
    "CS-AC-NaOH": "AC",
    "CS-AC-H3PO4": "AC",
    "CS-AC-H4P2O7": "AC",
    "TSAC": "AC",
    "MC350": "AC",
    "MC400": "AC",
    "MC450": "AC",
    "MC500": "AC",
    "MC550": "AC",
    "MC600": "AC",
    "MC0.75": "AC",
    "MC0.659": "AC",
    "MC0.569": "AC",
    "MC0.478": "AC",
    "MC20/1": "AC",
    "MC25/1": "AC",
    "MC30/1": "AC",
    "MC35/1": "AC",
    "MCNaOH10": "AC",
    "MCNaOH30": "AC",
    "MCNaOH40": "AC",
    "MCNaOH50": "AC",
    "GSAC": "AC",
    "CAS": "AC",
    "SAC": "AC",
    "HAC": "AC",
    "CAC": "AC",
    "CBAC": "AC",
    "VAC": "AC",
    "TRAC": "AC",
    "BGBHAC": "AC",
    "GSAC-Ce-1": "AC",
    "TWAC": "AC",
    "WSAC": "AC",
    "PSB": "Biochar",
    "PSB-LDHMgAl": "Biochar",
    "RH Biochar": "Biochar",
    "M-Biochar": "Biochar",
    "MN-Biochar": "Biochar",
    "MZ-Biochar": "Biochar",
}

DYE_TYPES = {
    'CR': 'Anionic',
    'FG FCF': 'Anionic',
    'MO': 'Anionic', 'NR': 'Anionic',
    'AR': 'Anionic', 'RB5': 'Anionic', 'RD': 'Anionic',
    'AB25': 'Anionic',
    'BV14': 'Cationic',
    'MB': 'Cationic',
    'SYF': 'Cationic', 'MV': 'Cationic',
    'GR': 'Cationic',
    'Rhd B': 'Cationic', 'YD': 'Cationic',
    'AM': 'Cationic'
}

def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    # setting sparse to True will return a scipy.sparse.csr.csr_matrix
    # not a numpy array
    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder


def _load_data(input_features:list=None)->pd.DataFrame:

    # read excel
    # our data is on the first sheet of both files
    ads_data = pd.read_excel('Adsorption and regeneration data_1007c.xlsx')
    dye_data = pd.read_excel('Dyes data.xlsx')

    # dropping unnecessary columns
    ads_data = ads_data.drop(columns=['final concentation', 'Volume (mL)',
                                  'Unnamed: 16', 'Unnamed: 17',
                                  'Unnamed: 18', 'Unnamed: 19',
                                  'Unnamed: 20', 'Unnamed: 21',
                                  'Unnamed: 22', 'Unnamed: 23',
                                  'Particle size'
                                  ])

    dye_data = dye_data.drop(columns=['C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C',
                                  'N/C', 'Average pore size',
                                  'rpm', 'g/L', 'Ion Concentration (M)',
                                  'Humic acid', 'wastewater type',
                                  'Adsorption type', 'Cf', 'Ref'
                                  ])

    # merging data
    data = pd.concat([ads_data, dye_data])
    data = data.dropna()

    #removing original index of both dataframes and assigning a new index
    data = data.reset_index(drop=True)

    data.columns = ['Adsorption Time (min)', 'Adsorbent', 'Pyrolysis Temperature',
                    'Pyrolysis Time (min)', 'Dye', 'Initial Concentration', 'Solution pH',
                    'Adsorbent Loading', 'Volume (L)', 'Adsorption Temperature',
                    'Surface Area', 'Pore Volume', 'Adsorption']

    # replacing a string 'Fast Green FCF' in features Dye with 'FG FCF' because it will
    # cause the scatter plot in SHAP to elongate.
    data['Dye'] = data['Dye'].str.replace('Fast Green FCF', 'FG FCF')

    target = ['Adsorption']
    if input_features is None:
        input_features = data.columns.tolist()[0:-1]
    else:
        assert isinstance(input_features, list)
        assert all([feature in data.columns for feature in input_features])

    return data[input_features + target]


def make_data(
        input_features:list = None,
        encode:bool = True)->Tuple[pd.DataFrame, Any, Any]:
    """
    prepares data for adsorption capacity prediction.

    Parameters
    ----------
    input_features : list
        names of variables to use as input. By default the following features
        are used as input features
            - Adsorption Time (min)
            - Adsorbent
            - Pyrolysis Temperature
            - Pyrolysis Time (min)
            - Dye
            - Initial Concentration
            - Solution pH
            - Adsorbent Loading
            - Volume (L)
            - Adsorption Temperature
            - Surface Area
            - Pore Volume

    encode : bool (default=True)
        whether to one hot encode the categorical variables or not

    Returns
    -------
    data : pd.DataFrame
        a pandas dataframe whose first 10 columns are numerical features
        and next columns contain categorical features. The last column is
        the target feature. If encode is True (default case) the returned
        dataframe has 75 columns. 0-10 numerical features, 11-58 adsorbents
        59-74: dyes 75th: target. If encode is False, then the returned
        dataframe will have 13 columns.

    Examples
    --------
    >>> data, ae, de = make_data()
    >>> data.shape
    (1514, 75)
    >>> len(ae.categories_[0])
    48
    to get the original adsorbent values we can do as below
    >>> ae.inverse_transform(data.iloc[:, 10:58].values)
    >>> len(de.categories_[0])
    16
    We can also convert the one hot encoded dye columns into original/string form as
    >>> de.inverse_transform(data.iloc[:, 58:-1].values)
    If we don't want to encode categorical features, we can set encode to False
    >>> data, _, _ = make_data(encode=False)
    >>> data.shape
    (1514, 13)
    """
    data = _load_data(input_features)

    adsorbent_encoder, dye_encoder = None, None
    if encode:
        # applying OHE
        data, _, adsorbent_encoder = _ohe_column(data, 'Adsorbent')

        data, _, dye_encoder = _ohe_column(data, 'Dye')

    # moving target to last
    target = data.pop('Adsorption')
    data['Adsorption'] = target

    return data, adsorbent_encoder, dye_encoder


def get_dataset():
    data, adsorbent_encoder, dye_encoder = make_data()

    dataset = DataSet(data=data,
                      seed=1575,
                      val_fraction=0.0,
                      split_random=True,
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      )
    return dataset, adsorbent_encoder, dye_encoder


def make_path():
    path = os.path.join(os.getcwd(), 'results', f'mlp_{dateandtime_now()}')
    os.makedirs(path)
    return path


def get_fitted_model(return_path=False,
                     model_type=None,
                     from_config=True):

    dataset, _, _ = get_dataset()

    X_train, y_train = dataset.training_data()

    if from_config:
        path = os.path.join(os.getcwd(), 'results', 'mlp_20221217_213202')
        cpath = os.path.join(path, 'config.json')
        if model_type == 'functional':
            model = f_model.from_config_file(config_path=cpath)
        else:
            model = Model.from_config_file(config_path=cpath)
        wpath = os.path.join(path, 'weights_585_1982.99475.hdf5')
        model.update_weights(wpath)
        fpath = os.path.join(path, 'losses.csv')
        df = pd.read_csv(fpath)[['loss', 'val_loss']]
        class History(object):
            def init(self):
                self.history = df.to_dict()

        h = History()
    else:
        path = make_path()
        if model_type=='functional':
            model = f_model(
                model=MLP(units=99, num_layers=4,
                          activation='relu'),
                lr=0.006440897421063212,
                input_features=dataset.input_features,
                output_features=dataset.output_features,
                epochs=400, batch_size=48,
                verbosity=0
            )
        else:
            model = Model(
                model=MLP(units=99, num_layers=4,
                          activation='relu'),
                lr=0.006440897421063212,
                input_features=dataset.input_features,
                output_features=dataset.output_features,
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
    for i in ['mse', 'rmse', 'r2', 'r2_score', 'mape', 'mae']:
        print(i, getattr(metrics, i)())
    return


def plot_violin_(feature_name, test_p, cut,
                 grid=None,
                 show_bar=False,
                 show_violin=True):

    dataset, _, _ = get_dataset()

    X_test, _ = dataset.test_data()

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

    if feature_name == 'Pyrolysis Temperature':
        df.drop(3, inplace=True)
        df['display_column'] = ['[25,550)', '[550,600)', '[600,700)', '[700,800)', '[800,900)']

    elif feature_name == 'Initial Concentration':
        df.drop(0, inplace=True)
        df['display_column'] = ['[1.01,10)', '[10,50)', '[50,100)', '[100,200)', '[200,300)', '[300,400)', '[400,900)']

    elif feature_name == 'Volume (L)':
        df.drop(1, inplace=True)
        df['display_column'] = ['[0.02,0.04)', '[0.04,0.05)', '[0.05,0.1)', '[0.1,0.25)', '[0.25,1)']

    elif feature_name == 'Adsorbent Loading':
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
    ax.set_xticklabels(list(preds.keys()), size=12, weight='bold')
    ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
    ax.set_title(feature_name, size=14, fontweight="bold")
    ax.set_facecolor("#fbf9f4")

    if show_violin:
        plt.show()
    return ax


def box_violin(ax, data, palette=None):
    if palette is None:
        palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.7)
    ax = sns.violinplot(orient='h', data=data,
                        palette=palette,
                        scale="width", inner=None,
                        ax=ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData))

    sns.boxplot(orient='h', data=data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return


def shap_interaction_all(shap_values_exp, feature, feature_names, CAT_FEATURES):
    inds = shap.utils.potential_interactions(shap_values_exp[:, feature], shap_values_exp)

    # make plots colored by each of the top three possible interacting features
    n, n_plots = 0, 0
    while n <= len(feature_names):
        if shap_values_exp.feature_names[inds[n]] not in CAT_FEATURES:
            sh_scatter(shap_values_exp[:, feature], show=False,
                    color=shap_values_exp[:, inds[n]],
                    )
            plt.tight_layout()
            plt.show()
            n_plots += 1

        if n_plots >= 10:
            break
        n += 1

    return


def shap_scatter(
        shap_values,  # SHAP values for a single feature
        feature_wrt:pd.Series = None,
        show_hist=True,
        show=True,
        is_categorical=False,
        palette_name = "tab10",
        s = 70,
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8,
        ax = None,
        **scatter_kws
):
    if ax is None:
        fig, ax = plt.subplots()

    if feature_wrt is None:
        c = None
    else:
        if is_categorical:
            if isinstance(palette_name, (tuple, list)):
                assert len(palette_name) == len(feature_wrt.unique())
                rgb_values = palette_name
            else:
                rgb_values = sns.color_palette(palette_name, feature_wrt.unique().__len__())
            color_map = dict(zip(feature_wrt.unique(), rgb_values))
            c= feature_wrt.map(color_map)
        else:
            c= feature_wrt.values.reshape(-1,)

    _, pc = scatter(
        shap_values.data,
        shap_values.values,
        c=c,
        s=s,
        marker="o",
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        show=False,
        **scatter_kws
    )

    if feature_wrt is not None:
        feature_wrt_name = ' '.join(feature_wrt.name.split('_'))
        if is_categorical:
            # add a legend
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                              label=k, markersize=8) for k, v in color_map.items()]

            ax.legend(title=feature_wrt_name,
                  handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                      title_fontsize=14
                      )
        else:
            cbar = plt.colorbar(pc, aspect=80)
            cbar.ax.set_ylabel(feature_wrt_name, rotation=90, labelpad=14,
                               fontsize=14, weight="bold")

            if 'volume' in feature_wrt_name.lower():
                ticks = np.round(cbar.ax.get_yticks(), 2)
                cbar.ax.set_yticklabels(ticks, size=12, weight='bold')
            else:
                cbar.ax.set_yticklabels(cbar.ax.get_yticks().astype(int), size=12, weight='bold')

            cbar.set_alpha(1)
            cbar.outline.set_visible(False)

    feature_name = ' '.join(shap_values.feature_names.split('_'))

    ax.set_xlabel(feature_name, fontsize=14, weight="bold")
    ax.set_ylabel(f"SHAP value for {feature_name}", fontsize=14, weight="bold")
    ax.axhline(0, color='grey', linewidth=1.3, alpha=0.3, linestyle='--')

    if 'volume' in feature_name.lower():
        ticks = np.round(ax.get_xticks(), 2)
        ax.set_xticklabels(ticks, size=12, weight='bold')
    else:
       ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')

    ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')

    if show_hist:
        x = shap_values.data

        if len(x) >= 500:
            bin_edges = 50
        elif len(x) >= 200:
            bin_edges = 20
        elif len(x) >= 100:
            bin_edges = 10
        else:
            bin_edges = 5

        ax2 = ax.twinx()

        xlim = ax.get_xlim()

        ax2.hist(x.reshape(-1,), bin_edges,
                 range=(xlim[0], xlim[1]),
                 density=False, facecolor='#000000', alpha=0.1, zorder=-1)
        ax2.set_ylim(0, len(x))
        ax2.set_yticks([])

    if show:
        plt.show()

    return ax


def _jitter_data(data, x_jitter, seed=None):

    s = np.random.RandomState(seed)
    s.random_sample([1, 2, 3])

    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = data.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            data += (s.random_sample(size = len(data))*jitter_amount) - (jitter_amount/2)

    return data


def bar_chart(
        values,
        labels=None,
        orient:str = 'h',
        sort:bool = False,
        max_bars:int = None,
        errors = None,
        color=None,
        cmap: Union[str, List[str]] = None,
        rotation:int = 0,
        bar_labels: Union[list, np.ndarray] = None,
        bar_label_kws=None,
        share_axes: bool = True,
        width = None,
        ax:plt.Axes = None,
        ax_kws: dict = None,
        show:bool = True,
        **kwargs
) -> Union[plt.Axes, List[plt.Axes]]:

    if labels is None:
        if hasattr(values, "index") and hasattr("values", "name"):
            labels = values.index

    naxes = 1
    ncharts = 1
    if is_1d(values):
        values = to_1d_array(values)
    else:
        values = np.array(values)
        ncharts = values.shape[1]
        if share_axes:
            kwargs['edgecolor'] = kwargs.get('edgecolor', 'k')
        else:
            naxes = values.shape[1]

    colors = get_color(cmap, color, ncharts, len(values))

    figsize = None
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')

    ax = maybe_create_axes(ax, naxes, figsize=figsize)

    if ncharts == 1:
        values, labels, bar_labels, colors = preprocess(values, labels,
                                                bar_labels, sort, max_bars, colors[0])
        ind = np.arange(len(values))
        bar_on_axes(ax[0], orient=orient, ax_kws=ax_kws, ind=ind,
                    values=values,
                    width=width, ticks=ind, labels=labels, color=colors,
                    bar_labels=bar_labels,
                   rotation=rotation, errors=errors,
                    bar_label_kws=bar_label_kws, kwargs=kwargs)

    elif share_axes:
        ind = np.arange(len(values))  # the label locations
        width = width or 1/ncharts * 0.9  # the width of the bars

        inds = []
        for idx in range(ncharts):
            if idx>0:
                ind = ind + width
            inds.append(ind)
        inds = np.column_stack(inds)
        ticks = np.mean(inds, axis=1)

        for idx in range(ncharts):

            _kwargs =kwargs.copy()
            _kwargs['label'] = _kwargs.get('label', idx)

            vals, labels, bar_labels, color = preprocess(values[:, idx], labels,
                                                  bar_labels, sort, max_bars, colors[idx])
            bar_on_axes(ax[0], orient, ax_kws,
                        inds[:, idx], vals, width, ticks, labels,
                        color, bar_labels,
                       rotation, errors, bar_label_kws, _kwargs)

    else:
        for idx in range(naxes):
            axes = ax[idx]
            data = values[:, idx]
            data, labels, bar_labels, color = preprocess(data, labels, bar_labels,
                                                  sort, max_bars, colors[idx])

            _kwargs = kwargs.copy()
            _kwargs['label'] = _kwargs.get('label', idx)

            ind = np.arange(len(data))
            bar_on_axes(axes, orient, ax_kws,
                        ind, data, width, ind, labels,
                        color, bar_labels,
                        rotation, errors,
                        bar_label_kws=bar_label_kws, kwargs=_kwargs)

    if show:
        plt.show()

    if len(ax) == 1:
        ax = ax[0]

    return ax


def maybe_create_axes(ax, naxes:int, figsize=None)->List[plt.Axes]:
    if ax is None:
        ax = plt.gca()
        if naxes>1:
            f, ax = create_subplots(ax=ax, naxes=naxes, figsize=figsize)
            ax = ax.flatten()
        else:
            if figsize:
                ax.figure.set_size_inches(figsize)
            ax = [ax]
    elif naxes>1:
        f, ax = create_subplots(ax=ax, naxes=naxes, figsize=figsize)
        ax = ax.flatten()
    else:
        if figsize:
            ax.figure.set_size_inches(figsize)
        ax = [ax]

    return ax


def handle_sort(sort, values, labels, bar_labels, color):
    if sort:
        sort_idx = np.argsort(values)
        values = values[sort_idx]
        labels = np.array(labels)[sort_idx]
        if bar_labels is not None:
            bar_labels = np.array(bar_labels)
            bar_labels = bar_labels[sort_idx]
            if 'float' in bar_labels.dtype.name:
                bar_labels = np.round(bar_labels, decimals=2)

        if isinstance(color, (list, np.ndarray, tuple)):
            if is_rgb(color[0]) or isinstance(color[0], str):
                color = np.array(color)[sort_idx]

    return values, labels, bar_labels, color


def handle_maxbars(max_bars, values, labels):
    if max_bars:
        n = len(values) - max_bars
        last_val = sum(values[0:-max_bars])
        values = values[-max_bars:]
        labels = labels[-max_bars:]
        values = np.append(last_val, values)
        labels = np.append(f"Rest of {n}", labels)
    return values, labels


def preprocess(values, labels, bar_labels, sort, max_bars, colors):
    if labels is None:
        labels = [f"F{i}" for i in range(len(values))]

    values, labels, bar_labels, colors = handle_sort(sort, values, labels, bar_labels, colors)

    values, labels = handle_maxbars(max_bars, values, labels)

    return values, labels, bar_labels, colors


def bar_on_axes(ax, orient, ax_kws, *args, **kwargs):
    if orient in ['h', 'horizontal']:
        horizontal_bar(ax, *args, **kwargs)
    else:
        vertical_bar(ax, *args, **kwargs)

    if ax_kws:
        process_axes(ax, **ax_kws)

    return


def horizontal_bar(ax, ind, values, width, ticks, labels, color, bar_labels,
                   rotation, errors, bar_label_kws, kwargs):

    if width:
        bar = ax.barh(ind, values, width, color=color, **kwargs)
    else:
        bar = ax.barh(ind, values, color=color, **kwargs)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, rotation=rotation)

    set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   values, ind)

    if 'label' in kwargs:
        ax.legend()
    return


def vertical_bar(ax, ind, values, width, ticks, labels, color, bar_labels,
                 rotation, errors, bar_label_kws, kwargs):

    bar = ax.bar(ind, values, width=width or 0.8, color=color, **kwargs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation)

    set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   ind, values)
    return


def set_bar_labels(bar, ax, bar_labels, bar_label_kws, errors,
                   values, ind):
    if bar_labels is not None:
        bar_label_kws = bar_label_kws or {'label_type': 'center'}
        if hasattr(ax, 'bar_label'):
            ax.bar_label(bar, labels=bar_labels, **bar_label_kws)
        else:
            bar.set_label(bar_labels)

    if errors is not None:
        ax.errorbar(values, ind, xerr=errors, fmt=".",
                    color="black")
    return


def is_1d(array):
    if isinstance(array, (KeysView, ValuesView)):
        array = np.array(list(array))
    else:
        array = np.array(array)
    if len(array)==array.size:
        return True
    return False


def get_color(cmap, color, ncharts, n_bars)->list:
    if not isinstance(cmap, list):
        cmap = [cmap for _ in range(ncharts)]

    if not isinstance(color, list):
        color = [color for _ in range(ncharts)]
    elif ncharts == 1:
        # the user has specified separate color for each bar
        # in next for loop we don't want to get just firs color from the list
        color = [color]

    colors = []
    for idx in range(ncharts):

        cm = make_cols_from_cmap(cmap[idx] or random.choice(BAR_CMAPS), n_bars, 0.2)

        clr = color[idx] if color[idx] is not None else cm

        colors.append(clr)

    return colors
