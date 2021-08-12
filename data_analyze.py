import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import NoReturn
import matplotlib.pyplot as plt

global gini
gini = lambda y,x: roc_auc_score(y, x)*2 - 1

class GenData:
    def __init__(self, data: dict = {}, trg_feature: str = None, date_feature: str = None, id_feature: str = None,
                 meta_features: list = [], ignore_dt: bool = True) -> NoReturn:
        self.train = data['train']
        self.valid = data['valid']

        self.date_feature = date_feature
        self.trg_feature = trg_feature
        self.id_feature = id_feature
        self.meta_features = meta_features
        self.long_list = [col for col in self.train.columns.to_list() if col not in
                          [self.date_feature, self.trg_feature, self.id_feature] + self.meta_features]
        self.bad_feats = {}

        self.feature_types = {}
        self.feature_stats = {'main': {}}
        self.trg_stats = {'main': {}}

        self.task_type = 'binary' if self.train[date_feature].nunique() == 2 else 'multi'
        self.ignore_dt = ignore_dt

    def define_types(self):
        cat_features = []
        numerical_features = []
        date_features = []

        for col in self.long_list:
            series = self.train[col]
            if (any([c in col for c in ['dt', 'date']]) \
                    or 'date' in str(series.dtype) or 'time' in str(series.dtype) \
                    or series.map(lambda x: '-' in str(x)).sum()/series.dropna().size > 0.96) and not self.ignore_dt:
                date_features.append(col)
            elif series.nunique() == 2 or \
                    any([c in col for c in ['string', 'code', 'cat', 'category']]) \
                    or series.nunique() < 10 and 'int' in str(series.dtype) \
                    or str(series.dtype) == 'O':
                cat_features.append(col)
            else:
                numerical_features.append(col)

        self.feature_types['categorical'] = cat_features
        self.feature_types['numerical'] = numerical_features
        self.feature_types['date'] = date_features

    def make_empty_dict(self, dict):
        for type in self.feature_types:
            if len(self.feature_types[type]) > 0:
                dict[type] = {}
                for col in self.feature_types[type]:
                    dict[type][col] = {}

    def calc_stats(self):
        """

        :return:
        """
        aggr_funcs_numerical = {'nans': lambda x: x.isna().sum(),
                              'max': lambda x: x.max(),
                              'min': lambda x: x.min(),
                              'std': lambda x: x.std(),
                              'mean': lambda x: x.mean(),
                              'median': lambda x: x.median(),
                              'q25': lambda x: x.quantile(0.25),
                              'q75': lambda x: x.quantile(0.75),
                              'q05': lambda x: x.quantile(1-0.9544),
                              'q95': lambda x: x.quantile(0.9544),
                              'skewness': lambda x: x.skew()
                              }

        aggr_funcs_cat = {'nans': lambda x: x.isna().sum(),
                          'nunique': lambda x: x.nunique(),
                          'n_obs': lambda x: x.value_counts().sort_values(ascending=False).to_dict(),
                          'most_frequent': lambda x: x.value_counts().sort_values(ascending=False)[:1].to_dict()
                          }

        aggr_funcs_date = {'nans': lambda x: x.isna().sum(),
                           'max': lambda x: pd.to_datetime(x).max(),
                           'min': lambda x: pd.to_datetime(x).min(),
                           }

        print('Calculating main features\'s stats')

        self.make_empty_dict(self.feature_stats['main'])
        for col in tqdm(self.long_list):
            series = self.train[col]
            if col in self.feature_types['numerical']:
                self.feature_stats['main']['numerical'][col].update({k: v(series) for k, v in aggr_funcs_numerical.items()})
            if col in self.feature_types['categorical']:
                self.feature_stats['main']['categorical'][col].update({k: v(series) for k, v in aggr_funcs_cat.items()})
            if col in self.feature_types['date']:
                self.feature_stats['main']['date'][col].update({k: v(series) for k, v in aggr_funcs_date.items()})

        if self.date_feature:
            print('Calculating features\'s stats grouped by dates')
            dates = np.sort(self.train[self.date_feature].unique())
            self.feature_stats.update({k: {} for k in dates})
            for date in tqdm(dates):
                self.make_empty_dict(self.feature_stats[date])
                sample = self.train[self.train[self.date_feature] == date]
                for col in self.long_list:
                    series = sample[col]
                    if col in self.feature_types['numerical']:
                        self.feature_stats[date]['numerical'][col].update({k: v(series) for k, v in aggr_funcs_numerical.items()})
                    if col in self.feature_types['categorical']:
                        self.feature_stats[date]['categorical'][col].update({k: v(series) for k, v in aggr_funcs_cat.items()})
                    if col in self.feature_types['date']:
                        self.feature_stats[date]['date'][col].update({k: v(series) for k, v in aggr_funcs_date.items()})

    def calc_stats_target(self):
        """

        :return:
        """
        y = self.train[self.trg_feature]
        trg_funcs_numerical = {'corr': lambda x: np.corrcoef(y,x)[0,1],
                             'gini': lambda x: gini(y,x.fillna(-9999)),
                             'max': lambda x: x[y == 1].max(),
                             'min': lambda x: x[y == 1].min(),
                             'std': lambda x: x[y == 1].std(),
                             'mean': lambda x: x[y == 1].mean(),
                             'median': lambda x: x[y == 1].median(),
                             'q25': lambda x: x[y == 1].quantile(0.25),
                             'q75': lambda x: x[y == 1].quantile(0.75),
                             'q05': lambda x: x[y == 1].quantile(1-0.9544),
                             'q95': lambda x: x[y == 1].quantile(0.9544),
                             'skewness': lambda x: x[y == 1].skew()
                            }

        trg_funcs_cat = {'n_pos': lambda x: {v: y[x == v].sum() for v in x.unique()},
                         'target_rate': lambda x: {v: y[x == v].mean() for v in x.unique()},
                         'most_frequent': lambda x: x[y == 1].value_counts().sort_values(ascending=False)[:1].to_dict()
                          }

        trg_funcs_date = {'max': lambda x: pd.to_datetime(x.loc[y == 1]).max(),
                          'min': lambda x: pd.to_datetime(x.loc[y == 1]).min(),
                           }

        self.make_empty_dict(self.trg_stats['main'])
        print('Calculating main TARGET-features\'s stats')
        for col in tqdm(self.long_list):
            series = self.train[col]
            if col in self.feature_types['numerical']:
                self.trg_stats['main']['numerical'][col].update({k: v(series) for k, v in trg_funcs_numerical.items()})
            if col in self.feature_types['categorical']:
                self.trg_stats['main']['categorical'][col].update({k: v(series) for k, v in trg_funcs_cat.items()})
            if col in self.feature_types['date']:
                self.trg_stats['main']['date'][col].update({k: v(series) for k, v in trg_funcs_date.items()})

        if self.date_feature:
            print('Calculating features\'s with TARGET stats grouped by dates')
            dates = np.sort(self.train[self.date_feature].unique())
            self.trg_stats.update({k: {} for k in dates})
            for date in tqdm(dates):
                self.make_empty_dict(self.trg_stats[date])
                sample = self.train[self.train[self.date_feature] == date]
                for col in self.long_list:
                    series = sample[col]
                    y = sample[self.trg_feature]
                    if col in self.feature_types['numerical']:
                        self.trg_stats[date]['numerical'][col].update({k: v(series) for k, v in trg_funcs_numerical.items()})
                    if col in self.feature_types['categorical']:
                        self.trg_stats[date]['categorical'][col].update({k: v(series) for k, v in trg_funcs_cat.items()})
                    if col in self.feature_types['date']:
                        self.trg_stats[date]['date'][col].update({k: v(series) for k, v in trg_funcs_date.items()})

    def find_bad(self, prc_nans: float = 0.95, prc_most_frequent: float = 0.95,
                 thr_const_float: float = 0.01, prc_outliers: float = 0.5):
        """

        :param prc_nans:
        :param prc_most_frequent:
        :param thr_corr:
        :return:
        """
        for col in tqdm(self.long_list):
            series = self.train[col]
            #############################
            # CONST FEATURES CATEGORICAL
            #############################
            for n, t in self.feature_types.items():
                if col in t:
                    f_type = n
                    break

            if series.nunique() == 1:
                self.bad_feats.update({col: 'const_categorical'})
                continue

            #############################
            # NaNs
            #############################
            if self.feature_stats['main'][f_type][col]['nans'] / series.size > prc_nans:
                self.bad_feats.update({col: 'nans'})
                continue

            if f_type == 'numerical':
                #############################
                # CONST FEATURES numerical
                #############################
                s_norm = MinMaxScaler().fit_transform(series.values.reshape(-1, 1))
                if (np.std(s_norm) < thr_const_float) or \
                        (self.feature_stats['main'][f_type][col]['q05'] == self.feature_stats['main'][f_type][col]['q95']):
                    self.bad_feats.update({col: 'const_numerical'})
                    continue

                #############################
                # OUTLIERS numerical
                #############################
                Q1 = self.feature_stats['main'][f_type][col]['q25']
                Q3 = self.feature_stats['main'][f_type][col]['q75']
                low = Q1 - 1.5 * (Q3 - Q1)
                high = Q3 + 1.5 * (Q3 - Q1)
                if ((series > high) + (series < low)).sum()/series.size > prc_outliers:
                    self.bad_feats.update({col: 'outliers'})
                    continue

            #############################
            # Most frequent value percent
            #############################
            if f_type == 'categorical':
                if list(self.feature_stats['main'][f_type][col]['most_frequent'].values())[0]/series.size > prc_most_frequent:
                    self.bad_feats.update({col: 'most_frequent'})
                    continue
        
        print(len(self.bad_feats), 'bad features was founded.')
        if len(self.bad_feats) < 50 and len(self.bad_feats) > 0:
            print(list(self.bad_feats.keys()))

    def plot_pie(self, pie_params: dict = {}, **kwargs):

        vals = [len(set(c) - set(self.bad_feats)) for c in self.feature_types.values()]
        keys = [k + f' = {vals[i]}' for i, k in enumerate(self.feature_types.keys()) if vals[i] > 0]
        vals = [v for v in vals if v > 0]

        if len(self.bad_feats) > 0:
            vals.append(len(self.bad_feats))
            keys.append(f'bad_feats = {len(self.bad_feats)}')

        plt.figure(figsize=(12, 8))
        plt.title('Feature\'s types count')
        plt.pie(vals, labels=keys, **pie_params)
        plt.show()

    def plot_dynamics(self, top: int = 5, feats_list: list = None, stats_plot: list = ['max', 'mean', 'std'],
                      dynamic_params: dict = {}, **kwargs):
        if top > 0 and top < 1:
            best_numeric = [col for col, vals in self.trg_stats['main']['numerical'].items() if abs(vals['gini']) >= top]
        elif top % 1 == 0:
            best_numeric = pd.Series({col: abs(vals['gini']) for col, vals in self.trg_stats['main']['numerical'].items()}).\
                               sort_values(ascending=False).index.to_list()[:top]
        elif feats_list is not None:
            best_numeric = feats_list

        stats = {}
        for stat in stats_plot:
            stats[stat] = {}
            for col in best_numeric:
                stats[stat][col] = {date: self.feature_stats[date]['numerical'][col][stat] for date in self.feature_stats
                                    if date != 'main'}

                plt.figure(figsize=(12, 8))
                plt.title(f'Dynamic of {col} feature: {stat} statistics')
                plt.plot(list(stats[stat][col].keys()), list(stats[stat][col].values()), label=f'{col}__{stat}',
                         **dynamic_params)
                plt.legend()
                plt.show()
        return stats

    def cat_hist_trg(self):
        for col in self.feature_types['categorical']:
            fig, ax = plt.subplots()
            x, y = zip(*sorted(self.trg_stats['main']['categorical'][col]['target_rate'].items()))
            ax.plot(x, y, color='yellow', zorder=1)
            ax.axhline(y=self.train[self.trg_feature].mean(), color='r', linestyle='--')
            pd.Series(self.feature_stats['main']['categorical'][col]['n_obs']).plot(kind='bar', secondary_y=True)
            plt.show()

    def visualize(self, style: str ='seaborn', pie_params: dict = {}, dynamics_params: dict = {}, **kwargs):
        print('Pie chart of features types')

        plt.style.use(style)

        self.plot_pie(pie_params=pie_params, **kwargs)
        stats = self.plot_dynamics(dynamic_params=dynamics_params, **kwargs)
        self.cat_hist_trg()
        return stats


    def run(self):
        self.define_types()
        self.calc_stats()
        self.calc_stats_target()
        self.find_bad()


    def find_patterns(self):
        pass