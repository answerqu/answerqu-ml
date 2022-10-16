import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import NoReturn
import matplotlib.pyplot as plt

#tqdm = lambda x: x

global gini
def gini(y,x): 
    try:
        return roc_auc_score(y, x) * 2 - 1
    except ValueError:
        return 0
            


class GenData:
    def __init__(self, data: dict = {}, trg_feature: str = None, date_feature: str = None, id_feature: str = None,
                 meta_features: list = [], score_name: str = 'score', ignore_dt: bool = True) -> NoReturn:
        self.data = {}
        self.data['train'] = data['train']
        self.data['valid'] = data.get('valid', pd.DataFrame(columns=data['train'].columns))
        self.data['oos'] = data.get('oos', pd.DataFrame(columns=data['train'].columns))
        self.data['oot'] = data.get('oot', pd.DataFrame(columns=data['train'].columns))
        
        self.date_feature = date_feature
        self.trg_feature = trg_feature
        self.id_feature = id_feature
        self.score = score_name
        self.score_list = [self.score] if self.score in self.data['train'].columns else []
        self.meta_features = meta_features
        self.long_list = [col for col in self.data['train'].columns.to_list() if col not in
                          [self.date_feature, self.trg_feature, self.id_feature, self.score] + self.meta_features]
        self.bad_feats = {}

        self.feature_types = {}
        if 'oot' in self.data:
            self.feature_stats = {'train': {}, 'oot': {}}
            self.non_trg_stats = {'train': {}, 'oot': {}}
            self.trg_stats = {'train': {}, 'oot': {}}
        else:
            self.feature_stats = {'train': {}}
            self.non_trg_stats = {'train': {}}
            self.trg_stats = {'train': {}}

        self.task_type = 'binary' if self.data['train'][date_feature].nunique() == 2 else 'multi'
        self.ignore_dt = ignore_dt

    def define_types(self):
        cat_features = []
        numerical_features = []
        date_features = []

        print('Definig types of features in process...')
        for col in self.long_list + self.score_list:
            series = self.data['train'][col]
            nunique = series.nunique()
            dtype = str(series.dtype)
            if not self.ignore_dt:
                if (any([c in col for c in ['dt', 'date']]) \
                        or 'date' in dtype or 'time' in dtype \
                        or series.map(lambda x: '-' in str(x)).sum() / series.dropna().size > 0.96):
                    date_features.append(col)
                elif nunique == 2 or \
                        any([c in col for c in ['string', 'code', 'cat', 'category', 'tag', 'flag', 'flg']]) \
                        or nunique < 10 and 'int' in dtype or dtype == 'O':
                    cat_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                if any([c in col for c in ['string', 'code', 'cat', 'category', 'tag', 'flag', 'flg']]) \
                        or nunique < 10 or dtype == 'O':
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
        aggr_funcs_numerical = {'gini': lambda x, y: gini(y, x.fillna(-9999)),
                                'corr': lambda x, y: np.corrcoef(y, x.fillna(0))[0,1],
                                'nans': lambda x, y: x.isna().mean(),
                                'max': lambda x, y: x.max(),
                                'min': lambda x, y: x.min(),
                                'std': lambda x, y: x.std(),
                                'mean': lambda x, y: x.mean(),
                                'median': lambda x, y: x.median(),
                                'q25': lambda x, y: x.quantile(0.25),
                                'q75': lambda x, y: x.quantile(0.75),
                                'q05': lambda x, y: x.quantile(1 - 0.95),
                                'q95': lambda x, y: x.quantile(0.95),
                                'skewness': lambda x, y: x.skew()
                                }

        aggr_funcs_cat = {'nans': lambda x, y: x.isna().sum(),
                          'nunique': lambda x, y: x.nunique(),                          
                          'n_obs': lambda x, y: x.value_counts().sort_values(ascending=False).to_dict(),
                          'target_rate': lambda x, y: {v: y[x == v].mean() for v in x.unique()},
                          'most_frequent': lambda x, y: x.value_counts().sort_values(ascending=False).iloc[:1].to_dict()
                          }

        aggr_funcs_date = {'nans': lambda x: x.isna().sum(),
                           'max': lambda x: pd.to_datetime(x).max(),
                           'min': lambda x: pd.to_datetime(x).min(),
                           }

        print('Calculating main features\'s stats')
        
        for sample_name in ['train', 'oot']:
            if sample_name == 'oot' and self.data['oot'].shape[0] == 0:
                continue
            sample = self.data[sample_name]
            y_sample = sample[self.trg_feature]
            self.make_empty_dict(self.feature_stats[sample_name])
            for col in tqdm(self.long_list + self.score_list):
                series = sample[col]
                if col in self.feature_types['numerical']:
                    self.feature_stats[sample_name]['numerical'][col].update({k: v(series, y_sample) for k, v in aggr_funcs_numerical.items()})
                if col in self.feature_types['categorical']:
                    self.feature_stats[sample_name]['categorical'][col].update({k: v(series, y_sample) for k, v in aggr_funcs_cat.items()})
                if col in self.feature_types['date']:
                    self.feature_stats[sample_name]['date'][col].update({k: v(series) for k, v in aggr_funcs_date.items()})

        
        full = pd.concat(list(self.data.values()))
        if self.date_feature:
            print('Calculating features\'s stats grouped by dates')
            dates = np.sort(full[self.date_feature].unique())
            self.feature_stats.update({k: {} for k in dates})
            for date in tqdm(dates):
                self.make_empty_dict(self.feature_stats[date])
                sample = full[full[self.date_feature] == date]
                y_sample = sample[self.trg_feature]
                                                           
                for col in self.long_list + self.score_list:
                    series = sample[col]
                    if series.size > 0:
                        if col in self.feature_types['numerical']:
                            self.feature_stats[date]['numerical'][col].update({k: v(series, y_sample) for k, v in aggr_funcs_numerical.items()})
                        if col in self.feature_types['categorical']:
                            self.feature_stats[date]['categorical'][col].update({k: v(series, y_sample) for k, v in aggr_funcs_cat.items()})
                        if col in self.feature_types['date']:
                            self.feature_stats[date]['date'][col].update({k: v(series) for k, v in aggr_funcs_date.items()})
                    else:
                        if col in self.feature_types['numerical']:
                            self.feature_stats[date]['numerical'][col].update({k: np.nan for k, v in aggr_funcs_numerical.items()})
                        if col in self.feature_types['categorical']:
                            self.feature_stats[date]['categorical'][col].update({k: np.nan for k, v in aggr_funcs_cat.items()})
                        if col in self.feature_types['date']:
                            self.feature_stats[date]['date'][col].update({k: np.nan for k, v in aggr_funcs_date.items()})
    
    def calc_stats_non_target(self):
        """

        :return:
        """
        trg_funcs_numerical = {'nans': lambda x, y: x.isna().mean(),
                                'max': lambda x, y: x.max(),
                                'min': lambda x, y: x.min(),
                                'std': lambda x, y: x.std(),
                                'mean': lambda x, y: x.mean(),
                                'median': lambda x, y: x.median(),
                                'q25': lambda x, y: x.quantile(0.25),
                                'q75': lambda x, y: x.quantile(0.75),
                                'q05': lambda x, y: x.quantile(1 - 0.95),
                                'q95': lambda x, y: x.quantile(0.95),
                                'skewness': lambda x, y: x.skew()
                                }

        trg_funcs_cat = {'nunique': lambda x, y: x.nunique(),   
                         'most_frequent': lambda x, y: x.value_counts().sort_values(ascending=False).iloc[:1].to_dict()
                         }

        trg_funcs_date = {'max': lambda x: pd.to_datetime(x).max(),
                          'min': lambda x: pd.to_datetime(x).min(),
                          }

        print('Calculating main non-TARGET-features\'s stats')
                
        for sample_name in ['train', 'oot']:
            if sample_name == 'oot' and self.data['oot'].shape[0] == 0:
                continue
            sample = self.data[sample_name]
            sample = sample[sample[self.trg_feature] == 0]
            y_sample = sample[self.trg_feature]
            self.make_empty_dict(self.non_trg_stats[sample_name])
            for col in tqdm(self.long_list + self.score_list):
                series = sample[col]
                if col in self.feature_types['numerical']:
                    self.non_trg_stats[sample_name]['numerical'][col].update({k: v(series, y_sample) for k, v in trg_funcs_numerical.items()})
                if col in self.feature_types['categorical']:
                    self.non_trg_stats[sample_name]['categorical'][col].update({k: v(series, y_sample) for k, v in trg_funcs_cat.items()})
                if col in self.feature_types['date']:
                    self.non_trg_stats[sample_name]['date'][col].update({k: v(series) for k, v in trg_funcs_date.items()})

    
    def calc_stats_target(self):
        """

        :return:
        """
        trg_funcs_numerical = {'nans': lambda x, y: x.isna().mean(),
                                'max': lambda x, y: x.max(),
                                'min': lambda x, y: x.min(),
                                'std': lambda x, y: x.std(),
                                'mean': lambda x, y: x.mean(),
                                'median': lambda x, y: x.median(),
                                'q25': lambda x, y: x.quantile(0.25),
                                'q75': lambda x, y: x.quantile(0.75),
                                'q05': lambda x, y: x.quantile(1 - 0.95),
                                'q95': lambda x, y: x.quantile(0.95),
                                'skewness': lambda x, y: x.skew()
                                }
        trg_funcs_cat = {'nunique': lambda x, y: x.nunique(),   
                         'most_frequent': lambda x, y: x.value_counts().sort_values(ascending=False).iloc[:1].to_dict()
                         }

        trg_funcs_date = {'max': lambda x: pd.to_datetime(x).max(),
                          'min': lambda x: pd.to_datetime(x).min(),
                          }
                                                           
                                                           
        print('Calculating main TARGET-features\'s stats')
                                                           
        for sample_name in ['train', 'oot']:
            if sample_name == 'oot' and self.data['oot'].shape[0] == 0:
                continue
            sample = self.data[sample_name]
            sample = sample[sample[self.trg_feature] == 1]
            y_sample = sample[self.trg_feature]
            self.make_empty_dict(self.trg_stats[sample_name])
            for col in tqdm(self.long_list + self.score_list):
                series = sample[col]
                if col in self.feature_types['numerical']:
                    self.trg_stats[sample_name]['numerical'][col].update({k: v(series, y_sample) for k, v in trg_funcs_numerical.items()})
                if col in self.feature_types['categorical']:
                    self.trg_stats[sample_name]['categorical'][col].update({k: v(series, y_sample) for k, v in trg_funcs_cat.items()})
                if col in self.feature_types['date']:
                    self.trg_stats[sample_name]['date'][col].update({k: v(series) for k, v in trg_funcs_date.items()})

        

    def find_bad(self, prc_nans: float = 0.9, prc_most_frequent: float = 0.997,
                 thr_const_float: float = 0.001, prc_outliers: float = 0.5):
        """

        :param prc_nans:
        :param prc_most_frequent:
        :param thr_corr:
        :return:
        """
        for col in tqdm(self.long_list):
            series = self.data['train'][col]
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
            if self.feature_stats['train'][f_type][col]['nans'] / series.size > prc_nans:
                self.bad_feats.update({col: 'nans'})
                continue

            if f_type == 'numerical':
                #############################
                # CONST FEATURES numerical
                #############################
                s_norm = MinMaxScaler().fit_transform(series.values.reshape(-1, 1))
                if (np.std(s_norm) < thr_const_float) or \
                        (self.feature_stats['train']['numerical'][col]['q05'] ==
                         self.feature_stats['train']['numerical'][col]['q95']):
                    self.bad_feats.update({col: 'const_numerical'})
                    continue

                #############################
                # OUTLIERS numerical
                #############################
                Q1 = self.feature_stats['train'][f_type][col]['q25']
                Q3 = self.feature_stats['train'][f_type][col]['q75']
                low = Q1 - 1.5 * (Q3 - Q1)
                high = Q3 + 1.5 * (Q3 - Q1)
                if ((series > high) | (series < low)).sum() / series.size > prc_outliers:
                    self.bad_feats.update({col: 'outliers'})
                    continue

            #############################
            # Most frequent value percent
            #############################
            if f_type == 'categorical':
                if list(self.feature_stats['train'][f_type][col]['most_frequent'].values())[0] / series.size > prc_most_frequent:
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
        if feats_list is not None:
            best_numeric = feats_list
        elif top > 0 and top < 1:
           best_numeric = [col for col, vals in self.feature_stats['train']['numerical'].items() if
                            abs(vals['gini']) >= top]
        elif top % 1 == 0:
            best_numeric = pd.Series(
                {col: abs(vals['gini']) for col, vals in self.feature_stats['train']['numerical'].items()}). \
                               sort_values(ascending=False).index.to_list()[:top]



        stats = {}
        
        fig, axes = plt.subplots(nrows=len(best_numeric), ncols=len(stats_plot), figsize=(12, 8*len(best_numeric)//5))


        for ax, stat in zip(axes[0], stats_plot):
            ax.set_title(f'{stat} statistic')
        for ax, col in zip(axes[:, 0], best_numeric):
            ax.set_ylabel(col)


        for i, stat in enumerate(stats_plot):
            stats[stat] = {}
            for j, col in enumerate(best_numeric):
                stats[stat][col] = {date: self.feature_stats[date]['numerical'][col][stat] for date in
                                    self.feature_stats if date not in ['train', 'oot']}
                #stats[stat][col].update({date: self.feature_stats[date]['numerical'][col][stat] for date in
                #                         self.feature_stats if date not in ['train', 'oot']})

                ax = axes[j, i]
                plt.setp(ax.get_xticklabels(), rotation=kwargs.get('rotation'))
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.plot(list(stats[stat][col].keys()), list(stats[stat][col].values()), **dynamic_params)
                ax.axhline(self.feature_stats['train']['numerical'][col][stat],  linestyle='--')
        fig.tight_layout()
        plt.show()
        return stats

    def cat_hist_trg(self):
        for col in self.feature_types['categorical']:
            if col not in list(self.bad_feats.keys()):
                fig, ax = plt.subplots()
                plt.title(f'{col} feature: TARGET RATE statistics')
                x, y = zip(*sorted(self.trg_stats['train']['categorical'][col]['target_rate'].items()))
                ax.plot(x, y, color='yellow', label='Target Rate')
                ax.axhline(y=self.data['train'][self.trg_feature].mean(), color='r', linestyle='--', label='Mean Target Rate')
                pd.Series(self.feature_stats['train']['categorical'][col]['n_obs']).plot(kind='bar', secondary_y=True,
                                                                                        label='N obs')
                plt.show()

    def visualize(self, rc: dict = {}, style: str = 'seaborn', pie_params: dict = {}, dynamic_params: dict = {}, **kwargs):
        print('Pie chart of features types')

        plt.style.use(style)
        for k in rc:
            plt.rc(k, **rc[k])

        self.plot_pie(pie_params=pie_params, **kwargs)
        stats = self.plot_dynamics(dynamic_params=dynamic_params, **kwargs)
        #self.cat_hist_trg()

        return stats

    def run(self):
        self.define_types()
        self.calc_stats()
        self.calc_stats_target()
        self.calc_stats_non_target()
        self.feature_types['numerical'].remove(self.score)
        self.find_bad()

    def find_patterns(self):
        pass