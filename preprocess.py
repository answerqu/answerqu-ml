import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook as tqdm
from datetime import date, timedelta
import os
from data_analyze import gini


class PreprocessData:
    def __init__(self, gen_data_obj = None, nan_val_num: int = None, nan_val_cat: int = None,
                 sampling_target_rate: float = None):
        self.gen_data = gen_data_obj

        self.train = gen_data_obj.train
        self.valid = gen_data_obj.valid

        self.nan_val_num = nan_val_num
        self.nan_val_cat = nan_val_cat

        self.sampling_target_rate = sampling_target_rate

    def remove_bad(self):
        self.train = self.gen_data.train.drop(columns=self.gen_data.bad_feats)
        self.valid = self.gen_data.valid.drop(columns=self.gen_data.bad_feats)

        self.gen_data.feature_types['categorical'] = [col for col in self.gen_data.feature_types['categorical'] if col not in self.gen_data.bad_feats]
        self.gen_data.feature_types['numerical'] = [col for col in self.gen_data.feature_types['numerical'] if col not in self.gen_data.bad_feats]
        self.gen_data.feature_types['date'] = [col for col in self.gen_data.feature_types['date'] if col not in self.gen_data.bad_feats]


    def encode(self):
        print('Encoding object type features...')
        for col in tqdm(self.gen_data.feature_types['categorical']):
            if self.train[col].dtype == 'O':
                le = LabelEncoder()
                s = self.train[col].fillna('NaN').replace(np.nan, 'NaN')
                self.train[col] = le.fit_transform(self.train[col])
                try:
                    self.valid[col] = le.transform(self.valid[col])
                except:
                    self.valid[col] = le.transform(self.valid[col].map(lambda x: x if x in list(s.unique())
                                                                       else np.nan))

    def encode_dates(self):
        print('Encoding dates...')
        date_train = self.train[self.gen_data.date_feature]
        date_valid = self.valid[self.gen_data.date_feature]
        for col in tqdm(self.gen_data.feature_types['date']):
            if self.train[col].dtype == 'O' or 'date' in str(self.train[col].dtype):
                self.train[col] = (date_train - pd.to_datetime(self.train[col])).days // 30.5 # may be error
                self.valid[col] = (date_valid - pd.to_datetime(self.valid[col])).days // 30.5 # may be error

        self.gen_data.feature_types['numerical'] += self.gen_data.feature_types['date']
        self.gen_data.feature_types['date'] = []
        old_long = self.gen_data.long_list.copy()
        self.gen_data.long_list = self.gen_data.feature_types[  'date']
        self.gen_data.calc_stats()
        self.gen_data.calc_stats_target()

        for group in self.gen_data.feature_stats:
            try:
                del self.gen_data.feature_stats[group]['date']
            except:
                pass
        for group in self.gen_data.trg_stats:
            try:
                del self.gen_data.trg_stats[group]['date']
            except:
                pass
        del self.gen_data.feature_types['date']

        self.gen_data.long_list = old_long

    def encode_enum(self):
        #for col in tqdm(self.gen_data.feature_types['categorical']):
        #    if self.train[col].dtype == 'O':
        #        if self.train[col].str.count(',').sum()/self.train.shape[0] > 0.5:
        pass

    def encode_nans(self):
        print('Encoding nans...')
        for col in tqdm(self.gen_data.long_list):
            if col in self.gen_data.feature_types['numerical'] and self.nan_val_num is not None:
                self.train[col] = self.train[col].map(lambda x: np.nan if x == self.nan_val_num else x)
                self.valid[col] = self.valid[col].map(lambda x: np.nan if x == self.nan_val_num else x)
            elif col in self.gen_data.feature_types['categorical'] and self.nan_val_cat is not None:
                self.train[col] = self.train[col].map(lambda x: np.nan if x == self.nan_val_cat else x)
                self.valid[col] = self.valid[col].map(lambda x: np.nan if x == self.nan_val_cat else x)

    def sampling(self):
        if self.sampling_target_rate is not None:
            pos = self.train[self.train[self.gen_data.trg_feature] == 1]
            neg = self.train[self.train[self.gen_data.trg_feature] == 0]
            n_pos = pos.shape[0]
            n_neg = neg.shape[0]
            m = self.train[self.gen_data.trg_feature].mean()
            frac = (m*(n_pos+n_neg) - self.sampling_target_rate*n_pos)/(self.sampling_target_rate*n_neg)
            sample_neg = neg.sample(frac=frac, random_state=42)
            self.train = pd.concat([pos, sample_neg]).sample(frac=1, random_state=42)

    def reduce_mem(self):
        print('TRAIN memory usage before:', np.round(self.train.memory_usage().sum() // 1024 / 1024, 2), 'Mb')
        print('VALID memory usage before:', np.round(self.valid.memory_usage().sum() // 1024 / 1024, 2), 'Mb')
        print('Changing categorical features type...')
        for col in tqdm(self.gen_data.feature_types['categorical']):
            s_train = self.train[col]
            s_valid = self.valid[col]
            if max(len(s_train.unique()), len(s_valid.unique()), abs(s_train).max(), abs(s_valid).max()) < 2**(8-1):
                if str(s_train.dtype) != 'int8':
                    s_train = s_train.astype('int8')
                    s_valid = s_valid.astype('int8')
            elif max(len(s_train.unique()), len(s_valid.unique()), abs(s_train).max(), abs(s_valid).max()) < 2**(16-1):
                if str(s_train.dtype) != 'int16':
                    s_train = s_train.astype('int16')
                    s_valid = s_valid.astype('int16')
            elif max(len(s_train.unique()), len(s_valid.unique()), abs(s_train).max(), abs(s_valid).max()) < 2**(32-1):
                if str(s_train.dtype) != 'int32':
                    s_train = s_train.astype('int32')
                    s_valid = s_valid.astype('int32')
            else:
                if str(s_train.dtype) != 'int64':
                    s_train = s_train.astype('int64')
                    s_valid = s_valid.astype('int64')
            self.train[col] = s_train
            self.valid[col] = s_valid

        print('TRAIN memory usage after:', np.round(self.train.memory_usage().sum() // 1024 / 1024, 4), 'Mb')
        print('VALID memory usage after:', np.round(self.valid.memory_usage().sum() // 1024 / 1024, 4), 'Mb')

    def run(self):
        self.remove_bad()
        self.encode()
        #self.encode_dates()
        #self.encode_enum()
        self.encode_nans()
        self.sampling()
        self.reduce_mem()