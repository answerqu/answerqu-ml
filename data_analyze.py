import pandas as pd
from typing_extensions import NoReturn


class GenData:
    def __init__(self, data: dict = {}, trg_feature: str = None, date_feature: str = None, id_feature: str = None,
                 meta_features: list = []) -> NoReturn:
        self.train = data['train']
        self.valid = data['valid']

        self.date_feature = date_feature
        self.trg_feature = trg_feature
        self.id_feature = id_feature
        self.meta_features = meta_features
        self.long_list = [col for col in self.train.columns.to_list() if col not in
                          [self.date_feature, self.trg_feature, self.id_feature] + self.meta_features]

        self.task_type = 'binary' if self.train[date_feature].nunique() == 2 else 'multi'

    def define_types(self):
        pass

    def calc_stats(self):
        pass

    def calc_stats_target(self):
        pass

    def find_bad(self):
        pass

    def vizualize(self):
        pass

    def find_patterns(self):
        pass