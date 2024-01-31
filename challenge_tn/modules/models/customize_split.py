from typing import List, Optional

from challenge_tn.modules.models.split import SplitTransformer


class OptimizationSplitTransformer(SplitTransformer):
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.sort_values(self.splitting_keys)
        sets_dict_first_split = self.fit(X).transform(X)
        valid_df = sets_dict_first_split["valid_sets"].copy()
        self.init_train_size = self.train_size
        self.train_size = 0.5
        sets_dict_second_split = self.fit(valid_df).transform(valid_df)
        self.train_size = self.init_train_size
        return {
            "train_sets" : sets_dict_first_split["train_sets"],
            "valid_sets" : sets_dict_second_split["train_sets"],
            "test_sets" : sets_dict_second_split["valid_sets"],
        }