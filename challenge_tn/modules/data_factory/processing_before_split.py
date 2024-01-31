import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class ProcessingBeforeSplitTransformer(BaseEstimator, TransformerMixin):
    
    COLUMNS_TO_KEEP = []
    # COLUMNS_TO_DROP = ["full_date", "composition", "way", "hour"]
    COLUMNS_TO_DROP = ["date", "full_date", "composition", "way", "hour"]

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    # TODO : 
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.drop(columns=self.COLUMNS_TO_DROP)
        self.station_le = LabelEncoder()
        X["station"] = self.station_le.fit_transform(X["station"])
        self.station_le_mapping = {elt : idx for idx, elt in enumerate(self.station_le.classes_)}
        return X