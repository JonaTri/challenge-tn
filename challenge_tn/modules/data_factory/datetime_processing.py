import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from challenge_tn.data_factory.feature_calendar_transformer import \
    FeatureCalendarTransformer


class CleanHourNaNTransformer(BaseEstimator, TransformerMixin):
    
    HOUR_PERIMETER = ["06:00:00", "07:00:00", "08:00:00", "09:00:00"]

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X["hour_float"] = X["hour"].str[:2].astype(float)
        train_hour_min_df = X.groupby(["date", "train"])["hour_float"].min().apply(lambda x: f"{str(int(x)).zfill(2)}:00:00").reset_index(name="hour_float_min")
        X = X.merge(train_hour_min_df, on=["date", "train"], how="left")
        where_first_station = X["p0q1"].isnull() & X["p0q2"].isnull() & X["p0q3"].isnull()
        X.loc[X["hour"].isnull() & where_first_station, "hour"] = X.loc[X["hour"].isnull() & where_first_station, "hour_float_min"]
        X = X[X["hour"].isin(self.HOUR_PERIMETER)]
        X.dropna(subset=["hour"], inplace=True)
        X.drop(columns=["hour_float", "hour_float_min"], inplace=True)
        return X
    
from typing import List


class ExtractFeatureCalendarTransformer(FeatureCalendarTransformer):

    CREATED_COLUMNS_TO_DROP: List[str] = [
            "unix_second", "year", "french_holiday_zone_a", "french_holiday_zone_b", "french_holiday_zone_c", 
            "is_weekend", "french_bank_holiday", "french_holiday_zone_at_least_in_one_zone", 
            "days_since_previous_french_bank_holiday", "hour_of_the_week", "hour_of_the_year"
        ]
    
    def __init__(self, include_cyclic_transform: bool = False):
        super().__init__("full_date", True, include_cyclic_transform)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.date_to_use = "full_date"
        self.create_hour_features = True
        X = X.assign(
            full_date=lambda x: x["date"] + " " + x["hour"],
        )
        X["full_date"] = pd.to_datetime(X["full_date"], format="%Y-%m-%d %H:%M:%S")
        X = self.fit(X).transform(X)
        X = X.drop(columns=self.CREATED_COLUMNS_TO_DROP)
        return X