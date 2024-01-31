
import itertools
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PqFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    
    P_COLUMN_NAMES: List = ["p0q3", "p0q1", "p0q2"]
    Q_COLUMN_NAMES: List = ["p3q0", "p1q0", "p2q0"]
    PQ_COLUMN_NAMES: List = ["p0q3", "p0q1", "p0q2", "p3q0", "p1q0", "p2q0"]

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.build_variation_pq_feature(X, "p0")
        X = self.build_variation_pq_feature(X, "q0")
        X = self.build_variation_pq_per_day_station(X)
        X = self.build_stats_features_on_pq(X)
        X = self.extract_nan_information_from_pq_features(X)
        X = self.replace_nan_by_zero(X)
        return X

    def build_variation_pq_feature(self, df: pd.DataFrame, which_target: str="p0")->pd.DataFrame: 
        if which_target == "p0":
            n1, n2, n3 = f"{which_target}q1", f"{which_target}q2", f"{which_target}q3"
        elif which_target == "q0":
            n1, n2, n3 = f"p1{which_target}", f"p2{which_target}", f"p3{which_target}"
        else:
            raise ValueError("'q0' or 'p0'")
        df[f"{which_target}_list"] = list(zip(df[n1].fillna(0), df[n2].fillna(0), df[n3].fillna(0)))
    
        df[f"{which_target}_combination_list"] = [
            list(itertools.combinations(elt, 2))
            for elt in df[f"{which_target}_list"]
        ]
        df[f"{which_target}_variation_max"] = [
            np.max([abs(a - b) for a, b in elt])
            for elt in df[f"{which_target}_combination_list"]
        ]
        df[f"{which_target}_variation_min"] = [
            np.min([abs(a - b) for a, b in elt])
            for elt in df[f"{which_target}_combination_list"]
        ]
        df.drop(columns=[f"{which_target}_list", f"{which_target}_combination_list"], inplace=True)
        return df
    
    def build_variation_pq_per_day_station(self, df: pd.DataFrame)->pd.DataFrame:
        variation_per_day_station = df.groupby(["station", "day_of_the_week", "hour"]).agg({"q0_variation_max" : np.mean, "q0_variation_min" : np.mean, "p0_variation_max" : np.mean, "p0_variation_min": np.mean}).reset_index()
        variation_per_day_station["station_variation_p0"] = variation_per_day_station[["p0_variation_max", "p0_variation_min"]].mean(axis=1)
        variation_per_day_station["station_variation_q0"] = variation_per_day_station[["q0_variation_max", "q0_variation_min"]].mean(axis=1)
        variation_per_day_station = variation_per_day_station[["station", "day_of_the_week", "hour", "station_variation_p0", "station_variation_q0"]]
        df = df.merge(variation_per_day_station, on=["station", "day_of_the_week", "hour"], how="left")
        return df
    
    def build_stats_features_on_pq(self, df: pd.DataFrame)->pd.DataFrame:
        df["mean_p0"] = df[self.P_COLUMN_NAMES].mean(axis=1, skipna=True).fillna(0)
        df["mean_q0"] = df[self.Q_COLUMN_NAMES].mean(axis=1, skipna=True).fillna(0)
        df["median_p0"] = df[self.P_COLUMN_NAMES].median(axis=1, skipna=True).fillna(0)
        df["median_q0"] = df[self.Q_COLUMN_NAMES].median(axis=1, skipna=True).fillna(0)
        df["std_p0"] = df[self.P_COLUMN_NAMES].std(axis=1, skipna=True).fillna(0)
        df["std_q0"] = df[self.Q_COLUMN_NAMES].std(axis=1, skipna=True).fillna(0)
        return df
    
    def extract_nan_information_from_pq_features(self, df: pd.DataFrame)->pd.DataFrame:
        df["info_missing_p0"] = df["p0q1"].isnull().astype(int) + df["p0q2"].isnull().astype(int) + df["p0q3"].isnull().astype(int)
        df["info_missing_q0"] = df["p1q0"].isnull().astype(int) + df["p2q0"].isnull().astype(int) + df["p3q0"].isnull().astype(int)
        df["start_q0"] = (df["info_missing_q0"] == 3).astype(int)
        df["start_p0"] = (df["info_missing_p0"] == 3).astype(int)
        return df
    
    def replace_nan_by_zero(self, df: pd.DataFrame)->pd.DataFrame:
        for pq_col in self.PQ_COLUMN_NAMES:
            df[pq_col] = df[pq_col].fillna(0)
        return df
    
class StationFeatureTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.build_nb_distinct_station_per_train(X)
        return X
    
    def build_nb_distinct_station_per_train(self, df: pd.DataFrame) -> pd.DataFrame:
        station_per_train = df.groupby(["train"]).agg({"station": "nunique"}).reset_index().rename(columns={"station" : "nb_distinct_station_per_train"})
        df = df.merge(station_per_train, on=["train"], how="left")
        return df
    
class AnomalicLoadingTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        mini = np.mean(X["p0q0"]) - np.std(X["p0q0"]) * 1.25
        maxi = np.mean(X["p0q0"]) + np.std(X["p0q0"]) * 1.5
        X["superior_anomalic_loading"] = (X["p0q0"] > maxi).astype(int) # 9 %
        X["inferior_anomalic_loading"] = (X["p0q0"] < mini).astype(int) # 7.6 %
        return X 