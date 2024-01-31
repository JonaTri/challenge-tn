import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)


def run_metrics(y_test, y_pred):
    return {  
        "MeanAE" : mean_absolute_error(y_test, y_pred),
        "MedAE" : median_absolute_error(y_test, y_pred),
        "R2" : r2_score(y_test, y_pred),
        "MSE" : mean_squared_error(y_test, y_pred),
        "RMSE" : np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    
class SimpleRegressionFeaturesTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, res_df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, res_df: pd.DataFrame, y=None) -> pd.DataFrame:
        self.retrive_datasets_for_training(res_df)
        models_informations = {}
        for idx in range(1, 4):
            lreg = LinearRegression()
            model_name = f"p0q0_hat_from_var_{idx}"
            lreg = lreg.fit(self.X_train[[f"p{idx}q0", f"p0q{idx}"]], self.y_train)
            self.X_train[model_name] = lreg.predict(self.X_train[[f"p{idx}q0", f"p0q{idx}"]])
            self.X_valid[model_name] = lreg.predict(self.X_valid[[f"p{idx}q0", f"p0q{idx}"]])
            self.X_test[model_name] = lreg.predict(self.X_test[[f"p{idx}q0", f"p0q{idx}"]])
            models_informations[model_name] = {
                "model" : lreg,
                "performance_valid" : run_metrics(self.y_valid, self.X_valid[model_name]),
                "performance_test" : run_metrics(self.y_test, self.X_test[model_name]),
            }
        return {
            "train_sets" : pd.concat([self.X_train, self.y_train], axis=1),
            "valid_sets" : pd.concat([self.X_valid, self.y_valid], axis=1),
            "test_sets" : pd.concat([self.X_test, self.y_test], axis=1),
            "simple_regression" : models_informations,
        } 
    
    def retrive_datasets_for_training(self, res_df):
        self.y_train = res_df["train_sets"]["p0q0"]
        self.X_train = res_df["train_sets"].drop(columns=["p0q0"])
        self.y_valid = res_df["valid_sets"]["p0q0"]
        self.X_valid = res_df["valid_sets"].drop(columns=["p0q0"])
        self.y_test = res_df["test_sets"]["p0q0"]
        self.X_test = res_df["test_sets"].drop(columns=["p0q0"])
