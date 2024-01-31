import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             precision_score, r2_score, recall_score)


def run_metrics(y_test, y_pred):
    return {  
        "MeanAE" : mean_absolute_error(y_test, y_pred),
        "MedAE" : median_absolute_error(y_test, y_pred),
        "R2" : r2_score(y_test, y_pred),
        "MSE" : mean_squared_error(y_test, y_pred),
        "RMSE" : np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    
def run_metrics_clf(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {
        "accuracy_score": accuracy_score(y_true, y_pred),
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1_score(y_true, y_pred),
        f"f{0.5}_score": ((1 + 0.5 ** 2) * precision * recall) / (0.5 ** 2 * precision + recall),
        f"f{2}_score": ((1 + 2 ** 2) * precision * recall) / (2 ** 2 * precision + recall),
    }
    
class AnomalicLoadingFeaturesTransformer(BaseEstimator, TransformerMixin):
    
    COLS_MIN_LOADING = ["p0q1", "p0q2", "p0q3", "station", "info_missing_p0", "start_p0", "station_variation_p0", "p0_variation_min", "p0_variation_max", "std_p0"]
    COLS_MAX_LOADING = ["p0q1", "p0q2", "p0q3", "p1q0", "p2q0", "p3q0", "mean_p0", "mean_q0", "station", "p0_variation_min", "station_variation_p0"]
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, res_df: pd.DataFrame, y=None) -> pd.DataFrame:
        self.retrive_datasets_for_training(res_df)
        models_informations = {}
        for model_name in ["inferior_anomalic_loading", "superior_anomalic_loading"]:
            cols_to_keep = self.COLS_MIN_LOADING if model_name == "inferior_anomalic_loading" else self.COLS_MAX_LOADING
            logreg = LogisticRegression()
            logreg = logreg.fit(self.X_train.filter(cols_to_keep), self.X_train[model_name])
            predict_name = model_name + "_hat"
            y_valid_hat = logreg.predict(self.X_valid.filter(cols_to_keep))
            y_test_hat = logreg.predict(self.X_test.filter(cols_to_keep))
            models_informations[predict_name] = {
                "model" : logreg,
                "performance_valid" : run_metrics_clf(self.X_valid[model_name], y_valid_hat),
                "performance_test" : run_metrics_clf(self.X_test[model_name], y_test_hat),
            }
            self.X_valid[model_name] = y_valid_hat
            self.X_test[model_name] = y_test_hat
        return {**res_df , **{"anomalic_loading" : models_informations}}
    
    def retrive_datasets_for_training(self, res_df):
        self.y_train = res_df["train_sets"]["p0q0"]
        self.X_train = res_df["train_sets"].drop(columns=["p0q0"])
        self.y_valid = res_df["valid_sets"]["p0q0"]
        self.X_valid = res_df["valid_sets"].drop(columns=["p0q0"])
        self.y_test = res_df["test_sets"]["p0q0"]
        self.X_test = res_df["test_sets"].drop(columns=["p0q0"])
