from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def check_columns_presence(df: pd.DataFrame, cols_to_check: List[str]) -> None:
    if any([col not in df.columns for col in cols_to_check]):
        raise ValueError(f"The dataframe must contain {cols_to_check} columns.")


def check_no_missing_values_cols(
    df: pd.DataFrame, cols_to_check: Optional[List[str]] = None
) -> None:
    if cols_to_check is None:
        if df.isnull().any(axis=None):
            raise ValueError("The dataset must not have any missing values.")
    else:
        if df.filter(cols_to_check).isnull().any(axis=None):
            raise ValueError(
                f"The dataset {cols_to_check} columns must not contain any missing values."
            )


def get_abs_delta(
    first_term: Union[float, int], second_term: Union[float, int]
) -> Union[float, int]:
    return np.abs(first_term - second_term)


def get_one_row_df_from_dict_with_multiple_lengths_values(
    dict_to_convert: Dict[str, Any]
) -> pd.DataFrame:
    """
    Convert a dictionary with values of different lengths into a one-row DataFrame.
    Return the resulting DataFrame.
    """
    df = pd.DataFrame.from_dict(dict_to_convert, orient="index")
    df = df.T

    return df



def check_df_is_sorted(
    df: pd.DataFrame, sorting_cols: List[str], ascending: Union[bool, List[bool]]
) -> None:
    """
    Check that `df` is sorted by `sorting_cols`.
    """
    if df.index.duplicated().any():
        raise ValueError("Dataframe index must not contain any duplicates.")
    test_index = df.sort_values(sorting_cols, kind="stable", ascending=ascending).index
    if (df.index != test_index).any():
        raise ValueError(f"The dataframe must have been sorted by {sorting_cols}.")

