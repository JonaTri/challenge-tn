from typing import Any, List, Optional, TypedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from challenge_tn.modules.models.data_transformers import (
    check_columns_presence,
    check_df_is_sorted,
    check_no_missing_values_cols,
    get_abs_delta,
)


class SplitDataSet(TypedDict):
    train_sets: List[pd.DataFrame]
    valid_sets: List[pd.DataFrame]


class SplitTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to split a DataFrame into a set of train sets and associated validation sets.

    Parameters
    ----------
    train_size: float
        Proportion of the train set. Must be in ]0,1[.
        The validation set proportion is computed as (1 - `train_size`).
    splitting_keys: Optional[List[str]] = None
        Columns to split by.
        If set to None, a random split is performed.
    n_splits: int = 100
        Number of candidate splits to perform.
        For small `n_splits` values, not all the possible splits combinations may be computed.
        If so, and no valid split is found, `n_splits` can be increased to look among more
        combination candidates.
    tol: float = 0.05
        Splitting tolerance level. Must be in [0,1].
        A splitting is valid if its train set size is in
        [`train_size` - `tol`, `train_size` + `tol`].
    random_state: int = 123
        Random seed state. Allows, for instance, for a set of given inputs and parameters, to
        always get the same outputs.
    """

    def __init__(
        self,
        train_size: float,
        splitting_keys: Optional[List[str]] = None,
        n_splits: int = 100,
        tol: float = 0.05,
        random_state: int = 123,
    ):
        """
        Instanciate a SplitTransformer class.
        Assign `train_size`, `splitting_keys`, `n_splits`, `tol` and `random_state` values to
        instance attributes.
        """
        self.__parameters_validity_check(train_size, tol)
        self.train_size = train_size
        self.splitting_keys = splitting_keys
        self.n_splits = n_splits
        self.tol = tol
        self.random_state = random_state

    def __parameters_validity_check(self, train_size: float, tol: float) -> None:
        """
        Check for class instanciation parameters validity. Validity criteria are described in the
        class docstring.
        """
        if not (0 < train_size < 1):
            raise (f"`train_size` must be in ]0, 1[.")
        if not (0 <= tol <= 1):
            raise (f"`tol` must be in [0, 1].")

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        """
        Unused method. Necessary for TransformerMixin daughter classes.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> SplitDataSet:
        """
        Split the input `X` dataset into a list of train sets and a list of associated validation sets.

        If the `splitting_keys` attribute is set to None, a random split is performed.
        Otherwise, `X` is split by `splitting_keys`. The train sets and the associated validation sets
        display exclusive `splitting_keys` columns values (no overlapping).

        If no computed split match the `train_set` and `tol` criteria, one can increase `n_splits`
        and/or `tol` values.

        For each train and validation sets couples, if multiple splits match the `train_set` and `tol`
        criteria, keep the one minimizing the absolute difference between the train set size and
        `train_size` attribute value.

        Return the resulting split datasets as a SplitDataSet TypedDict.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame to split.
            If the `splitting_keys` attribute is not set to None, `X` must :
            - Contain `splitting_keys` columns
            - Display no `splitting_keys` missing values
            - Be ascendingly sorted by `splitting_keys`

        y: Optional[Any] = None
            Unused argument. Necessary for transform methods of TransformerMixin daughter classes.

        Returns
        -------
        SplitDataSet
            TypedDict with 2 keys:
            - "train_sets": list of train sets
            - "valid_sets": list of the associated validation sets
        """
        self.__check_X_validity(X)
        if self.splitting_keys is None:
            splitter = ShuffleSplit(
                n_splits=self.n_splits, train_size=self.train_size, random_state=self.random_state
            )
            split_idxs = list(splitter.split(X))
        else:
            groups = X[self.splitting_keys].apply(tuple, axis=1).values
            splitter = GroupShuffleSplit(
                n_splits=self.n_splits, train_size=self.train_size, random_state=self.random_state
            )
            split_idxs = list(splitter.split(X, groups=groups))
        full_dataset_size = len(X)
        # Keep only valid indexes
        split_idxs = [
            idxs for idxs in split_idxs if self.__is_trainset_idx_valid(idxs[0], full_dataset_size)
        ]
        if not split_idxs:
            raise ValueError(
                "No eligible splits for the given train/validation sizes. Try to increase `n_splits` and/or `tol`."
            )
        # Keep the index set which minimizes the absolute difference between train set size and
        # self.train_size
        size_deltas_list = [
            get_abs_delta(len(indexes[0]) / full_dataset_size, self.train_size)
            for indexes in split_idxs
        ]
        train_index, valid_index = split_idxs[np.argmin(size_deltas_list)]

        train_sets = X.iloc[train_index]
        valid_sets = X.iloc[valid_index]

        return {
            "train_sets": train_sets,
            "valid_sets": valid_sets,
        }

    def __check_X_validity(self, X: pd.DataFrame) -> None:
        """
        Check if DataFrame `X` meets structure requirements.
        See `transform` method docstring for requirements descriptions.
        """
        if self.splitting_keys is not None:
            check_columns_presence(X, self.splitting_keys)
            check_no_missing_values_cols(X, self.splitting_keys)
            check_df_is_sorted(X, self.splitting_keys, ascending=True)

    def __is_trainset_idx_valid(self, trainset_idx: pd.Index, full_dataset_size: int) -> bool:
        """
        Check if `trainset_idx` matches `train_size` and `tol` attributes criteria.
        I.e. if `trainset_idx` relative length is in [`train_size` - `tol`, `train_size` + `tol`].

        Return the resulting boolean.
        """
        valid_index = False
        idx_proportion = len(trainset_idx) / full_dataset_size
        if (idx_proportion >= self.train_size - self.tol) and (
            idx_proportion <= self.train_size + self.tol
        ):
            valid_index = True

        return valid_index
