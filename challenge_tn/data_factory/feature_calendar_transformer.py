from typing import Any, Optional, Union

import holidays
import numpy as np
import pandas as pd
from jours_feries_france import JoursFeries
from pandas.api.types import is_datetime64_any_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from vacances_scolaires_france import SchoolHolidayDates

from challenge_tn.data_factory.parameters import PARAMETERS

params = PARAMETERS["feature_calendar_transformer"]


class FeatureCalendarTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class which allow to transform a DataFrame with at least
    one date column into a dataframe with multiple calendar features generated
    from a specific date column.
    """

    def __init__(
        self,
        date_to_use: str,
        create_hour_features: bool = False,
        include_cyclic_transform: bool = False,
    ):
        """
        Instanciate a FeatureCalendarTransformer class instance.

        Parameters
        ----------
        date_to_use: str
            The column name to use to build calendar features.
        create_hour_feartures: bool
            Indicate if the user want to create hour feature.
        include_cyclic_transform: bool
            Indicate if the user want cyclic tranform of some feature.
        """
        self.date_to_use = date_to_use
        self.unix_value = params["EPOCH"]
        self.create_hour_features = create_hour_features
        if self.create_hour_features:
            self.unix_value = params["EPOCH_HOUR"]
        self.include_cyclic_transform = include_cyclic_transform
        self.date_format = "%Y-%m-%d"

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the input dataframe by adding calendar features generated from
        `date_to_use` attribute column.
        1. Prepare the `date_to_use` attribute value as column date and the
        attribute values to build calendar features.
        2. Create an array containing daily calendar feature.
        3. Create and add to the array hour calendar feature.
        4. Create and add to the array cyclic calendar feature, if
        `include_cyclic_transform` attribute value is set to True.
        5. Create and add to the array hour cyclic calendar feature, if
        `include_cyclic_transform` attribute value is set to True and
        create_hour_features is set to True.
        6. Concatenate the input DataFrame and the generated calendar feature.
        """
        X = self.initialize_date(X, True).reset_index(drop=True)
        self.df_bank_holidays = self.create_french_bank_holidays_df()
        features = np.array(
            [
                self.unix_second(),  # number of second since 1970-01-01 00:00:00
                np.array(self.dt.dayofweek) + 1,  # day of the week
                np.array(self.dt.day),  # day of the month
                np.array(self.dt.dayofyear),  # day of the year
                self.week_of_year(),  # week of the year
                np.array(self.dt.month),  # month of the year
                np.array(self.dt.quarter),  # quarter
                np.array(self.dt.year),  # year
                self.is_weekend(),  # is weekend
                self.french_bank_holiday(),  # french bank holiday
                *self.french_holiday(),  # french holiday in zone A, B or C, and at least in one of the zone
                self.european_central_bank_holidays(),  # european bank holiday based on TARGET2
                *self.days_since_french_bank_holidays(),  # distance in day until, since and close to french bank holiday
            ]
        )
        colnames_output = params["COLNAMES_FEATURES_CALENDAR"].copy()
        if self.create_hour_features:
            features_hour = np.array(
                [
                    np.array(self.dt.hour),  # hour of the day
                    self.hour_of_the_week(),  # hour of the week
                    self.hour_of_the_month(),  # hour of the month
                    self.hour_of_the_year(),  # hour of the year
                ]
            )
            features = np.append(features, features_hour, axis=0)
            colnames_output += params["COLNAMES_HOUR_FEATURES_CALENDAR"]
        if self.include_cyclic_transform:
            features_cyclic = self.__add_cyclic_transformation()
            features = np.append(features, features_cyclic, axis=0)
            colnames_output += params["COLNAMES_CYCLIC_TRANSFORM"]
            if self.create_hour_features:
                colnames_output += params["COLNAMES_HOUR_CYCLIC_TRANSFORM"]
        features = np.transpose(features)
        features = pd.DataFrame(features, columns=colnames_output)
        X = pd.concat([X, features], axis=1)
        return X

    def initialize_date(
        self, X: pd.DataFrame, return_output: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Process the dataframe and prepare the attribute values to build
        calendar feature.

        Parameters
        ----------
        X: pd.DataFrame
            Input DataFrame containing `date_to_use` attribute value as column
            date.
        return_output: bool
            Generate an output DataFrame with the process on `date_to_use`
            attribute value column date.
        """
        self._check_nan_presence(X[self.date_to_use])
        self._check_date_to_use_validity(X)
        self._check_datetime(X[self.date_to_use])
        self.date = X[self.date_to_use]
        self.dt = self.date.dt
        if return_output:
            return X

    def __add_cyclic_transformation(self) -> np.array:
        features_cyclic = np.array(
            [
                *self.__cyclic_transform(
                    self.dt.dayofyear, params["DAYS_IN_YEAR"]
                ),  # day of the year
                *self.__cyclic_transform(
                    self.dt.dayofweek + 1, params["DAYS_IN_WEEK"]
                ),  # day of the week
                *self.__cyclic_transform(
                    self.dt.month, params["MONTH_IN_YEAR"]
                ),  # month of the year
                *self.__cyclic_transform(
                    self.dt.day, params["DAYS_IN_MONTH"]
                ),  # day of the month
                *self.__cyclic_transform(
                    self.week_of_year(), params["WEEK_OF_YEAR"]
                ),  # week of the year
                *self.__cyclic_transform(
                    self.dt.quarter, params["QUARTER"]
                ),  # quarter of the year
            ]
        )
        if self.create_hour_features:
            features_hour = np.array(
                [
                    *self.__cyclic_transform(
                        self.dt.hour, params["HOURS_IN_DAY"]
                    ),  # hour of the day
                    *self.__cyclic_transform(
                        self.hour_of_the_week(),
                        params["DAYS_IN_WEEK"] * params["HOURS_IN_DAY"],
                    ),  # hour of the week
                    *self.__cyclic_transform(
                        self.hour_of_the_month(),
                        params["DAYS_IN_MONTH"] * params["HOURS_IN_DAY"],
                    ),  # hour of the month
                    *self.__cyclic_transform(
                        self.hour_of_the_year(),
                        params["DAYS_IN_YEAR"] * params["HOURS_IN_DAY"],
                    ),  # hour of the year
                ]
            )
            features_cyclic = np.append(features_cyclic, features_hour, axis=0)
        return features_cyclic

    @staticmethod
    def __cyclic_transform(unit: np.array, period: Union[float, int]) -> np.array:
        """
        Transform a calendar feature into two columns using a cyclic
        transformation taking account of the periodicity of the value.

        Parameters
        ----------
        unit: np.array
            The feature on which we apply the cyclic transform.
        period: Union[float, int]
            The periodicity of the unit input feature calendar.
            It match with the maximum value possible in unit.
        """
        return np.array(
            [
                np.sin(2 * np.pi * unit / period),
                np.cos(2 * np.pi * unit / period),
            ]
        )

    def hour_of_the_week(self) -> np.array:
        """Generate array with hour of the week from each values of the targeted datetime column"""
        return np.array(self.dt.dayofweek * params["HOURS_IN_DAY"] + self.dt.hour)

    def hour_of_the_month(self) -> np.array:
        """Generate array with hour of the month from each values of the targeted datetime column"""
        return np.array(self.dt.day * params["HOURS_IN_DAY"] + self.dt.hour)

    def hour_of_the_year(self) -> np.array:
        """Generate array with hour of the year from each values of the targeted datetime column"""
        return np.array(self.dt.dayofyear * params["HOURS_IN_DAY"] + self.dt.hour)

    def unix_second(self) -> np.array:
        """Generate array with the unix second from each values of the targeted datetime column"""
        return np.array(
            (self.date - pd.Timestamp(self.unix_value)) // pd.Timedelta("1s")
        )

    def is_weekend(self) -> np.array:
        """Generate array which is 1 if the date is in the weekend and 0 else for each values of the targeted datetime column"""
        return np.array((self.dt.dayofweek + 1 > params["WEEKEND_START"]).astype(int))

    def week_of_year(self) -> np.array:
        """Generate array with the week of year from each values of the targeted datetime column"""
        return np.array(self.dt.isocalendar().week.astype(float))

    def create_french_bank_holidays_df(self) -> pd.DataFrame:
        """Create a dataframe with all the french bank holidays from the years mentionned inside the targeted datetime column"""
        distinct_year = self.dt.year.unique()
        bank_holiday = np.array(
            [
                [value.strftime(self.date_format), 1]
                for year in distinct_year
                for value in JoursFeries.for_year(int(year)).values()
            ]
        )
        return pd.DataFrame(bank_holiday, columns=[self.date_to_use, "bank_holiday"])

    def french_bank_holiday(self) -> np.array:
        """Generate array with the french bank holiday from each values of the targeted datetime column"""
        serie_date_to_use = self.dt.strftime(self.date_format).astype(str)
        df_with_holidays = self.df_bank_holidays.merge(
            serie_date_to_use, on=[self.date_to_use], how="right"
        )
        df_with_holidays["bank_holiday"] = (
            df_with_holidays["bank_holiday"].fillna(0).astype(int)
        )
        return np.array(df_with_holidays["bank_holiday"])

    def days_since_french_bank_holidays(self) -> np.array:
        """Generate the distance in day since, until a french bank holidays, plus the minimum of the two, for each values of the targeted datetime column"""
        bank_holiday_date_colname = self.date_to_use + "_bh"
        df_bank_holidays_adapted = self.df_bank_holidays.rename(
            columns={self.date_to_use: bank_holiday_date_colname}
        ).sort_values(bank_holiday_date_colname)
        df_bank_holidays_adapted = pd.to_datetime(
            df_bank_holidays_adapted[bank_holiday_date_colname], format=self.date_format
        )
        date_df = self.dt.strftime(self.date_format).astype(str)
        date_df = pd.to_datetime(date_df, format=self.date_format)
        reference_df = date_df.sort_values().drop_duplicates()
        reference_df = pd.merge_asof(
            reference_df,
            df_bank_holidays_adapted,
            left_on=self.date_to_use,
            right_on=bank_holiday_date_colname,
            direction="forward",
        )
        reference_df = pd.merge_asof(
            reference_df,
            df_bank_holidays_adapted,
            left_on=self.date_to_use,
            right_on=bank_holiday_date_colname,
        )
        reference_df["days_until_next_french_bank_holiday"] = (
            reference_df.pop(bank_holiday_date_colname + "_x")
            .sub(reference_df[self.date_to_use])
            .dt.days
        )
        reference_df["days_since_previous_french_bank_holiday"] = (
            reference_df[self.date_to_use]
            .sub(reference_df.pop(bank_holiday_date_colname + "_y"))
            .dt.days
        )
        reference_df["distance_in_days_from_french_bank_holiday"] = reference_df[
            [
                "days_until_next_french_bank_holiday",
                "days_since_previous_french_bank_holiday",
            ]
        ].min(axis=1)
        date_df = reference_df.merge(date_df, on=[self.date_to_use], how="right")
        return np.array(
            [
                np.array(date_df["days_until_next_french_bank_holiday"]),
                np.array(date_df["days_since_previous_french_bank_holiday"]),
                np.array(date_df["distance_in_days_from_french_bank_holiday"]),
            ]
        )

    def european_central_bank_holidays(self) -> np.array:
        """Generate arrays with the european central bank holidays from each values of the targeted datetime column"""
        return np.array(
            self.date.apply(
                lambda datetime: datetime in holidays.EuropeanCentralBank()
            ).astype(int)
        )

    def french_holiday(self) -> np.array:
        """Generate arrays of holidays in zone A, B and C, plus at least in one zone, for each values of the targeted datetime column"""
        serie_date_to_use = self.dt.strftime(self.date_format).astype(str)
        dfh = (
            pd.DataFrame(SchoolHolidayDates().data)
            .T.drop(columns=["nom_vacances"])
            .rename(
                columns={
                    "date": self.date_to_use,
                    "vacances_zone_a": "holidays_zone_a",
                    "vacances_zone_b": "holidays_zone_b",
                    "vacances_zone_c": "holidays_zone_c",
                }
            )
        )
        list_of_holidays_zone = [
            "holidays_zone_a",
            "holidays_zone_b",
            "holidays_zone_c",
        ]
        dfh[self.date_to_use] = dfh[self.date_to_use].astype(str)
        dfh[list_of_holidays_zone] = dfh[list_of_holidays_zone].astype(int)
        dfh["is_holiday_all_zone"] = np.ones(dfh.shape[0])
        dfh = dfh.merge(serie_date_to_use, on=[self.date_to_use], how="right")
        dfh = dfh.fillna(0)
        return np.array(
            dfh[list_of_holidays_zone + ["is_holiday_all_zone"]].T.values.tolist()
        )

    def _check_date_to_use_validity(self, X):
        if self.date_to_use not in X.columns:
            raise ValueError(
                f"The columns `{self.date_to_use}` is not in the input "
                "DataFrame colnames."
            )

    @staticmethod
    def _check_nan_presence(date_serie: pd.Series):
        if date_serie.isnull().values.any():
            raise ValueError(
                "The column date contains NaN, you must process it before the "
                "use of the class."
            )

    @staticmethod
    def _check_datetime(date_serie: pd.Series):
        if not is_datetime64_any_dtype(date_serie):
            raise ValueError(
                "Incorrect date column: you must give a datetime column in input."
            )
