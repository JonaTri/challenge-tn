PARAMETERS = {
    # For the prepation of the output
    "feature_calendar_transformer": {
        "EPOCH_HOUR": "1970-01-01 00:00:00",
        "EPOCH": "1970-01-01",
        "HOURS_IN_DAY": 24,
        "DAYS_IN_WEEK": 7,
        "DAYS_IN_MONTH": 31,
        "DAYS_IN_YEAR": 366,
        "WEEKEND_START": 5,
        "WEEK_OF_YEAR": 53,
        "MONTH_IN_YEAR": 12,
        "QUARTER": 4,
        "COLNAMES_FEATURES_CALENDAR": [
            "unix_second",
            "day_of_the_week",
            "day_of_the_month",
            "day_of_the_year",
            "week_of_the_year",
            "month_of_the_year",
            "quarter",
            "year",
            "is_weekend",
            "french_bank_holiday",
            "french_holiday_zone_a",
            "french_holiday_zone_b",
            "french_holiday_zone_c",
            "french_holiday_zone_at_least_in_one_zone",
            "european_bank_holiday_target2",
            "days_until_next_french_bank_holiday",
            "days_since_previous_french_bank_holiday",
            "distance_in_days_from_french_bank_holiday",
        ],
        "COLNAMES_HOUR_FEATURES_CALENDAR": [
            "hour_of_the_day",
            "hour_of_the_week",
            "hour_of_the_month",
            "hour_of_the_year",
        ],
        "COLNAMES_CYCLIC_TRANSFORM": [
            "day_of_the_year_sin",
            "day_of_the_year_cos",
            "day_of_the_week_sin",
            "day_of_the_week_cos",
            "month_of_the_year_sin",
            "month_of_the_year_cos",
            "day_of_the_month_sin",
            "day_of_the_month_cos",
            "week_of_the_year_sin",
            "week_of_the_year_cos",
            "quarter_sin",
            "quarter_cos",
        ],
        "COLNAMES_HOUR_CYCLIC_TRANSFORM": [
            "hour_of_the_day_sin",
            "hour_of_the_day_cos",
            "hour_of_the_week_sin",
            "hour_of_the_week_cos",
            "hour_of_the_month_sin",
            "hour_of_the_month_cos",
            "hour_of_the_year_sin",
            "hour_of_the_year_cos",
        ],
    }
}
