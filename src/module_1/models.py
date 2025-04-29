from typing import Dict, Any, Literal
import pandas as pd


# Type aliases for better readability
JSONType = Dict[str, Any]
DataFrameType = pd.DataFrame
FrequencyType = Literal[
    "B",  # business day frequency
    "C",  # custom business day frequency (experimental)
    "D",  # calendar day frequency
    "W",  # weekly frequency
    "M",  # month end frequency
    "SM",  # semi-month end frequency (15th and end of month)
    "BME",  # business month end frequency
    "CBM",  # custom business month end frequency
    "MS",  # month start frequency
    "SMS",  # semi-month start frequency (1st and 15th)
    "BMS",  # business month start frequency
    "CBMS",  # custom business month start frequency
    "Q",  # quarter end frequency
    "BQ",  # business quarter endfrequency
    "QS",  # quarter start frequency
    "BQS",  # business quarter start frequency
    "A",  # year end frequency
    "BA",  # business year end frequency
    "BY",  # business year end frequency (alias)
    "AS",  # year start frequency
    "YS",  # year start frequency (alias)
    "BAS",  # business year start frequency
    "BYS",  # business year start frequency (alias)
    "BH",  # business hour frequency
    "h",  # hourly frequency
    "T",  # minutely frequency
    "min",  # minutely frequency (alias)
    "S",  # secondly frequency
    "L",  # milliseconds
    "ms",  # milliseconds (alias)
    "U",  # microseconds
    "us",  # microseconds (alias)
    "N",  # nanoseconds
]
