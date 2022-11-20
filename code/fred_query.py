# Author: Ruoyu Lin
# Docstring style: Sphinx

import os
import requests
import pandas as pd
from typing import List


# Define global utility variables

# Change this line if you saved your FRED API key under a different name
FRED_API_KEY = os.getenv("FRED_API_KEY") 
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations?"
OBSEVATION_FIELD_STR = "observations"
TICKERS = ["GS"+str(n) for n in [1, 2, 3, 5, 7, 10]]


def get_fred_series_df(series_id: str) -> pd.DataFrame:
    """
    Queries one FRED series via a ticker.

    :param series_id: the series_id identifier
    :type series_id: str
    :rtype: pd.DataFrame
    """

    # use requests library to query
    url = FRED_BASE_URL + \
        f"series_id={series_id}&api_key={FRED_API_KEY}" + \
        "&file_type=json"
    response = requests.get(url)

    # cast target field into pd.DataFrame
    df = pd.DataFrame(dict(response.json())[OBSEVATION_FIELD_STR])

    # rename value to be the series_id
    df.rename({"value": series_id}, axis=1, inplace=True)

    # drop everything but date and value
    df = df.loc[:, ["date", series_id]]

    # set date as index for later concat
    df.set_index("date", inplace=True)

    return df


def get_all_series(id_list: List[str], as_ret: bool = False) -> pd.DataFrame:
    """
    Get all series from FRED as specified in input and return a dataframe.

    :param id_list: a list of strs containing all pertinent series_id
    :type id_list: List[str]
    :param as_ret: a flag to specify whether to return returns rather than prices
    :type as_ret: bool
    :rtype: pd.DataFrame
    """
    # intialize an empty dataframe
    df = pd.DataFrame()

    # query and concatenate all series_id
    for id in id_list:
        df = df.merge(get_fred_series_df(id), how="outer",
                      left_index=True, right_index=True)

    # drop NaN values
    df.dropna(how="any", axis=0, inplace=True)

    # cast data to float64 for consistency
    df = df.astype("float64")

    if not as_ret:
        return df
    else:
        return df.pct_change().dropna(how="any", axis=0)

if __name__ == "__main__":
    df = get_all_series(TICKERS, as_ret=True)
    print(df)
