import pandas as pd
from typing import Optional 
def open_dataframe(PATH:str)->pd.DataFrame:
    extension = PATH.split(".")[-1]
    if extension == "zip" or extension == "csv":
        df = pd.read_csv(PATH)
    elif extension == "pkl":
        df = pd.read_pickle(PATH)
    else:
        raise  "Error: unknown file type : ." + extension
    # Normalize columns name
    df.columns = df.columns.str.lower()
    if "close" not in df.columns:
        if "open_price" in df.columns:
            df["close"] = df["open_price"].shift(-1)
        else:
            raise "impossible to find an open price to create close price"
    return df

def y_build(df:pd.DataFrame,column:str)->pd.DataFrame:
    assert column in df.columns
    df[f"{column}_label"]=(df[column].diff(1)>0).astype(int)
    return df

def filter_dataframe_by_date_range(dataframe:pd.DataFrame,begin_date:Optional[str]=None,end_date:Optional[str]=None):
    """
    Args:
        dataframe (pd.Dataframe): the dataframe to be filtered
        begin_date (str or None): the starting date, or None if not specified
        end_date (str or None): the ending date, or None if not specified

    Returns:

    """
    if begin_date is not None and end_date is not None:
        df = dataframe[(dataframe['open_date'] >= begin_date) & (dataframe['open_date'] <= end_date)]
    elif begin_date is not None:
        df = dataframe[(dataframe['open_date'] >= begin_date)]
    elif end_date is not None:
        df = dataframe[(dataframe['open_date'] <= end_date)]
    else:
        df = dataframe
    return df.reset_index(drop=True)
if __name__=="__main__":
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
    df = filter_dataframe_by_date_range(y_build(open_dataframe(PATH=PATH),column="close"),begin_date="2023-04-03",end_date="2023-05-06")
    print(df.head())
