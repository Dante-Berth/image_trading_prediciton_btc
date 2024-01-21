import pandas as pd 
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
if __name__=="__main__":
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
    df = y_build(open_dataframe(PATH=PATH),column="close")
    print(df.head())
