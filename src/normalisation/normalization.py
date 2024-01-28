import numpy as np
from typing import Optional,Union
import pandas as pd

class normalizer:
    @staticmethod    
    def log_division_normalization(df: pd.DataFrame,row: int = 0, value:bool=True, not_relative:bool=True):
        df = df.copy()
        if value:
            value = df.iloc[0,row] if df.iloc[0,row] !=0 else 1
            df = df.apply(lambda x: x / value) if not_relative else df.apply(lambda x: x / value) - 1
        else:
            df = df.apply(lambda x: x / df.iloc[:,row]) if not_relative else df.apply(lambda x: x / df.iloc[row]) - 1

        return np.log(df)
    

    
    