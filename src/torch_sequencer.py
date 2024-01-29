import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils_df import y_build,open_dataframe,filter_dataframe_by_date_range
from normalisation.normalization import normalizer
from typing import Optional
class AtomicSequencer(Dataset):
    def __init__(self, PATH,begin_date:Optional[str]=None,end_date:Optional[str]=None,time_steps=64):
        self.column = "close"
        self.df = filter_dataframe_by_date_range(y_build(open_dataframe(PATH=PATH),column=self.column),begin_date=begin_date,end_date=end_date)
        self.column_names = ["open", "high", "low", "close"]
        self.n = len(self.df)
        # Define a list of valid indexes based on the specified margin
        self.time_steps = time_steps
        self.valid_indexes = range(self.time_steps+1,self.n-1-self.time_steps)

    def __len__(self):
        return len(self.valid_indexes)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indexes[idx]
        data = normalizer.log_division_normalization(df=self.df[self.column_names].iloc[actual_idx-self.time_steps:actual_idx],row=0)
        x = torch.tensor(data.values,dtype=torch.float32)
        y = torch.tensor(self.df[f"{self.column}_label"].iloc[actual_idx],dtype=torch.float32)
        return x,y
    
if __name__=="__main__":
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
    Sequencer = AtomicSequencer(PATH=PATH)
    item = Sequencer.__getitem__(42)
    print(item)