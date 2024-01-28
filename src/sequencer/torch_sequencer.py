import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils_df import y_build,open_dataframe
from typing import Optional
class AtomicSequencer(Dataset):
    def __init__(self, PATH,begin_date:Optional[str]=None,end_date:Optional[str]=None,time_steps=128):
        raise "To be implemented"
    def __len__(self):
        return len(self.valid_indexes)
    def __getitem__(self, idx):
        raise "To be implemented"