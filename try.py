import pandas as pd
import numpy as np
import os

columns = ['檔名', '時間', 'epoch', 'val_loss', '額外備註']

df = pd.read_csv('outcome.csv')

df.loc[len(df.index)] = [1,1,1,1, 1, 1,'e,e,e']

df.to_csv(rf'csvfile/{123}.csv', index=False)