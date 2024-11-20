import pandas as pd
import numpy as np
import os

df1 = pd.read_csv('upload(no answer).csv')

df2 = pd.read_csv('csvfile/template/2024-11-17T15_59_13Z.csv')

df1['答案'] = df2['答案']

df1.to_csv(rf'answer.csv', index=False, mode='w')