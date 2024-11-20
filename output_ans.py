
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import R2Score
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.optimizers import Adafactor
from tensorflow.keras.utils import custom_object_scope

import numpy as np
import pandas as pd
import os

#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數


#%%
#============================備註============================
from datetime import datetime
NowDateTime = '2024-11-17T15_59_13Z'
file_name = f'{NowDateTime}'


# %%
#============================預測數據============================

#載入模型
regressor = load_model(rf'D:\NCNU\aicup\2024\LSTM\model\{NowDateTime}.h5')
# regressor = load_model('WheatherLSTM_2024-09-21T03_25_16Z.h5')

#載入測試資料
DataName = os.getcwd()+r'\upload(no answer).csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values
print(len(EXquestion))

#%%
inputs = []#存放參考資料
PredictOutput = [] #存放預測值

count = 0
while(count < len(EXquestion)):
  print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode
  DataName = os.getcwd()+'\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[['Power(mW)']].values
  
  inputs = []#重置存放參考資料

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      inputs = np.append(inputs, ReferData[DaysCount])
  
  if len(inputs) == 0:
    DataName = os.getcwd()+'\ExampleTrainData(IncompleteAVG)\IncompleteAvgDATA_'+ strLocationCode +'_2024.csv'
    SourceData = pd.read_csv(DataName, encoding='utf-8')
    ReferTitle = SourceData[['Serial']].values
    ReferData = SourceData[['Power(mW)']].values
    for DaysCount in range(len(ReferTitle)):
      if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
        inputs = np.append(inputs, ReferData[DaysCount])
  
  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs = np.append(inputs, PredictOutput[i-1])
    
    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
    # 在預測前增加數據檢查
    if len(inputs) < LookBackNum:
        print(f"Warning: 序號 {LocationCode} 缺少足夠參考數據")
        # 可以跳過此序號或填充預設值
        continue

    #Reshaping
    NewTest = np.array(X_test)
    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 1))
    predicted = regressor.predict(NewTest, verbose=0)
    PredictOutput.append(round(predicted[0,0], 2))
  
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48

#%%
#寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
# df = pd.DataFrame(np.array([EXquestion.reshape(9600), PredictOutput]).T, columns=['序號', '答案'])
# df['序號'] = df['序號'].astype(np.int64)

df = pd.DataFrame(PredictOutput, columns=['答案'])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv(rf'csvfile/template/{NowDateTime}.csv', index=False) 
print('Output CSV File Saved')


# Warning: 序號 20240927090004 缺少足夠參考數據