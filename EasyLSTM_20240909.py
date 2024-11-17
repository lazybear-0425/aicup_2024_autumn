
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
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import custom_object_scope

import numpy as np
import pandas as pd
import os

#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數

#載入訓練資料
DataName = os.getcwd()+'\ExampleTrainData(AVG)\AvgDATA_17.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')

#選擇要留下來的資料欄位(發電量)
target = ['Power(mW)']
AllOutPut = SourceData[target].values

X_train = []
y_train = []

#設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum,len(AllOutPut)):
  X_train.append(AllOutPut[i-LookBackNum:i, 0])
  y_train.append(AllOutPut[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
# Reshaping
X_train = np.reshape(X_train,(X_train.shape [0], X_train.shape [1], 1))
print(X_train.shape)




#%%
#============================備註============================
from datetime import datetime
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
epoch = 1000
other = 'LossScaleOptimizer(RMSprop) + R2score + EarlyStopping, patience=20, min_delta=0.0001'
file_name = f'{NowDateTime}'






#%%
#============================建置&訓練模型============================
#建置LSTM模型
regressor = Sequential ()
# , activation='ReLU'
regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(LSTM(units = 64))

regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))
plot_model(regressor, to_file='model.jpg')

regressor.compile(optimizer = LossScaleOptimizer(inner_optimizer=RMSprop()), loss = 'mean_squared_error'
                  , metrics=R2Score())

#開始訓練
callback = EarlyStopping(monitor='r2_score', mode='max', 
                         patience=20,min_delta=0.0001,restore_best_weights=True)
history = regressor.fit(X_train, y_train, epochs = epoch, 
                        batch_size = 128,  verbose = 2, 
                        validation_split=0.2, callbacks=[callback])

#保存模型
regressor.save(rf'model/'+NowDateTime+'.h5')
print('Model Saved')

#%%
#====================繪製圖表=============================================
import matplotlib.pyplot as plt

m1 = 'r2_score'
m2 = 'val_'+m1

plt.plot(history.history[m1], color='b', label=m1)
plt.plot(history.history[m2], color='r', label=m2)
plt.title(other)
plt.xlabel('epoch')
plt.ylabel(f'{m1} vs. {m2}')
plt.legend()
plt.savefig(rf'pic/{NowDateTime}')

best_loss = 0
best_epoch = 0
if m1 == 'r2_score':
  best_loss = np.max(np.array(history.history[m2]))
  best_epoch = np.argmax(np.array(history.history[m2]))
else: # loss
  best_loss = np.min(np.array(history.history[m2]))
  best_epoch = np.argmin(np.array(history.history[m2]))



# %%
#============================預測數據============================

#載入模型

with custom_object_scope({
    'RMSprop': RMSprop,
    'LossScaleOptimizer': LossScaleOptimizer
}):
  regressor = load_model(rf'D:\NCNU\aicup\2024\LSTM\model\{NowDateTime}.h5')
# regressor = load_model('WheatherLSTM_2024-09-21T03_25_16Z.h5')

#載入測試資料
DataName = os.getcwd()+r'\ExampleTestData\upload.csv'
SourceData = pd.read_csv(DataName, encoding='utf-8')
target = ['序號']
EXquestion = SourceData[target].values

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
         
  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :

    #print(i)
    
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs = np.append(inputs, PredictOutput[i-1])
    
    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[0+i:LookBackNum+i])
    
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
df.to_csv(rf'csvfile/{NowDateTime}.csv', index=False) 
print('Output CSV File Saved')
del df

# %%
print(f'lowest {m2}: \033[33m{best_loss}\033[0m in epoch \033[33m{best_epoch}\033[0m')
df = pd.read_csv('outcome.csv')
# 檔名, 時間, epoch,val_loss, min_val_loss, min_loss_epoch, 額外備註
df.loc[len(df.index)] = [file_name, NowDateTime, epoch, m2, history.history[m2][-1], best_loss, best_epoch, other]
df.to_csv('outcome.csv', index=False, mode='w')
# %%