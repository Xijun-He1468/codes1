# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:35:25 2022

@author: zyk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks,regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional,LSTM,Flatten

df = pd.read_csv( r"D:\mywork\发论文\地震数据\lastdata.csv")


#数据处理
for_training=df[:-300]
for_testing=df[-300:]

scaler = MinMaxScaler(feature_range=(0,1))
training_scaled = scaler.fit_transform(for_training)
testing_scaled = scaler.fit_transform(for_testing)

def createXY(dataset,n_past,col):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,col])
    return np.array(dataX),np.array(dataY)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


#震级预测
mg_trainX,mg_trainY=createXY(training_scaled,100,3)
mg_testX,mg_testY=createXY(testing_scaled,100,3)


model = Sequential()
model.add(LSTM(64,input_shape=(100,4),return_sequences=False)) 
model.add(Dropout(0.25))
model.add(Dense(units=1))
model.add(Activation("relu"))
#model.compile(loss='mse', optimizer='adam'(lr = 0.001))
model.compile (optimizer=Adam(lr=1e-3), loss='mse')
model.summary()
history =model.fit(mg_trainX,mg_trainY, epochs= 300, batch_size=20, validation_data=(mg_testX,mg_testY), verbose=2, shuffle=False)
plt.figure()
plt.plot(history.history["loss"], label = "train")
plt.plot(history.history["val_loss"], label = "test")
plt.legend()
plt.show()

prediction_mg=model.predict(mg_testX)
columns = df.columns
mg_pre1=prediction_mg[:,0]
MAX=np.max(for_testing[columns[3]])
MIN=np.min(for_testing[columns[3]])
mg_pre=mg_pre1*(MAX-MIN)+MIN
mg_ori=np.array(for_testing)[-200:,3]
plt.plot(mg_ori, color = 'green', label = 'Real value')
plt.plot(mg_pre, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()


#def mape(y_true, y_pred):
#    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

print('mse:',mean_squared_error(mg_ori,mg_pre))
print('mae:',mean_absolute_error(mg_ori,mg_pre))
print("error_mg:",np.average(np.abs(mg_ori - mg_pre) /mg_ori, axis=0))
print("r2:",r2_score(mg_ori,mg_pre))
print("rmse:",np.sqrt(mean_squared_error(mg_ori,mg_pre)))
print("mape:",mape(mg_ori,mg_pre))
print("平均绝对误差",mean_absolute_error(mg_ori,mg_pre))


#纬度预测
lon_trainX,lon_trainY=createXY(training_scaled,100,1)
lon_testX,lon_testY=createXY(testing_scaled,100,1)

model = Sequential()
model.add(LSTM(64, input_shape = (100, 4),return_sequences=True)) 
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(loss = "mse",optimizer= "adam")
model.summary()
history=model.fit(lon_trainX,lon_trainY,batch_size=20,epochs=200, verbose=2, validation_split=(0.05))
plt.figure()
plt.plot(history.history["loss"], label = "train")
plt.plot(history.history["val_loss"], label = "test")
plt.legend()
plt.show()

prediction_lon=model.predict(lon_testX)
columns = df.columns
lon_pre1=prediction_lon[:,0]
MAX=np.max(for_testing[columns[1]])
MIN=np.min(for_testing[columns[1]])
lon_pre=lon_pre1*(MAX-MIN)+MIN
lon_ori=np.array(for_testing)[-200:,1]
plt.plot(lon_ori, color = 'green', label = 'Real value')
plt.plot(lon_pre, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


print('mse:',mean_squared_error(lon_ori,lon_pre))
print('mae:',mean_absolute_error(lon_ori,lon_pre))
print("error_mg:",np.average(np.abs(lon_ori - lon_pre) /lon_ori, axis=0))
print("r2:",r2_score(lon_ori,lon_pre))
print("rmse:",np.sqrt(mean_squared_error(lon_ori,lon_pre)))
print("mape:",mape(lon_ori,lon_pre))
print("平均绝对误差",mean_absolute_error(lon_ori,lon_pre))

#经度预测
lat_trainX,lat_trainY=createXY(training_scaled,100,2)
lat_testX,lat_testY=createXY(testing_scaled,100,2)

model = Sequential()
model.add(LSTM(64, input_shape = (100, 4),return_sequences=True)) 
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss = "mse",optimizer= "adam")
model.summary()
history=model.fit(lat_trainX,lat_trainY,batch_size=20,epochs=200, verbose=2, validation_split=(0.05))
plt.figure()
plt.plot(history.history["loss"], label = "train")
plt.plot(history.history["val_loss"], label = "test")
plt.legend()
plt.show()

prediction_lat=model.predict(lat_testX)
columns = df.columns
lat_pre1=prediction_lat[:,0]
MAX=np.max(for_testing[columns[2]])
MIN=np.min(for_testing[columns[2]])
lat_pre=lat_pre1*(MAX-MIN)+MIN
lat_ori=np.array(for_testing)[-200:,2]
plt.plot(lat_ori, color = 'green', label = 'Real value')
plt.plot(lat_pre, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()



print('mse:',mean_squared_error(lat_ori,lat_pre))
print('mae:',mean_absolute_error(lat_ori,lat_pre))
print("error_mg:",np.average(np.abs(lat_ori - lat_pre) /lat_ori, axis=0))
print("r2:",r2_score(lat_ori,lat_pre))
print("rmse:",np.sqrt(mean_squared_error(lat_ori,lat_pre)))
print("mape:",mape(lat_ori,lat_pre))
print("平均绝对误差",mean_absolute_error(lat_ori,lat_pre))



#时间差预测

date_trainX,date_trainY=createXY(training_scaled,100,0)
date_testX,date_testY=createXY(testing_scaled,100,0)

model = Sequential()
model.add(LSTM(64, input_shape = (100, 4),return_sequences=True)) 
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss = "mse",optimizer= "adam")
model.summary()
history=model.fit(date_trainX,date_trainY,batch_size=20,epochs=200, verbose=2, validation_split=(0.05))
plt.figure()
plt.plot(history.history["loss"], label = "train")
plt.plot(history.history["val_loss"], label = "test")
plt.legend()
plt.show()


prediction_date=model.predict(date_testX)
columns = df.columns
date_pre1=prediction_date[:,0]
MAX=np.max(for_testing[columns[0]])
MIN=np.min(for_testing[columns[0]])
date_pre=date_pre1*(MAX-MIN)+MIN
date_ori=np.array(for_testing)[-200:,0]
plt.plot(date_ori, color = 'green', label = 'Real value')
plt.plot(date_pre, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('datediff')
plt.legend()
plt.show()


print('mse:',mean_squared_error(date_ori,date_pre))
print('mae:',mean_absolute_error(date_ori,prediction_date))
print("error_datediff:",np.average(np.abs(date_ori - date_pre) /date_ori, axis=0))
print("r2:",r2_score(date_ori,prediction_date))
print("rmse:",np.sqrt(mean_squared_error(date_ori,date_pre)))
print("mape:",mape(date_ori,date_pre))
print("平均绝对误差",mean_absolute_error(date_ori,date_pre))














































