# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:24:38 2022

@author: zyk
"""

from PyEMD import CEEMDAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional,LSTM,Flatten
from tensorflow.keras import regularizers

#导入数据
df = pd.read_csv(r"D:\mywork\发论文\地震数据\lastdata.csv")
lat =np.array(df)[:,2]

IImfs=[]
ceemdan = CEEMDAN()
ceemdan.ceemdan(lat)
imfs, res = ceemdan.get_imfs_and_residue()
plt.figure(figsize=(12,9))
plt.subplots_adjust(hspace=0.1)
plt.subplot(imfs.shape[0]+2, 1, 1)
plt.plot(lat,'r')
for i in range(imfs.shape[0]):
    plt.subplot(imfs.shape[0]+2,1,i+2)
    plt.plot(imfs[i], 'g')
    plt.ylabel("IMF %i" %(i+1))
    plt.locator_params(axis='x', nbins=10)
    IImfs.append(imfs[i])
plt.xlabel("Time [s]")

df_rest=df.drop(columns='latitude')

#IMF1-LSTM
data1_lat=pd.DataFrame(IImfs[0],columns=['lat1'])
data1=pd.concat([df_rest,data1_lat],axis=1)

for_training1=data1[:-300]
for_testing1=data1[-300:]

scaler1 = MinMaxScaler(feature_range=(0,1))
training_scaled1 = scaler1.fit_transform(for_training1)
testing_scaled1 = scaler1.fit_transform(for_testing1)
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i,3:])
    return np.array(dataX),np.array(dataY)

trainX1,trainY1=createXY(training_scaled1,100)
testX1,testY1=createXY(testing_scaled1,100)

model = Sequential()
model.add(LSTM(72,input_shape=(100,4),return_sequences=True))  #返回所有节点的输出
model.add(LSTM(36,return_sequences=False))  #返回最后一个节点的输出
model.add(Dropout(0.3))
model.add(Dense(units=1))
#model.add(Activation("relu"))
model.compile(loss='mse', optimizer='adam')
model.summary()

history1=model.fit(trainX1,trainY1,batch_size=25,epochs=250, verbose=2, validation_split=(0.05))

plt.figure()
plt.plot(history1.history["loss"], label = "train")
plt.plot(history1.history["val_loss"], label = "test")
plt.legend()
plt.show()

prediction1=model.predict(testX1)
columns1 = data1.columns
b=prediction1[:,0]
MAX1=np.max(for_testing1[columns1[3]])
MIN1=np.min(for_testing1[columns1[3]])
IMFS1_lat_pred=b*(MAX1-MIN1)+MIN1
IMFS1_lat_orgi=np.array(data1_lat)[-200:,]
plt.plot(IMFS1_lat_orgi[:], color = 'green', label = 'Real value')
plt.plot(IMFS1_lat_pred[:], color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()


#error_lon1=np.average(np.abs(IMFS1_lon_orgi - IMFS1_lon_pred) /IMFS1_lon_orgi, axis=0)
mse_test1=mean_squared_error(IMFS1_lat_orgi,IMFS1_lat_pred)
mae_test1=mean_absolute_error(IMFS1_lat_orgi,IMFS1_lat_pred)
print('mse:',mse_test1)
print('mae:',mae_test1)



def myLSTM(mydata):
    for_training=mydata[:-300]
    for_testing=mydata[-300:]

    scaler = MinMaxScaler(feature_range=(0,1))
    training_scaled = scaler.fit_transform(for_training)
    testing_scaled = scaler.fit_transform(for_testing)
    def createXY(dataset,n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,3:])
        return np.array(dataX),np.array(dataY)

    trainX,trainY=createXY(training_scaled,100)
    testX,testY=createXY(testing_scaled,100)
    model = Sequential()
    model.add(LSTM(64,input_shape=(100,4),return_sequences=True))  #返回所有节点的输出
    model.add(LSTM(32,return_sequences=False))  #返回最后一个节点的输出
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.add(Activation("relu"))
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    history=model.fit(trainX,trainY,batch_size=20,epochs=100, verbose=2, validation_split=(0.05))

    prediction=model.predict(testX)
    columns = mydata.columns
    a=prediction[:,0]
    MAX=np.max(for_testing[columns[3]])
    MIN=np.min(for_testing[columns[3]])
    lat_pred=a*(MAX-MIN)+MIN
    
    return lat_pred

#IMF2-LSTM
data2_lat=pd.DataFrame(IImfs[1],columns=['lat2'])
data2=pd.concat([df_rest,data2_lat],axis=1)
IMFS2_lat_pred=myLSTM(data2)

plt.plot(np.array(data2_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS2_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()


#IMF3-LSTM
data3_lat=pd.DataFrame(IImfs[2],columns=['lat3'])
data3=pd.concat([df_rest,data3_lat],axis=1)
IMFS3_lat_pred=myLSTM(data3)

plt.plot(np.array(data3_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS3_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()


#IMF4-LSTM
data4_lat=pd.DataFrame(IImfs[3],columns=['lat4'])
data4=pd.concat([df_rest,data4_lat],axis=1)
IMFS4_lat_pred=myLSTM(data4)

plt.plot(np.array(data4_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS4_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()

#IMF5-LSTM
data5_lat=pd.DataFrame(IImfs[4],columns=['lat5'])
data5=pd.concat([df_rest,data5_lat],axis=1)
IMFS5_lat_pred=myLSTM(data5)

plt.plot(np.array(data5_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS5_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()

#IMF6-LSTM
data6_lat=pd.DataFrame(IImfs[5],columns=['lat6'])
data6=pd.concat([df_rest,data6_lat],axis=1)
IMFS6_lat_pred=myLSTM(data6)

plt.plot(np.array(data6_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS6_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()

#IMF7-LSTM
data7_lat=pd.DataFrame(IImfs[6],columns=['lat7'])
data7=pd.concat([df_rest,data7_lat],axis=1)
IMFS7_lat_pred=myLSTM(data7)

plt.plot(np.array(data7_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS7_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()

#IMF8-LSTM
data8_lat=pd.DataFrame(IImfs[7],columns=['lat8'])
data8=pd.concat([df_rest,data8_lat],axis=1)
IMFS8_lat_pred=myLSTM(data8)

plt.plot(np.array(data8_lat)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS8_lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()


lat_pred=IMFS8_lat_pred+IMFS7_lat_pred+IMFS6_lat_pred+IMFS5_lat_pred+IMFS4_lat_pred+IMFS3_lat_pred+IMFS2_lat_pred+IMFS1_lat_pred

lat_=np.array(lat)[-200:]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
plt.plot(np.array(lat_), color = 'green', label = 'Real value')
plt.plot(lat_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('latitude')
plt.legend()
plt.show()
#评价
mse_lat=mean_squared_error(lat_,lat_pred)
mae_lat=mean_absolute_error(lat_,lat_pred)
print('mse:',mse_lat)
print('mae:',mae_lat)