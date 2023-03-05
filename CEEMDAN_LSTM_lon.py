# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:27:05 2022

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
lon =np.array(df)[:,1]

IImfs=[]
ceemdan = CEEMDAN()
ceemdan.ceemdan(lon)
imfs, res = ceemdan.get_imfs_and_residue()
plt.figure(figsize=(12,9))
plt.subplots_adjust(hspace=0.1)
plt.subplot(imfs.shape[0]+2, 1, 1)
plt.plot(lon,'r')
for i in range(imfs.shape[0]):
    plt.subplot(imfs.shape[0]+2,1,i+2)
    plt.plot(imfs[i], 'g')
    plt.ylabel("IMF %i" %(i+1))
    plt.locator_params(axis='x', nbins=10)
    IImfs.append(imfs[i])
plt.xlabel("Time [s]")



df_rest=df.drop(columns='longitude')

#IMF1-LSTM
data1_lon=pd.DataFrame(IImfs[0],columns=['lon1'])
data1=pd.concat([df_rest,data1_lon],axis=1)

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
model.add(LSTM(64,input_shape=(100,4),return_sequences=True))  #返回所有节点的输出
model.add(LSTM(32,return_sequences=True))  #返回最后一个节点的输出
model.add(LSTM(16,return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(units=1))
model.add(Activation("relu"))
model.compile(loss='mse', optimizer='adam')
model.summary()

history1=model.fit(trainX1,trainY1,batch_size=20,epochs=250, verbose=2, validation_split=(0.05))

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
IMFS1_lon_pred=b*(MAX1-MIN1)+MIN1
IMFS1_lon_orgi=np.array(data1_lon)[-200:,]
plt.plot(IMFS1_lon_orgi, color = 'green', label = 'Real value')
plt.plot(IMFS1_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


#error_lon1=np.average(np.abs(IMFS1_lon_orgi - IMFS1_lon_pred) /IMFS1_lon_orgi, axis=0)
mse_test1=mean_squared_error(IMFS1_lon_orgi,IMFS1_lon_pred)
mae_test1=mean_absolute_error(IMFS1_lon_orgi,IMFS1_lon_pred)
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
    Mg_pred=a*(MAX-MIN)+MIN
    
    return Mg_pred

#IMF2-LSTM
data2_lon=pd.DataFrame(IImfs[1],columns=['lon2'])
data2=pd.concat([df_rest,data2_lon],axis=1)
IMFS2_lon_pred=myLSTM(data2)

plt.plot(np.array(data2_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS2_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()

#IMF3-LSTM
data3_lon=pd.DataFrame(IImfs[2],columns=['lon3'])
data3=pd.concat([df_rest,data3_lon],axis=1)
IMFS3_lon_pred=myLSTM(data3)

plt.plot(np.array(data3_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS3_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()



#IMF4-LSTM
data4_lon=pd.DataFrame(IImfs[3],columns=['lon4'])
data4=pd.concat([df_rest,data4_lon],axis=1)
IMFS4_lon_pred=myLSTM(data4)

plt.plot(np.array(data4_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS4_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()

#IMF5-LSTM
data5_lon=pd.DataFrame(IImfs[4],columns=['lon5'])
data5=pd.concat([df_rest,data5_lon],axis=1)
IMFS5_lon_pred=myLSTM(data5)

plt.plot(np.array(data5_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS5_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()

#IMF6-LSTM
data6_lon=pd.DataFrame(IImfs[5],columns=['lon6'])
data6=pd.concat([df_rest,data6_lon],axis=1)
IMFS6_lon_pred=myLSTM(data6)

plt.plot(np.array(data6_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS6_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


#IMF7-LSTM
data7_lon=pd.DataFrame(IImfs[6],columns=['lon7'])
data7=pd.concat([df_rest,data7_lon],axis=1)
IMFS7_lon_pred=myLSTM(data7)

plt.plot(np.array(data7_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS7_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


#IMF8-LSTM
data8_lon=pd.DataFrame(IImfs[7],columns=['lon8'])
data8=pd.concat([df_rest,data8_lon],axis=1)
IMFS8_lon_pred=myLSTM(data8)

plt.plot(np.array(data8_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS8_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


#IMF9-LSTM
data9_lon=pd.DataFrame(IImfs[8],columns=['lon9'])
data9=pd.concat([df_rest,data9_lon],axis=1)
IMFS9_lon_pred=myLSTM(data9)

plt.plot(np.array(data9_lon)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS9_lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()


lon_pred=IMFS9_lon_pred+IMFS8_lon_pred+IMFS7_lon_pred+IMFS6_lon_pred+IMFS5_lon_pred+IMFS4_lon_pred+IMFS3_lon_pred+IMFS2_lon_pred+IMFS1_lon_pred

lon_=np.array(lon)[-200:]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
plt.plot(np.array(lon_), color = 'green', label = 'Real value')
plt.plot(lon_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('longitude')
plt.legend()
plt.show()
#评价
mse_lon=mean_squared_error(lon_,lon_pred)
mae_lon=mean_absolute_error(lon_,lon_pred)
print('mse:',mse_lon)
print('mae:',mae_lon)