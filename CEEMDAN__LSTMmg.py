# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:42:22 2022

@author: 28581
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
Mg =np.array(df)[:,3]

IImfs=[]
ceemdan = CEEMDAN()
ceemdan.ceemdan(Mg)
imfs, res = ceemdan.get_imfs_and_residue()
plt.figure(figsize=(12,9))
plt.subplots_adjust(hspace=0.1)
plt.subplot(imfs.shape[0]+2, 1, 1)
plt.plot(Mg,'r')
for i in range(imfs.shape[0]):
    plt.subplot(imfs.shape[0]+2,1,i+2)
    plt.plot(imfs[i], 'g')
    plt.ylabel("IMF %i" %(i+1))
    plt.locator_params(axis='x', nbins=10)
    IImfs.append(imfs[i])
plt.xlabel("Time [s]")
IMF1=IImfs[0]




df_rest=df.drop(columns='Mg')

#IMF1-LSTM
data1_Mg=pd.DataFrame(IImfs[0],columns=['Mg1'])
data1=pd.concat([df_rest,data1_Mg],axis=1)

for_training1=data1[:-300]S
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
model.add(LSTM(32,return_sequences=False)) 
model.add(Dropout(0.3))
model.add(Dense(units=1))
model.add(Activation("relu"))
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
IMFS1_Mg_pred=b*(MAX1-MIN1)+MIN1
IMFS1_Mg_orgi=np.array(data1_Mg)[-200:,]
plt.plot(IMFS1_Mg_orgi, color = 'green', label = 'Real value')
plt.plot(IMFS1_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()



mse_test1=mean_squared_error(IMFS1_Mg_orgi,IMFS1_Mg_pred)
mae_test1=mean_absolute_error(IMFS1_Mg_orgi,IMFS1_Mg_pred)
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
data2_Mg=pd.DataFrame(IImfs[1],columns=['Mg2'])
data2=pd.concat([df_rest,data2_Mg],axis=1)
IMFS2_Mg_pred=myLSTM(data2)

plt.plot(np.array(data2_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS2_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#IMF3-LSTM
data3_Mg=pd.DataFrame(IImfs[2],columns=['Mg3'])
data3=pd.concat([df_rest,data3_Mg],axis=1)
IMFS3_Mg_pred=myLSTM(data3)

plt.plot(np.array(data3_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS3_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#IMF4-LSTM
data4_Mg=pd.DataFrame(IImfs[3],columns=['Mg4'])
data4=pd.concat([df_rest,data4_Mg],axis=1)
IMFS4_Mg_pred=myLSTM(data4)

plt.plot(np.array(data4_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS4_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#IMF5-LSTM
data5_Mg=pd.DataFrame(IImfs[4],columns=['Mg5'])
data5=pd.concat([df_rest,data5_Mg],axis=1)
IMFS5_Mg_pred=myLSTM(data5)

plt.plot(np.array(data5_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS5_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#IMF6-LSTM
data6_Mg=pd.DataFrame(IImfs[5],columns=['Mg6'])
data6=pd.concat([df_rest,data6_Mg],axis=1)
IMFS6_Mg_pred=myLSTM(data6)

plt.plot(np.array(data6_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS6_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()


#IMF7-LSTM
data7_Mg=pd.DataFrame(IImfs[6],columns=['Mg7'])
data7=pd.concat([df_rest,data7_Mg],axis=1)
IMFS7_Mg_pred=myLSTM(data7)

plt.plot(np.array(data7_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS7_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()


#IMF8-LSTM
data8_Mg=pd.DataFrame(IImfs[7],columns=['Mg8'])
data8=pd.concat([df_rest,data8_Mg],axis=1)
IMFS8_Mg_pred=myLSTM(data8)

plt.plot(np.array(data8_Mg)[-200:], color = 'green', label = 'Real value')
plt.plot(IMFS8_Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#重构
Mg_pred=IMFS8_Mg_pred+IMFS7_Mg_pred+IMFS6_Mg_pred+IMFS5_Mg_pred+IMFS4_Mg_pred+IMFS3_Mg_pred+IMFS2_Mg_pred+IMFS1_Mg_pred

Mg_=np.array(Mg)[-200:]     

plt.plot(np.array(Mg_), color = 'green', label = 'Real value')
plt.plot(Mg_pred, color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

from matplotlib.pyplot import MultipleLocator
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
plt.plot(np.array(Mg_)[:100], color = 'green', label = 'real value')
plt.plot(Mg_pred[:100], color = 'black', label = 'Predicted value')
plt.title('comparisiom chart')
'''
x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
'''
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

plt.plot(np.array(Mg_)[100:], color = 'green', label = 'Real value')
plt.plot(Mg_pred[100:], color = 'black', label = 'Predicted value')
plt.title('comparision chart')
plt.xlabel('Time')
plt.ylabel('Mg')
plt.legend()
plt.show()

#评价
mse=mean_squared_error(Mg_,Mg_pred)
mae=mean_absolute_error(Mg_,Mg_pred)
print('mse:',mse)
print('mae:',mae)