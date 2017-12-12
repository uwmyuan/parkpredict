#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:18:27 2017

@author: yunyuan
"""
#%%input
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#import seaborn as sns
import numpy as np
import scipy as sp
#from datetime import timedelta
#import datetime
import platform
#df=ticket user
if (platform.node()=='YundeMacBook-Pro.local'):
    df=pd.read_csv('./LTR.csv')
else:
    df=pd.read_csv('D:\\LTR.csv')
#df.head()
df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
df[['Entr_Time','Exit_Time']]=df[['Entr_Time','Exit_Time']].apply(pd.to_datetime)
df.dtypes
#count
df_entr=df[['Rate','Entr_Time']].set_index('Entr_Time')
df_exit=df[['Rate','Exit_Time']].set_index('Exit_Time')
if (platform.node()=='YundeMacBook-Pro.local'):
    df2=pd.read_csv('./LPA.csv')
else:
    df2=pd.read_csv('D:\\LPA.csv')
#df2.head()
df2[['Date and Time']]=df2[['Date and Time']].apply(pd.to_datetime)
df2_entr=df2[['Date and Time','Lot']][(df2.Direction=='In')&(df2.Allowed=='Yes')]
#df2_entr.head()
df2_exit=df2[['Date and Time','Lot']][(df2.Direction=='Out')&(df2.Allowed=='Yes')]
#df2_exit.head()
df2_entr=df2_entr[['Date and Time','Lot']].set_index('Date and Time')
df2_exit=df2_exit[['Date and Time','Lot']].set_index('Date and Time')
#%%NN regression

#定义拟合误差
#均方误差根  
def rmse(y_test, y):  
    return sp.sqrt(sp.mean((y_test - y) *(y_test - y)))  
  
#与均值相比的优秀程度，介于[0~1]0表示不如均值1表示完美预测.这个版本的实现是参考scikit-learn官网文档 
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)*(y_test - y_true)).sum() / ((y_true - y_true.mean())*(y_true - y_true.mean())).sum()  
  
  
#Conway&White机器学习使用案例解析里的版本 
def R22(y_test, y_true):  
    y_mean = np.array(y_true)  
    y_mean[:] = y_mean.mean()  
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true) 

def create_dataset(X, Y, loop_back=3):
    dataX, dataY = [], []
    for i in range(len(X) - loop_back):
        dataX.append(X[i:(i+loop_back)])
        dataY.append(Y[i+loop_back])
    return np.array(dataX), np.array(dataY)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#%%
#df_entr=df_entr['2017-04-1':'2017-04-30']
#df_exit=df_exit['2017-04-1':'2017-04-30']
#df2_entr=df2_entr['2017-04-1':'2017-04-30']
#df2_exit=df2_exit['2017-04-1':'2017-04-30']
#df_entr=df_entr['2017-04-10']
#df_exit=df_exit['2017-04-10']
#df2_entr=df2_entr['2017-04-10']
#df2_exit=df2_exit['2017-04-10']
intervals='15'
entr_count=np.array(df_entr.resample(intervals+'T').count(),dtype=int)
exit_count=np.array(df_exit.resample(intervals+'T').count(),dtype=int)
entr_count2=np.array(df2_entr.resample(intervals+'T').count(),dtype=int)
exit_count2=np.array(df2_exit.resample(intervals+'T').count(),dtype=int)
#%%preprocessing
length=np.min((entr_count.shape[0],exit_count.shape[0],entr_count2.shape[0],exit_count2.shape[0]))
dataX=np.hstack((entr_count[len(entr_count)-length:len(entr_count)],exit_count[len(exit_count)-length:len(exit_count)],entr_count2[len(entr_count2)-length:len(entr_count2)],exit_count2[len(exit_count2)-length:len(exit_count2)]))
dataY=np.array(np.vstack((dataX[1:len(dataX)],np.zeros(4))),dtype=float)
#crossvaladate
train_size = np.int(np.round(len(dataX) * 0.7))
train_dataX = dataX[0:train_size]
train_dataY = dataY[0:train_size]
test_dataX=dataX[-train_size:]
test_dataY=dataY[-train_size:]
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_dataX=scaler.fit_transform(train_dataX)
train_dataY=scaler.fit_transform(train_dataY)
test_dataX=scaler.fit_transform(test_dataX)
test_dataY=scaler.fit_transform(test_dataY)
#create training data with lookback
trainX,trainY=create_dataset(train_dataX,train_dataY)
testX,testY=create_dataset(test_dataX,test_dataY)
#model definition
model = Sequential()
#model.add(Embedding(input_dim=3,output_dim=3))
model.add(LSTM(128,input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True, activation='softsign'))
model.add(LSTM(64,return_sequences=True, activation='softsign'))
model.add(LSTM(32,return_sequences=True, activation='softsign'))
model.add(LSTM(16, activation='softsign'))
model.add(Dense(dataY.shape[1], activation='softsign'))
#model compiling
model.compile(loss='mse', optimizer='adam', metrics=['acc'])


#binary_crossentropy
#model training
early_stopping=EarlyStopping(monitor='loss',patience=3)
#plot_model(model,show_shapes=True,show_layer_names=True,to_file='model.png')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
#model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)
#model evaluating
print(model.evaluate(trainX, trainY))
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
print(R22(trainPredict,trainY))
print(rmse(trainPredict,trainY))
for i in range(dataY.shape[1]):
    plt.figure(figsize=(8,4))
    plt.plot(trainY[:,i],label='trainY')
    plt.plot(trainPredict[:,i],label='predicted trainY')
    plt.legend(bbox_to_anchor=(1,1))    
    plt.savefig('D:\\'+intervals+'entr_exit_TR_PA_prediction_train '+str(i),dpi=600)
    plt.show()
    plt.figure(figsize=(8,4))
    plt.plot(testY[:,i],label='testY')
    plt.plot(testPredict[:,i],label='predicted testY')
    plt.title(i)
    plt.legend(bbox_to_anchor=(1,1))    
    plt.savefig('D:\\'+intervals+'entr_exit_TR_PA_prediction_test '+str(i),dpi=600)
    plt.show()

