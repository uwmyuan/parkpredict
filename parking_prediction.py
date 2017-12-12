# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:11:19 2017

@author: Yun
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
import timeit
import time
def input_data(filename):
    #df.head()
    df=pd.read_csv(filename)
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
    #preprocessing
    length=np.min((entr_count.shape[0],exit_count.shape[0],entr_count2.shape[0],exit_count2.shape[0]))
    dataX=np.hstack((entr_count[len(entr_count)-length:len(entr_count)],exit_count[len(exit_count)-length:len(exit_count)],entr_count2[len(entr_count2)-length:len(entr_count2)],exit_count2[len(exit_count2)-length:len(exit_count2)]))
    dataY=np.array(np.vstack((dataX[1:len(dataX)],np.zeros(4))),dtype=float)
    return dataX,dataY

#NN regression
def createModel(shape1,shape2,shape3):
    model=Sequential()
    #model.add(Embedding(input_dim=3,output_dim=3))
    model.add(LSTM(256,input_shape=(shape1,shape2),return_sequences=True, activation='softsign'))
    model.add(Bidirectional(LSTM(256,return_sequences=True, activation='softsign')))
    model.add(Bidirectional(LSTM(256,return_sequences=True, activation='softsign')))
#    model.add(LSTM(256,return_sequences=True, activation='softsign'))
    model.add(Bidirectional(LSTM(256, activation='softsign')))
    model.add(Dense(shape3, activation='softsign'))
    #model compiling
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    return model

def rmse(y_test, y):  
    return sp.sqrt(sp.mean((y_test - y) *(y_test - y)))  
  
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)*(y_test - y_true)).sum() / ((y_true - y_true.mean())*(y_true - y_true.mean())).sum()  
  
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
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
#from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#from keras.utils.np_utils import to_categorical
def training(train_dataX,train_dataY):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_dataX=scaler.fit_transform(train_dataX)
    train_dataY=scaler.fit_transform(train_dataY)    
    #create training data with lookback
    trainX,trainY=create_dataset(train_dataX,train_dataY)   
    #model definition
    model=createModel(trainX.shape[1],trainX.shape[2],train_dataY.shape[1])    
    #early stopping
    early_stopping=EarlyStopping(monitor='loss',patience=3)
    #model training
    
    history=model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
    #plot model    
    #plot_model(model,show_shapes=True,show_layer_names=True,to_file='model.png')
    #model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)#fitting with fixed hyperparas
    #model evaluating
    evals=model.evaluate(trainX, trainY)
    print('loss:'+str(evals[0])+'\n'+'acc:'+str(evals[1]))
    # make predictions
    trainPredict = model.predict(trainX)
    print('train rmse:'+str(rmse(trainPredict,trainY)))
    print('train R2:'+str(R2(trainPredict,trainY)))
    print('train R22:'+str(R22(trainPredict,trainY)))
    trainY=scaler.inverse_transform(trainY)
    trainPredict=scaler.inverse_transform(trainPredict)
    for i in range(train_dataY.shape[1]):
        plt.figure(figsize=(8,4))
        plt.plot(trainY[:,i],label='trainY')
        plt.plot(trainPredict[:,i],label='predicted trainY')
        plt.title(i)
        plt.legend(bbox_to_anchor=(1,1))    
        plt.savefig(time.strftime('%Y-%m-%d %H%M%S')+' 4-11_train '+str(i),dpi=600)
        plt.show()
    return model
def testing(model,test_dataX,test_dataY):
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_dataX=scaler.fit_transform(test_dataX)
    test_dataY=scaler.fit_transform(test_dataY)   
    testX,testY=create_dataset(test_dataX,test_dataY)
    testPredict = model.predict(testX)
    print('test R2:'+str(R2(testPredict,testY)))
    print('test R22:'+str(R22(testPredict,testY)))
    testY=scaler.inverse_transform(testY)
    testPredict=scaler.inverse_transform(testPredict)
    for i in range(test_dataY.shape[1]):
        plt.figure(figsize=(8,4))
        plt.plot(testY[:,i],label='testY')
        plt.plot(testPredict[:,i],label='predicted testY')
        plt.title(i)
        plt.legend(bbox_to_anchor=(1,1))    
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S')+' 4-11_test '+str(i),dpi=600)
        plt.show()

        
def save_model(model,filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")    
    print("Saved")

def load_model(filename):
    from keras.models import model_from_json
    # load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded")
    return loaded_model
def tuning(dataX,dataY):
    from hyperopt import Trials, STATUS_OK, tpe
    from hyperas import optim
    from hyperas.distributions import choice, uniform, conditional
    best_run, best_model = optim.minimize(model=tuning_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
def data():
    filename='linkmin.csv'
    df=pd.read_csv(filename)
    df.head()
    df.columns
    #crossid,linkid,lgid,stage,stagestart,flow,traveltime,signaldelay 
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['starttime','endtime']]=df[['starttime','endtime']].apply(pd.to_datetime)
    df[['linkid']]=df[['linkid']].astype(str)
    df.dtypes
    
    df_link=df.drop(['endtime','linkid'],1)
    df_link.head()
    df_link=df_link.set_index('starttime')
    df_link=df_link.rolling(window=5,center=True).mean()
    data=np.array(df_link,dtype='float')
    dataX=data[2:len(data)-3]
    dataY=data[3:len(data)-2]
    rate=0.7
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataX=scaler.fit_transform(dataX)
    dataY=scaler.fit_transform(dataY) 
    train_size = np.int(np.round(len(dataX) * rate))
    train_dataX = dataX[0:train_size]
    train_dataY = dataY[0:train_size]
    test_dataX=dataX[train_size:]
    test_dataY=dataY[train_size:]   
    #create training data with lookback
    loop_back=3
    dataX1, dataY1 = [], []
    for i in range(len(train_dataX) - loop_back):
        dataX1.append(train_dataX[i:(i+loop_back)])
        dataY1.append(train_dataY[i+loop_back])
    trainX,trainY= np.array(dataX1), np.array(dataY1)
    dataX2, dataY2 = [], []
    for i in range(len(test_dataX) - loop_back):
        dataX2.append(test_dataX[i:(i+loop_back)])
        dataY2.append(test_dataY[i+loop_back])
    testX,testY= np.array(dataX2), np.array(dataY2)
    return trainX,trainY,testX,testY
def tuning_model(trainX,trainY,testX,testY):
    shape1=trainX.shape[1]
    shape2=trainX.shape[2]
    shape3=trainY.shape[1]
    model=Sequential()
    #model.add(Embedding(input_dim=3,output_dim=3))
    model.add(LSTM(512,input_shape=(shape1,shape2),return_sequences=True, activation='softsign'))
    model.add(LSTM(512,return_sequences=True, activation={{choice(['softsign','relu','tanh'])}}))
    model.add(LSTM(512,return_sequences=True, activation={{choice(['softsign','relu','tanh'])}}))
    model.add(LSTM(512, activation={{choice(['softsign','relu','tanh'])}}))
    model.add(Dense(shape3, activation={{choice(['softsign','relu','tanh','signmax'])}}))
    #model compiling
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    early_stopping=EarlyStopping(monitor='loss',patience=3)
    history=model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
    score, acc = model.evaluate(testX, testY, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
def main():
    start = timeit.default_timer()

    #input a csv file
    dataX,dataY=input_data('LTR.csv')
#
#    #for cross validation
    train_size = np.int(np.round(len(dataX) * 0.7))
#    #train
    train_dataX = dataX[0:train_size]
    train_dataY = dataY[0:train_size]    
    model=training(train_dataX,train_dataY)
#    #test
    test_dataX=dataX[train_size:]
    test_dataY=dataY[train_size:]
    testing(model,test_dataX,test_dataY)
    
    #save model to file
    save_model(model,"model.json")
    save_model(model,time.strftime('%Y-%m-%d %H%M%S')+"model.json")
    
    #load model
    model=load_model('model.json')
       
    #tune
    model=tuning(dataX,dataY)
    elapsed = (timeit.default_timer() - start)
    print(str(elapsed)+' seconds')
if __name__ == '__main__':
    main()