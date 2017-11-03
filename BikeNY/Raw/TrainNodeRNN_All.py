#!/usr/bin/python
'''
This code is used to predict the crime time series for the ZipCode region: 90003.
Diurnal cumulative and temporal superresolution.
Hourly, Daily, Weekly features.
External features.
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.ndimage
import csv
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import argparse, h5py

parser = argparse.ArgumentParser()
cor = []

def hisAver(numEvents, testN):
    train = numEvents[:-testN]
    test = numEvents[-testN:]
    testPredict = np.zeros(test.shape)
    for i in range(testN):
        j = i%24
        testPredict[i] = np.mean(train[range(j,train.shape[0],24)])
    #plt.plot(testPredict)
    #plt.show()
    testRMSE = math.sqrt(mean_squared_error(test, testPredict))
    print 'Historical average = ', testRMSE

#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

#Convert data into (feature, label) format
def ConvertSeriesToMatrix(numEvents, Time, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range(len(numEvents) - len2):
        tmp = []	
        tmp.append(Time[i+len2])    #Time
    	#Weekly dependence
    	for j in range(numWeek):
    	    tmp.append(numEvents[i+len2-(j+1)*7*TimeEachDay])
    	#Daily dependence
    	for j in range(numDay):
    	    tmp.append(numEvents[i+len2-(j+1)*TimeEachDay])
    	#Hourly dependence
    	for j in range(i+len2-len1+1, i+len2):
    	    tmp.append(numEvents[j])
    	
    	#Label
    	tmp.append(numEvents[i+len2])
    	matrix.append(tmp)
    return matrix

#RNN predictor
def RNNPrediction(numEvents, Time, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    numEvents = scaler1.fit_transform(numEvents)

    #Dependence
    numWeek = 4; numDay = 4; numHour = 5
    sequence_length1 = numHour + 1
    sequence_length2 = numWeek*7*TimeEachDay + 1
    matrix = ConvertSeriesToMatrix(numEvents, Time, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
    matrix = np.asarray(matrix)
    
    #Split dataset: the last 10 days for testing
    train_set = matrix[:-10*24, :]
    test_set = matrix[-10*24:, :]
     
    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    #Transform the training set into the LSTM format (number of samples, the dim of each elements)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print x_train.shape, x_test.shape, y_train.shape, y_test.shape
    
    #Build the deep learning model
    model = Sequential()
    #Layer1: LSTM
    model.add(LSTM(input_dim = 1, output_dim = 32, return_sequences = True))
    model.add(Dropout(0.2))
    #Layer2: LSTM
#    model.add(LSTM(input_dim = 64, output_dim = 128, return_sequences = True))
#    model.add(Dropout(0.2))
    #Layer3: LSTM
    model.add(LSTM(output_dim = 64, return_sequences = False))
    model.add(Dropout(0.2))
    #Layer4: fully connected
    model.add(Dense(output_dim = 1, activation = 'sigmoid'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "mse", optimizer = adam)
    
    #Training the model
    model.fit(x_train, y_train, batch_size = 64, nb_epoch = 300, validation_split = 0.1, verbose = 1)

    #save the model
    model_json = model.to_json()
    with open("Saved_models/model{0}_{1}.json".format(cor[0], cor[1]), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Saved_models/model{0}_{1}.h5".format(cor[0], cor[1]))
    print("Saved model to disk")

    #Prediction
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    
    #Invert the prediction
    trainPredict = scaler1.inverse_transform(trainPredict)
    testPredict = scaler1.inverse_transform(testPredict)
    
    train = scaler1.inverse_transform(np.array(y_train))
    test = scaler1.inverse_transform(np.array(y_test))

    print train.shape, test.shape

    np.savetxt('test.csv', test)
    np.savetxt('testPredict.csv', testPredict)
    
    #Calculate the error
    trainScore = math.sqrt(mean_squared_error(train, trainPredict))
    trainScore2 = np.average(train - trainPredict)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test, testPredict))
    testScore2 = np.average(test-testPredict)
    print('Test Score: %.2f RMSE' % (testScore))
        
#The main function
if __name__ == '__main__':

    f = h5py.File('../NYC14_M16x8_T60_NewEnd.h5')
    data = f['data']
    data = np.asarray(data)
    #plt.plot(data[:,0,12,5])
    #plt.show()

    #the following grids are all 0: (12,6), (4,6), (12,7)

    for cor0 in [(4,5),(12,5)]: #(8,4), (4,2), (12,2), (4,5),(12,5)
        cor = cor0
        ndays = 183
        TimeEachDay = 24
        
        #Events
        numEvents = data[:,0,cor[0],cor[1]]
        print 'numEvents Size: ', numEvents.shape
        numEvents = np.asarray(numEvents)
        #plt.plot(range(240),numEvents[0:240])
        #plt.show()

        #Use periodic time as the only external feature.
        Time = ndays*range(1,25)
        Time = np.asarray(Time)
        Time = Time/24.0

        #train, and save the model
        print 'Begin Training node ', cor
        hisAver(numEvents, 10*24);
        res = RNNPrediction(numEvents, Time, TimeEachDay)

