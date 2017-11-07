#!/usr/bin/python
'''
This code is used to predict the Taxi flow time series in Beijing.
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
import argparse
import h5py
from keras.callbacks import ModelCheckpoint # save checkpoint

cor = [16,16]
cor_around = [[cor[0]-1,cor[1]], [cor[0]+1,cor[1]], [cor[0],cor[1]-1], [cor[0],cor[1]+1]]

def hisAver(data):
    testN = data.shape[0] - int(round(0.9*data.shape[0]))
    train = data[:-testN,:]
    test = data[-testN:,:]
    testPredict = np.zeros(test.shape)
    for i in range(testN):
        j = i%24
        testPredict[i,:] = np.mean(train[range(j,train.shape[0],24),:],axis=0)
    #plt.plot(testPredict)
    #plt.show()
    testRMSE = math.sqrt(mean_squared_error(test, testPredict))
    print 'Historical average = ', testRMSE

#Convert data into (feature, label) format
def ConvertSeriesToMatrix(data, external, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range((data.shape[0]) - len2):
    	tmp = []
        # external feature		
        tmp = tmp + list(external[i+len2,:])
    	#Weekly dependence
    	for j in range(numWeek):
    	    tmp.append(data[i+len2-(j+1)*7*TimeEachDay,0,cor[0],cor[1]])
            tmp.append(data[i+len2-(j+1)*7*TimeEachDay,1,cor[0],cor[1]])
    	#Daily dependence
    	for j in range(numDay):
    	    tmp.append(data[i+len2-(j+1)*TimeEachDay,0,cor[0],cor[1]])
            tmp.append(data[i+len2-(j+1)*TimeEachDay,1,cor[0],cor[1]])
    	#Hourly dependence
    	for j in range(i+len2-len1, i+len2):
    	    tmp.append(data[j,0,cor[0],cor[1]])
            tmp.append(data[j,1,cor[0],cor[1]])

        #Weekly dependence around 
        for j in range(numWeek):
            for c in cor_around:
                tmp.append(data[i+len2-(j+1)*7*TimeEachDay,0,c[0],c[1]])
                tmp.append(data[i+len2-(j+1)*7*TimeEachDay,1,c[0],c[1]])
        #Daily dependence around
        for j in range(numDay):
            for c in cor_around:
                tmp.append(data[i+len2-(j+1)*TimeEachDay,0,c[0],c[1]])
                tmp.append(data[i+len2-(j+1)*TimeEachDay,1,c[0],c[1]])
        #Hourly dependence around
        for j in range(i+len2-len1+1, i+len2):
            for c in cor_around:
                tmp.append(data[j,0,c[0],c[1]])
                tmp.append(data[j,1,c[0],c[1]])

    	#Label
    	tmp.append(data[i+len2,0,cor[0],cor[1]])
        tmp.append(data[i+len2,1,cor[0],cor[1]])
        matrix.append(tmp)
    return matrix

#RNN predictor
def RNNPrediction(data_4years, External_4years, TimeEachDay):
    #Normalize data to (0, 1)
    MIN_TAXI = 10000
    MAX_TAXI = 0
    for data in data_4years:
        MIN_TAXI = min(MIN_TAXI, data[:,:,cor[0],cor[1]].min())
        MAX_TAXI = max(MAX_TAXI, data[:,:,cor[0],cor[1]].max())
        for c in cor_around:
            MIN_TAXI = min(MIN_TAXI, data[:,:,c[0],c[1]].min())
            MAX_TAXI = max(MAX_TAXI, data[:,:,c[0],c[1]].max())
    for i in range(4):
        data_4years[i][:,:,cor[0],cor[1]] = (data_4years[i][:,:,cor[0],cor[1]] - MIN_TAXI)/(MAX_TAXI - MIN_TAXI)
        for c in cor_around:
            data_4years[i][:,:,c[0],c[1]] = (data_4years[i][:,:,c[0],c[1]] - MIN_TAXI)/(MAX_TAXI - MIN_TAXI)

    #convert the data in each year into matrix, and then concate
    for i in range(4):
        #Dependence
        numWeek = 4; numDay = 4; numHour = 5
        sequence_length1 = numHour
        sequence_length2 = numWeek*7*TimeEachDay + 1
        if i==0:
            matrix = ConvertSeriesToMatrix(data_4years[i], External_4years[i], sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
            matrix = np.asarray(matrix)
        else:
            matrix_temp = ConvertSeriesToMatrix(data_4years[i], External_4years[i], sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
            matrix_temp = np.asarray(matrix_temp)
            matrix = np.concatenate((matrix,matrix_temp),axis=0)
    print matrix.shape
    #Split dataset: 90% for training and 10% for testing
    train_row = int(round(0.9*matrix.shape[0]))
    train_set = matrix[:train_row, :]
    test_set = matrix[train_row:, :]

    x_train = train_set[:, :-2]
    y_train = train_set[:, -2:]
    x_test = test_set[:, :-2]
    y_test = test_set[:, -2:]

    #Transform the training set into the LSTM format (number of samples, the dim of each elements)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    #Build the deep learning model
    model = Sequential()
    #Layer1: LSTM
    model.add(LSTM(input_dim = 1, output_dim = 64, return_sequences = True))
    model.add(Dropout(0.2))
    #Layer2: LSTM
#    model.add(LSTM(input_dim = 64, output_dim = 128, return_sequences = True))
#    model.add(Dropout(0.2))
    #Layer3: LSTM
    model.add(LSTM(output_dim = 128, return_sequences = False))
    model.add(Dropout(0.2))
    #Layer4: fully connected
    model.add(Dense(output_dim = 2, activation = 'sigmoid'))
    #model.compile(loss = "mse", optimizer = "adam")
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "mse", optimizer = adam)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    
    #Training the model
    model.fit(x_train, y_train, batch_size = 128, nb_epoch = 1, validation_split = 0.2, verbose = 1, callbacks=[checkpointer])

    #save the model
    model_json = model.to_json()
    with open("Saved_models/model{0}_{1}.json".format(cor[0],cor[1]), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Saved_models/model{0}_{1}.h5".format(cor[0],cor[1]))
    print("Saved model to disk")

    #Prediction
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    
    #Invert the prediction
    trainPredict = trainPredict*(MAX_TAXI - MIN_TAXI) + MIN_TAXI
    testPredict = testPredict*(MAX_TAXI - MIN_TAXI) + MIN_TAXI
    
    train = y_train*(MAX_TAXI - MIN_TAXI) + MIN_TAXI
    test = y_test*(MAX_TAXI - MIN_TAXI) + MIN_TAXI
    
    #Calculate the error
    trainScore = math.sqrt(mean_squared_error(train, trainPredict))
    trainScore2 = np.average(train - trainPredict)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(test, testPredict))
    testScore2 = np.average(test-testPredict)
    print('Test Score: %.2f RMSE' % (testScore))
    np.savetxt('testPredict.csv',testPredict)
    print train.shape, test.shape

#The main function
if __name__ == '__main__':
    TimeEachDay = 48
    
    #Events
    data_4years = []
    data_4years_around = []
    External_4years = []
    External_4years_around = []
    # there are 4 years of data.
    for year in [13,14,15,16]:
        f = h5py.File('../Data/BJ{0}_complete_0.h5'.format(year),'r')
        data = f['data']
        data = np.asarray(data)
        f.close()
        #print data.shape
        #plt.plot(data[0:48*30,0,16,16])
        #plt.show()
        data_4years.append(data)
        External = np.genfromtxt('../Data/External_feature{0}.csv'.format(year))
        External_4years.append(External)
    hisAver(data_4years[3][:,:,cor[0],cor[1]])
    res = RNNPrediction(data_4years, External_4years, TimeEachDay)