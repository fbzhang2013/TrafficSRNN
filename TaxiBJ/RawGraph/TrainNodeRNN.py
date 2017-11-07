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

#Resize the image
def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

#Convert data into (feature, label) format
def ConvertSeriesToMatrix(numTaxi, external, len1, len2, numWeek, numDay, TimeEachDay):
    matrix = []
    #We need to discard the data 0 ~ len2-1
    for i in range((numTaxi.shape[0]) - len2):
    	tmp = []
        # external feature		
        tmp = tmp + list(external[i+len2,:])
    	#Weekly dependence
    	for j in range(numWeek):
    	    tmp.append(numTaxi[i+len2-(j+1)*7*TimeEachDay,0])
    	#Daily dependence
    	for j in range(numDay):
    	    tmp.append(numTaxi[i+len2-(j+1)*TimeEachDay,0])
    	#Hourly dependence
    	for j in range(i+len2-len1, i+len2):
    	    tmp.append(numTaxi[j,0])
    	#Label
    	tmp.append(numTaxi[i+len2,0])
        matrix.append(tmp)
    return matrix

#RNN predictor
def RNNPrediction(data_4years, External_4years, TimeEachDay):
    #Normalize data to (0, 1)
    scaler1 = MinMaxScaler(feature_range = (0, 1))
    len_4years = [0]
    for i in range(4):
        if i==0:
            all_data = np.array(data_4years[i])
            all_external = np.array(External_4years[i])
            len_4years.append(data_4years[i].shape[0])
        else:
            all_data = np.concatenate((all_data,data_4years[i]), axis=0)
            all_external = np.concatenate((all_external,External_4years[i]), axis=0)
            len_4years.append(len_4years[-1]+data_4years[i].shape[0])
    print all_data.shape, all_external.shape, len_4years
    
    temp = np.concatenate((all_data[:,0],all_data[:,1]),axis = 0)
    temp = temp.reshape(-1,1)
    temp = scaler1.fit_transform(temp)
    all_data1 = np.vstack((temp[:temp.shape[0]/2,0], temp[temp.shape[0]/2:,0])).transpose()
    all_external1 = scaler1.fit_transform(all_external)

    #convert the data in each year into matrix.
    for i in range(4):
        numTaxi = all_data1[len_4years[i]:len_4years[i+1],:]
        external = all_external1[len_4years[i]:len_4years[i+1],:]
        #Dependence
        numWeek = 4; numDay = 4; numHour = 5
        sequence_length1 = numHour
        sequence_length2 = numWeek*7*TimeEachDay + 1
        if i==0:
            matrix = ConvertSeriesToMatrix(numTaxi, external, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
            matrix = np.asarray(matrix)
        else:
            matrix_temp = ConvertSeriesToMatrix(numTaxi, external, sequence_length1, sequence_length2, numWeek, numDay, TimeEachDay)
            matrix_temp = np.asarray(matrix_temp)
            matrix = np.concatenate((matrix,matrix_temp),axis=0)
    print matrix.shape
    #Split dataset: 90% for training and 10% for testing
    train_row = int(round(0.9*matrix.shape[0]))
    train_set = matrix[:train_row, :]
    test_set = matrix[train_row:, :]

    x_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

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
    model.add(Dense(output_dim = 1, activation = 'sigmoid'))
    #model.compile(loss = "mse", optimizer = "adam")
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "mse", optimizer = adam)
    
    #Training the model
    model.fit(x_train, y_train, batch_size = 128, nb_epoch = 200, validation_split = 0.2, verbose = 1)

    #save the model
    model_json = model.to_json()
    with open("Saved_models/model{0}.json".format(FLAGS.train_n), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Saved_models/model{0}.h5".format(FLAGS.train_n))
    print("Saved model to disk")

    #Prediction
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    
    #Invert the prediction
    trainPredict = scaler1.inverse_transform(trainPredict)
    testPredict = scaler1.inverse_transform(testPredict)
    
    train = scaler1.inverse_transform(np.array(y_train))
    test = scaler1.inverse_transform(np.array(y_test))
    
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
    cor = [16,16]
    data_4years = []
    External_4years = []
    for year in [13,14,15,16]:
        f = h5py.File('Data/BJ{0}_complete.h5'.format(year),'r')
        data = f['data']
        data = np.asarray(data)
        f.close()
        #print data.shape
        #plt.plot(data[0:48*30,0,16,16])
        #plt.show()
        data_4years.append(data[:,:,cor[0],cor[1]])
        External = np.genfromtxt('External_feature{0}.csv'.format(year))
        External_4years.append(External)
    res = RNNPrediction(data_4years, External_4years, TimeEachDay)