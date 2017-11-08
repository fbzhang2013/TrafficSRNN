import numpy as np
import matplotlib.pyplot as plt
import time
import math
import csv
import pandas as pd
import argparse, h5py
from datetime import date
from sklearn.preprocessing import MinMaxScaler

f = h5py.File('BJ_Meteorology.h5', 'r')
print f.keys()
Time = f['date']
Time = np.asarray(Time)
#print np.where(Time == '2015121929')
#print f['WindSpeed'][50444:50473], f['WindSpeed'][50473:50502]
#print Time[50444:50473], Time[50473:50502]

Time = np.delete(Time, range(50444,50473), axis = 0)
Temperature = np.delete(f['Temperature'],range(50444,50473), axis = 0)
Weather = np.delete(f['Weather'],range(50444,50473), axis = 0)
WindSpeed = np.delete(f['WindSpeed'],range(50444,50473), axis = 0)
f.close()

'''
#check if it is corrected:
for i in range(1,Time.shape[0]):
	day_miss = date(int(Time[i][0:4]), int(Time[i][4:6]), int(Time[i][6:8])) - date(int(Time[i-1][0:4]), int(Time[i-1][4:6]), int(Time[i-1][6:8]))
	gap = day_miss.days*48 + int(Time[i][-2:]) - int(Time[i-1][-2:])
	if gap!=1:
		print 'missing {2} time slots between: {0} and {1}'.format(Time[i-1], Time[i], gap-1)#, total_len, i
'''

f1 = h5py.File('BJ_Meteorology_corrected.h5', 'w')
f1.create_dataset('Temperature', data=Temperature)
f1.create_dataset('Weather', data=Weather)
f1.create_dataset('WindSpeed', data=WindSpeed)
f1.create_dataset('date', data=Time)
f1.close()

text_file = open("BJ_Holiday.txt", "r")
holidays = text_file.read().split('\n')


for year in [13,14,15,16]:
	print '================================='
	print 'Year 20{0}'.format(year)
	f = h5py.File('BJ{0}_M32x32_T30_InOut.h5'.format(year),'r')
	start = f['date'][0]
	end = f['date'][-1]
	start_index = np.where(Time == start)[0][0]
	end_index = np.where(Time == end)[0][0]
	Date_year = Time[start_index:end_index+1]
	#Tempreture, Weather, WindSpeed, Time feature
	Tem = Temperature[start_index:end_index+1]
	Wea = Weather[start_index:end_index+1,:]
	WSpeed = WindSpeed[start_index:end_index+1]
	print Tem.shape, Wea.shape, WSpeed.shape
	#Holiday
	date_start = date(int(start[0:4]), int(start[4:6]), int(start[6:8]))
	date_end = date(int(end[0:4]), int(end[4:6]), int(end[6:8]))
	print date_start, date_end
	N = (date_end - date_start).days*48 + int(end[-2:]) - int(start[-2:]) + 1
	BJ_Holiday = np.zeros((N,))
	for hol in holidays:
		date_hol = date(int(hol[0:4]), int(hol[4:6]), int(hol[6:8]))
		if date_hol >= date_start and date_hol <= date_end:
			k = (date_hol - date_start).days*48 - int(start[-2:]) + 1
			#print k, Time_year[k]
			BJ_Holiday[k:k+48] = 1
	#Time
	firstday = range(int(start[-2:]),49)
	lastday = range(1, int(end[-2:])+1)
	between_n = (N - len(firstday) - len(lastday))/48.0
	TimePeriod = firstday + int(between_n)*range(1,49) + lastday
	TimePeriod = np.asarray(TimePeriod)
	#Concate together
	Tem = Tem.reshape(Tem.shape[0],1)
	WSpeed = WSpeed.reshape(WSpeed.shape[0],1)
	BJ_Holiday = BJ_Holiday.reshape(BJ_Holiday.shape[0],1)
	TimePeriod = TimePeriod.reshape(TimePeriod.shape[0],1)
	External_feature = np.concatenate((Tem, WSpeed,BJ_Holiday,Wea,TimePeriod),axis = 1)

	print External_feature.shape, External_feature.max(), External_feature.min()

	np.savetxt('External_feature{0}.csv'.format(year), External_feature)













