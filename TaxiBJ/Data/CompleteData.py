import numpy as np
import matplotlib.pyplot as plt
import time
import math
import csv
import pandas as pd
import argparse, h5py
from datetime import date

for year in [13,14,15,16]:
	print '=============================='
	print 'Year 20{0}'.format(year)
	f = h5py.File('BJ{0}_M32x32_T30_InOut.h5'.format(year),'r')
	data = f['data']
	Time = f['date']
	data = np.asarray(data)
	Time = np.asarray(Time)

	Temperature = []

	print data.shape
	print 'start: ', Time[0], 'end: ', Time[-1]

	[a,b,c] = data[0,:,:,:].shape

	total_len = 1
	for i in range(1,Time.shape[0]):
		day_miss = date(int(Time[i][0:4]), int(Time[i][4:6]), int(Time[i][6:8])) - date(int(Time[i-1][0:4]), int(Time[i-1][4:6]), int(Time[i-1][6:8]))
		gap = day_miss.days*48 + int(Time[i][-2:]) - int(Time[i-1][-2:])
		if gap!=1:
			print 'missing {2} time slots between: {0} and {1}'.format(Time[i-1], Time[i], gap-1)#, total_len, i
			data = np.concatenate((data[:total_len], np.zeros((gap-1,a,b,c)), data[total_len:]), axis = 0) #complete missing timeslots with '0'

			total_len += gap-1
		total_len += 1

	print 'shape after completion:', data.shape

	f1 = h5py.File('BJ{0}_complete.h5'.format(year),'w')
	f1.create_dataset('data', data=data)