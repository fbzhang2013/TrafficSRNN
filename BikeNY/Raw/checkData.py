import numpy as np
import h5py

f = h5py.File('../NYC14_M16x8_T60_NewEnd.h5')
data = f['data']
data = np.asarray(data)
InvalidIn = {}
InvalidOut = {}

for i in range(16):
	for j in range(8):
		inflow = data[:,0,i,j]
		outflow = data[:,1,i,j]
		if np.sum(inflow) < 1000:
			InvalidIn[(i,j)] = np.sum(inflow)
		if np.sum(outflow) < 1000:
			InvalidOut[(i,j)] = np.sum(outflow)
print 'Inflow zeros slot:sum': InvalidIn
print 'Outflow zeros slot:sum': InvalidOut