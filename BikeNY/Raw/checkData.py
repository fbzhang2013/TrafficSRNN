import numpy as np
import h5py

def findInvalid():
	f = h5py.File('../NYC14_M16x8_T60_NewEnd.h5')
	data = f['data']
	data = np.asarray(data)
	InvalidIn = []
	InvalidOut = []

	for i in range(16):
		for j in range(8):
			inflow = data[:,0,i,j]
			outflow = data[:,1,i,j]
			if np.sum(inflow) < 1000:
				InvalidIn.append([i,j])
			if np.sum(outflow) < 1000:
				InvalidOut.append([i,j])
	return InvalidIn, InvalidOut