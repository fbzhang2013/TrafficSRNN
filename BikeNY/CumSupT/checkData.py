import numpy as np
import h5py
import scipy.ndimage

def img_enlarge(img, factor = 2.0, order = 2):
    return scipy.ndimage.zoom(img, factor, order = order)

f = h5py.File('../NYC14_M16x8_T60_NewEnd.h5')
data = f['data']
data = np.asarray(data)

ndays = 183
TimeEachDay = 24
numEvents = data[:,0,4,5]
numEvents = np.asarray(numEvents)
numEvents2 = np.zeros(ndays*TimeEachDay)    #cdf of the events
for i in range(len(numEvents)):
    if i%24 == 0:
         numEvents2[i] = numEvents[i]
    else:
         numEvents2[i] = numEvents[i]+numEvents2[i-1]
numEvents3 = img_enlarge(numEvents2, factor = 2.0, order = 2)   #Superresolve

test = numEvents3[-480:]
testCum = test[1::2]
print testCum[-24:]
testExact = np.zeros(testCum.shape)
T = 24
for i in range(10):
    for j in range(1,24):
        testExact[i*T+j]=testCum[i*T+j]-testCum[i*T+j-1]

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
print InvalidIn
print InvalidOut