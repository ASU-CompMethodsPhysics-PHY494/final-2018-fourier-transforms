import time
import numpy as np
import matplotlib.pyplot as plt

Dt = 1
s = 20
timesFFT = np.empty(shape=(s,1))
timesAlg = np.empty(shape=(s,1))
size = np.empty_like(timesFFT)
for h in range(s):
    startFFT = time.time()
    t = np.linspace(-10,10,20/Dt)
    x1 = np.sqrt(2)*np.sin(np.pi*t)

    y = np.fft.fftshift(np.fft.fft(x1))
    m = y
    m = np.abs(m)

    f2 = np.linspace(-1/(2*Dt),1/(2*Dt),len(y))*(2*np.pi)

    endFFT = time.time()



    timesFFT[h] = endFFT-startFFT
    size[h] = 20/Dt
    Dt = 0.5*Dt

plt.plot(size,timesFFT,color='cyan',label='FFT')
plt.legend()
plt.xlabel('Size of Array')
plt.ylabel('Run time in seconds')
plt.show()

    #time it and assign the time into a list
    #assign Dt into a list
    #divide Dt into half
