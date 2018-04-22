#for finding transform of a position space wave function with DFT and FFT

import numpy as np
import matplotlib.pyplot as plt

import numpy.fft as fft
import scipy as sp

hbar=6.62606957*(10**-34.0)/(2*np.pi)
x = np.linspace(-10,10,2**12)

#insert the wave function to be transformed here:
y2btransformed = np.exp(-x**2)
k = np.fft.fftshift(np.fft.fftfreq(len(x)))
ytransformcontrol = np.sqrt(np.pi) * np.exp(- np.pi**2 * x**2)
yt1 = np.fft.fft(y2btransformed)
yt = np.abs(yt1)
ytransformed = 0.005 * np.fft.fftshift(yt)
yuntrans = np.fft.fftshift(np.abs(np.fft.ifft(ytransformed)))/0.01

plt.subplot(2,2,1)
plt.plot(x,y2btransformed)
plt.subplot(2,2,2)
plt.plot(k,ytransformcontrol)
plt.subplot(2,2,3)
plt.plot(k,ytransformed)
plt.subplot(2,2,4)
plt.plot(x,yuntrans)
plt.show()
