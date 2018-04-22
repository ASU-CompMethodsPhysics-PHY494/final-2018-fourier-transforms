import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack



t = np.arange(-9.99,9.99,0.01)
x = np.exp(-t**2)

ycontrol = np.sqrt(np.pi) *np.exp(-np.pi**2 * t**2)
y = np.fft.fftshift(np.fft.fft(x))
m = 0.01*np.abs(y)

f = np.arange(-len(y)/2,len(y)/2)*100/len(y)

plt.subplot(1,2,1)
plt.plot(f,ycontrol)
plt.subplot(1,2,2)
plt.plot(f,m)
plt.show()
