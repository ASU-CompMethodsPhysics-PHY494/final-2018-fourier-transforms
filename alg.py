#homemade algorithm for computing fourier transforms of position space wave functions

import numpy as np
import matplotlib.pyplot as plt

def wavefunction(x):
    return 1/np.pi**0.25 / np.sqrt(8) * (4*x**2 - 2) * np.exp(-x**2 / 2)

def wavetransform(f,xmin=-10,xmax=10,kmin=-10,kmax=10,nx=2000,nk=2000):
    k = np.linspace(kmin,kmax,nk)
    x = np.linspace(xmin,xmax,nx)
    phi = np.empty_like(k)
    for m in range(len(k)):
        gx = f(x)*np.exp(-1j*k[m]*x)
        area = np.sum(gx)*(xmax-xmin)/nx
        phi[m] = area
        phi = np.real(phi)
    return k, phi


k, phi = wavetransform(wavefunction)
#phicontrol = np.sqrt(np.pi) * np.exp(- k**2 / 4)
#plt.subplot(1,2,1)
#plt.plot(k,phicontrol)
plt.subplot(1,2,2)
plt.plot(k,phi)
plt.show()
