import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


Dt = np.pi/20

def wavefunction1(x):
    return np.exp(-x**2)


def wavetransform(f1,xmin=-10,xmax=10,kmin=-10,kmax=10,nx=20/Dt,nk=20/Dt):
    k = np.linspace(kmin,kmax,nk)
    x = np.linspace(xmin,xmax,nx)
    phiReal = np.empty_like(k)
    phiImag = np.empty_like(k)
    for m in range(len(k)):
        gxreal = f1(x)*np.cos(k[m]*x)
        areaReal = np.sum(gxreal)*(xmax-xmin)/nx
        gximag = f1(x)*np.sin(k[m]*x)
        areaImag = np.sum(gximag)*(xmax-xmin)/nx
        phiReal[m] = areaReal
        phiImag[m] = areaImag
    return k, phiReal, phiImag


k, phiReal, phiImag = wavetransform(wavefunction1)
phi = phiReal + 1j*phiImag
phi = np.abs(phi)

plt.subplot(2,1,1)
plt.title('K-Space Transform')
plt.plot(k,phi,'-',color='firebrick',label='Algorithm Transform')
plt.xlim(-10,10)
plt.ylim(0,2)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.legend()
plt.show()
