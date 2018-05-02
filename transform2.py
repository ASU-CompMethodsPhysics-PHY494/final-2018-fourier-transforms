import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


Dt = 0.0001
t = np.arange(-10,10,Dt)
x = np.exp(-np.abs(t))

y = np.fft.fftshift(np.fft.fft(x))
m = Dt*np.abs(y)

f2 = np.arange(-len(y)/2,len(y)/2)/Dt/len(y)*(2*np.pi)

def wavefunction(x):
    return np.exp(-np.abs(x))

def wavetransform(f,xmin=-10,xmax=10,kmin=-10,kmax=10,nx=20000,nk=20000):
    k = np.linspace(kmin,kmax,nk)
    x = np.linspace(xmin,xmax,nx)
    phiReal = np.empty_like(k)
    phiImag = np.empty_like(k)
    for m in range(len(k)):
        gxreal = f(x)*np.cos(k[m]*x)
        areaReal = np.sum(gxreal)*(xmax-xmin)/nx
        gximag = f(x)*np.sin(k[m]*x)
        areaImag = np.sum(gximag)*(xmax-xmin)/nx
        phiReal[m] = areaReal
        phiImag[m] = areaImag
    return k, phiReal, phiImag


k, phiReal, phiImag = wavetransform(wavefunction,nx=2000,nk=2000)
phi = phiReal + 1j*phiImag
phi = np.abs(phi)
ycontrolalg = np.abs(2/(k**2+1))


plt.subplot(2,1,1)
plt.plot(k,ycontrolalg, '-', color='rebeccapurple',label='Calculated by Hand')
plt.title('Calculated by Hand')
plt.xlim(-6,6)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.title('K-Space Transforms for the Delta Potential Bound State')
plt.legend()
plt.subplot(2,1,2)
plt.plot(f2,m,'o',color='cornflowerblue',label='FFT Output')
plt.plot(k,phi,'-',color='firebrick',label='Algorithm Output')
plt.xlim(-6,6)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.legend()
#plt.title('FFT Version Calculated by Hand')
#plt.subplot(2,2,2)
#plt.plot(f2,m)
#plt.title('Calculated with FFT')
#plt.subplot(2,2,3)
#plt.plot(k,ycontrolalg)
#plt.title('Algorithm Version Calculated by Hand')
#plt.subplot(2,2,4)
#plt.plot(k,phi)
#plt.title('Calculated with Algorithm')
plt.show()
