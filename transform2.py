import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


Dt = 0.001
t = np.arange(-10,10,Dt)
x = np.exp(-t**2)

y = np.fft.fftshift(np.fft.fft(x))
m = Dt*np.abs(y)

f = np.arange(-len(y)/2,len(y)/2)/Dt/len(y)
f2 = np.arange(-len(y)/2,len(y)/2)/Dt/len(y)*(2*np.pi)
ycontrolfft = np.sqrt(np.pi) * np.exp(-np.pi**2 * f**2)

def wavefunction(x):
    return np.exp(-x**2)

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


k, phi = wavetransform(wavefunction,nx=2000,nk=2000)
ycontrolalg = np.sqrt(np.pi) *np.exp(- k**2 /4)

plt.plot(k,ycontrolalg, '-', color='rebeccapurple',label='Calculated by Hand')
plt.plot(f2,m,'o',color='cornflowerblue',label='FFT Output')
plt.plot(k,phi,'-',color='firebrick',label='Algorithm Output')
plt.xlim(-6,6)
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
