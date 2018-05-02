import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


Dt = 0.001
t = np.arange(-10,10,Dt)
x1 = np.exp(-np.abs(t))
#x2 = t*np.exp(-t**2)
#x3 = (t**2-1)*np.exp(-t**2)

def wavefunction1(x):
    return np.exp(-np.abs(x))
#def wavefunction2(x):
#    return x*np.exp(-x**2)
#def wavefunction3(x):
#    return (x**2-1)*np.exp(-x**2)


def FT(f1,g1,xmin=-10,xmax=10):

    y = np.fft.fftshift(np.fft.fft(g1))
    m = Dt*np.abs(y)

    f2 = np.arange(-len(y)/2,len(y)/2)/Dt/len(y)*(2*np.pi)

    def wavetransform(f1,xmin=-10,xmax=10,kmin=-10,kmax=10,nx=20000,nk=20000):
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


    k, phiReal, phiImag = wavetransform(f1,xmin,xmax)
    phi = phiReal + 1j*phiImag
    phi = np.abs(phi)
    return k, phi, m, f2

k, yalg1, m1, f2 = FT(wavefunction1,x1,xmin=-10,xmax=10)
#k, yalg2, m2, f2 = FT(wavefunction2,x2,xmin=-10,xmax=10)
#k, yalg3, m3, f2 = FT(wavefunction3,x3,xmin=-10,xmax=10)
ycontrol1 = 2/(k**2+1)
#ycontrol2 = np.abs(-0.5*1j*k*np.sqrt(np.pi) * np.exp(-k**2 / 4))
#ycontrol3 = np.abs(-0.25*np.sqrt(np.pi) *(k**2 +2) * np.exp(-k**2 / 4))

plt.subplot(2,1,1)
plt.plot(k,ycontrol1, '-', color='rebeccapurple',label='Ground State')
#plt.plot(k,ycontrol2, '-', color='red',label='First Excited State')
#plt.plot(k,ycontrol3, '-', color='blue',label='Second Excited State')
plt.title('Calculated by Hand')
plt.xlim(-10,10)
plt.ylim(0,2.5)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.title('K-Space Transforms for the Harmonic Oscillator')
plt.legend()
plt.subplot(2,1,2)
plt.plot(f2,m1,'o',color='cornflowerblue',label='FFT Ground State')
plt.plot(k,yalg1,'-',color='firebrick',label='Algorithm Ground State')
#plt.plot(f2,m2,'o',color='green',label='FFT First Excited State')
#plt.plot(k,yalg2,'-',color='orchid',label='Algorithm First Excited State')
#plt.plot(f2,m3,'o',color='orange',label='FFT Second Excited State')
#plt.plot(k,yalg3,'-',color='black',label='Algorithm Second Excited State')
plt.xlim(-10,10)
plt.ylim(0,2.5)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.title('FFT and Algorithm Outputs')
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
