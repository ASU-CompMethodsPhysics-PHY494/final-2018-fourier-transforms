import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


Dt = 0.001
t = np.linspace(-10,10,20/Dt)
x1 = np.sqrt(2)*np.sin(np.pi*t)
x2 = np.sqrt(2)*np.sin(2 * np.pi*t)
x3 = np.sqrt(2)*np.sin(3 * np.pi*t)

def wavefunction1(x):
    return np.sqrt(2)*np.sin(np.pi*x)
def wavefunction2(x):
    return np.sqrt(2)*np.sin(2*np.pi*x)
def wavefunction3(x):
    return np.sqrt(2)*np.sin(3*np.pi*x)


def FT(f1,g1,xmin=-15,xmax=15):

    y = np.fft.fftshift(np.fft.fft(g1))
    m = y
    m = np.abs(m)

    f2 = np.linspace(-1/(2*Dt),1/(2*Dt),len(y))*(2*np.pi)

    def wavetransform(f1,xmin=xmin,xmax=xmax,kmin=-15,kmax=15,nx=2000,nk=2000):
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
k, yalg2, m2, f2 = FT(wavefunction2,x2,xmin=0,xmax=1)
k, yalg3, m3, f2 = FT(wavefunction3,x3,xmin=0,xmax=1)
ycontrol1 = np.abs(-np.sqrt(2)*np.pi*(1-np.exp(-1j*k))/((k**2)-(np.pi**2)))
ycontrol2 = np.abs(-np.sqrt(2)*np.pi*(2-2*np.exp(-1j*k))/((k**2)-(4*np.pi**2)))
ycontrol3 = np.abs(-3*np.sqrt(2)*np.pi*(1+np.exp(-1j*k))/((k**2)-(9*np.pi**2)))

plt.subplot(2,1,1)
plt.plot(k,ycontrol1, '-', color='rebeccapurple',label='Ground State')
plt.plot(k,ycontrol2, '-', color='red',label='First Excited State')
plt.plot(k,ycontrol3, '-', color='blue',label='Second Excited State')
plt.title('Calculated by Hand')
plt.xlim(-15,15)
plt.ylim(0,10)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.title('K-Space Transforms for the Infinite Square Well')
plt.legend()
plt.xticks([np.pi,-np.pi,2*np.pi,-2*np.pi,3*np.pi,-3*np.pi,0],['$\pi$','$-\pi$','$2\pi$','$-2\pi$','$3\pi$','$-3\pi$','0'])
plt.subplot(2,1,2)
plt.plot(f2,m1,'o',color='cornflowerblue',label='FFT Ground State')
plt.plot(k,yalg1,'-',color='firebrick',label='Algorithm Ground State')
plt.plot(f2,m2,'-',color='green',label='FFT First Excited State')
plt.plot(k,yalg2,'-',color='orchid',label='Algorithm First Excited State')
plt.plot(f2,m3,'-',color='orange',label='FFT Second Excited State')
plt.plot(k,yalg3,'-',color='black',label='Algorithm Second Excited State')
plt.xlim(-15,15)
plt.ylim(0,10)
plt.xlabel('k')
plt.ylabel('Phi(k)')
plt.title('FFT and Algorithm Outputs')
plt.legend()
plt.xticks([np.pi,-np.pi,2*np.pi,-2*np.pi,3*np.pi,-3*np.pi,0],['$\pi$','$-\pi$','$2\pi$','$-2\pi$','$3\pi$','$-3\pi$','0'])
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
