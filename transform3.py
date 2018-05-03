import time
import numpy as np
import matplotlib.pyplot as plt

Dt = 1
s = 12
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

    startAlg = time.time()

    def wavefunction1(x):
        return np.sqrt(2)*np.sin(np.pi*x)

    def wavetransform(f1,xmin=-10,xmax=10,kmin=-15,kmax=15,nx=20/Dt,nk=20/Dt):
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

    endAlg = time.time()
    timesFFT[h] = endFFT-startFFT
    timesAlg[h] = endAlg-startAlg
    size[h] = 20/Dt
    Dt = 0.5*Dt

plt.plot(size,timesFFT,color='cyan',label='FFT')
plt.plot(size,timesAlg,color='orchid',label='Home-made Algorithm')
plt.legend()
plt.xlabel('Size of Array')
plt.ylabel('Run time in seconds')
plt.show()

    #time it and assign the time into a list
    #assign Dt into a list
    #divide Dt into half
