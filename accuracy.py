import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

Dt = 0.001
t = np.arange(-10,10,Dt)
x1 = np.exp(-t**2)                          #
x2 = t*np.exp(-t**2)
x3 = (t**2-1)*np.exp(-t**2)

def wavefunction1(x):
    return np.exp(-x**2)
def wavefunction2(x):
    return x*np.exp(-x**2)
def wavefunction3(x):
    return (x**2-1)*np.exp(-x**2)


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
k, yalg2, m2, f2 = FT(wavefunction2,x2,xmin=-10,xmax=10)
k, yalg3, m3, f2 = FT(wavefunction3,x3,xmin=-10,xmax=10)
ycontrol1 = np.abs(np.sqrt(np.pi) * np.exp(-k**2 / 4))
ycontrol2 = np.abs(-0.5*1j*k*np.sqrt(np.pi) * np.exp(-k**2 / 4))
ycontrol3 = np.abs(-0.25*np.sqrt(np.pi) *(k**2 +2) * np.exp(-k**2 / 4))

y1 = sorted(ycontrol1, reverse=True)
y1 = y1[:1]
y1 = np.asarray(y1)
yalg1 = sorted(yalg1, reverse=True)
yalg1 = yalg1[:1]
yalg1 = np.asarray(yalg1)
m1 = sorted(m1, reverse=True)
m1 = m1[:1]
m1 = np.asarray(m1)
zalg1 = np.linalg.norm(y1-yalg1)
zfft1 = np.linalg.norm(y1-m1)

y2 = sorted(ycontrol2, reverse=True)
y2 = y2[:1]
y2 = np.asarray(y2)
yalg2 = sorted(yalg2, reverse=True)
yalg2 = yalg2[:1]
yalg2 = np.asarray(yalg2)
m2 = sorted(m2, reverse=True)
m2 = m2[:1]
m2 = np.asarray(m2)
zalg2 = np.linalg.norm(y2-yalg2)
zfft2 = np.linalg.norm(y2-m2)

y3 = sorted(ycontrol3, reverse=True)
y3 = y3[:1]
y3 = np.asarray(y3)
yalg3 = sorted(yalg3, reverse=True)
yalg3 = yalg3[:1]
yalg3 = np.asarray(yalg3)
m3 = sorted(m3, reverse=True)
m3 = m3[:1]
m3 = np.asarray(m3)
zalg3 = np.linalg.norm(y3-yalg3)
zfft3 = np.linalg.norm(y3-m3)
