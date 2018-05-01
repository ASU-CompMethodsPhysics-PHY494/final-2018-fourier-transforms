import numpy as np
hbar = 6.626e-34 / (2*np.pi)



def Phi(a,p,t,n):
    (n*hbar**(3/2)*np.sqrt(np.pi*a))/(p**2*a**2-n**2*np.pi**2*hbar**2)((-1)**n*np.exp(1j*p*a/hbar)-1)*np.exp(-1j*n**2*np.pi**2*hbar/(2*m*a**2)*t)
    return PHI
