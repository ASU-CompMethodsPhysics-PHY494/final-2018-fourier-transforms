Dt = 1
for k in 10:
    t = np.linspace(-10,10,20/Dt)
    x1 = np.sqrt(2)*np.sin(np.pi*t)


    def wavefunction1(x):
        return np.sqrt(2)*np.sin(np.pi*x)



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

    #time it and assign the time into a list
    #assign Dt into a list
    #divide Dt into half
