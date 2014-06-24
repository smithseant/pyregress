from numpy import ndarray, zeros, empty, tile, abs, nanmax
from numpy.random import randn

from pyregress import *

Nx = 10
dx = 1e-4  # This doesn't seem to behave well smaller than 1e-4.

def test_func(X, aniso=None):
    Nx, Ndim = X.shape[0], X.shape[1]
    if not isinstance(aniso, ndarray):
        f = (X**2).sum(axis=1)
    else:
        f = zeros(Nx)
        for i in xrange(Ndim):
            f += X[:,i]**2/aniso[i]
    return f

def radius(X, Y, aniso=None):
    Ndim = X.shape[1]
    Nx, Ny = X.shape[0], Y.shape[0]
    Rk = empty((Nx, Ny, Ndim))
    for k in xrange(Ndim):
        Rk[:, :, k] = tile(X[:,[k]], (1, Ny)) - tile(Y[:,[k]].T, (Nx, 1))
        if isinstance(aniso, ndarray):
            Rk[:, :, k] /= aniso[k]
    return Rk

X = randn(Nx, 2)
Rk = radius(X, X)

#my_kernel = Noise(w=Constant(guess=1.5))
my_kernel = SquareExp(w=Constant(guess=1.5), l=2.0)
#my_kernel = SquareExp(w=1.5, l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, 2.5])
#my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=1.5, l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)])
#my_kernel = GammaExp(w=Constant(guess=1.5), l=2.0, gamma=1.5)
#my_kernel = GammaExp(w=1.5, l=Constant(guess=2.0), gamma=1.5)
#my_kernel = GammaExp(w=Constant(guess=1.5), l=[2.0, 2.5], gamma=1.5)
#my_kernel = GammaExp(w=1.5, l=[Constant(guess=2.0), 2.5], gamma=1.5)
#my_kernel = GammaExp(w=1.5, l=[2.0, Constant(guess=2.5)], gamma=1.5)
#my_kernel = GammaExp(w=1.5, l=2.0, gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=1.5, l=[2.0, 2.5], gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=Constant(guess=1.5), l=Constant(guess=2.0), gamma=1.5)
#my_kernel = GammaExp(w=Constant(guess=1.5), l=[Constant(guess=2.0), 2.5], gamma=1.5)
#my_kernel = GammaExp(w=Constant(guess=1.5), l=[2.0, Constant(guess=2.5)], gamma=1.5)
#my_kernel = GammaExp(w=Constant(guess=1.5), l=2.0, gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=Constant(guess=1.5), l=[2.0, 2.5], gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)], gamma=1.5)
#my_kernel = GammaExp(w=1.5, l=Constant(guess=2.0), gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=1.5, l=[Constant(guess=2.0), 2.5], gamma=Constant(guess=1.5))
#my_kernel = GammaExp(w=1.5, l=[2.5, Constant(guess=2.5)], gamma=Constant(guess=1.5))
#my_kernel = RatQuad(w=Constant(guess=1.5), l=2.0, alpha=1.5)
#my_kernel = RatQuad(w=1.5, l=Constant(guess=2.0), alpha=1.5)
#my_kernel = RatQuad(w=Constant(guess=1.5), l=[2.0, 2.5], alpha=1.5)
#my_kernel = RatQuad(w=1.5, l=[Constant(guess=2.0), 2.5], alpha=1.5)
#my_kernel = RatQuad(w=1.5, l=[2.0, Constant(guess=2.5)], alpha=1.5)
#my_kernel = RatQuad(w=1.5, l=2.0, alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=1.5, l=[2.0, 2.5], alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=Constant(guess=1.5), l=Constant(guess=2.0), alpha=1.5)
#my_kernel = RatQuad(w=Constant(guess=1.5), l=[Constant(guess=2.0), 2.5], alpha=1.5)
#my_kernel = RatQuad(w=Constant(guess=1.5), l=[2.0, Constant(guess=2.5)], alpha=1.5)
#my_kernel = RatQuad(w=Constant(guess=1.5), l=2.0, alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=Constant(guess=1.5), l=[2.0, 2.5], alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)], alpha=1.5)
#my_kernel = RatQuad(w=1.5, l=Constant(guess=2.0), alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=1.5, l=[Constant(guess=2.0), 2.5], alpha=Constant(guess=1.5))
#my_kernel = RatQuad(w=1.5, l=[2.5, Constant(guess=2.5)], alpha=Constant(guess=1.5))
#my_kernel = SquareExp(w=1.5, l=Constant(guess=0.8)) + SquareExp(w=1.5, l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=1.5, l=Constant(guess=0.8)) * SquareExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)])

my_gpr = GPP(X, test_func(X), my_kernel)
my_gpb = GPP(X, test_func(X), my_kernel, explicit_basis=[0, 1])

K, Kp, Kpp = my_kernel(Rk, grad='Hess', r_grad=True, data=True)
my_hyper[0] += dx
Kplus = my_kernel(Rk, data=True)
my_hyper[0] -= dx
Kminus = my_kernel(Rk, data=True)
my_hyper[0] += dx
Kd = (Kplus - Kminus)/(2.0*dx)
print 'Kernel Gradient Error (1st dim.):', nanmax(abs(Kd - Kp[:,:,0])/Kd)
Kdd = (Kplus - 2.0*K + Kminus)/dx**2
print 'Kernel Hessian Error (1st dim.):', nanmax(abs(Kdd - Kpp[:,:,0,0])/Kdd)

if len(my_hyper) > 1:
    my_hyper[1] += dx
    Kplus = my_kernel(Rk, data=True)
    my_hyper[1] -= dx
    
    my_hyper[1] -= dx
    Kminus = my_kernel(Rk, data=True)
    my_hyper[1] += dx
        
    Kd = (Kplus - Kminus)/(2.0*dx)
    print 'Kernel Gradient Error (2nd dim.):', nanmax(abs(Kd - Kp[:,:,1])/Kd)
    Kdd = (Kplus - 2.0*K + Kminus)/dx**2
    print 'Kernel Hessian Error (2nd dim.):', nanmax(abs(Kdd - Kpp[:,:,1,1])/Kdd)
    
    my_hyper[0] += dx
    my_hyper[1] += dx
    Kplusplus = my_kernel(Rk, data=True)
    my_hyper[0] -= dx
    my_hyper[1] -= dx
    
    my_hyper[0] += dx
    my_hyper[1] -= dx
    Kplusminus = my_kernel(Rk, data=True)
    my_hyper[0] -= dx
    my_hyper[1] += dx
    
    my_hyper[0] -= dx
    my_hyper[1] += dx
    Kminusplus = my_kernel(Rk, data=True)
    my_hyper[0] += dx
    my_hyper[1] -= dx
    
    my_hyper[0] -= dx
    my_hyper[1] -= dx
    Kminusminus = my_kernel(Rk, data=True)
    my_hyper[0] += dx
    my_hyper[1] += dx
    
    Kdd = (Kplusplus - Kplusminus - Kminusplus + Kminusminus)/(2.0*dx)**2
    print 'Kernel Hessian Error (cross dim.):', nanmax(abs(Kdd - Kpp[:,:,0,1])/Kdd)












