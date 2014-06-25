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

X = randn(Nx, 1)
Rk = radius(X, X)

#my_kernel = SquareExp(w=Constant(guess=1.5), l=2.0)
#my_kernel = SquareExp(w=1.5, l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, 2.5])
#my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=1.5, l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)])

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

 # sum and prod not set up yet
#my_kernel = SquareExp(w=1.5, l=Constant(guess=0.8)) + SquareExp(w=1.5, l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=1.5, l=Constant(guess=0.8)) * SquareExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)])

my_hyper, hyper_bounds = my_kernel._map_hyper()
my_gpr = GPP(X, test_func(X), my_kernel)
my_gpb = GPP(X, test_func(X), my_kernel, explicit_basis=[0, 1])

K, Kp, Kpp = my_kernel(Rk, grad_r='Hess', data=True)

Rk += dx
Kplus = my_kernel(Rk, data=True)
Rk -= dx

Rk -= dx 
Kminus = my_kernel(Rk, data=True)
Rk += dx

Kd = (Kplus - Kminus)/(2.0*dx)
print 'Kernel Gradient Error (1st dim.):', nanmax(abs(Kd - Kp[:,:,0])/Kd)
Kdd = (Kplus - 2.0*K + Kminus)/dx**2
print 'Kernel Hessian Error (1st dim.):', nanmax(abs(Kdd - Kpp[:,:,0,0])/Kdd)
Kdd = (Kplus - 2.0*K + Kminus)/dx**2
print 'Kernel Hessian Error (1st dim.):', nanmax(abs(Kdd - Kpp[:,:,0,0])/Kdd)
