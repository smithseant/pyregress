from numpy import ndarray, zeros, empty, tile, abs, nanmax
from numpy.random import randn

from pyregress import *

n_pts = 10
δx = 1e-4  # This doesn't seem to behave well smaller than 1e-4.

def test_func(X, aniso=None):
    n_pts, n_dims = X.shape[0], X.shape[1]
    if not isinstance(aniso, ndarray):
        f = (X**2).sum(axis=1)
    else:
        f = zeros(n_pts)
        for i in range(n_dims):
            f += X[:,i]**2/aniso[i]
    return f

def radius(X, Y, aniso=None):
    n_dims = X.shape[1]
    n_xpts, n_ypts = X.shape[0], Y.shape[0]
    Rk = empty((n_xpts, n_ypts, n_dims))
    for k in range(n_dims):
        Rk[:, :, k] = tile(X[:,[k]], (1, n_ypts)) - tile(Y[:,[k]].T, (n_xpts, 1))
        if isinstance(aniso, ndarray):
            Rk[:, :, k] /= aniso[k]
    return Rk

X = randn(n_pts, 1)
Rk = radius(X, X)

my_kernel = SquareExp(w=Jeffreys(guess=1.5), l=2.0)
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

#my_hyper, hyper_bounds = my_kernel._map_hyper()
my_gpr = GPI(X, test_func(X), my_kernel)
my_gpb = GPI(X, test_func(X), my_kernel, explicit_basis=[0])

K, Kp, Kpp = my_kernel(Rk, grad_r='Hess')

Rk += δx
Kplus = my_kernel(Rk)
Rk -= δx

Rk -= δx 
Kminus = my_kernel(Rk)
Rk += δx

Kd = (Kplus - Kminus)/(2.0*δx)
print('Kernel Gradient Error (1st dim.):', nanmax(abs(Kd - Kp[:,:,0])/Kd))
Kdd = (Kplus - 2.0*K + Kminus)/δx**2
print('Kernel Hessian Error (1st dim.):', nanmax(abs(Kdd - Kpp[:,:,0,0])/Kdd))
Kdd = (Kplus - 2.0*K + Kminus)/δx**2
print('Kernel Hessian Error (1st dim.):', nanmax(abs(Kdd - Kpp[:,:,0,0])/Kdd))
