# -*- coding: utf-8 -*-
"""


Created on Thu Apr 17 15:48:02 2014
@author: Sean T. Smith, University of Utah
"""
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
        for i in range(Ndim):
            f += X[:,i]**2 / aniso[i]
    return f

def radius(X, Y, aniso=None):
    Ndim = X.shape[1]
    Nx, Ny = X.shape[0], Y.shape[0]
    Rk = empty((Nx, Ny, Ndim))
    for k in range(Ndim):
        Rk[:, :, k] = tile(X[:,[k]], (1, Ny)) - tile(Y[:,[k]].T, (Nx, 1))
        if isinstance(aniso, ndarray):
            Rk[:, :, k] /= aniso[k]
    return Rk

X = randn(Nx, 2)
Rk = radius(X, X)

my_kernel = [Noise(w=1.5),
             Noise(w=Jeffreys(1.5)),
             SquareExp(w=Jeffreys(1.5), l=2.0),
             SquareExp(w=1.5, l=Jeffreys(2.0)),
             SquareExp(w=Jeffreys(1.5), l=[2.0, 2.5]),
             SquareExp(w=1.5, l=[Jeffreys(2.0), 2.5]),
             SquareExp(w=1.5, l=[2.0, Jeffreys(2.5)]),
             SquareExp(w=Jeffreys(1.5), l=Jeffreys(2.0)),
             SquareExp(w=Jeffreys(1.5), l=[Jeffreys(2.0), 2.5]),
             SquareExp(w=Jeffreys(1.5), l=[2.0, Jeffreys(2.5)]),
             SquareExp(w=1.5, l=[Jeffreys(2.0), Jeffreys(2.5)]),
             GammaExp(w=Jeffreys(1.5), l=2.0, γ=1.5),
             GammaExp(w=1.5, l=Jeffreys(2.0), γ=1.5),
             GammaExp(w=Jeffreys(1.5), l=[2.0, 2.5], γ=1.5),
             GammaExp(w=1.5, l=[Jeffreys(2.0), 2.5], γ=1.5),
             GammaExp(w=1.5, l=[2.0, Jeffreys(2.5)], γ=1.5),
             GammaExp(w=1.5, l=2.0, γ=Uniform(2, 1.5)),
             GammaExp(w=1.5, l=[2.0, 2.5], γ=Uniform(2, 1.5)),
             GammaExp(w=Jeffreys(1.5), l=Jeffreys(2.0), γ=1.5),
             GammaExp(w=Jeffreys(1.5), l=[Jeffreys(2.0), 2.5], γ=1.5),
             GammaExp(w=Jeffreys(1.5), l=[2.0, Jeffreys(2.5)], γ=1.5),
             GammaExp(w=Jeffreys(1.5), l=2.0, γ=Uniform(2, 1.5)),
             GammaExp(w=Jeffreys(1.5), l=[2.0, 2.5], γ=Uniform(2, 1.5)),
             GammaExp(w=1.5, l=[Jeffreys(2.0), Jeffreys(2.5)], γ=1.5),
             GammaExp(w=1.5, l=Jeffreys(2.0), γ=Uniform(2, 1.5)),
             GammaExp(w=1.5, l=[Jeffreys(2.0), 2.5], γ=Uniform(2, 1.5)),
             GammaExp(w=1.5, l=[2.5, Jeffreys(2.5)], γ=Uniform(2, 1.5)),
             RatQuad(w=Jeffreys(1.5), l=2.0, α=1.5),
             RatQuad(w=1.5, l=Jeffreys(2.0), α=1.5),
             RatQuad(w=Jeffreys(1.5), l=[2.0, 2.5], α=1.5),
             RatQuad(w=1.5, l=[Jeffreys(2.0), 2.5], α=1.5),
             RatQuad(w=1.5, l=[2.0, Jeffreys(2.5)], α=1.5),
             RatQuad(w=1.5, l=2.0, α=Jeffreys(1.5)),
             RatQuad(w=1.5, l=[2.0, 2.5], α=Jeffreys(1.5)),
             RatQuad(w=Jeffreys(1.5), l=Jeffreys(2.0), α=1.5),
             RatQuad(w=Jeffreys(1.5), l=[Jeffreys(2.0), 2.5], α=1.5),
             RatQuad(w=Jeffreys(1.5), l=[2.0, Jeffreys(2.5)], α=1.5),
             RatQuad(w=Jeffreys(1.5), l=2.0, α=Jeffreys(1.5)),
             RatQuad(w=Jeffreys(1.5), l=[2.0, 2.5], α=Jeffreys(1.5)),
             RatQuad(w=1.5, l=[Jeffreys(2.0), Jeffreys(2.5)], α=1.5),
             RatQuad(w=1.5, l=Jeffreys(2.0), α=Jeffreys(1.5)),
             RatQuad(w=1.5, l=[Jeffreys(2.0), 2.5], α=Jeffreys(1.5)),
             RatQuad(w=1.5, l=[2.5, Jeffreys(2.5)], α=Jeffreys(1.5)),
             Noise(w=Jeffreys(1.5)) +
                SquareExp(w=1.5, l=[2.0, Jeffreys(2.5)]),
             SquareExp(w=1.5, l=Jeffreys(0.8)) *
                SquareExp(w=1.5, l=Jeffreys(2.5))]

my_hyper = my_kernel.get_φ()
my_gpr = GPI(X, test_func(X), my_kernel, optimize=False)
my_gpb = GPI(X, test_func(X), my_kernel, explicit_basis=[0, 1],
             optimize=False)

K, Kp = my_kernel(Rk, grad=True, block_diag=True)
P, Pp = my_gpr.posterior_φ(my_hyper, grad=True)
B, Bp = my_gpb.posterior_φ(my_hyper, grad=True)

my_hyper[0] += dx
Kplus = my_kernel(Rk, block_diag=True)
Pplus = my_gpr.posterior_φ(my_hyper, grad=False)
Bplus = my_gpb.posterior_φ(my_hyper, grad=False)
my_hyper[0] -= dx

my_hyper[0] -= dx
Kminus = my_kernel(Rk, block_diag=True)
Pminus = my_gpr.posterior_φ(my_hyper, grad=False)
Bminus = my_gpb.posterior_φ(my_hyper, grad=False)
my_hyper[0] += dx

Kd = (Kplus - Kminus) / (2 * dx)
print('Kernel Gradient Error (1st dim.):', nanmax(abs(Kd - Kp[:,:,0])/Kd))
Pd = (Pplus - Pminus) / (2 * dx)
print('Posterior Gradient Error (1st dim.):', abs((Pd[0,0] - Pp[0])/Pd[0,0]))
Bd = (Bplus - Bminus) / (2 * dx)
print('Posterior (w/ bases) Gradient Error (1st dim.):', abs((Bd[0,0] - Bp[0])/Bd[0,0]))

if len(my_hyper) > 1:
    my_hyper[1] += dx
    Kplus = my_kernel(Rk, block_diag=True)
    Pplus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] -= dx
    
    my_hyper[1] -= dx
    Kminus = my_kernel(Rk, block_diag=True)
    Pminus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] += dx
        
    Kd = (Kplus - Kminus)/(2.0*dx)
    print ('Kernel Gradient Error (2nd dim.):', nanmax(abs(Kd - Kp[:,:,1])/Kd))
    Pd = (Pplus - Pminus)/(2.0*dx)
    print ('Posterior Gradient Error (2nd dim.):', abs((Pd[0,0] - Pp[1])/Pd[0,0]))
