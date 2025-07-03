# -*- coding: utf-8 -*-
"""
Not fully developed!

Created on Thu Apr 17 15:48:02 2014
@author: Sean T. Smith, University of Utah
"""
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
            f += X[:,i]**2 / aniso[i]
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

X = randn(n_pts, 2)
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
             Noise(w=Jeffreys(1.5)) + SquareExp(w=1.5, l=[2.0, Jeffreys(2.5)]),
             SquareExp(w=1.5, l=Jeffreys(0.8)) * SquareExp(w=1.5, l=Jeffreys(2.5))]

my_hyper = my_kernel.get_φ()
my_gpr = GPI(X, test_func(X), my_kernel, optimize=False)
my_gpb = GPI(X, test_func(X), my_kernel, explicit_basis=[0, 1], optimize=False)

K, Kp = my_kernel(Rk, grad=True, block_diag=True)
P, Pp = my_gpr.posterior_φ(my_hyper, grad=True)
B, Bp = my_gpb.posterior_φ(my_hyper, grad=True)

my_hyper[0] += δx
Kplus = my_kernel(Rk, block_diag=True)
Pplus = my_gpr.posterior_φ(my_hyper, grad=False)
Bplus = my_gpb.posterior_φ(my_hyper, grad=False)
my_hyper[0] -= δx

my_hyper[0] -= δx
Kminus = my_kernel(Rk, block_diag=True)
Pminus = my_gpr.posterior_φ(my_hyper, grad=False)
Bminus = my_gpb.posterior_φ(my_hyper, grad=False)
my_hyper[0] += δx

Kd = (Kplus - Kminus) / (2 * δx)
print('Kernel Gradient Error (1st dim.):', nanmax(abs(Kd - Kp[:,:,0])/Kd))
Pd = (Pplus - Pminus) / (2 * δx)
print('Posterior Gradient Error (1st dim.):', abs((Pd[0,0] - Pp[0])/Pd[0,0]))
Bd = (Bplus - Bminus) / (2 * δx)
print('Posterior (w/ bases) Gradient Error (1st dim.):', abs((Bd[0,0] - Bp[0])/Bd[0,0]))

if len(my_hyper) > 1:
    my_hyper[1] += δx
    Kplus = my_kernel(Rk, block_diag=True)
    Pplus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] -= δx
    
    my_hyper[1] -= δx
    Kminus = my_kernel(Rk, block_diag=True)
    Pminus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] += δx
        
    Kd = (Kplus - Kminus)/(2.0*δx)
    print ('Kernel Gradient Error (2nd dim.):', nanmax(abs(Kd - Kp[:,:,1])/Kd))
    Pd = (Pplus - Pminus)/(2.0*δx)
    print ('Posterior Gradient Error (2nd dim.):', abs((Pd[0,0] - Pp[1])/Pd[0,0]))
