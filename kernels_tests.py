# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:48:02 2014

@author: sean
"""
from numpy import ndarray, zeros, abs
from numpy.random import randn

from pyregress0 import *
from hyper_params import *
from kernels import *

Nx = 10
dx = 1e-3  # doesn't seem to behave well smaller than 1e-4.

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

#my_kernel = SquareExp(w=Constant(guess=1.5), l=2.0)
#my_kernel = SquareExp(w=1.5, l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, 2.5])
#my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=1.5, l=[2.0, Constant(guess=2.5)])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=Constant(guess=2.0))
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[Constant(guess=2.0), 2.5])
#my_kernel = SquareExp(w=Constant(guess=1.5), l=[2.0, Constant(guess=2.5)])
my_kernel = SquareExp(w=1.5, l=[Constant(guess=2.0), Constant(guess=2.5)])

my_hyper = my_kernel._map_hyper()
my_gpr = GPR(X, test_func(X), my_kernel)

K, Kp, Kpp = my_kernel(Rk, grad='Hess')
P, Pp, Ppp = my_gpr.hyper_posterior(my_hyper, grad='Hess')

my_hyper[0] += dx
Kplus = my_kernel(Rk)
Pplus = my_gpr.hyper_posterior(my_hyper, grad=False)
my_hyper[0] -= dx

my_hyper[0] -= dx
Kminus = my_kernel(Rk)
Pminus = my_gpr.hyper_posterior(my_hyper, grad=False)
my_hyper[0] += dx

Kd = (Kplus - Kminus)/(2.0*dx)
print 'Kernel Gradient Error (1st dim.):', abs(Kd - Kp[:,:,0]).max()

Kdd = (Kplus - 2.0*K + Kminus)/dx**2
print 'Kernel Hessian Error (1st dim.):', abs(Kdd - Kpp[:,:,0,0]).max()

Pd = (Pplus - Pminus)/(2.0*dx)
print 'Posterior Gradient Error (1st dim.):', abs(Pd[0,0] - Pp[0])

Pdd = (Pplus - 2.0*P + Pminus)/dx**2
print 'Posterior Hessian Error (1st dim.):', abs(Pdd[0,0] - Ppp[0,0])


if len(my_hyper) > 1:
    my_hyper[1] += dx
    Kplus = my_kernel(Rk)
    Pplus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] -= dx
    
    my_hyper[1] -= dx
    Kminus = my_kernel(Rk)
    Pminus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[1] += dx
        
    Kd = (Kplus - Kminus)/(2.0*dx)
    print 'Kernel Gradient Error (2nd dim.):', abs(Kd - Kp[:,:,1]).max()
    
    Kdd = (Kplus - 2.0*K + Kminus)/dx**2
    print 'Kernel Hessian Error (2nd dim.):', abs(Kdd - Kpp[:,:,1,1]).max()
    
    Pd = (Pplus - Pminus)/(2.0*dx)
    print 'Posterior Gradient Error (2nd dim.):', abs(Pd[0,0] - Pp[1])
    
    Pdd = (Pplus - 2.0*P + Pminus)/dx**2
    print 'Posterior Hessian Error (2nd dim.):', abs(Pdd[0,0] - Ppp[1,1])
    
    my_hyper[0] += dx
    my_hyper[1] += dx
    Kplusplus = my_kernel(Rk)
    Pplusplus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[0] -= dx
    my_hyper[1] -= dx
    
    my_hyper[0] += dx
    my_hyper[1] -= dx
    Kplusminus = my_kernel(Rk)
    Pplusminus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[0] -= dx
    my_hyper[1] += dx
    
    my_hyper[0] -= dx
    my_hyper[1] += dx
    Kminusplus = my_kernel(Rk)
    Pminusplus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[0] += dx
    my_hyper[1] -= dx
    
    my_hyper[0] -= dx
    my_hyper[1] -= dx
    Kminusminus = my_kernel(Rk)
    Pminusminus = my_gpr.hyper_posterior(my_hyper, grad=False)
    my_hyper[0] += dx
    my_hyper[1] += dx
    
    Kdd = (Kplusplus - Kplusminus - Kminusplus + Kminusminus)/(2.0*dx)**2
    print 'Kernel Hessian Error (cross dim.):', abs(Kdd - Kpp[:,:,0,1]).max()
    
    Pdd = (Pplusplus - Pplusminus - Pminusplus + Pminusminus)/(2.0*dx)**2
    print 'Posterior Hessian Error (cross dim.):', abs(Pdd[0,0] - Ppp[0,1])