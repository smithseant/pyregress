# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:22:40 2014

@author: sean
"""
from numpy import array, empty, expand_dims, linspace, meshgrid, reshape, vstack

from transforms import Logarithm, Probit, ProbitBeta, Logit

dx = 1e-6
x1 = linspace(2.0*dx, 1.0-2.0*dx, 20)
x2 = linspace(2.0*dx, 2.0-2.0*dx, 20)
x1mesh, x2mesh = meshgrid(x1, x2)
X = vstack((reshape(x1mesh, -1), reshape(x2mesh, -1))).T

def y_func(x):
    return expand_dims( x.dot(array([0.5, 0.25])), 1)
Y = y_func(X)
my_probit = Probit(Y)
Zprobit = my_probit(Y)

grad_Zprobit = empty(X.shape)
for j in range(X.shape[1]):
    Xminus, Xplus = X.copy(), X.copy()
    Xminus[:, j] -= dx
    Xplus[:, j] += dx
    grad_Zprobit[:, j] = ((my_probit(y_func(Xplus)) - my_probit(y_func(Xminus))) / (2.0*dx))[:,0]

Yout, Ygrad_out = my_probit(Zprobit, inverse=True, grad_z=grad_Zprobit)

print Ygrad_out