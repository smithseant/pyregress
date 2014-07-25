# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:22:40 2014

@author: sean
"""
from numpy import array, empty, expand_dims, linspace, meshgrid, reshape, vstack

from transforms import Logarithm, Probit, ProbitBeta, Logit

dx = 1e-5
x1 = linspace(1e2*dx, 1.0-1e2*dx, 40)
x2 = linspace(1e2*dx, 2.0-1e2*dx, 40)
x1mesh, x2mesh = meshgrid(x1, x2)
X = vstack((reshape(x1mesh, -1), reshape(x2mesh, -1))).T

def fy(x):
    return expand_dims(x.dot(array([0.5, 0.25])), 1)
Y = fy(X)
#trans = Logarithm(Y)
trans = Probit(Y)
#trans = ProbitBeta(Y)
#trans = Logit(Y)
Zprobit = trans(Y)

grad_Zprobit = empty(X.shape)
for j in range(X.shape[1]):
    Xminus, Xplus = X.copy(), X.copy()
    Xminus[:, j] -= dx
    Xplus[:, j] += dx
    grad_Zprobit[:, j] = (0.5*(trans(fy(Xplus)) - trans(fy(Xminus))) /dx)[:, 0]

Yout, Ygrad_out = trans(Zprobit, inverse=True, grad_z=grad_Zprobit)

print Ygrad_out