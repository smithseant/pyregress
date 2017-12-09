# -*- coding: utf-8 -*-
"""
This is a demonstration of Gaussian processes for which the length scale
is large and two data points are very close. The behavior of the error
is shown as as the smallest kernel eigenvalue drops below the effective
machine precision
"""
from numpy import array, empty, zeros, linspace, sin, exp, pi as π
import matplotlib.pyplot as plt
from pyregress import GPI, SquareExp

L = 2.0  # kernel length scale
δ = 2e-3  # narrow spacing
Δ = 10 * δ  # wide spacing

def source(x):
    return sin(π * x / L) * exp(-x / L) - 0.2

# Two similar points near 0.5:
Xd0 = array([0, 0.25, 0.5 - Δ/2, 0.5 + Δ/2, 0.75, 1])  # wide enough spacing
Xd1 = array([0, 0.25, 0.5 - δ/2, 0.5 + δ/2, 0.75, 1])  # too narrow spacing
Yd0 = source(Xd0)
Yd1 = source(Xd1)

# Generate the GPs:
kernel = SquareExp(w=1, l=L)
my_gpi0 = GPI(Xd0.reshape((-1, 1)), Yd0, kernel, fast=False)
print('spaced far has {} eigenvalues'.format(my_gpi0.LKdd[0].shape))
my_gpi1 = GPI(Xd1.reshape((-1, 1)), Yd1, kernel, fast=False)
print('spaced near has {} eigenvalues'.format(my_gpi1.LKdd[0].shape))

# Interpolate with error estimates:
Xi = empty(400)
Xi[:100] = linspace(0, 0.5 - 0.6*Δ, 100, endpoint=False)
Xi[100:300] = linspace(0.5 - 0.6*Δ, 0.5 + 0.6*Δ, 200, endpoint=False)
Xi[-100:] = linspace(0.5 + 0.6*Δ, 1.0, 100, endpoint=True)
Y = source(Xi)
Yi0, εi0 = my_gpi0(Xi, infer_std=True)
Yi1, εi1 = my_gpi1(Xi, infer_std=True)

# Plot the results
fig = plt.figure(figsize=(15, 7))
fig.add_subplot(1, 2, 1)
plt.plot(Xi, Y, color='k', linewidth=0.5, label='exact')
plt.plot(Xd0, Yd0, 'b.')
plt.plot(Xi, Yi0, color='b', linewidth=2.0, label='spaced far')
plt.plot(Xi, Yi0 + εi0, linestyle='--', color='b', linewidth=1.0)
plt.plot(Xi, Yi0 - εi0, linestyle='--', color='b', linewidth=1.0)
plt.plot(Xd1, Yd1, 'r.')
plt.plot(Xi, Yi1, color='r', linewidth=2.0, label='spaced close')
plt.plot(Xi, Yi1 + εi1, linestyle='--', color='r', linewidth=1.0)
plt.plot(Xi, Yi1 - εi1, linestyle='--', color='r', linewidth=1.0)
plt.xlim((0, 1))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function')
fig.add_subplot(1, 2, 2)
plt.plot(Xi, zeros(Xi.shape), 'k--', linewidth=0.5)
plt.plot(Xd0, zeros(Xd0.shape), 'b.')
plt.plot(Xi, εi0, linestyle='--', color='b', linewidth=1.0)
plt.plot(Xd1, zeros(Xd1.shape), 'r.')
plt.plot(Xi, εi1, linestyle='--', color='r', linewidth=1.0)
plt.xlim((0, 1))
plt.ylim((0, None))
plt.xlabel('x')
plt.ylabel('$\epsilon$')
plt.title('Error')
plt.show()