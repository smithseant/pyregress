# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:43:06 2014

@author: ben
"""
from numpy import zeros, linspace, sort
from numpy.random import random
import matplotlib.pyplot as plt

from pyregress import *

samples = 7
points = 30
Xs = linspace(0.0, 2.0, points)  # Evenly spaced points
#Xs =  sort(2.0*(random(points))).reshape((-1, 1))  # Random points

fig1 = plt.figure(figsize=(10, 8), dpi=150)
fig1.subplots_adjust(left=0.08, right=0.95,
                     bottom=0.08, top=0.90,
                     hspace=0.30, wspace=0.20)
plt.subplot(2, 2, 1)
my_K = SquareExp(w=2.0, l=0.25) 
my_gp1 = GPI(zeros((0, 1)), zeros(0), my_K)
Ys1 = my_gp1.sample(Xs,n_samples=samples)
plt.plot(Xs, Ys1.T, '.-', markersize=6)
plt.title('Squared Exp - w=2.0, l=0.25', fontsize=10)

plt.subplot(2, 2, 2)
my_K = SquareExp(w=0.5, l=1.0) 
my_gp2 = GPI(zeros((0, 1)), zeros(0), my_K)
Ys2 = my_gp2.sample(Xs,n_samples=samples)
plt.plot(Xs, Ys2.T, '.-', markersize=6)
plt.title('Squared Exp - w=0.5, l=1.0', fontsize=10)

plt.subplot(2, 2, 3)
my_K = RatQuad(w=0.6, l=0.3, α=1.0) 
my_gp3 = GPI(zeros((0, 1)), zeros(0), my_K)
Ys3 = my_gp3.sample(Xs,n_samples=samples)
plt.plot(Xs, Ys3.T, '.-', markersize=6)
plt.title('Rational Quad - w=0.6, l=0.3, alpha=1.0', fontsize=10)

plt.subplot(2, 2, 4)
my_K = GammaExp(w=0.6, l=0.3, γ=1.5) 
my_gp4 = GPI(zeros((0, 1)), zeros(0), my_K)
Ys4 = my_gp4.sample(Xs,n_samples=samples)
plt.plot(Xs, Ys4.T, '.-', markersize=6)
plt.title('Gamma Exp - w=0.5, l=1.0, gamma=1.5', fontsize=10)

fig1.text(0.45, 0.01, 'Sample Points', fontsize=16)
fig1.text(0.35, 0.95, 'Random Samples from GP', fontsize=20)

plt.show()