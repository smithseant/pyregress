# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:43:06 2014

@author: ben
"""
from numpy import zeros, linspace, sort
from numpy.random import random
from pyregress import *
import matplotlib.pyplot as plt

samples = 7
points = 30
Xs = linspace(0.0, 2.0, points).reshape((-1, 1))  # Evenly spaced points
#Xs =  sort(2.0*(random(points))).reshape((-1, 1))  # Random points

fig1 = plt.figure(figsize=(10, 8), dpi=150)
fig1.subplots_adjust(left=0.08, right=0.95,
                     bottom=0.08,top=0.90,
                     hspace=0.30, wspace=0.20)
plt.subplot(2,2,1)
myK = SquareExp(w=2.0, l=0.25) 
sourceGP1 = GPP(zeros((0, 1)), zeros(0), myK)
Ys1 = sourceGP1.sample(Xs, Nsamples=samples)
plt.plot(Xs, Ys1, '.-', markersize=6)
plt.title('Squared Exp - w=2.0, l=0.25', fontsize=10)

plt.subplot(2, 2, 2)
myK = SquareExp(w=0.5, l=1.0) 
sourceGP2 = GPP(zeros((0, 1)), zeros(0), myK)
Ys2 = sourceGP2.sample(Xs, Nsamples=samples)
plt.plot(Xs, Ys2, '.-', markersize=6)
plt.title('Squared Exp - w=0.5, l=1.0', fontsize=10)

plt.subplot(2, 2, 3)
myK = RatQuad(w=0.6, l=0.3, alpha=1.0) 
sourceGP3 = GPP(zeros((0, 1)), zeros(0), myK)
Ys3 = sourceGP3.sample(Xs, Nsamples=samples)
plt.plot(Xs, Ys3, '.-', markersize=6)
plt.title('Rational Quad - w=0.6, l=0.3, alpha=1.0', fontsize=10)

plt.subplot(2,2,4)
myK = GammaExp(w=0.6, l=0.3, gamma=1.5) 
sourceGP4 = GPP(zeros((0,1)), zeros(0), myK)
Ys4 = sourceGP4.sample(Xs, Nsamples=samples)
plt.plot(Xs, Ys4, '.-', markersize=6)
plt.title('Gamma Exp - w=0.5, l=1.0, gamma=1.5', fontsize=10)

fig1.text(0.45, 0.01, 'Sample Points', fontsize=16)
fig1.text(0.35, 0.95, 'Random Samples from GP', fontsize=20)

plt.show()