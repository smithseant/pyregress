# -*- coding: utf-8 -*-
"""
This script demonstrates how data generated as a sum of multiple kernels can
be decomposed, by inference, into the individual kernel contributions.

Created on Mon Jun 16 13:58:59 2014
@author: Sean T. Smith
"""

from numpy import zeros, linspace, expand_dims, maximum, abs, sqrt
import matplotlib.pyplot as plt
from pyregress import *

# Create a data kernel as the total of a high- and a low-frequency kernels
Ka = SquareExp(w=0.5, l=3.0)
Kb = SquareExp(w=2.0, l=12.0)
Kt = Ka + Kb

# Generate the training data from a source GP
Nd = 10
Xd = expand_dims(linspace(0.0, 50.0, Nd), 1)
sourceGP = GPP(zeros((0, 1)), zeros(0), Kt)
Yd = sourceGP.sample(Xd)

# Calculate the posterior GPs based on each of the individual kernels
Ni = 200
Xi = expand_dims(linspace(-20.0, 70.0, Ni), 1)
myGP = GPP(Xd, Yd, Kt)
meanYt_i, stdYt_i = myGP.inference(Xi, infer_std=True)
meanYa_i, stdYa_i = myGP.inference(Xi, infer_std=True, sum_terms=0)
meanYb_i, stdYb_i = myGP.inference(Xi, infer_std=True, sum_terms=1)

# Test if the decomposition is consistent, and reformat data for plotting
meanYtest_i = meanYa_i + meanYb_i
Xi, Xd, Yd = Xi.reshape(Ni), Xd.reshape(Nd), Yd.reshape(Nd)
meanYt_i, stdYt_i = meanYt_i.reshape(Ni), stdYt_i.reshape(Ni)
meanYa_i, stdYa_i = meanYa_i.reshape(Ni), maximum(0.0, stdYa_i.reshape(Ni))
meanYb_i, stdYb_i = meanYb_i.reshape(Ni), maximum(0.0, stdYb_i.reshape(Ni))
meanYtest_i = meanYtest_i.reshape(Ni)

# Plotting
fig1 = plt.figure(figsize=(15,4.5), dpi=100)
plt.subplot(1, 3, 1)
plt.plot(Xd, Yd, linestyle='None', marker='o', color='k', label='Data')
plt.plot(Xi, meanYt_i, color='b', label='Interpolant')
plt.fill_between(Xi, meanYt_i-stdYt_i, meanYt_i+stdYt_i, alpha=0.25)
plt.title('Original Interpolation')
plt.legend(numpoints=1)
plt.xlim(-20.0, 70.0)
plt.subplot(1, 3, 2)
plt.plot(Xi, meanYa_i, label='High-Freq. Contrib.')
plt.fill_between(Xi, meanYa_i-stdYa_i, meanYa_i+stdYa_i, alpha=0.25)
plt.plot(Xi, meanYb_i, label='Low-Freq. Contrib.')
plt.fill_between(Xi, meanYb_i-stdYb_i, meanYb_i+stdYb_i, alpha=0.25,
                 facecolor='g', edgecolor='g')
plt.title('Decomposition')
plt.legend()
plt.xlim(-20.0, 70.0)
plt.subplot(1, 3, 3)
plt.plot(Xi, abs(meanYt_i - meanYtest_i), label='Mean Err.')
#plt.plot(Xi, abs(sqrt(varYt_i) - sqrt(varYtest_i)), label='Std. Err.')
plt.title('Reconstruction Error')
plt.legend()
plt.xlim(-20.0, 70.0)
plt.show()