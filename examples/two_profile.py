# -*- coding: utf-8 -*-
"""
Example/test of the pyregress package.

This demonstration generates a random sample from a 2D Gaussian process.
Then, using the same kernel with all parameters known except two,
the posterior of this hyper-parameter is calculated and maximized.
"""
import cProfile#, pyprof2calltree
import numpy as np
from numpy.random import random, randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyregress import *

# Setup the source GP with exactly two hyper-parameters
Nt = Nd = 5*2**4
Xt = Xd = 8.0*(random(2*Nd)).reshape((-1, 2))

myK = Noise(w=0.1) + SquareExp(w=1.0, l=[Constant(0.7), Constant(1.1)])

# Generate the testing data from sourceGP
sourceGP = GPP(Xt, np.zeros((Nt, 1)), myK)
Yt = Yd = sourceGP.Kdd.dot(randn(Nt)).reshape((Nt, 1))
Xt, Yt = Xd.T, Yd.reshape(Nt)

# Setup the GPR object
myGP = GPP(Xd, Yd, myK)

# Posterior of the hyper-parameter
Nh = (60, 60)
hyper1, hyper2 = np.linspace(0.2, 2.0, Nh[0]), np.linspace(0.2, 2.0, Nh[1])
h_post = np.empty(Nh)
h_grad1, h_grad2 = np.empty(Nh), np.empty(Nh)

prof = cProfile.Profile()
prof.enable()
for i in xrange(Nh[0]):
    for j in xrange(Nh[1]):
        params = np.array([hyper1[i], hyper2[j]])
        h_post[i,j], h_grad = myGP.hyper_posterior(params, p_mapped)
        h_grad1[i,j], h_grad2[i,j] = h_grad[0], h_grad[1]
prof.create_stats()
prof.dump_stats('hyper_posterior1.raw.prof')
#pyprof2calltree.convert(prof.getstats(), 'hyper_posterior.kgrind')

# Check that the posterior and its gradient are consistent
test_hyper = np.array([0.9, 0.9])
delta = 1e-5
d1, d2 = np.array([delta, 0.0]), np.array([0.0, delta])
h_post_t0, h_grad_t0 = myGP.hyper_posterior(test_hyper, p_mapped)
h_post_t1, h_grad_t1 = myGP.hyper_posterior(test_hyper+d1, p_mapped)
h_post_t2, h_grad_t2 = myGP.hyper_posterior(test_hyper+d2, p_mapped)
grad = 0.5*np.array([h_grad_t0[0] + h_grad_t1[0], h_grad_t0[1] + h_grad_t2[1]])
finite_diff = np.array([h_post_t1[0,0] - h_post_t0[0,0], h_post_t2[0,0] - h_post_t0[0,0]])/delta
print 'Gradient:    ', grad
print 'Finite diff.:', finite_diff
print 'Error (abs.):', abs(grad - finite_diff)
print 'Error (rel.):', abs(1.0 - finite_diff/grad)
print ' '
# I previously used chech_grad (from scipy.optimize import check_grad),
# but I don't trust it anymore.

# Maximize the hyper-parameter posterior
p_mapped[:] = 0.9
myK.map_hyper(p_mapped, unmap=True)
myGPR, param = myGPR.maximize_hyper_posterior()
print 'Optimized value of the hyper-parameters:', param

# Inference over the entire domain
Ni = (75, 75)
xi1, xi2 = np.linspace(0.0, 8.0, Ni[0]), np.linspace(0.0, 8.0, Ni[1])
Xi1, Xi2 = np.meshgrid(xi1, xi2, indexing='ij')
Xi = np.hstack([Xi1.reshape((-1,1)), Xi2.reshape((-1,1))])

prof = cProfile.Profile()
prof.enable()
post_mean = myGPR.inference(Xi)
prof.create_stats()
prof.dump_stats('inference1.raw.prof')
#pyprof2calltree.convert(prof.getstats(), 'inference.kgrind')