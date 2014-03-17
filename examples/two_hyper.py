# -*- coding: utf-8 -*-
"""
Example/test of the pyregress package.

This demonstration generates a random sample from a 2D Gaussian process.
Then, using the same kernel with all parameters known except two,
the posterior of this hyper-parameter is calculated and maximized.
"""
import numpy as np
from numpy.random import random, randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyregress import *
plt.close('all')

# Setup the source sGP with exactly two hyper-parameters
Nt = Nd = 5*2**4
Xt = Xd = 8.0*(random(2*Nd)).reshape((-1,2))

myK = Noise([0.1])+SquareExp([1.0, [0.7, 1.1]])
myHyper = [[False], [False, logNormal(mean=0.2,std=0.25)]]

# Setup hyper-parameters in the BaseKernels and map to a single array
myK.declare_hyper(myHyper)
p_mapped = np.empty(2)
i = 0
myK.map_hyper(p_mapped)

# Generate the testing data from the source GP
sourceGPR = GPR(Xt, np.zeros((Nt, 1)), myK, anisotropy=False)
Yt = Yd = sourceGPR.Kdd.dot(randn(Nt)).reshape((Nt, 1))
(Xt, Yt) = (Xd.T, Yd.reshape(Nt))

# Setup the GPR object
myGPR = GPR(Xd, Yd, myK, anisotropy=False)

# Posterior of the hyper-parameter
Nh = (60, 60)
(hyper1, hyper2) = (np.linspace(0.2, 2.0, Nh[0]), np.linspace(0.2, 2.0, Nh[1]))
h_post = np.empty(Nh)
(h_grad1, h_grad2) = (np.empty(Nh), np.empty(Nh))
for i in xrange(Nh[0]):
    for j in xrange(Nh[1]):
        params = np.array([hyper1[i], hyper2[j]])
        (h_post[i,j], h_grad) = myGPR.hyper_posterior(params, p_mapped)
        (h_grad1[i,j], h_grad2[i,j]) = (h_grad[0], h_grad[1])

# Check that the posterior and its gradient are consistent
test_hyper = np.array([0.9, 0.9])
delta = 1e-5
(d1, d2) = (np.array([delta, 0.0]), np.array([0.0, delta]))
(h_post_t0, h_grad_t0) = myGPR.hyper_posterior(test_hyper, p_mapped)
(h_post_t1, h_grad_t1) = myGPR.hyper_posterior(test_hyper+d1, p_mapped)
(h_post_t2, h_grad_t2) = myGPR.hyper_posterior(test_hyper+d2, p_mapped)
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
(myGPR, param) = myGPR.maximize_hyper_posterior()
print 'Optimized value of the hyper-parameters:', param

# Inference over the entire domain
Ni = (75, 75)
(xi1, xi2) = (np.linspace(0.0, 8.0, Ni[0]), np.linspace(0.0, 8.0, Ni[1]))
(Xi1, Xi2) = np.meshgrid(xi1, xi2, indexing='ij')
Xi = np.hstack([Xi1.reshape((-1,1)), Xi2.reshape((-1,1))])
post_mean = myGPR.inference(Xi, infer_std=False)
post_mean = post_mean.reshape(Ni)


# Visualize
fig1 = plt.figure(figsize=(10,8), dpi=150)
plt.subplot(2,2,1)
plt.pcolormesh(hyper1, hyper2, -h_post.T)
plt.clim(np.max(-h_post)-20.0, None)
plt.colorbar()
plt.title('Log Hyper-Parameter Posterior', fontsize=14)
plt.xlabel('Hyper-parameter 1', fontsize=12)
plt.ylabel('Hyper-parameter 2', fontsize=12)
plt.plot(param[0], param[1], 'kx', label='Maximum')

plt.subplot(2,2,3)
plt.pcolormesh(hyper1, hyper2, -h_grad1.T)
plt.clim(-20, 20)
plt.colorbar()
plt.title('Gradient wrt. Param. 1', fontsize=14)
plt.xlabel('Hyper-parameter 1', fontsize=12)
plt.ylabel('Hyper-parameter 2', fontsize=12)
plt.plot(param[0], param[1], 'kx', label='Maximum')

plt.subplot(2,2,4)
plt.pcolormesh(hyper1, hyper2, -h_grad2.T)
plt.clim(-20, 20)
plt.colorbar()
plt.title('Gradient wrt. Param. 2', fontsize=14)
plt.xlabel('Hyper-parameter 1', fontsize=12)
plt.ylabel('Hyper-parameter 2', fontsize=12)
plt.plot(param[0], param[1], 'kx', label='Maximum')

fig1.subplots_adjust(left=0.08, right=0.95,
                     bottom=0.08,top=0.90,
                     hspace=0.30, wspace=0.20)

fig2 = plt.figure(figsize=(8,5), dpi=150)
ax = fig2.gca(projection='3d')
ax.plot_surface(Xi1, Xi2, post_mean, alpha=0.75,
                linewidth=0.5, cmap=mpl.cm.jet, rstride=1, cstride=1)
ax.scatter(Xt[0,:], Xt[1,:],Yt, c='black', s=50)
ax.set_title('Inference', fontsize=16)
ax.set_xlabel('Independent variable, X1', fontsize=12)
ax.set_ylabel('Independent variable, X2', fontsize=12)
ax.set_zlabel('Dependent variable, Y', fontsize=12)

plt.show()