# -*- coding: utf-8 -*-
"""
Example/test of the pyregress package.

This demonstration generates a random sample from a 2D Gaussian process.
Then, using the same kernel with all parameters known except two,
the posterior of this hyper-parameter is calculated and maximized.
"""
from numpy import zeros, linspace, empty, array, meshgrid, hstack
from numpy.random import random, randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyregress import *

# Setup the source sGP with exactly two hyper-parameters
Nt = Nd = 5*2**4
Xd = 8.0*(random(2*Nd)).reshape((-1, 2))

myK = Noise(w=0.1) + SquareExp(w=1.0, l=[0.7, 1.1])

# Generate the testing data from a source GP
sourceGP = GPP(zeros((0, 2)), zeros(0), myK)
Yd = sourceGP.sample(Xd)
Xt, Yt = (Xd.T, Yd.reshape(Nt))

# Setup the GPP object
myK = SquareExp(w=1.0, l=[LogNormal(guess=.4, std=.25),
                          LogNormal(guess=1.4, std=.25)]) + Noise(w=0.1)
myGP = GPP(Xd, Yd, myK)

# Inference over the entire domain
Ni = (75, 75)
xi1, xi2 = linspace(0.0, 8.0, Ni[0]), linspace(0.0, 8.0, Ni[1])
Xi1, Xi2 = meshgrid(xi1, xi2, indexing='ij')
Xi = hstack([Xi1.reshape((-1,1)), Xi2.reshape((-1,1))])
post_mean = myGP.inference(Xi, infer_std=False)
post_mean = post_mean.reshape(Ni)

# Maximize the hyper-parameter posterior
hopt_post, hopt_grad = myGP.hyper_posterior()
param = myGP.kernel.get_hp()

# Check that the posterior and its gradient are consistent
test_hyper, bounds = myGP.kernel._map_hyper()
test_hyper[:] = array([0.9, 0.9])
delta = 1e-5
d1, d2 = array([delta, 0.0]), array([0.0, delta])
h_post_t0, h_grad_t0 = myGP.hyper_posterior(test_hyper)
test_hyper += d1
h_post_t1, h_grad_t1 = myGP.hyper_posterior(test_hyper)
test_hyper += d2 - d1
h_post_t2, h_grad_t2 = myGP.hyper_posterior(test_hyper)
grad = 0.5*array([h_grad_t0[0] + h_grad_t1[0], h_grad_t0[1] + h_grad_t2[1]])
finite_diff = array([h_post_t1[0,0] - h_post_t0[0,0], h_post_t2[0,0] - h_post_t0[0,0]])/delta
print('Gradient:    ', grad)
print('Finite diff.:', finite_diff)
print('Error (abs.):', abs(grad - finite_diff))
print('Error (rel.):', abs(1.0 - finite_diff/grad))
print(' ')
# I previously used chech_grad (from scipy.optimize import check_grad),
# but I don't trust it anymore.

# Posterior of the hyper-parameter
Nh = (60, 60)
hyper1, hyper2 = linspace(0.2, 2.0, Nh[0]), linspace(0.2, 2.0, Nh[1])
h_post = empty(Nh)
h_grad1, h_grad2 = empty(Nh), empty(Nh)
for i in range(Nh[0]):
    for j in range(Nh[1]):
        test_hyper[:] = array([hyper1[i], hyper2[j]])
        h_post[i,j], h_grad = myGP.hyper_posterior(test_hyper)
        h_grad1[i,j], h_grad2[i,j] = h_grad[0], h_grad[1]


Yd_pred, Yd_std, std_res = myGP.loo(return_data=True, plot_results=True)


# Visualize
fig1 = plt.figure(figsize=(10,8), dpi=150)
plt.subplot(2,2,1)
plt.pcolormesh(hyper1, hyper2, -h_post.T)
plt.clim((-h_post).max()-20.0, None)
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