# -*- coding: utf-8 -*-
"""
Example/test of the pygpr package.

This demonstration generates a random sample from a 1D Gaussian process.
Then, using the same kernel with all parameters known except one,
the posterior of this hyper-parameter is calculated and maximized.
"""
import numpy as np
from numpy.random import random, randn
import matplotlib.pyplot as plt
from pygpr import *

# Setup the Training Data
Nt = Nd = 5*2**1
Xt = Xd = 8.0*(random(Nd)).reshape((-1,1))
#Xt = Xd = np.linspace(0.0, 8.0, Nt).reshape((-1,1))
Yt_prior = Yd_prior = 5.0*(Xt/8.0 - 0.5)

#(myK, myHyper) = (OU([1.0, 1.0]), [True, False])
#(myK, myHyper) = (GammaExp([1.0, 1.0, 2.0]), [True, False, False])
(myK, myHyper) = (SquareExp([1.0, [1.0]]), [False, [True]])
#(myK, myHyper) = (RatQuad([1.0, 1.0, 5.0]), [True, False, False])
#i_hyper = myHyper.index(True)
i_hyper = 1

trainingGPR = GPR(Xt, np.zeros(np.shape(Xt)), [myK], anisotropy=False)
Yt = Yd = trainingGPR.Kdd.dot(randn(Nt)).reshape((-1,1)) + Yt_prior
(Xt, Yt) = (Xd.reshape(-1), Yd.reshape(-1))

# Setup the GPR object
myGPR = GPR(Xd, Yd, [myK], anisotropy=False, Yd_mean=Yd_prior)
(myK.Nhyper, myK.hyper) = (1, myHyper)
p_mapped = np.empty(1)
myK.p[i_hyper] = p_mapped[0:1]

# Posterior of the hyper-parameter
#hyper = np.logspace(sp.log10(0.1),sp.log10(0.8),200)
hyper = np.linspace(0.2, 2.0, 100)
h_post = np.empty(np.shape(hyper))
h_grad = np.empty(np.shape(hyper))
for i in xrange(len(hyper)):
    (h_post[i], h_grad[i]) = myGPR.hyper_posterior(hyper[i:i+1], p_mapped)

# Check that the posterior and its gradient are consistent
test_hyper = np.array([0.9])
delta = 1e-5
(h_post_t0, h_grad_t0) = myGPR.hyper_posterior(test_hyper, p_mapped)
(h_post_t1, h_grad_t1) = myGPR.hyper_posterior(test_hyper+delta, p_mapped)
grad = 0.5*(h_grad_t0[0] + h_grad_t1[0])
finite_diff = (h_post_t1[0,0] - h_post_t0[0,0])/delta
print 'Gradient:    ', grad
print 'Finite diff.:', finite_diff
print 'Error (abs.):', abs(grad - finite_diff)
print 'Error (rel.):', abs(1.0 - finite_diff/grad)
print ' '
# I previously used chech_grad (from scipy.optimize import check_grad),
# but I don't trust it anymore.

# Maximize the hyper-parameter posterior
#myK.p[i_hyper] = 0.9
myK.p[i_hyper] = [0.9]
myGPR.maximize_hyper_posterior()
print 'Optimized value of the hyper-parameter:', myK.p[i_hyper]
(hopt_post, hopt_grad) = myGPR.hyper_posterior(myK.p[i_hyper:i_hyper+1], p_mapped)

# Inference over the entire domain
Ni = 100
Xi = np.linspace(0.0, 8.0, Ni).reshape((-1,1))
Yi_prior = 5.0*(Xi/8.0 - 0.5)
(post_mean, post_std) = myGPR.inference(Xi, Yi_mean=Yi_prior, infer_std=True)
Xi = Xi.reshape(-1)
(post_mean, post_std) = (post_mean.reshape(-1), post_std.reshape(-1))


# Visualize
fig1 = plt.figure(figsize=(5,4), dpi=150)
plot1 = fig1.add_subplot(2,1,1)
plot1.plot(hyper,-h_post, linewidth=2.0, label='Probability')
plot2 = fig1.add_subplot(2,1,2)
plot2.plot(hyper,-h_grad, linewidth=2.0)
plot1.xaxis.set_visible(False)
fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.1,top=0.9, hspace=0.1)
plot1.set_title('Hyper-Parameter Posterior', fontsize=16)
plot1.set_ylabel('ln(P)', fontsize=12)
plot2.set_ylabel('d/dl ln(P)', fontsize=12)
plot2.set_xlabel('hyper-parameter', fontsize=12)

plot1.plot(myK.p[i_hyper],-hopt_post, 'ro', label='Maximum')
plot2.plot(myK.p[i_hyper],-hopt_grad, 'ro')
plot1.legend(loc='lower center',  prop={'size':10}, numpoints=1)

fig2 = plt.figure(figsize=(5,3), dpi=150)
p1, = plt.plot(Xt,Yt, 'ko')
p2, = plt.plot(Xi,post_mean, linewidth=2.0, color='blue')
plt.fill_between(Xi, post_mean-post_std, post_mean+post_std, alpha=0.25)    
p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue', alpha=0.25)
fig2.subplots_adjust(left=0.15, right=0.95, bottom=0.15,top=0.9)
plt.title('Inference', fontsize=16)
plt.xlabel('Independent Variable, X', fontsize=12)
plt.ylabel('Dependent Variable, Y', fontsize=12)
plt.legend([p1,p2,p3], ('Data', 'Inferred mean', 'Uncertainty (1*sigma)'),
           numpoints=1, loc='best', prop={'size':8})

plt.show()