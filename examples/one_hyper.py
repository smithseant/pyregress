# -*- coding: utf-8 -*-
"""
Example/test of the pyregress package.

This demonstration generates a random sample from a 1D Gaussian process.
Then, using the same kernel with all parameters known except one,
the posterior of this hyper-parameter is calculated and maximized.
"""
import numpy as np
from numpy.random import random, randn
import matplotlib.pyplot as plt
from pyregress import *
plt.close('all')

# Setup the source GP with any single hyper-parameter
Nt = Nd = 5*2**1
Xd = Xt =  2.0*(random(Nd)).reshape((-1,1))

def prior_mean(inputs):
    return 2.0*(inputs/5.0 - 0.5)

#myK = RatQuad(w=1.0, l=0.5, alpha=1.0)
myK = SquareExp(w=1.0, l=0.5) 

# Generate the testing data from the source GP
sourceGP = GPR(Xt, np.ones((Nt, 1)), myK)
Yt = Yd = sourceGP.Kdd.dot(randn(Nt)).reshape((Nt, 1)) + prior_mean(Xt)

#Yd = sourceGP.sample(Xt) + prior_mean(Xt)
#Yt = Yd = sourceGP.Kdd.dot(randn(Nt)).reshape((Nt, 1))
(Xt, Yt) = (Xd.reshape(Nt), Yd.reshape(Nt))

# Setup the GPR object
#myK = RatQuad(w=Constant(1.0), l=0.5, alpha=1.0) + Noise(w=0.1)
myK = SquareExp(w=1.0, l=Constant(0.5)) + Noise(w=0.1)

myGPR = GPR(Xd, Yd, myK, Yd_mean=prior_mean)
#param = myGPR.kernel.p[myGPR.kernel.hp_id[0]]
param = myGPR.kernel.terms[0].p[myGPR.kernel.terms[0].hp_id[0]]
(hopt_post, hopt_grad) = myGPR.hyper_posterior(param)

# Inference over the entire domain
Ni = 100
Xi = np.linspace(0.0, 2.0, Ni).reshape((-1,1))
(post_mean, post_std) = myGPR.inference(Xi, infer_std=True)
Xi = Xi.reshape(-1)
(post_mean, post_std) = (post_mean.reshape(-1), post_std.reshape(-1))

# Check that the posterior and its gradient are consistent
test_hyper = myGPR.kernel._map_hyper()    
test_hyper[:] = 0.9
(h_post_t0, h_grad_t0) = myGPR.hyper_posterior(test_hyper)
delta = 1e-5
test_hyper += delta
(h_post_t1, h_grad_t1) = myGPR.hyper_posterior(test_hyper)
grad = 0.5*(h_grad_t0[0] + h_grad_t1[0])
finite_diff = (h_post_t1[0,0] - h_post_t0[0,0])/delta
print 'Gradient:    ', grad
print 'Finite diff.:', finite_diff
print 'Error (abs.):', abs(grad - finite_diff)
print 'Error (rel.):', abs(1.0 - finite_diff/grad)
print ' '

## Posterior of the hyper-parameter
hyper = np.linspace(0.2, 3.0, 100)
h_post = np.empty(np.shape(hyper))
h_grad = np.empty(np.shape(hyper))
for i in xrange(len(hyper)):
    test_hyper[:] = hyper[i:i+1]
    (h_post[i], h_grad[i]) = myGPR.hyper_posterior(test_hyper)


# Visualize
fig1 = plt.figure(figsize=(5,4), dpi=150)
plot1 = fig1.add_subplot(2,1,1)
plot1.plot(hyper,-h_post, linewidth=2.0, label='Probability')
plot2 = fig1.add_subplot(2,1,2)
plot2.plot(hyper,-h_grad, linewidth=2.0)
plot1.xaxis.set_visible(False)
fig1.subplots_adjust(left=0.18, right=0.95, bottom=0.15,top=0.9, hspace=0.1)
plot1.set_title('Hyper-Parameter Posterior', fontsize=16)
plot1.set_ylabel('ln(P)', fontsize=12)
plot2.set_ylabel('d/dl ln(P)', fontsize=12)
plot2.set_xlabel('hyper-parameter', fontsize=12)

plot1.plot(param,-hopt_post, 'ro', label='Maximum')
plot2.plot(param,-hopt_grad, 'ro')
plot1.legend(loc='best',  prop={'size':10}, numpoints=1)

fig2 = plt.figure(figsize=(5,3), dpi=150)
p1, = plt.plot(Xt,Yt, 'ko')
p2, = plt.plot(Xi, post_mean, linewidth=2.0, color='blue')
plt.fill_between(Xi, post_mean-2.0*post_std, post_mean+2.0*post_std, alpha=0.25)    
p3 = plt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor='blue', alpha=0.25)
fig2.subplots_adjust(left=0.15, right=0.95, bottom=0.15,top=0.9)
plt.title('Inference', fontsize=16)
plt.xlabel('Independent Variable, X', fontsize=12)
plt.ylabel('Dependent Variable, Y', fontsize=12)
plt.legend([p1,p2,p3], ('Data', 'Inferred mean', 'Inference +/-2 sigma'),
           numpoints=1, loc='best', prop={'size':8})

plt.show()
