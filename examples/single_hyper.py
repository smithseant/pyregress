# -*- coding: utf-8 -*-
"""
Example/test of the pygpr package.

This demonstration generates a random sample from a Gaussian process.
Then, using the same kernel with all parameters know except lengthscale,
the posterior of the lengthscale is calculated and maximized.
"""
import numpy as np
from numpy.random import random, randn
from scipy.optimize import check_grad
import matplotlib.pyplot as plt
from pygpr import *

# Setup the Training Data
Nt = Nd = 5*2**1
Xt = Xd = 3.0*(random(Nd)).reshape((-1,1))

#Xt = Xd = np.linspace(0.0, 3.0, Nt).reshape((-1,1))
#myK = OU([1.0, 0.4])
#hyper_spec = {'OU':[False, True]}
#myK = GammaExp([1.0, 0.4, 2.0])
#hyper_spec = {'GammaExp':[False, True, False]}
myK = SquareExp([1.0, 0.4])
hyper_spec = {'SquareExp':[False, True]}
#myK = RatQuad([1.0, 0.4, 5.0])
#hyper_spec = {'RatQuad':[False, True, False]}

trainingGPR = GPR(Xt, np.zeros(np.shape(Xt)), [myK], anisotropy=False)
Yt = Yd = trainingGPR.Kdd.dot(randn(Nt)).reshape((-1,1))
(Xt, Yt) = (Xd.reshape(-1), Yd.reshape(-1))

# Setup the GPR object
myGPR = GPR(Xd, Yd, [myK], anisotropy=False)
(myK.Nhyper, myK.hyper[1]) = (1, True)
p_mapped = np.array([0.1])
myK.p[1] = p_mapped[0:1]

# Posterior of the hyper-parameter - length
#length = np.logspace(sp.log10(0.1),sp.log10(0.8),200)
length = np.linspace(0.2,0.8,300)
h_post = np.empty(np.shape(length))
h_grad = np.empty(np.shape(length))
for i in xrange(len(length)):
    (h_post[i], h_grad[i]) = myGPR.hyper_posterior(length[i:i+1], p_mapped)

# Check that the posterior and its gradient are consistent
def f(p, p_mapped):
    (func, grad) = myGPR.hyper_posterior(p, p_mapped)
    return func
def fprime(p, p_mapped):
    (func, grad) = myGPR.hyper_posterior(p, p_mapped)
    return grad
print 'Error between provided gradient and finite difference:'
print check_grad(f, fprime, np.array([0.3]), p_mapped)
print ' '

# Maximize the hyper-parameter posterior
myK.p[1] = 0.3
myGPR.maximize_hyper_posterior(hyper_spec)
print 'Optimized value of lengthscale:', myK.p[1]
(hopt_post, hopt_grad) = myGPR.hyper_posterior(myK.p[1:2], p_mapped)

# Inference over the entire domain
Ni = 100
Xi = np.linspace(0.0, 3.0, Ni).reshape((-1,1))
(post_mean, post_std) = myGPR.inference(Xi, infer_std=True)
Xi = Xi.reshape(-1)
(post_mean, post_std) = (post_mean.reshape(-1), post_std.reshape(-1))


# Visualize
fig1 = plt.figure(figsize=(5,4), dpi=150)
plot1 = fig1.add_subplot(2,1,1)
plot1.plot(length,-h_post, linewidth=2.0, label='Probability')
plot2 = fig1.add_subplot(2,1,2)
plot2.plot(length,-h_grad, linewidth=2.0)
plot1.xaxis.set_visible(False)
fig1.subplots_adjust(left=0.15, right=0.95, bottom=0.1,top=0.9, hspace=0.1)
plot1.set_title('Hyper-Parameter Posterior', fontsize=16)
plot1.set_ylabel('ln(P)', fontsize=12)
plot2.set_ylabel('d/dl ln(P)', fontsize=12)
plot2.set_xlabel('Lengthscale, l', fontsize=12)

plot1.plot(myK.p[1],-hopt_post, 'ro', label='Maximum')
plot2.plot(myK.p[1],-hopt_grad, 'ro')
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