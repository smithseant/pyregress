# -*- coding: utf-8 -*-
"""
Verifcation test for pyregress package.
"""
from numpy import empty, array, linspace
from scipy import sin, exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyregress import *

# True function (create training data)
def funcIn(x):
    return sin(5.*x) * exp(.25*x)

# Setup source GP with multiple hyper-parameter
x_all = linspace(1.0,2.0).T
y_all = funcIn(x_all)
Xd1 = array([[ 1.0], [1.1], [1.3], [1.9], [2.0]])
Yd1 = funcIn(Xd1)

#myK1 = Noise(w=0.1) + SquareExp(w=1.0, l=0.1)

#shift = shift_to_zero()

# Setup the GPR object
myGPR1 = GPP(Xd1, Yd1, Noise(w=0.1) + SquareExp(w=1.0, l=0.1))

# Infered points
Xi1 = linspace(1.0, 2.0, 100)
(Yi1, Yi1std) = myGPR1.inference(Xi1, infer_std=True)
(Yi1, Yi1std) = (Yi1.reshape(-1), Yi1std.reshape(-1))
## Plot
#fig1 = plt.figure(figsize=(5,3), dpi=150)
#plt.plot(Xd1,Yd1,'ko',label='Y1 data')
#plt.plot(x_all,y_all,'k',label='Y1 func')
#plt.plot(Xi1,Yi1,'-b',label='Mean')
#plt.fill_between(Xi1, Yi1-Yi1std, Yi1+Yi1std, alpha=0.25)
#plt.Rectangle((0.0, 0.0), 2.0, 50.0, facecolor='blue', alpha=0.25)
#plt.legend(numpoints=1, loc='best', prop={'size':8})
#plt.xlim([0.9,2.1])
#plt.show()

#------------------------------------------------------------------
#------------------------------------------------------------------

# Setup source GP with multiple hyper-parameter
#myK2 = Noise(w=0.1) + SquareExp(w=1.0, l=0.1)
myK2 = Noise(w=0.1) + SquareExp(w=1.0, l=Jeffreys(guess=0.5))
#myHyper2 = [[False], [False, Gamma(1.,2.)]]
#myHyper2 = [[False], [False, LogNormal(mean=.5,std=.2)]]

# Setup hyper-parameters in the kernel and map to an array
#myK2.declare_hyper(myHyper2)
#p_mapped2 = empty(1)
#myK2.map_hyper(p_mapped2)

# Setup the GPR object
myGPR2 = GPP(Xd1, Yd1, myK2)

### Posterior of the hyper-parameter
hyper = linspace(.1, 1., 100)
#h_post = empty(shape(hyper))
#h_grad = empty(shape(hyper))
#for i in xrange(len(hyper)):
#    (h_post[i], h_grad[i]) = myGPR2.hyper_posterior(hyper[i:i+1], p_mapped2)

#fig3 = plt.figure(figsize=(5,4), dpi=150)
#plot1 = fig3.add_subplot(2,1,1)
#plot1.plot(hyper,-h_post, linewidth=2.0, label='Probability')
#plot2 = fig3.add_subplot(2,1,2)
#plot2.plot(hyper,-h_grad, linewidth=2.0)
#plot1.xaxis.set_visible(False)
#fig3.subplots_adjust(left=0.15, right=0.95, bottom=0.1,top=0.9, hspace=0.1)
#plot1.set_title('Hyper-Parameter Posterior', fontsize=16)
#plot1.set_ylabel('ln(P)', fontsize=12)
#plot2.set_ylabel('d/dl ln(P)', fontsize=12)
#plot2.set_xlabel('hyper-parameter', fontsize=12)
#
##plot1.plot(param,-hopt_post, 'ro', label='Maximum')
##plot2.plot(param,-hopt_grad, 'ro')
#plot1.legend(loc='lower center',  prop={'size':10}, numpoints=1)
#plt.show()

## Maximize the hyper-parameter posterior
p_mapped2[0] = hyper[len(hyper)/2]
myK2.map_hyper(p_mapped2, unmap=True)
(myGPR2, param) = myGPR2.maximize_hyper_posterior()
print('Optimized value of the hyper-parameter:', param)
myK2.map_hyper(p_mapped2)
(hopt_post, hopt_grad) = myGPR2.hyper_posterior(param, p_mapped2)

# Infered points
(Yi2, Yi2std) = myGPR2(Xi1, infer_std=True)
(Yi2, Yi2std) = (Yi2.reshape(-1), Yi2std.reshape(-1))

# Plot
fig2 = plt.figure(figsize=(7,3), dpi=150)
plota = fig2.add_subplot(2,1,1)
plt.plot(Xd1,Yd1,'ko',label='Y1 data')
plt.plot(x_all,y_all,'k',label='Y1 func')
plt.plot(Xi1,Yi1,'-b',label='Mean')
plt.fill_between(Xi1, Yi1-Yi1std, Yi1+Yi1std, alpha=0.25)
plt.Rectangle((0.0, 0.0), 2.0, 50.0, facecolor='blue', alpha=0.25)
plt.legend(numpoints=1, loc='best', prop={'size':8})
plt.xlim([0.9,2.1])

plotb = fig2.add_subplot(2,1,2)
plt.plot(Xd1,Yd1,'ko')
plt.plot(x_all,y_all,'k')
plt.plot(Xi1,Yi2,'-b',label='Mean (max post-hyper)')
plt.fill_between(Xi1, Yi2-Yi2std, Yi2+Yi2std, alpha=0.25)
plt.Rectangle((0.0, 0.0), 2.0, 50.0, facecolor='green', alpha=0.25)
plt.legend(numpoints=1, loc='best', prop={'size':8})
plt.xlim([0.9,2.1])

plt.show()
