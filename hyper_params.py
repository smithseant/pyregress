# -*- coding: utf-8 -*-
"""
Docstring for the features module

Includes additional features for pyregress includeing prior distributions for
    hyper-parameters and a class for derivative inputs

Provided prior distributions (log(P) and d_log(P))
    Currently include log-Normal, Jeffreys', and constant distributions.
    Marginalized class also created to hold space of hyper-parameters that are 
    not being explored.
"""

from numpy import array, sum, divide, concatenate, squeeze, mean, amin, copy
from scipy import log, sqrt, pi
from scipy.special import gamma

class shift_to_zero:
    """shift data above zero and (optional) scale be mean value"""
    def __init__(self):
        self._shift = 0.
        self._scale = 1.
    
    def __call__(self,original_data,scale=False):
        data = copy(original_data)
        if (data < 0.).any():
            self._shift = amin(data)
            data -= self._shift
        if scale == True:
            self._scale = mean(data)
            data /= self._scale
        return data

    def reverse(self,data):
        return (data*self._scale)+self._shift


class derivative:
    """Derivative input class."""
    def __init__(self, *args):
        #deduce if three arays were passed for an array of three values
        self._derivative_position = 0
        
        if len(args) == 3:
            if isinstance(args[0],float):
                self._x    = array([args[0]])
                self._dy   = array([args[1]])
                self._xref = array([args[2]])
            else:
                self._x    = args[0]
                self._dy   = args[1]
                self._xref = args[2]
        # TODO: output error is more than 3 args
    def merge(self,x,y):
        self._derivative_position = len(y)
        new_x = concatenate((squeeze(x.T),self._x),axis=1)
        new_y = concatenate((squeeze(y),self._dy),axis=1)
        return new_x,new_y
        
    def separate(self,x,y):
        old_x = x[:self._derivative_position]
        old_y = y[:self._derivative_position]
        return old_x,old_y
    
"""Prior hyper-parameter distributions"""

class HyperPrior:
    """Base class for hyper-parameter prior distriubution classes
        HyperPrior subclasses provide different probability distributions
        and take the general form of:
        Init: 
            guess - inital guess for hyper-parameter
            args** - additional arguments that are distribtuion specific
        Call:
            Arguments
            ---------
                x - current hyper-parameter value
                grad - bool (optional), when grad is True also return dlnpdf,
                        and when grad is 'Hess' also return d2lnpdf.
            Returns
            -------
                lnprior - ln of hyper-parameter prior
                dlnprior - derivative of lnprior (optional, grad=True)
                d2lnprior - 2nd derivative of lnprior (optional, grad='Hess')
            """
    def __init__(self, guess=1.):
        self.guess = guess
    
    def __call__(self):
        pass
    

class LogNormal(HyperPrior):
    """Log Normal distribution class for hyper-parameter priors.
        f(x;mu,sigma) = 1/(x sigma sqrt(2 pi) *
                        exp(-(ln(x)-mu)^2/(2 sigma^2)) , x > 0"""
    def __init__(self, guess=1., **args):
        self.guess = guess
        if args.has_key( "mean" ) and args.has_key( "std" ):
            self._mu = log(args["mean"]**2/ 
                        (args["std"]**2 + args["mean"]**2))
            self._sigma = sqrt(log(1 + args["std"]**2/args["mean"]**2))   
    
    def auto_fill(self,Rk2):
        mean = sum(Rk2)/(Rk2 != 0).sum()
        std = sum((Rk2-mean)**2)/((Rk2 != 0).sum()-1.0)
        self._mu = log(mean**2/(std**2 + mean**2))
        self._sigma = sqrt(log(1 + std**2/mean**2))
        
    def __call__(self, x, grad=False):
        # TODO: if mean and std not defined, use auto_fill with Rk2      
        #if not hasattr(self,'mu'):
        #   self.auto_fill(self.Rk2)        
        
        twosigsqr = 2.0*self._sigma**2
        lnpdf = (-log(self._sigma*x)-0.5*log(2.0*pi)- 
                    (log(x)-self._mu)**2/twosigsqr)
        if not grad: 
            return lnpdf
        dlnpdf = (-self._mu + self._sigma**2 + 
                    log(x))/(self._sigma**2 * x)
        if grad == True:
            return lnpdf, dlnpdf
        if grad == 'Hess':             
            d2lnpdf = ((self._mu - self._sigma**2 -log(x)+1.) 
                        /(self._sigma*x)**2)
            return lnpdf, dlnpdf, d2lnpdf
            
class Jeffreys(HyperPrior):
    """Jefferys' distribution class for hyper-parameter priors.
        Non informative prior
        f(x) = 1/x"""
    def __call__(self, x, grad=False):
        lnpdf = -log(x)
        if not grad:
            return lnpdf
        dlnpdf = divide(-1.0,x)
        if grad == True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = divide(1.0,x**2)
            return lnpdf, dlnpdf, d2lnpdf
        
class Constant(HyperPrior):
    """Constant class for hyper-parameter
       f = const"""
    def __call__(self, x, grad=False):
        if not grad:
            return array([1.0])
        if grad == True:
            return array([1.0]),array([0.0])
        if grad == 'Hess': 
            return array([1.0]),array([0.0]),array([0.0])
            
class Beta(HyperPrior):
    """Beta distribution class for hyper-parameter priors.
        For parameters bounded by [0,2]
        f(x;a,b) = const * x^(a-1) * (1-x)^(b-1)"""
    def __init__(self, a, b, guess=1.):
        self.guess = guess
        self._a = a
        self._b = b
        
    def __call__(self, x, grad=False):
        x = x/2.
        a = self._a-1.
        b = self._b-1.
        lnpdf = a*log(x) + b*log(1.-x) + log(.5)
        if not grad:
            return lnpdf
        dlnpdf = a/x + b/(x-1.)
        if grad == True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = -a/x**2 -b/(x-1.)**2
            return lnpdf, dlnpdf, d2lnpdf
            
class Gamma(HyperPrior):
    """Gamma distribution class for hyper-parameter priors
        f(x;k,theta) = x^(k-1) * exp(-x/theta)/(theta^k * Gamma(k))"""
    def __init__(self, k, theta, guess=1.):
        self.guess = guess
        self._k = k
        self._theta = theta
        self._demonenator = k*log(theta) +  log(gamma(k))
        
    def __call__(self, x, grad=False):
        k = self._k - 1.
        lnpdf = k*log(x) -x/self._theta - self._demonenator
        if not grad:
            return lnpdf
        dlnpdf = k/x - 1./self._theta
        if grad == True:
            return lnpdf, dlnpdf
        if grad == 'Hess':
            d2lnpdf = -k/x**2
            return lnpdf, dlnpdf, d2lnpdf